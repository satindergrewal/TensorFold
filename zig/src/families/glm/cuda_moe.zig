//! GLM-5.3-Flash's MoE on CUDA, a rank's share of the Python family's moe_block: top-8 plus shared, grouped NVFP4.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");

/// TF_GLM_EXL3_MMA=0 falls back to the runMixed decode kernel (the A/B switch; default on when built).
var mma_cache: ?bool = null;
fn mmaOn() bool {
    if (mma_cache == null) {
        const v = std.c.getenv("TF_GLM_EXL3_MMA");
        mma_cache = !(v != null and std.mem.eql(u8, std.mem.span(v.?), "0"));
    }
    return mma_cache.?;
}
const experts = @import("cuda_experts.zig");
const x3 = @import("cuda_weights_exl3.zig");
const b16 = @import("cuda_b16.zig");


/// One MoE layer on a rank: router (bf16) and fp32 bias, routed experts, shared [gate | up] (or EXL3's two) and down.
pub const QuantW = struct { w: u64, s: u64, bits: u8 };

pub const Layer = struct {
    router: u64,
    bias: u64,
    routed: cuda.experts.Layer,
    count: usize,
    shared_gu: u64,
    shared_gate: ?cuda.qlinear.Weight = null,
    shared_up: ?cuda.qlinear.Weight = null,
    shared_down: b16.Proj,
    shared_down_mm: u64 = 0, // the tiled bf16 rows for glue.Mm16 (0: the b16 path)
    shared_down_q: ?QuantW = null, // affine-quantized (TF_GLM_DENSE_BITS): codes + scales + bits
    shared: usize,
    top_k: u32,
    // the mixed-width mma decode pass's per-expert flat scale tables (half): [experts][k or n]
    mma_suh_g: u64 = 0,
    mma_suh_u: u64 = 0,
    mma_svh_g: u64 = 0,
    mma_svh_u: u64 = 0,
    mma_suh_d: u64 = 0,
    mma_svh_d: u64 = 0,
    scale: f32,
    norm: bool,
};

/// The block's scratch for up to `rows` rows (forward.Buffers' MoE part).
pub const Scratch = struct {
    mlog: u64, // fp32 [rows][count]
    router_part: u64, // fp32 [router_ks][rows][count]
    pick: u64, // i32 [rows][top_k + 1]
    wts: u64, // fp32 [rows][top_k + 1]
    plan: cuda.grouped.Plan,
    eact: u64, // bf16 [rows (top_k + 1)][width]
    ey: u64, // [rows][top_k + 1][dims]: fp32, bf16 in a prompt chunk
    sgu: u64, // bf16 [rows][2 shared]
    sact: u64, // bf16 [rows][shared]
    sxs: u64, // fp32 [rows][shared / 64]
    sy: u64, // fp32 [rows][dims]
    mm_part: u64, // fp32 split-K partials of the shared expert's matmuls
    mma_xg: u64 = 0, // the mma decode pass: the pairs' gate rotation rows [pairs][hidden] half
    mma_xu: u64 = 0,
    mma_xd: u64 = 0, // the down pass's rotated activations [pairs][width] half
    mma_z: u64 = 0, // fp32 partials [mats][SK][pairs][n] of a pass
};

/// The kernels the block runs: the captured Triton set, the grouped NVFP4 experts, the plan, and glm_ops.
pub const Kernels = struct { set: glue.Set, experts: *const experts.Kernels, put_slot: cuda.Function };

/// forward.moe_block: out [rows][dims] fp32 partial; `prompt`: 64-pair items, the prompt kernels, bf16 slot rows.
pub fn block(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool, out: u64) !void {
    try body(k, stream, l, s, x, x_stride, rows, prompt);
    try fill(k, stream, l, s, x, x_stride, 0, rows, prompt, out);
}

/// The block up to its fill: the router, the top-k pick, the plan and the routed experts into the slot rows.
pub fn body(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, rows: usize, prompt: bool) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const shared_id: i32 = @intCast(l.count);
    try glue.router(k.set, stream, x, x_stride, l.router, s.mlog, s.router_part, rows, d, l.count);
    try glue.select(k.set, stream, s.mlog, l.bias, s.pick, s.wts, rows, l.top_k, l.count, l.scale, l.norm);
    // the mma pass reads 64-pair items once the window has more than a 16-pair item's rows (the amortization);
    // routing reads plan.prompt for its tile, and both kernel paths take any item count, so a wide decode plan
    // simply says prompt
    // TF_GLM_MMA_MT picks the mma item shape (16 default; 32/64 experiment — 64 regressed, kept for data)
    const mt: @import("cuda").exl3_experts.Experts.MmaTile = blk: {
        const v = std.c.getenv("TF_GLM_MMA_MT");
        const t = if (v) |p| std.mem.span(p) else "32";
        break :blk if (std.mem.eql(u8, t, "32")) .mt32 else if (std.mem.eql(u8, t, "64")) .mt64 else .mt16;
    };
    const wide = !prompt and mt != .mt16 and mmaOn() and l.mma_suh_g != 0 and s.mma_z != 0;
    const plan: experts.Plan = .{ .route = s.plan, .slots = slots, .experts = l.count + 1, .prompt = prompt };
    try k.experts.router.route(stream, s.pick, rows * slots, l.count + 1, if (prompt) experts.staged_tile else if (wide) switch (mt) {
        .mt32 => 32,
        .mt64 => experts.staged_tile,
        .mt16 => experts.tile,
    } else experts.tile, plan.route);
    try experts.route(k.experts, stream, s.pick, rows * slots, plan);
    const items = experts.maxItems(rows * slots, l.count + 1, switch (mt) {
        .mt16 => experts.tile,
        .mt32 => 32,
        .mt64 => experts.staged_tile,
    });
    if (!prompt and mmaOn() and l.mma_suh_g != 0 and s.mma_z != 0) {
        try k.experts.exl3.mmaMixed(stream, x, x_stride, rows, slots, s.pick, l.mma_suh_g, l.mma_suh_u, l.mma_svh_g, l.mma_svh_u, l.mma_suh_d, l.mma_svh_d, l.routed.mats_up, l.routed.mats_down, plan.route, items, s.mma_z, s.mma_xg, s.mma_xu, s.mma_xd, s.ey, l.routed.width, d, l.routed.limit, @as(c_uint, @intCast(@backingInt(l.routed.exl3.codebook))), shared_id, if (wide) mt else .mt16);
        return;
    }
    try experts.run(k.experts, stream, .gate_up, l.routed, plan, x, x_stride, rows, s.eact, shared_id);
    try experts.run(k.experts, stream, if (prompt) .down else .down_f32, l.routed, plan, s.eact, l.routed.width, rows, s.ey, shared_id);
}

/// moe_block's fill of rows lo..hi: shared expert into the last slot, slot sum into out fp32 (any range, same bits).
pub fn fill(k: Kernels, stream: cuda.Stream, l: Layer, s: Scratch, x: u64, x_stride: usize, lo: usize, hi: usize, prompt: bool, out: u64) !void {
    const d = l.routed.dims;
    const slots = l.top_k + 1;
    const n = hi - lo;
    const eb: usize = if (prompt) 2 else 4; // the slot rows: bf16 in a prompt chunk
    const gu = s.sgu + lo * 2 * l.shared * 2;
    if (l.shared_gate) |g| {
        // EXL3: gate and up apart, each through s.sact into its half of the [gate | up] rows
        for ([_]cuda.qlinear.Weight{ g, l.shared_up orelse return error.MissingSharedUp }, 0..) |w, j| {
            try b16.proj(k.set, stream, x + lo * x_stride * 2, x_stride, .{ .q = w }, s.sact + lo * l.shared * 2, s.mm_part, n, l.shared, d, false);
            try b16.copyRows(k.set, stream, s.sact + lo * l.shared * 2, l.shared * 2, gu + j * l.shared * 2, 2 * l.shared * 2, n, l.shared * 2);
        }
    } else try b16.matmul(k.set, stream, x + lo * x_stride * 2, x_stride, l.shared_gu, gu, s.mm_part, n, 2 * l.shared, d, false);
    try glue.swiglu(k.set, stream, s.sgu + lo * 2 * l.shared * 2, s.sact + lo * l.shared * 2, s.sxs + lo * (l.shared / 64) * 4, l.routed.limit, n, l.shared);
    if (l.shared_down_q) |qw| {
        try k.set.glue.mm.run(stream, s.sact + lo * l.shared * 2, l.shared, n, d, l.shared, qw.w, qw.s, qw.bits, s.sy + lo * d * 4, s.mm_part, true);
    } else if (!prompt and l.shared_down_mm != 0 and glue.mm16On()) {
        // DECODE-ONLY: the standalone checks (mm16-check, live-call diff, graph capture) are all
        // bit-exact, but prompt fills (pieced, captured) corrupt long-prompt replies - until that
        // is root-caused, mm16 runs decode windows only, where it is verified exact
        try k.set.glue.mm.run(stream, s.sact + lo * l.shared * 2, l.shared, n, d, l.shared, l.shared_down_mm, 0, 16, s.sy + lo * d * 4, s.mm_part, true);
    } else try b16.proj(k.set, stream, s.sact + lo * l.shared * 2, l.shared, l.shared_down, s.sy + lo * d * 4, s.mm_part, n, d, l.shared, true);
    var a: cuda.Args = .{};
    a.add(s.sy + lo * d * 4);
    a.add(s.ey + lo * slots * d * eb);
    for ([_]usize{ n, d, slots, l.top_k, @intFromBool(prompt) }) |v| a.add(@as(c_int, @intCast(v)));
    try cuda.launch.launch(k.put_slot, .{ .grid = .{ .x = @intCast((n * d + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    try glue.combine(k.set, stream, s.ey + lo * slots * d * eb, !prompt, s.wts + lo * slots * 4, out, n, d, slots);
}
