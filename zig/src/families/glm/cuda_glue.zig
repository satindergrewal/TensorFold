//! GLM-5.3-Flash's glue on CUDA: the Python family's cuda/glue.py on glm_glue.cu, whose kernels keep Triton's bytes.

const std = @import("std");
const cuda = @import("cuda");

/// What the forward's kernels launch from: our glue's functions.

/// The dense linear on tensor cores for BF16 weights (ported from pr/glm-flash-cuda's glm/mm.cu,
/// BITS=16): weights tiled at load into mma B-fragment order (glm_mm_tile16), one kernel per row
/// bucket, the K slices' partials reduced in order. TF_GLM_MM16=0 falls back to the b16 path.
pub const Mm16 = struct {
    k16: cuda.Function,
    k32: cuda.Function,
    k64: cuda.Function,
    q4_16: cuda.Function = undefined,
    q4_32: cuda.Function = undefined,
    q4_64: cuda.Function = undefined,
    q8_16: cuda.Function = undefined,
    q8_32: cuda.Function = undefined,
    q8_64: cuda.Function = undefined,
    reduce: cuda.Function,
    tile16: cuda.Function,
    tile4: cuda.Function = undefined,
    tile8: cuda.Function = undefined,

    pub fn load(m: cuda.Module) !Mm16 {
        var r: Mm16 = .{
            .k16 = try m.function("glm_mm16_16"),
            .k32 = try m.function("glm_mm16_32"),
            .k64 = try m.function("glm_mm16_64"),
            .reduce = try m.function("glm_mm_reduce"),
            .tile16 = try m.function("glm_mm_tile16"),
        };
        r.q4_16 = m.function("glm_mm4_16") catch r.k16;
        r.q4_32 = m.function("glm_mm4_32") catch r.k32;
        r.q4_64 = m.function("glm_mm4_64") catch r.k64;
        r.q8_16 = m.function("glm_mm8_16") catch r.k16;
        r.q8_32 = m.function("glm_mm8_32") catch r.k32;
        r.q8_64 = m.function("glm_mm8_64") catch r.k64;
        r.tile4 = m.function("glm_mm_tile4") catch r.tile16;
        r.tile8 = m.function("glm_mm_tile8") catch r.tile16;
        return r;
    }

    /// out [rows][n] (fp32 with `wide`, else bf16) = x [rows][k] rows `ldx` apart times the tiled w.
    /// Their launch contract verbatim (kern.zig mm()): the row bucket picks the instance, the grid
    /// is (n/64, rows/bm), dynamic shared = 3*(bm*128 + 8*32*8*4 + 128); SK=1 writes out directly.
    /// bits: 16 = the bf16 path (w is tiled bf16, s unused); 4/8 = affine (w = tiled codes, s = tiled scales).
    pub fn run(mm: *const Mm16, stream: cuda.Stream, x: u64, ldx: usize, rows: usize, n: usize, k: usize, w: u64, s: u64, bits: u8, out: u64, part: u64, wide: bool) !void {
        _ = part;
        if (k % 64 != 0 or n % 8 != 0) return error.Mm16Shape;
        const bm: usize = if (rows <= 16) 16 else if (rows <= 32) 32 else if (rows <= 64) 64 else 128;
        const wpc: usize = if (bits == 16) 8 else bits / 2;
        const f = switch (bm) {
            16 => if (bits == 4) mm.q4_16 else if (bits == 8) mm.q8_16 else mm.k16,
            32 => if (bits == 4) mm.q4_32 else if (bits == 8) mm.q8_32 else mm.k32,
            else => if (bits == 4) mm.q4_64 else if (bits == 8) mm.q8_64 else mm.k64,
        };
        const smem: u32 = @intCast(3 * (bm * 128 + 8 * 32 * wpc * 4 + 128));
        try f.allowDynamicShared(smem);
        var a: cuda.Args = .{};
        a.add(x);
        a.add(@as(i64, @intCast(ldx)));
        a.add(w);
        a.add(s);
        a.add(out);
        a.add(out); // part: unused at SK 1
        a.add(@as(c_int, @intFromBool(wide)));
        a.add(@as(c_int, @intCast(rows)));
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        a.add(@as(c_int, 1));
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast((n + 63) / 64), .y = @intCast((rows + bm - 1) / bm), .z = 1 }, .block = .{ .x = if (bm <= 32) 128 else 256 }, .shared = smem }, stream, &a);
    }

    /// Tiling at load: bf16 rows [n][k] device-uploaded at `src` -> the tiled chunks at `dst` (same size).
    pub fn tile(mm: *const Mm16, stream: cuda.Stream, src: u64, dst: u64, n: usize, k: usize) !void {
        const chunks = (n / 8) * (k / 64) * 32;
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(@as(c_int, @intCast(n)));
        a.add(@as(c_int, @intCast(k)));
        try cuda.launch.launch(mm.tile16, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    }
};

/// TF_GLM_MM16=1 opts the tiled-bf16 tensor-core path in (the shared expert's down projection and
/// the KDA out projection): faster (+2.5-5%) but GSM-class accuracy drops (5/8 vs 8/8 spot) - a
/// per-workload speed-vs-accuracy choice, default off.
pub fn mm16On() bool {
    if (mm16_on == null) {
        const v = std.c.getenv("TF_GLM_MM16");
        mm16_on = v != null and std.mem.eql(u8, std.mem.span(v.?), "1");
    }
    return mm16_on.?;
}

var mm16_on: ?bool = null;

pub const Set = struct { glue: *const Kernels, linear: cuda.qlinear.Linear = .{}, copy_rows: ?cuda.Function = null, bf16_kv: bool = false };

/// Our glm_glue.cu's functions.
pub const Kernels = struct {
    tree_mask: u64 = 0,
    keep_rows: cuda.Function,
    mm: Mm16 = undefined,
    rmsnorm_4096: cuda.Function,
    rmsnorm_1536: cuda.Function,
    rmsnorm_512: cuda.Function,
    residual_add: cuda.Function,
    stream_mean: cuda.Function,
    rank_sum: cuda.Function,
    hc_post: cuda.Function,
    combine: cuda.Function,
    slice_sum: cuda.Function,
    top1: cuda.Function,
    top2: cuda.Function,
    pick_top2: cuda.Function,
    top1_rows: cuda.Function,
    pick_greedy: cuda.Function,
    bell_wait: cuda.Function,
    put_words: cuda.Function,
    embed_b16: cuda.Function,
    conv_shift: cuda.Function,
    swiglu_512: cuda.Function,
    swiglu_64: cuda.Function,
    group_sums: cuda.Function,
    topk: cuda.Function,
    router_part: cuda.Function,
    router_tile: cuda.Function,
    hc_partial: cuda.Function,
    hc_partial8: cuda.Function,
    hc_finish: cuda.Function,
    absorb_q: cuda.Function,
    expand_v: cuda.Function,
    lat_write: cuda.Function,
    lat_write_fp8: ?cuda.Function = null,
    chunks16_fp8: ?cuda.Function = null,
    lat_write_nvfp4: ?cuda.Function = null,
    chunks16_nvfp4: ?cuda.Function = null,
    chunks32_nvfp4: ?cuda.Function = null,
    chunks_16_s32: ?cuda.Function = null, // the 32-row sliced decode form (bf16)
    l2pf: ?cuda.Function = null, // TF_GLM_L2PF: the side-stream weight-prefetch kernel
    chunks16_nvfp4_s32: ?cuda.Function = null,
    lat_dequant_nvfp4: ?cuda.Function = null,
    chunks32_fp8: ?cuda.Function = null,
    lat_dequant: ?cuda.Function = null,
    lat_dequant_tok: ?cuda.Function = null,
    index_write: cuda.Function,
    pool_keys: cuda.Function,
    select_rows: cuda.Function,
    bmm_decode: cuda.Function,
    bmm_rows: cuda.Function,
    bmm_tile: cuda.Function,
    bmm_tile1: cuda.Function,
    bmm_d32s1: cuda.Function,
    bmm_d32s2: cuda.Function,
    bmm_d32s3: cuda.Function,
    bmm_d64s2: cuda.Function,
    chunks_16: cuda.Function,
    chunks_32: cuda.Function,
    merge: cuda.Function,
    scores: cuda.Function,

    pub fn load(m: cuda.Module) !Kernels {
        var k: Kernels = undefined;
        inline for (@typeInfo(Kernels).@"struct".field_names) |name| {
            if (comptime !std.mem.eql(u8, name, "mm") and !std.mem.eql(u8, name, "l2pf") and !std.mem.eql(u8, name, "tree_mask") and !std.mem.endsWith(u8, name, "_fp8") and !std.mem.endsWith(u8, name, "_nvfp4") and !std.mem.endsWith(u8, name, "_s32") and !std.mem.startsWith(u8, name, "lat_dequant")) @field(k, name) = try m.function("tf_glm_" ++ name);
        }
        k.mm = try Mm16.load(m);
        // KV cache quantization kernels: load what exists, skip what doesn't
        const kv_fp8 = std.c.getenv("TF_GLM_KV_FP8") != null;
        const kv_nvfp4 = std.c.getenv("TF_GLM_KV_NVFP4") != null;
        if (kv_fp8) {
            k.lat_write_fp8 = m.function("tf_glm_lat_write_fp8") catch null;
            k.lat_dequant = m.function("tf_glm_lat_dequant") catch null;
            k.lat_dequant_tok = m.function("tf_glm_lat_dequant_tok") catch null;
        }
        if (kv_nvfp4) {
            k.lat_write_nvfp4 = m.function("tf_glm_lat_write_nvfp4") catch null;
            k.lat_dequant_nvfp4 = m.function("tf_glm_lat_dequant_nvfp4") catch null;
            k.chunks16_nvfp4 = m.function("tf_glm_chunks16_nvfp4") catch null;
            k.chunks32_nvfp4 = m.function("tf_glm_chunks32_nvfp4") catch null;
            // a device that refuses the opt-in (sm_120's 99KB) leaves the form null: the other head
            // width serves those launches, or the dequant path takes over
            if (k.chunks16_nvfp4) |*f| f.allowDynamicShared(chunksSmemNvfp4(16)) catch {
                k.chunks16_nvfp4 = null;
            };
            if (k.chunks32_nvfp4) |*f| f.allowDynamicShared(chunksSmemNvfp4(32)) catch {
                k.chunks32_nvfp4 = null;
            };
            k.chunks16_nvfp4_s32 = m.function("tf_glm_chunks16_nvfp4_s32") catch null;
            if (k.chunks16_nvfp4_s32) |*f| f.allowDynamicShared(chunksSmemNvfp4Slice()) catch {
                k.chunks16_nvfp4_s32 = null;
            };
        }
        try k.bmm_tile.allowDynamicShared(bmm_tile_smem);
        try k.bmm_tile1.allowDynamicShared(bmm_tile1_smem[1]);
        k.chunks_16_s32 = m.function("tf_glm_chunks_16_s32") catch null;
        k.l2pf = m.function("tf_glm_l2pf") catch null;
        if (k.chunks_16_s32) |*f| f.allowDynamicShared(chunksSmem(16)) catch {
            k.chunks_16_s32 = null;
        };
        try k.chunks_16.allowDynamicShared(chunksSmem(16));
        try k.chunks_32.allowDynamicShared(chunksSmem(32));
        return k;
    }
};

/// An element-wise kernel's launch: a block of 256 columns, `rows` x the blocks to cover `d`.
fn rowsLaunch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize) !void {
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 256) }, .block = .{ .x = 256 } }, stream, a);
}

/// As above, 8 columns a thread (16-byte loads), 1024 a block of 128 (Triton's BLOCK); d % 8 == 0, rows 16B-aligned.
fn rows8Launch(f: cuda.Function, stream: cuda.Stream, a: *cuda.Args, rows: usize, d: usize, ptrs: anytype) !void {
    if (d % 8 != 0) return error.WidthNotEights;
    inline for (ptrs) |p| if (p % 16 != 0) return error.RowsUnaligned;
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(d, 1024) }, .block = .{ .x = 128 } }, stream, a);
}

fn i64of(v: usize) i64 {
    return @intCast(v);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

fn pow2(d: usize) i64 {
    return @intCast(std.math.ceilPowerOfTwoAssert(usize, d));
}

/// glue.rmsnorm: bf16(w * bf16(x / sqrt(mean(x^2) + eps))) into `out`, with `xs` each 64 outputs' fp32 sum.
pub fn rmsnorm(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, o_stride: usize, xs: ?u64, eps: f32, rows: usize, d: usize) !void {
    const f, const warps: u32 = switch (d) {
        4096 => .{ set.glue.rmsnorm_4096, 8 },
        1536 => .{ set.glue.rmsnorm_1536, 4 },
        512 => .{ set.glue.rmsnorm_512, 4 },
        else => return error.NoRmsnormWidth,
    };
    var a: cuda.Args = .{};
    a.add(x);
    a.add(@as(i64, @intCast(x_stride)));
    a.add(w);
    a.add(out);
    a.add(@as(i64, @intCast(o_stride)));
    a.add(xs orelse @as(u64, 0));
    a.add(eps);
    a.add(@as(c_int, @intFromBool(xs != null)));
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = warps * 32 } }, stream, &a);
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// swiglu's column block: 512 where it divides the width, else 64 (TP=3's 704 and 640, the latter 128 in glue.py).
fn swigluBlock(w: usize) !usize {
    if (w % 512 == 0) return 512;
    if (w % 64 == 0) return 64;
    return error.NoSwigluBlock;
}

/// glue.swiglu: [gate | up] -> bf16(bf16(silu(min(g, L))) * clip(u, -L, L)), 64-sums in `xs`.
pub fn swiglu(set: Set, stream: cuda.Stream, gu: u64, out: u64, xs: u64, limit: f32, rows: usize, w: usize) !void {
    const block = try swigluBlock(w);
    const f, const threads: u32 = if (block == 512) .{ set.glue.swiglu_512, 128 } else .{ set.glue.swiglu_64, 64 };
    var a: cuda.Args = .{};
    inline for (.{ gu, out, xs, limit, i64of(w) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(w / block) }, .block = .{ .x = threads } }, stream, &a);
}

/// glm_glue_bmm.cuh's Tile::smem(f32) of tf_glm_bmm_tile: 3 stages of 128 x and 64 w rows, 64 bf16 a row.
pub const bmm_tile_smem: u32 = 3 * (128 + 64) * 64 * 2;
/// tf_glm_bmm_tile1's [bf16, fp32 out]: 3 stages of 128 x and 128 w rows, 32 bf16 a row, or the fp32 tile if larger.
pub const bmm_tile1_smem = [2]u32{ 3 * (128 + 128) * 32 * 2, 128 * (128 + 4) * 4 };
/// glm_glue_attn.cuh's wide_smem<hb>(): q (hb x 512), two 32-key tiles, p (hb x 32) in bf16, then max/sum scratch.
pub fn chunksSmem(comptime hb: u32) u32 {
    return (hb * 512 + 2 * 32 * 512 + hb * 32) * 2 + (hb * 4 + 2 * hb) * 4;
}

/// TF_GLM_L2PF: prefetch `n` pieces of the table on `side` (the caller forks it from the main stream first).
pub fn l2pf(set: Set, side: cuda.Stream, table: u64, first: usize, n: usize) !void {
    const f = set.glue.l2pf orelse return error.NoL2pf;
    if (n == 0) return;
    var a: cuda.Args = .{};
    a.add(table + first * 16);
    a.add(@as(c_int, @intCast(n)));
    const blocks: u32 = @intCast(@min(48, (n + 255) / 256));
    try cuda.launch.launch(f, .{ .grid = .{ .x = blocks }, .block = .{ .x = 256 } }, side, &a);
}

/// NVFP4 chunks kernels add a double-buffered staging tile of the packed e2m1 halves (2 * 32 * 256
/// bytes) and the 16-entry bf16 LUT.
pub fn chunksSmemNvfp4(comptime hb: u32) u32 {
    return chunksSmem(hb) + 2 * 32 * 256 + 32;
}

/// The 32-row sliced form stages one tile: the packed e2m1 halves, the 32 B/row of e4m3 scales, the LUT.
pub fn chunksSmemNvfp4Slice() u32 {
    return chunksSmem(16) + 32 * 256 + 32 * 32 + 32;
}

/// K slices of the router's dot (glue.ROUTER_KS): fixed by the shape, as the sliced and the single pass differ in bits.
pub const router_ks = 8;

/// glue.router: out [m][ne] fp32 = x . w rows; d a multiple of router_ks x 64: K slices in `part`, added in order.
pub fn router(set: Set, stream: cuda.Stream, x: u64, x_stride: usize, w: u64, out: u64, part: u64, m: usize, d: usize, ne: usize) !void {
    if (d % (router_ks * 64) != 0) return error.RouterWidth; // the Python family's single-pass _router: no GLM width
    var a: cuda.Args = .{};
    inline for (.{ x, w, part, i64of(m), i64of(x_stride), i64of(d), i64of(ne), @as(c_int, router_ks) }) |v| a.add(v);
    if (x % 16 == 0 and x_stride % 8 == 0) // rows on 16 bytes: 64-row tiles staged by cp.async (decode's rows past m load as zeros)
        try cuda.launch.launch(set.glue.router_tile, .{ .grid = .{ .x = cdiv(m, 64), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a)
    else
        try cuda.launch.launch(set.glue.router_part, .{ .grid = .{ .x = cdiv(m, 16), .y = cdiv(ne, 32), .z = router_ks }, .block = .{ .x = 128 } }, stream, &a);
    try sliceSum(set, stream, part, out, m * ne, router_ks, true);
}

/// glue.select: top_k by sigmoid + bias (lower id on ties), normed if `norm`, times `scale`, then the shared expert.
pub fn select(set: Set, stream: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, top_k: usize, experts: usize, scale: f32, norm: bool) !void {
    if (experts > 512 or top_k >= 32) return error.TopkTooWide;
    var a: cuda.Args = .{};
    inline for (.{ logits, bias, pick, wts, scale, cint(experts), cint(top_k), @as(c_int, @intFromBool(norm)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.topk, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &a);
}

/// glue.combine: out [rows][d] fp32 = sum over slots in order of wts[r][k] y[r][k] (y fp32 in decode, bf16 in prefill).
pub fn combine(set: Set, stream: cuda.Stream, y: u64, y_f32: bool, wts: u64, out: u64, rows: usize, d: usize, slots: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ y, @as(c_int, @intFromBool(y_f32)), wts, out, i64of(d), cint(slots) }) |v| a.add(v);
    try rows8Launch(set.glue.combine, stream, &a, rows, d, .{ y, out });
}

/// K blocks of the hyper-connection mixing dots (glue.HC_BLOCKS): fixed by the shape, like the router's slices.
pub const hc_blocks = 16;

/// The mixing kernels' scratch: rows x hc_blocks x 32 fp32 (24 dots and the sum of squares of each K block).
pub fn hcPartBytes(rows: usize) usize {
    return rows * hc_blocks * 32 * 4;
}

/// glue.hc_pre: streams [rows][s d] -> the collapsed RMS-normed row and its 64-sums `xs`; `post`, `comb` (Sinkhorn).
pub fn hcPre(set: Set, stream: cuda.Stream, x: u64, fn_w: u64, base: u64, scale: u64, norm_w: u64, out: u64, xs: u64, post: u64, comb: u64, part: u64, eps: f32, hc_eps: f32, iters: usize, rows: usize, d: usize, s: usize) !void {
    if (d != 4096 or s != 4) return error.HcShape;
    var a: cuda.Args = .{};
    inline for (.{ x, fn_w, part, i64of(s * d), @as(c_int, hc_blocks) }) |v| a.add(v);
    // Decode windows (16 blocks a row) take each block's 8 steps at once; prompt rows 4 rows a 128-thread block.
    if (rows < 64) {
        try cuda.launch.launch(set.glue.hc_partial8, .{ .grid = .{ .x = @intCast(rows), .y = hc_blocks }, .block = .{ .x = 1024 } }, stream, &a);
    } else {
        a.add(i64of(rows));
        try cuda.launch.launch(set.glue.hc_partial, .{ .grid = .{ .x = cdiv(rows, 4), .y = hc_blocks }, .block = .{ .x = 128 } }, stream, &a);
    }
    var b: cuda.Args = .{};
    inline for (.{ x, part, base, scale, norm_w, out, xs, post, comb, eps, hc_eps, @as(c_int, hc_blocks), cint(iters) }) |v| b.add(v);
    try cuda.launch.launch(set.glue.hc_finish, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, stream, &b);
}

/// glue.hc_post: branch = bf16(`g`, rank 0 first); xout[s] = bf16(post[s] branch + sum_t comb[t][s] x[t]); x may alias.
pub fn hcPost(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, post: u64, comb: u64, rows: usize, d: usize, s: usize, world: usize) !void {
    if (s != 4) return error.StreamsNotFour;
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, post, comb, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(set.glue.hc_post, stream, &a, rows, d, .{ x, xout, g });
}

/// glue.rank_sum: out bf16 = partials summed rank 0 first, `me`'s from `own`, others' from `peers` in rank order.
pub fn rankSum(set: Set, stream: cuda.Stream, own: u64, peers: u64, peer_stride: usize, out: u64, me: usize, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ own, peers, out, i64of(peer_stride), i64of(d), cint(world), cint(me) }) |v| a.add(v);
    try rows8Launch(set.glue.rank_sum, stream, &a, rows, d, .{ own, peers, out });
}

/// glue.residual_add (the MTP block's plain residual): xout = bf16(x + bf16(the partials `g` summed rank 0 first)).
pub fn residualAdd(set: Set, stream: cuda.Stream, x: u64, xout: u64, g: u64, rows: usize, d: usize, world: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, xout, g, i64of(rows * d), i64of(d), cint(world) }) |v| a.add(v);
    try rows8Launch(set.glue.residual_add, stream, &a, rows, d, .{ x, xout, g });
}

/// glue.stream_mean: out [rows][d] = bf16 of the s streams' mean (the final collapse).
pub fn streamMean(set: Set, stream: cuda.Stream, x: u64, out: u64, rows: usize, d: usize, s: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ x, out, i64of(d), cint(s) }) |v| a.add(v);
    try rows8Launch(set.glue.stream_mean, stream, &a, rows, d, .{ x, out });
}

/// glue.embed of a BF16 table: each row's token (ids i32) as `copies` streams of d values, out [rows][copies d].
pub fn embedB16(set: Set, stream: cuda.Stream, ids: u64, table: u64, out: u64, rows: usize, d: usize, copies: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ ids, table, out, i64of(d), cint(copies) }) |v| a.add(v);
    try rowsLaunch(set.glue.embed_b16, stream, &a, rows, d);
}

/// forward._shift_conv: each layer's 3-row conv window becomes rows keep .. keep + 2 of [window; projection rows].
pub fn convShift(set: Set, stream: cuda.Stream, conv: u64, proj: u64, keep: usize, proj_layer: usize, proj_row: usize, layers: usize, c: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ conv, proj, i64of(keep), i64of(3 * c), i64of(proj_layer), i64of(proj_row), i64of(c) }) |v| a.add(v);
    try rowsLaunch(set.glue.conv_shift, stream, &a, layers, c);
}

/// glue._router_sum and qmm._reduce: `total` values summed over `ks` slices in slice order, fp32 or bf16 out.
pub fn sliceSum(set: Set, stream: cuda.Stream, part: u64, out: u64, total: usize, ks: usize, wide: bool) !void {
    var a: cuda.Args = .{};
    inline for (.{ part, out, i64of(total), cint(ks), @as(c_int, @intFromBool(wide)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.slice_sum, .{ .grid = .{ .x = cdiv(total, 1024) }, .block = .{ .x = 128 } }, stream, &a);
}

test "the norm's block is the next power of two of the width" {
    try std.testing.expectEqual(@as(i64, 4096), pow2(4096));
    try std.testing.expectEqual(@as(i64, 8192), pow2(6144));
}

test "the wrappers' shape rules as glue.py's" {
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(6144)); // swiglu's block at the dense TP=2 width
    try std.testing.expectEqual(@as(usize, 512), try swigluBlock(1024)); // a routed expert's rows at TP=2
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(704)); // TP=3's first two ranks
    try std.testing.expectEqual(@as(usize, 64), try swigluBlock(640)); // TP=3's last rank
    try std.testing.expectError(error.NoSwigluBlock, swigluBlock(96));
    try std.testing.expectEqual(@as(i64, 512), pow2(288 + 1)); // _topk's BLOCK for 288 experts
    try std.testing.expectEqual(@as(i64, 16), pow2(8 + 1)); // its SLOTP for top-8 and the shared slot
}