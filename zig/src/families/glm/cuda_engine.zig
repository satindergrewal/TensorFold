//! The GLM CUDA engine: weights, kernels and one window's buffers; the
//! backbone walk (hc boundary, KDA or MLA attention, hc boundary, dense or MoE
//! MLP), the head. Built block-ranged for the box's two-GPU split; this file
//! is the single-device walk.

const std = @import("std");
const cuda = @import("cuda");
const exl3 = cuda.exl3;
const cfg = @import("config.zig");
const weights = @import("cuda_weights.zig");
const hc = @import("cuda_hc.zig");
const glue = @import("glm_glue.zig");
const kda = @import("kda.zig");

const Config = cfg.Config;

pub const Options = struct {
    blocks_lo: usize = 0,
    blocks_hi: usize = 1 << 16,
    context: usize = 4096, // the KV caches' capacity
    skip_mla: bool = false, // bring-up: attention passes through with a zero branch
    cont: bool = false, // the streams/branch/post/comb staged; skip embed
    stop_after_blocks: bool = false, // leave the walk mid-model for the handoff
    dsa: ?@import("dsa.zig").Kernels = null,
    index: ?@import("index.zig").Kernels = null,
// bring-up: attention passes through with a zero branch
};

/// One GEMM shape's plan + the shared workspace.
const Gemms = struct {
    lt: glue.Library,
    map: std.AutoHashMap(u64, cuda.cublaslt.Linear),
    ws: cuda.DeviceBuffer,

    fn key(m: u64, n: u64, k: u64, f32_out: bool) u64 {
        return (m << 34) | (n << 14) | k | (@as(u64, @intFromBool(f32_out)) << 62);
    }

    fn init(gpa: std.mem.Allocator, d: *const cuda.Driver) !Gemms {
        return .{
            .lt = try glue.Library.open(),
            .map = std.AutoHashMap(u64, cuda.cublaslt.Linear).init(gpa),
            .ws = try cuda.DeviceBuffer.alloc(d, 64 << 20),
        };
    }

    fn deinit(g: *Gemms) void {
        var it = g.map.valueIterator();
        while (it.next()) |l| l.deinit();
        g.map.deinit();
        g.ws.free();
        g.lt.close();
    }

    /// D . W^T with independent row strides (bytes) for x, W and out.
    fn mmStrided(g: *Gemms, s: cuda.Stream, x: u64, xs: usize, w: u64, ws_: usize, out: u64, os: usize, rows: usize, n: usize, k: usize) !void {
        // a dedicated Linear per stride combo
        const m: usize = if (rows == 1 and mpad_on) 2 else rows; // 1-row pads to the 2-row plan (spec-decode bit parity)
        const id = key(m | (xs << 8), n, k, false) ^ (ws_ << 20) ^ (os << 26);
        const gop = try g.map.getOrPut(id);
        if (!gop.found_existing) {
            // build with custom layouts (the shared init hardcodes ld=k)
            var lin: cuda.cublaslt.Linear = undefined;
            lin = try cuda.cublaslt.Linear.initStrided(&g.lt, m, n, k, .bf16, 64 << 20, xs / 2, ws_ / 2, os / 2);
            gop.value_ptr.* = lin;
        }
        gop.value_ptr.run(x, w, out, g.ws.ptr, g.ws.len, s.handle) catch |err| {
            std.debug.print("mmS FAIL rows={d} n={d} k={d}" ++ "\n", .{ rows, n, k });
            return err;
        };
    }

    /// D[rows, n] = X[rows, k] . W[k, n] (W UNtransposed: row-major x @ W).
    fn mmNT(g: *Gemms, s: cuda.Stream, x: u64, xs: usize, w: u64, ws_: usize, out: u64, os: usize, rows: usize, n: usize, k: usize) !void {
        const m: usize = if (rows == 1 and mpad_on) 2 else rows; // 1-row pads to the 2-row plan (spec-decode bit parity)
        const id = key(m | (xs << 8), n, k, false) ^ (ws_ << 20) ^ (os << 26) ^ 0x5A5A;
        const gop = try g.map.getOrPut(id);
        if (!gop.found_existing)
            gop.value_ptr.* = try cuda.cublaslt.Linear.initNT(&g.lt, m, n, k, .bf16, 64 << 20, xs / 2, ws_ / 2, os / 2);
        gop.value_ptr.run(x, w, out, g.ws.ptr, g.ws.len, s.handle) catch |err| {
            std.debug.print("mmNT FAIL rows={d} n={d} k={d}" ++ "\n", .{ rows, n, k });
            return err;
        };
    }

    /// D[rows, n] bf16 (or fp32) = X[rows, k] . W[n, k]^T.
    fn mm(g: *Gemms, s: cuda.Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize, f32_out: bool) !void {
        // m=1 pads to an m=2 plan: 1-row and 2-row windows then share GEMM bits
        // for row 0, so speculative verify rounds agree with plain greedy rounds
        const m: usize = if (rows == 1 and mpad_on) 2 else rows;
        const id = key(m, n, k, f32_out);
        const gop = try g.map.getOrPut(id);
        if (!gop.found_existing)
            gop.value_ptr.* = try cuda.cublaslt.Linear.init(&g.lt, m, n, k, if (f32_out) .f32 else .bf16, 64 << 20);
        gop.value_ptr.run(x, w, out, g.ws.ptr, g.ws.len, s.handle) catch |err| {
            std.debug.print("mm FAIL rows={d} n={d} k={d} f32={}" ++ "\n", .{ rows, n, k, f32_out });
            return err;
        };
    }
};

pub var mpad_on: bool = true; // perf probe: the rows==1->m=2 spec-parity padding
pub var group_stage: u8 = 3; // bisect: 1=gate grouped only, 3=all

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    block_sums: ?[]f64 = null, // set by the test: per-block x-stream fingerprints
    cur_bi: usize = 0,
    kda_st_cur: []usize = &.{},
    attn_sums: ?[]f64 = null,
    fwd_count: usize = 0,
    dump_epoch: u8 = 1, // resetState bumps: window-2 probe dumps get a 2 suffix
    quiet: bool = false, // serving: no probe dumps or debug prints
    dump_on: bool = false, // test: dump this window's probes regardless of fwd_count
    hc_calls: usize = 0, // probe: counts hcGates calls within a forward
    dump_serial: usize = 0, // probe: monotonic per-write serial; filenames carry it (no pairing ambiguity)
    f32_streams: bool = true, // set by the test: stream means at their L-points
    d: *const cuda.Driver,
    stream: cuda.Stream,
    c: Config,
    w: *weights.Weights,
    ex: exl3.Kernels,
    hck: hc.Kernels,
    ops: glue.Ops,
    kdk: kda.Kernels,
    g: Gemms,
    // one window's buffers
    rows_max: usize,
    ids: cuda.DeviceBuffer, // u32 [rows_max]
    x: [2]cuda.DeviceBuffer, // the hc streams [rows, 4, hidden] bf16
    xi: usize = 0,
    branch: cuda.DeviceBuffer, // [rows, hidden]
    normed: cuda.DeviceBuffer, // [rows, hidden]
    part: cuda.DeviceBuffer, // hc partials [rows, 16, 25] f32
    post: cuda.DeviceBuffer, // [rows, 4] f32
    comb: cuda.DeviceBuffer, // [rows, 16] f32
    y: cuda.DeviceBuffer, // [rows, hidden] the attention/MLP outs
    proj: cuda.DeviceBuffer, // the KDA stacked projections [rows, kdaProj]
    qkv: cuda.DeviceBuffer, // the conv'd [rows, 3W]
    fg: cuda.DeviceBuffer, // f_b/g_b outs [rows, 2W]
    bv: cuda.DeviceBuffer, // b out [rows, H]
    kda_st: [2][]f32, // host staging? no - device below
    kda_st0: cuda.DeviceBuffer, // [n_kda, H, D, D] fp32
    kda_st1: cuda.DeviceBuffer,
    kda_cs: cuda.DeviceBuffer, // [n_kda, TAPS-1, 3W] bf16 history
    kda_cw: cuda.DeviceBuffer, // [TAPS, 3W] bf16, transposed at load
    kda_conv_p: cuda.DeviceBuffer, // [rows, 3W]: the conv reads cs history + p
    logits: cuda.DeviceBuffer, // [rows, vocab] bf16
    picks: cuda.DeviceBuffer, // u32 [rows]
    eps_buf: cuda.DeviceBuffer,
    fn_zero: cuda.DeviceBuffer,
    moe_pick: cuda.DeviceBuffer,
    moe_pick_hb: cuda.HostBuffer = undefined,
    moe_pick_h: []u8 = &.{},
    // grouped-expert path (rows==1): per-slot outputs + tables per family
    moe_xt: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true }, // [topk, k_max]
    moe_yg: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true }, // [topk, inter_max]
    moe_yu: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    moe_act_g: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    moe_tbl_gate: u64 = 0,
    moe_tbl_up: u64 = 0,
    moe_tbl_down: u64 = 0,

    // #741 mixed path
    m41_on: bool = false,
    m41_stage_n: u8 = 0,
    m41_rounds: usize = 0,
    m41_up: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_up_h: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_down: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_down_h: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_desc_up: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_desc_down: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_xe: ?@import("cuda").exl3_experts.Experts = null,
    m41_router: ?@import("cuda").grouped.Router = null,
    m41_router_mod: ?cuda.Module = null,
    m41_members: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_items: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_counts: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_rank: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_hist: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_out: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_act: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    m41_inter_max: usize = 0,

    moe_k_blocks: usize = 0,
    moe_tn_i: usize = 0,
    moe_tn_h: usize = 0,
    moe_inter_max: usize = 0,
    moe_wts: cuda.DeviceBuffer,
    moe_logits: cuda.DeviceBuffer,
    moe_ys: cuda.DeviceBuffer,
    moe_ye: cuda.DeviceBuffer,
    moe_gu: cuda.DeviceBuffer,
    moe_act: cuda.DeviceBuffer,
    // MLA: per-layer caches sized by context
    mla_keys: cuda.DeviceBuffer,
    mla_ik: cuda.DeviceBuffer,
    mla_ig: cuda.DeviceBuffer,
    mla_pool: cuda.DeviceBuffer,
    mla_iw: cuda.DeviceBuffer,
    mla_xp: cuda.DeviceBuffer,
    mla_qr: cuda.DeviceBuffer,
    mla_qp: cuda.DeviceBuffer,
    mla_ql: cuda.DeviceBuffer,
    mla_att: cuda.DeviceBuffer,
    mla_vals: cuda.DeviceBuffer,
    mla_scores: cuda.DeviceBuffer,
    mla_idx: cuda.DeviceBuffer,
    mla_scale: cuda.DeviceBuffer,
    mla_scratch_ikraw: cuda.DeviceBuffer,
    mla_klen: cuda.DeviceBuffer,
    mlp_tmp: cuda.DeviceBuffer, // [rows, 2*dense_inter] bf16
    mlp_act: cuda.DeviceBuffer, // [rows, dense_inter] bf16
    scratch: cuda.DeviceBuffer, // exl3/experts scratch
    dsa: ?@import("dsa.zig").Kernels = null,
    index: ?@import("index.zig").Kernels = null,
    context: usize = 4096,
    skip_mla: bool = false,
    cont: bool = false,
    stop_after_blocks: bool = false,
    // the MTP draft head (the block trailing the backbone, when in range)
    mtp_li: ?usize = null, // its index in w.layers
    mtp_mi: usize = 0, // its MLA cache plane (local)
    mtp_eh: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    mtp_x: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    mtp_out: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    mtp_pick: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    mtp_emb: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true }, // bf16 [1, D]: the last token's embedding
    kda_cs_bak: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true }, // spec-decode rollback
    // #741 mixed path: per-layer pointer desc tables (no stacks, no repack)
    m41_layer_descs: ?std.AutoHashMapUnmanaged(usize, u64) = null,
    // prefill cache: snapshots of everything a decode continuation needs
    pf_on: bool = false,
    pf_st0: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_st1: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_cs: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_x0: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_x1: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_branch: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_post: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_comb: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_mla_keys: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_mla_ik: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_mla_ig: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_mla_pool: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    pf_y: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true }, // the tail stream-mean: the MTP draft's h input
    mtp_dump_nth: usize = 0, // probe: dump the Nth mtpDraft call (0 = off)
    mtp_calls: usize = 0, // probe: mtpDraft call counter within the request
    mtp_tag: u32 = 0, // probe: filename tag (request counter)

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, opts: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const w = try weights.load(gpa, io, d, dir, .{ .lo = opts.blocks_lo, .hi = opts.blocks_hi });
        errdefer {
            w.deinit();
            gpa.destroy(w);
        }
        const rows_max: usize = 24;
        e.* = .{
            .gpa = gpa,
            .io = io,
            .d = d,
            .stream = undefined,
            .c = w.config,
            .w = w,
            .ex = undefined,
            .hck = undefined,
            .ops = undefined,
            .kdk = undefined,
            .g = undefined,
            .rows_max = rows_max,
            .ids = undefined,
            .x = undefined,
            .branch = undefined,
            .normed = undefined,
            .part = undefined,
            .post = undefined,
            .comb = undefined,
            .y = undefined,
            .proj = undefined,
            .qkv = undefined,
            .fg = undefined,
            .bv = undefined,
            .kda_st = .{ &.{}, &.{} },
            .kda_st0 = undefined,
            .kda_st1 = undefined,
            .kda_cs = undefined,
            .kda_cw = undefined,
            .kda_conv_p = undefined,
            .logits = undefined,
            .picks = undefined,
            .eps_buf = undefined,
            .fn_zero = undefined,
            .moe_pick = undefined,
            .moe_wts = undefined,
            .moe_logits = undefined,
            .moe_ys = undefined,
            .moe_ye = undefined,
            .moe_gu = undefined,
            .moe_act = undefined,
            .mla_keys = undefined,
            .mla_ik = undefined,
            .mla_ig = undefined,
            .mla_pool = undefined,
            .mla_iw = undefined,
            .mla_xp = undefined,
            .mla_qr = undefined,
            .mla_qp = undefined,
            .mla_ql = undefined,
            .mla_att = undefined,
            .mla_vals = undefined,
            .mla_scores = undefined,
            .mla_idx = undefined,
            .mla_scale = undefined,
            .mla_scratch_ikraw = undefined,
            .mla_klen = undefined,
            .mlp_tmp = undefined,
            .mlp_act = undefined,
            .scratch = undefined,
            .skip_mla = opts.skip_mla,
            .cont = opts.cont,
            .stop_after_blocks = opts.stop_after_blocks,
            .context = opts.context,
        };
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.ex = try exl3.Kernels.load(d);
        e.hck = try hc.Kernels.load(d);
        e.ops = try glue.Ops.load(d);
        e.kdk = try kda.Kernels.load(d);
        e.dsa = try @import("dsa.zig").Kernels.load(d);
        e.index = try @import("index.zig").Kernels.load(d);
        e.g = try Gemms.init(gpa, d);

        const hidden = e.c.hidden;
        const n_kda = e.c.countKind(.kda);
        const D: usize = e.c.kda_dim;
        const W: usize = e.c.kdaWidth();
        const rows = rows_max;
        e.ids = try cuda.DeviceBuffer.alloc(d, rows * 4);
        e.x[0] = try cuda.DeviceBuffer.alloc(d, rows * 4 * hidden * 4); // fp32 streams
        e.x[1] = try cuda.DeviceBuffer.alloc(d, rows * 4 * hidden * 4);
        e.branch = try cuda.DeviceBuffer.alloc(d, rows * hidden * 2);
        e.normed = try cuda.DeviceBuffer.alloc(d, rows * hidden * 2);
        e.part = try cuda.DeviceBuffer.alloc(d, rows * hc.partBytes(1));
        e.post = try cuda.DeviceBuffer.alloc(d, rows * 4 * 4);
        e.comb = try cuda.DeviceBuffer.alloc(d, rows * 16 * 4);
        e.y = try cuda.DeviceBuffer.alloc(d, rows * @max(hidden, W) * 2); // the KDA step writes [rows, Wq]
        e.proj = try cuda.DeviceBuffer.alloc(d, rows * e.c.kdaProj() * 2);
        e.qkv = try cuda.DeviceBuffer.alloc(d, rows * 3 * W * 2);
        e.fg = try cuda.DeviceBuffer.alloc(d, rows * 2 * W * 2);
        e.bv = try cuda.DeviceBuffer.alloc(d, rows * e.c.kda_heads * 2);
        e.kda_st0 = try cuda.DeviceBuffer.alloc(d, n_kda * e.c.kda_heads * D * D * 4);
        e.kda_st1 = try cuda.DeviceBuffer.alloc(d, n_kda * e.c.kda_heads * D * D * 4);
        e.kda_cs = try cuda.DeviceBuffer.alloc(d, n_kda * (4 - 1) * 3 * W * 2);
        e.kda_conv_p = try cuda.DeviceBuffer.alloc(d, (3 + rows) * 3 * W * 2);
        e.logits = try cuda.DeviceBuffer.alloc(d, rows * e.c.vocab * 2);
        e.picks = try cuda.DeviceBuffer.alloc(d, rows * 4);
        e.mlp_tmp = try cuda.DeviceBuffer.alloc(d, rows * 2 * e.c.dense_inter * 2);
        e.mlp_act = try cuda.DeviceBuffer.alloc(d, rows * e.c.dense_inter * 2);
        e.scratch = try cuda.DeviceBuffer.alloc(d, 256 << 20);
        // the recurrent state and conv history start zeroed
        try e.post.fill32(0, null);
        try e.comb.fill32(0, null);
        try e.kda_st0.fill32(0, null);
        try e.kda_st1.fill32(0, null);
        e.kda_st_cur = try gpa.alloc(usize, n_kda);
        @memset(e.kda_st_cur, 0);
        try e.kda_cs.fill8(0, null);
        // the KDA conv weights: [TAPS, 3W] transposed from the per-tensor
        // [W, 1, TAPS] uploads (element (c, j) at c*TAPS + j)
        {
            const Wq: usize = e.c.kdaWidth();
            e.kda_cw = try cuda.DeviceBuffer.alloc(d, n_kda * 4 * 3 * Wq * 2);
            var ki: usize = 0;
            for (e.w.layers) |*L| {
                const wk = switch (L.attn) {
                    .kda => |*x| x,
                    .mla => continue,
                };
                const srcs = [3]u64{ wk.q_conv, wk.k_conv, wk.v_conv };
                const taps_g: usize = 4;
                const cw_at = e.kda_cw.ptr + ki * 4 * 3 * Wq * 2;
                for (0..taps_g) |j| for (0..3) |part| {
                    try e.ops.gather(e.stream, srcs[part] + j * 2, cw_at + (j * 3 + part) * Wq * 2, Wq, 4);
                };
                ki += 1;
            }
            try e.stream.synchronize();
        }
        e.moe_pick = try cuda.DeviceBuffer.alloc(d, e.rows_max * 8 * 4);
        e.moe_wts = try cuda.DeviceBuffer.alloc(d, e.rows_max * 8 * 4);
        e.moe_logits = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.experts * 4);
        e.moe_ys = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.hidden * 2);
        e.moe_ye = try cuda.DeviceBuffer.alloc(d, e.rows_max * 8 * e.c.hidden * 2);
        e.moe_gu = try cuda.DeviceBuffer.alloc(d, 2 * e.c.moe_inter * 2);
        e.moe_act = try cuda.DeviceBuffer.alloc(d, e.c.moe_inter * 2);
        if (e.w.layers.len > 0 and e.w.layers[e.w.layers.len - 1].hc_attn == null) {
            // the MTP draft head trails the backbone: 1-row draft buffers + its own MLA plane
            e.mtp_li = e.w.layers.len - 1;
            for (e.w.layers[0 .. e.w.layers.len - 1]) |*L2| {
                if (L2.attn == .mla) e.mtp_mi += 1;
            }
            e.mtp_eh = try cuda.DeviceBuffer.alloc(d, 2 * e.c.hidden * 2);
            e.mtp_x = try cuda.DeviceBuffer.alloc(d, e.c.hidden * 2);
            e.mtp_out = try cuda.DeviceBuffer.alloc(d, e.c.hidden * 2);
            e.mtp_pick = try cuda.DeviceBuffer.alloc(d, 4);
            e.mtp_emb = try cuda.DeviceBuffer.alloc(d, e.c.hidden * 2);
        }
        {
            const cap = opts.context;
            const nm: usize = e.c.countKind(.mla);
            e.mla_keys = try cuda.DeviceBuffer.alloc(d, nm * cap * e.c.kv_lora * 2);
            e.mla_ik = try cuda.DeviceBuffer.alloc(d, nm * cap * e.c.i_dim * 2);
            e.mla_ig = try cuda.DeviceBuffer.alloc(d, nm * cap * e.c.i_dim * 4);
            e.mla_pool = try cuda.DeviceBuffer.alloc(d, nm * (cap / 4) * e.c.i_dim * 2);
            e.mla_iw = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.i_heads * 2);
            e.mla_xp = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.xProj() * 2);
            e.mla_qr = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.q_lora * 2);
            e.mla_qp = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.qrProj() * 2);
            e.mla_ql = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.mla_heads * e.c.kv_lora * 2);
            e.mla_att = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.mla_heads * e.c.kv_lora * 2);
            e.mla_vals = try cuda.DeviceBuffer.alloc(d, e.rows_max * e.c.mla_heads * e.c.v_dim * 2);
            e.mla_scores = try cuda.DeviceBuffer.alloc(d, e.rows_max * (cap / 4 + 1) * 4);
            e.mla_idx = try cuda.DeviceBuffer.alloc(d, e.rows_max * 2051 * 4);
            const sc = try gpa.alloc(f32, 1);
            defer gpa.free(sc);
            sc[0] = 1.0 / 16.0;
            e.mla_scale = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(sc));
            const eps2 = try gpa.alloc(f32, 2);
            defer gpa.free(eps2);
            eps2[0] = e.c.eps;
            eps2[1] = e.c.hc_eps;
            e.eps_buf = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(eps2));
            e.mla_scratch_ikraw = try cuda.DeviceBuffer.alloc(d, nm * cap * e.c.i_dim * 2);
            const kl = try gpa.alloc(i32, 1);
            defer gpa.free(kl);
            kl[0] = 0;
            e.mla_klen = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(kl));
        }
        {
            const z = try gpa.alloc(u8, 24 * 4 * e.c.hidden * 2);
            defer gpa.free(z);
            @memset(z, 0);
            e.fn_zero = try cuda.DeviceBuffer.fromHost(d, z);
        }
        return e;
    }


    /// One hc boundary: expand (pending) or first-mix, then the split; the
    /// post/comb pair persists across boundaries in the engine.

    /// A block-0 fingerprint of one buffer's row 0 (the lane-side prints' counterpart).
    // serving: wipe all recurrent state so a new request starts from scratch.
    // MLA caches need no wipe: forward(ids, rows, pos) re-masks by pos each call.
    pub fn resetState(e: *Engine) !void {
        // fills MUST ride e.stream: it is non-blocking, so null-stream memsets
        // are unordered against in-flight work and the reset would race.
        try e.kda_st0.fill32(0, e.stream.handle);
        try e.kda_st1.fill32(0, e.stream.handle);
        try e.kda_cs.fill8(0, e.stream.handle);
        @memset(e.kda_st_cur, 0);
        e.xi = 0;
        e.fwd_count = 0;
        e.dump_epoch += 1;
        try e.stream.synchronize();

    }

    fn b0Print(e: *Engine, what: []const u8, buf: cuda.DeviceBuffer) !void {
        try e.stream.synchronize();
        const b = try e.gpa.alloc(u8, e.c.hidden * 2);
        defer e.gpa.free(b);
        try buf.download(0, b[0..@min(b.len, buf.len)]);
        var acc: f64 = 0;
        const n = @min(e.c.hidden, buf.len / 2);
        for (0..n) |d| {
            const h: u16 = std.mem.readInt(u16, b[d * 2 ..][0..2], .little);
            acc += @as(f32, @bitCast(@as(u32, h) << 16));
        }
        std.debug.print("  [b0] {s} {d:.4}\n", .{ what, acc });
    }

    /// Dump n bytes at a raw device pointer to a file in the cwd (probes only; raw posix write).
    fn dumpBin(e: *Engine, name: []const u8, ptr: u64, n: usize) !void {
        try e.stream.synchronize();
        const hb = try e.gpa.alloc(u8, n);
        defer e.gpa.free(hb);
        try (cuda.DeviceBuffer{ .d = e.d, .ptr = ptr, .len = n, .borrowed = true }).download(0, hb);
        var z: [64:0]u8 = undefined;
        if (name.len >= z.len) return;
        @memcpy(z[0..name.len], name);
        z[name.len] = 0;
        const fd = std.posix.openatZ(std.posix.AT.FDCWD, z[0..name.len :0].ptr, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o664) catch return;
        defer _ = std.os.linux.close(fd);
        var off: usize = 0;
        while (off < hb.len) {
            const wn = std.os.linux.write(fd, hb.ptr + off, hb.len - off);
            const wr: isize = @bitCast(wn);
            if (std.os.linux.errno(wr) == .INTR) continue;
            if (wr <= 0) return;
            off += @as(usize, @intCast(wr));
        }
    }

    /// Count nonzero f32 in a whole device range (probes only).
    fn nzFull(e: *Engine, ptr: u64, n: usize) !usize {
        const hb = try e.gpa.alloc(u8, n);
        defer e.gpa.free(hb);
        try (cuda.DeviceBuffer{ .d = e.d, .ptr = ptr, .len = n, .borrowed = true }).download(0, hb);
        const fv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, hb));
        var c: usize = 0;
        for (fv) |v| if (v != 0) {
            c += 1;
        };
        return c;
    }

    /// Per-KDA-layer probe dump: sd{li}_{tag}.bin.
    fn dumpPb2(e: *Engine, li: usize, tag: []const u8, ptr: u64, n: usize) !void {
        var name: [24]u8 = undefined;
        const nm = std.fmt.bufPrint(&name, "sd{d}_{s}.bin", .{ li, tag }) catch return;
        try e.dumpBin(nm, ptr, n);
    }

    /// Per-block probe dump: pb{bi}_{tag}.bin, rows*hidden bf16 at a raw pointer.
    fn dumpPb(e: *Engine, bi: usize, tag: []const u8, ptr: u64, rows: usize) !void {
        var name: [24]u8 = undefined;
        e.dump_serial += 1;
        const n = std.fmt.bufPrint(&name, "pb{d}_{s}S{d}.bin", .{ bi, tag, e.dump_serial }) catch return;
        try e.dumpBin(n, ptr, rows * e.c.hidden * 2);
    }

    /// Count nonzero f32 in the first 128 bytes at a raw device pointer (probes only).
    fn nzHead(e: *Engine, ptr: u64) !usize {
        const hb = try e.gpa.alloc(u8, 128);
        defer e.gpa.free(hb);
        try (cuda.DeviceBuffer{ .d = e.d, .ptr = ptr, .len = 128, .borrowed = true }).download(0, hb);
        const fv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, hb));
        var c: usize = 0;
        for (fv) |v| if (v != 0) {
            c += 1;
        };
        return c;
    }

    fn hcGates(e: *Engine, rows: usize, hc_w: weights.Hc, norm_w: u64) !void {
        const x_old = e.x[e.xi].ptr;
        e.hc_calls += 1;
        if (e.dump_on and e.hc_calls == 1) { // the very first gates call: the x entering block 0's attention
            try e.stream.synchronize();
            const n = rows * 4 * e.c.hidden * 4;
            const xb = try e.gpa.alloc(u8, n);
            defer e.gpa.free(xb);
            try e.x[e.xi].download(0, xb);
            var nm7: [32]u8 = undefined;
            e.dump_serial += 1;
            const f7 = std.fmt.bufPrint(&nm7, "kxoS{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
            var dir7 = std.Io.Dir.cwd();
            dir7.writeFile(e.io, .{ .sub_path = f7, .data = xb }) catch {};
        }
        const x_new = e.x[1 - e.xi].ptr;
        const layer_h = hc_w;
        // glue's hc_pre: this boundary's gates + normed from the x entering it;
        // the mixer's branch folds LATER, in hcFold, with THESE gates
        if (e.f32_streams) {
            try e.hck.expandMixF32(e.stream, x_old, e.branch.ptr, e.post.ptr, e.comb.ptr, layer_h.fnp, x_new, e.part.ptr, rows, false);
            if (e.dump_on and rows == 2 and e.hc_calls == 1) {
                    {
                        try e.stream.synchronize();
                        const cb = try e.gpa.alloc(u8, 3 * 3 * 12288 * 2);
                        defer e.gpa.free(cb);
                        try (cuda.DeviceBuffer{ .d = e.d, .ptr = e.kda_cs.ptr, .len = cb.len, .borrowed = true }).download(0, cb);
                        var nmX: [40]u8 = undefined;
                        e.dump_serial += 1;
                        const fX = std.fmt.bufPrint(&nmX, "ksh{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                        var dirX = std.Io.Dir.cwd();
                        dirX.writeFile(e.io, .{ .sub_path = fX, .data = cb }) catch {};
                    }
            }
            try e.hck.splitF32(e.stream, x_old, e.part.ptr, layer_h.scale, layer_h.base, norm_w, e.eps_buf.ptr, e.normed.ptr, e.post.ptr, e.comb.ptr, rows);
        } else {
            try e.hck.expandMix(e.stream, x_old, e.branch.ptr, e.post.ptr, e.comb.ptr, layer_h.fnp, x_new, e.part.ptr, rows, false);
            try e.hck.split(e.stream, x_old, e.part.ptr, layer_h.scale, layer_h.base, norm_w, e.eps_buf.ptr, e.normed.ptr, e.post.ptr, e.comb.ptr, rows);
        }
    }

    /// glue's hc_post: fold the mixer's branch with the gates the last hcGates
    /// computed (each branch folds with its OWN boundary's gates, right after
    /// its mixer - not one boundary later).
    fn hcFold(e: *Engine, fn_packed: u64, rows: usize) !void {
        const x_old = e.x[e.xi].ptr;
        const x_new = e.x[1 - e.xi].ptr;
        if (e.f32_streams) {
            try e.hck.expandMixF32(e.stream, x_old, e.branch.ptr, e.post.ptr, e.comb.ptr, fn_packed, x_new, e.part.ptr, rows, true);
        } else {
            try e.hck.expandMix(e.stream, x_old, e.branch.ptr, e.post.ptr, e.comb.ptr, fn_packed, x_new, e.part.ptr, rows, true);
        }
        e.xi = 1 - e.xi;
    }

    /// KDA attention block `li` (its dense kind index) on normed -> branch.
    pub var opt_gates: i128 = 0;
    pub var opt_conv: i128 = 0;
    pub var opt_step: i128 = 0;
    pub var opt_moe: i128 = 0;
    pub var opt_gemms: i128 = 0;
    pub var op_timing: bool = false;
    fn tns() i128 {
        var ts: std.posix.timespec = undefined;
        _ = std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts);
        return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
    }

    fn kdaLayer(e: *Engine, li: usize, w: *const weights.Kda, rows: usize) !void {
        if (e.dump_on and li == 0) {
            try e.stream.synchronize();
            {
                const npt = rows * 16 * 25 * 4;
                if (e.part.len >= npt) {
                    const pb2 = try e.gpa.alloc(u8, npt);
                    defer e.gpa.free(pb2);
                    try e.part.download(0, pb2);
                    var nm6: [32]u8 = undefined;
                    e.dump_serial += 1;
                    const f6 = std.fmt.bufPrint(&nm6, "kpartS{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                    var dir6 = std.Io.Dir.cwd();
                    dir6.writeFile(e.io, .{ .sub_path = f6, .data = pb2 }) catch {};
                }
                const nn = rows * e.c.hidden * 2;
                const nb = try e.gpa.alloc(u8, nn);
                defer e.gpa.free(nb);
                try e.normed.download(0, nb);
                var nm5: [32]u8 = undefined;
                e.dump_serial += 1;
                const f5 = std.fmt.bufPrint(&nm5, "knaS{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                var dir5 = std.Io.Dir.cwd();
                dir5.writeFile(e.io, .{ .sub_path = f5, .data = nb }) catch {};
            }
            {
                const np = rows * @as(usize, e.c.kdaProj()) * 2;
                if (e.proj.len >= np) {
                    const pb = try e.gpa.alloc(u8, np);
                    defer e.gpa.free(pb);
                    try e.proj.download(0, pb);
                    var nm3: [32]u8 = undefined;
                    const f3 = std.fmt.bufPrint(&nm3, "kproj{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial + 1 }) catch unreachable;
                    var dir = std.Io.Dir.cwd();
                    dir.writeFile(e.io, .{ .sub_path = f3, .data = pb }) catch {};
                }
                const nc = 3 * 12288 * 2;
                const cb = try e.gpa.alloc(u8, nc);
                defer e.gpa.free(cb);
                try (cuda.DeviceBuffer{ .d = e.d, .ptr = e.kda_cs.ptr, .len = nc, .borrowed = true }).download(0, cb);
                var nm4: [32]u8 = undefined;
                const f4 = std.fmt.bufPrint(&nm4, "kcs{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial + 2 }) catch unreachable;
                var dir2 = std.Io.Dir.cwd();
                dir2.writeFile(e.io, .{ .sub_path = f4, .data = cb }) catch {};
            }
            // one-shot repro: the SAME three mmStrided proj calls into scratch
            {
                const Wq2 = e.c.kdaWidth();
                const ps2 = e.c.kdaProj();
                const xst2 = e.c.hidden * 2;
                const wst2 = e.c.hidden * 2;
                const ost2 = ps2 * 2;
                try e.g.mmStrided(e.stream, e.normed.ptr, xst2, w.q, wst2, e.scratch.ptr, ost2, rows, Wq2, e.c.hidden);
                try e.g.mmStrided(e.stream, e.normed.ptr, xst2, w.k, wst2, e.scratch.ptr + Wq2 * 2, ost2, rows, Wq2, e.c.hidden);
                try e.g.mmStrided(e.stream, e.normed.ptr, xst2, w.v, wst2, e.scratch.ptr + 2 * Wq2 * 2, ost2, rows, Wq2, e.c.hidden);
                try e.stream.synchronize();
                const ns = rows * @as(usize, ps2) * 2;
                const sb = try e.gpa.alloc(u8, ns);
                defer e.gpa.free(sb);
                try e.scratch.download(0, sb);
                var nm8: [32]u8 = undefined;
                e.dump_serial += 1;
                const f8 = std.fmt.bufPrint(&nm8, "krep{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                var dir8 = std.Io.Dir.cwd();
                dir8.writeFile(e.io, .{ .sub_path = f8, .data = sb }) catch {};
                // POISON TEST: overwrite normed row 1, re-run, compare row 0 stability
                {
                    const rb2 = e.c.hidden * 2; // one row, bytes
                    const save = try e.gpa.alloc(u8, rb2);
                    defer e.gpa.free(save);
                    try e.stream.synchronize();
                    try e.normed.download(rb2, save);
                    const poison = try e.gpa.alloc(u8, rb2);
                    defer e.gpa.free(poison);
                    @memset(poison, 0x49);
                    try e.normed.upload(rb2, poison);
                    try e.g.mmStrided(e.stream, e.normed.ptr, xst2, w.q, wst2, e.scratch.ptr, ost2, rows, Wq2, e.c.hidden);
                    try e.g.mmStrided(e.stream, e.normed.ptr, xst2, w.k, wst2, e.scratch.ptr + Wq2 * 2, ost2, rows, Wq2, e.c.hidden);
                    try e.g.mmStrided(e.stream, e.normed.ptr, xst2, w.v, wst2, e.scratch.ptr + 2 * Wq2 * 2, ost2, rows, Wq2, e.c.hidden);
                    try e.stream.synchronize();
                    const sb2 = try e.gpa.alloc(u8, ns);
                    defer e.gpa.free(sb2);
                    try e.scratch.download(0, sb2);
                    var changed: usize = 0;
                    const half = @min(sb.len, sb2.len) / 2;
                    var q: usize = 0;
                    while (q * 2 < half * 2) : (q += 1) {
                        const v1 = std.mem.readInt(u16, sb[q * 2 ..][0..2], .little);
                        const v2 = std.mem.readInt(u16, sb2[q * 2 ..][0..2], .little);
                        if (v1 != v2) changed += 1;
                    }
                    var tot: usize = 0;
                    const r0 = @min(ns / 2, @as(usize, e.c.kdaProj()) * 2); // row 0 span in bytes
                    q = 0;
                    var ch0: usize = 0;
                    while (q * 2 + 2 <= r0) : (q += 1) {
                        const v1 = std.mem.readInt(u16, sb[q * 2 ..][0..2], .little);
                        const v2 = std.mem.readInt(u16, sb2[q * 2 ..][0..2], .little);
                        if (v1 != v2) ch0 += 1;
                        tot += 1;
                    }
                    std.debug.print("  POISON rows={d}: whole {d}/{d} changed, ROW0 {d}/{d} changed\n", .{ rows, changed, half, ch0, tot });
                    try e.normed.upload(rb2, save); // restore
                }
            }
            const nq = rows * 3 * @as(usize, e.c.kdaWidth()) * 2;
            if (e.qkv.len >= nq) {
                const qb = try e.gpa.alloc(u8, nq);
                defer e.gpa.free(qb);
                try e.qkv.download(0, qb);
                var nm1: [32]u8 = undefined;
                const f1 = std.fmt.bufPrint(&nm1, "kqkv{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial + 3 }) catch unreachable;
                var dir = std.Io.Dir.cwd();
                dir.writeFile(e.io, .{ .sub_path = f1, .data = qb }) catch {};
            }
        }
        const c = &e.c;
        const Wq: usize = c.kdaWidth();
        const D: usize = c.kda_dim;
        const ps = c.kdaProj();
        const H: usize = c.kda_heads;
        const st_bytes: usize = H * D * D * 4;
        // the stacked projections: rows land in the ps-strided fused buffer the conv reads
        const xst = c.hidden * 2;
        const wst = c.hidden * 2;
        const ost = ps * 2;
        try e.g.mmStrided(e.stream, e.normed.ptr, xst, w.q, wst, e.proj.ptr, ost, rows, Wq, c.hidden);
        try e.g.mmStrided(e.stream, e.normed.ptr, xst, w.k, wst, e.proj.ptr + Wq * 2, ost, rows, Wq, c.hidden);
        try e.g.mmStrided(e.stream, e.normed.ptr, xst, w.v, wst, e.proj.ptr + 2 * Wq * 2, ost, rows, Wq, c.hidden);
        try e.g.mmStrided(e.stream, e.normed.ptr, xst, w.f_a, wst, e.proj.ptr + 3 * Wq * 2, ost, rows, D, c.hidden);
        try e.g.mmStrided(e.stream, e.normed.ptr, xst, w.g_a, wst, e.proj.ptr + (3 * Wq + D) * 2, ost, rows, D, c.hidden);
        try e.g.mmStrided(e.stream, e.normed.ptr, xst, w.b, wst, e.proj.ptr + (3 * Wq + 2 * D) * 2, ost, rows, H, c.hidden);
        // f_b / g_b on the f_a/g_a slices; the step kernel reads fg interleaved [rows, 2Wq]
        try e.g.mmStrided(e.stream, e.proj.ptr + 3 * Wq * 2, ost, w.f_b, D * 2, e.fg.ptr, 2 * Wq * 2, rows, Wq, D);
        try e.g.mmStrided(e.stream, e.proj.ptr + (3 * Wq + D) * 2, ost, w.g_b, D * 2, e.fg.ptr + Wq * 2, 2 * Wq * 2, rows, Wq, D);
        if (li == 0 and !e.quiet) {
            try e.stream.synchronize();
            const fb2 = try e.gpa.alloc(u8, 2 * Wq * 2);
            defer e.gpa.free(fb2);
            try e.fg.download(0, fb2);
            var fb_s: f64 = 0;
            var gb_s: f64 = 0;
            for (0..Wq) |d| {
                const h: u16 = std.mem.readInt(u16, fb2[d * 2 ..][0..2], .little);
                fb_s += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            for (0..Wq) |d| {
                const h: u16 = std.mem.readInt(u16, fb2[(Wq + d) * 2 ..][0..2], .little);
                gb_s += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            std.debug.print("  [kda0] fb {d:.2} gb {d:.2}\n", .{ fb_s, gb_s });
        }
        // b packed from the proj slice
        try e.ops.copy2d(e.stream, e.proj.ptr + (3 * Wq + 2 * D) * 2, e.bv.ptr, rows, H, ps);
        if (li == 0 and !e.quiet) {
            try e.stream.synchronize();
            const bb2 = try e.gpa.alloc(u8, ps * 2);
            defer e.gpa.free(bb2);
            try e.proj.download(0, bb2);
            var q_s: f64 = 0;
            var k_s: f64 = 0;
            var v_s: f64 = 0;
            var fa_s: f64 = 0;
            var ga_s: f64 = 0;
            var b_s: f64 = 0;
            for (0..Wq) |d| {
                const h: u16 = std.mem.readInt(u16, bb2[d * 2 ..][0..2], .little);
                q_s += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            for (0..Wq) |d| {
                const h: u16 = std.mem.readInt(u16, bb2[(Wq + d) * 2 ..][0..2], .little);
                k_s += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            for (0..Wq) |d| {
                const h: u16 = std.mem.readInt(u16, bb2[(2 * Wq + d) * 2 ..][0..2], .little);
                v_s += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            for (0..D) |d| {
                const h: u16 = std.mem.readInt(u16, bb2[(3 * Wq + d) * 2 ..][0..2], .little);
                fa_s += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            for (0..D) |d| {
                const h: u16 = std.mem.readInt(u16, bb2[(3 * Wq + D + d) * 2 ..][0..2], .little);
                ga_s += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            for (0..H) |d| {
                const h: u16 = std.mem.readInt(u16, bb2[(3 * Wq + 2 * D + d) * 2 ..][0..2], .little);
                b_s += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            std.debug.print("  [kda0] q {d:.2} k {d:.2} v {d:.2} fa {d:.2} ga {d:.2} b {d:.2}\n", .{ q_s, k_s, v_s, fa_s, ga_s, b_s });
        }
        // conv: history per layer, weights the engine built at load (kda_cw)
        const cs_at = e.kda_cs.ptr + li * 3 * 3 * Wq * 2;
        try e.kdk.conv(e.stream, e.proj.ptr, ps, cs_at, e.kda_cw.ptr + li * 4 * 3 * Wq * 2, e.qkv.ptr, rows);
        if (li == 0 and !e.quiet) {
            try e.stream.synchronize();
            const qb2 = try e.gpa.alloc(u8, 3 * Wq * 2);
            defer e.gpa.free(qb2);
            try e.qkv.download(0, qb2);
            var qq: f64 = 0;
            var kk: f64 = 0;
            var vv: f64 = 0;
            for (0..Wq) |d| {
                const h: u16 = std.mem.readInt(u16, qb2[d * 2 ..][0..2], .little);
                qq += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            for (0..Wq) |d| {
                const h: u16 = std.mem.readInt(u16, qb2[(Wq + d) * 2 ..][0..2], .little);
                kk += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            for (0..Wq) |d| {
                const h: u16 = std.mem.readInt(u16, qb2[(2 * Wq + d) * 2 ..][0..2], .little);
                vv += @as(f32, @bitCast(@as(u32, h) << 16));
            }
            std.debug.print("  [kda0] convq {d:.2} convk {d:.2} convv {d:.2}\n", .{ qq, kk, vv });
        }
        // the step: this layer's own parity so windows chain (layer li's next
        // read is always the buffer it last wrote)
        const cur = e.kda_st_cur[li];
        if (e.dump_on and li == 0) {
            std.debug.print("  STEP cur[0]={d} fwd={d}\n", .{ cur, e.fwd_count });
            const sb = try e.gpa.alloc(u8, 8192);
            defer e.gpa.free(sb);
            try e.stream.synchronize();
            try (cuda.DeviceBuffer{ .d = e.d, .ptr = e.kda_st0.ptr, .len = 8192, .borrowed = true }).download(0, sb);
            var nmS0: [40]u8 = undefined;
            e.dump_serial += 1;
            const fS0 = std.fmt.bufPrint(&nmS0, "kst0aS{d}.bin", .{e.dump_serial}) catch unreachable;
            var dirS = std.Io.Dir.cwd();
            dirS.writeFile(e.io, .{ .sub_path = fS0, .data = sb }) catch {};
            try (cuda.DeviceBuffer{ .d = e.d, .ptr = e.kda_st1.ptr, .len = 8192, .borrowed = true }).download(0, sb);
            var nmS1: [40]u8 = undefined;
            const fS1 = std.fmt.bufPrint(&nmS1, "kst1aS{d}.bin", .{e.dump_serial}) catch unreachable;
            dirS.writeFile(e.io, .{ .sub_path = fS1, .data = sb }) catch {};
        }
        const st_in = if (cur == 0) e.kda_st0.ptr else e.kda_st1.ptr;
        if (li <= 4 and (e.fwd_count == 1 and !e.quiet)) {
            try e.stream.synchronize();
            const hb = try e.gpa.alloc(u8, 256);
            defer e.gpa.free(hb);
            try (cuda.DeviceBuffer{ .d = e.d, .ptr = st_in, .len = 256, .borrowed = true }).download(0, hb);
            const fv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, hb));
            var nz: usize = 0;
            for (fv) |v| if (v != 0) {
                nz += 1;
            };
            std.debug.print("  pre-step st_in nonzero {d}/64 first {d:.4}\n", .{ nz, fv[0] });

        }
        const st_out = if (cur == 0) e.kda_st1.ptr else e.kda_st0.ptr;
        try e.kdk.step(e.stream, e.qkv.ptr, e.fg.ptr, e.bv.ptr, w.a, w.dt_bias, st_in + li * st_bytes, w.o_norm, c.lower_bound, c.hc_eps, rows, st_out + li * st_bytes, e.y.ptr);
        if (li <= 4 and (e.fwd_count == 1 and !e.quiet)) {
            try e.stream.synchronize();
            const c0 = try e.nzHead(e.kda_st0.ptr);
            const c1 = try e.nzHead(e.kda_st1.ptr);
            std.debug.print("  post-step st0 nz {d}/32 st1 nz {d}/32\n", .{ c0, c1 });

        }
        e.kda_st_cur[li] = 1 - cur;
        // conv-history rotate: cs becomes the last TAPS-1 rows of [cs; proj],
        // so the next window's conv sees this window's tail (their _conv_shift)
        {
            const cs_at2 = e.kda_cs.ptr + li * 3 * 3 * Wq * 2;
            if (rows >= 3) {
                try e.ops.copy2d(e.stream, e.proj.ptr + (rows - 3) * ps, cs_at2, 3, 3 * Wq, ps);
            } else {
                const keep = 3 - rows;
                // the kept slice is cs[rows..3) (the LAST keep rows). dst overlaps
                // src (a downward shift) and copy2d blocks run CONCURRENTLY - an
                // in-place copy races (block y+1 clobbers block y's source row).
                // Stage through scratch: two disjoint copies are always safe.
                if (e.dump_on and li == 0) {
                    std.debug.print("  ROT args rows={d} keep={d} cols={d} cs={x} src={x} scratch={x}\n", .{ rows, keep, 3 * Wq, cs_at2, cs_at2 + rows * 3 * Wq * 2, e.scratch.ptr });
                }
                try e.ops.copy2d(e.stream, cs_at2 + rows * 3 * Wq * 2, e.scratch.ptr, keep, 3 * Wq, 3 * Wq);
                if (e.dump_on and li == 0) {
                    try e.stream.synchronize();
                    const nb = try e.gpa.alloc(u8, keep * 3 * Wq * 2);
                    defer e.gpa.free(nb);
                    try (cuda.DeviceBuffer{ .d = e.d, .ptr = e.scratch.ptr, .len = nb.len, .borrowed = true }).download(0, nb);
                    var nmS: [40]u8 = undefined;
                    e.dump_serial += 1;
                    const fS = std.fmt.bufPrint(&nmS, "ksc1{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                    var dirS = std.Io.Dir.cwd();
                    dirS.writeFile(e.io, .{ .sub_path = fS, .data = nb }) catch {};
                }
                try e.ops.copy2d(e.stream, e.scratch.ptr, cs_at2, keep, 3 * Wq, 3 * Wq);
                try e.ops.copy2d(e.stream, e.proj.ptr, cs_at2 + keep * 3 * Wq * 2, rows, 3 * Wq, ps);
            }
            if (e.dump_on and li == 0) {
                try e.stream.synchronize();
                const nc = 3 * Wq * 2;
                const cb = try e.gpa.alloc(u8, nc);
                defer e.gpa.free(cb);
                try (cuda.DeviceBuffer{ .d = e.d, .ptr = cs_at2, .len = nc, .borrowed = true }).download(0, cb);
                var nmR: [40]u8 = undefined;
                e.dump_serial += 1;
                const fR = std.fmt.bufPrint(&nmR, "krot{s}S{d}r{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial, rows }) catch unreachable;
                var dirR = std.Io.Dir.cwd();
                dirR.writeFile(e.io, .{ .sub_path = fR, .data = cb }) catch {};
            }
        }
        // o_proj -> branch
        try e.g.mm(e.stream, e.y.ptr, w.o, e.branch.ptr, rows, c.hidden, Wq, false);
        if (e.dump_on and li == 0) {
            try e.stream.synchronize();
            const ep = @as([]const u8, if (e.cont) "b" else "a");
            const W3 = @as(usize, c.kdaWidth());
            inline for (.{ "kqkvT", "kfgT", "kbvT", "kyT", "kprojT", "kcsT" }, .{ 3 * W3, 2 * W3, c.kda_heads, W3, @as(usize, c.kdaProj()), 3 * W3 }) |tag, wid| {
                const n = if (comptime std.mem.eql(u8, tag, "kcsT")) 3 * wid * 2 else if (comptime std.mem.eql(u8, tag, "kprojT")) 2 * wid * 2 else rows * wid * 2;
                const buf = try e.gpa.alloc(u8, n);
                defer e.gpa.free(buf);
                const src = if (comptime std.mem.eql(u8, tag, "kqkvT")) e.qkv.ptr else if (comptime std.mem.eql(u8, tag, "kfgT")) e.fg.ptr else if (comptime std.mem.eql(u8, tag, "kbvT")) e.bv.ptr else if (comptime std.mem.eql(u8, tag, "kprojT")) e.proj.ptr else if (comptime std.mem.eql(u8, tag, "kcsT")) (e.kda_cs.ptr + li * 3 * W3 * 2) else e.y.ptr;
                try (cuda.DeviceBuffer{ .d = e.d, .ptr = src, .len = n, .borrowed = true }).download(0, buf);
                var nm: [40]u8 = undefined;
                e.dump_serial += 1;
                const fname = std.fmt.bufPrint(&nm, "{s}{s}S{d}.bin", .{ tag, ep, e.dump_serial }) catch unreachable;
                var dir = std.Io.Dir.cwd();
                dir.writeFile(e.io, .{ .sub_path = fname, .data = buf }) catch {};
            }
        }
        if (li <= 4 and (e.fwd_count == 1 and !e.quiet)) {
            try e.stream.synchronize();
            std.debug.print("  ptrs st_in {x} st_out {x} st0 {x} st1 {x} sb {d}\n", .{ st_in, st_out, e.kda_st0.ptr, e.kda_st1.ptr, st_bytes });
            const z0 = try e.nzFull(e.kda_st0.ptr, e.kda_st0.len);
            const z1 = try e.nzFull(e.kda_st1.ptr, e.kda_st1.len);
            std.debug.print("  census st0 {d}/{d} st1 {d}/{d}\n", .{ z0, e.kda_st0.len / 4, z1, e.kda_st1.len / 4 });

            try e.dumpPb2(li, "qkv", e.qkv.ptr, rows * 3 * Wq * 2);
            try e.dumpPb2(li, "fg", e.fg.ptr, rows * 2 * Wq * 2);
            try e.dumpPb2(li, "bv", e.bv.ptr, rows * H * 2);
            try e.dumpPb2(li, "a", w.a, H * 4);
            try e.dumpPb2(li, "dtb", w.dt_bias, H * D * 4);
            try e.dumpPb2(li, "onw", w.o_norm, D * 2);
            try e.dumpPb2(li, "y", e.y.ptr, rows * Wq * 2);
            try e.dumpBin("sd_stout.bin", st_out + li * st_bytes, H * D * D * 4);
            try e.dumpBin("sd_stin.bin", st_in + li * st_bytes, H * D * D * 4);
            std.debug.print("  step io dumped rows={d}", .{rows});
        }
        if (li <= 4 and (e.fwd_count == 1 and !e.quiet)) {
            try e.stream.synchronize();
            const hb = try e.gpa.alignedAlloc(u8, .@"2", D * 2);
            defer e.gpa.free(hb);
            try (cuda.DeviceBuffer{ .d = e.d, .ptr = e.y.ptr, .len = D * 2, .borrowed = true }).download(0, hb);
            var acc: f64 = 0;
            const fv = std.mem.bytesAsSlice(u16, hb);
            for (fv) |h16| acc += @as(f32, @bitCast(@as(u32, h16) << 16));
            try e.dumpBin("sd_y2.bin", e.y.ptr, rows * Wq * 2);
            std.debug.print("  y r0 h0 sum {d:.5}\n", .{acc});

        }
        if (e.dump_on and li == 0) {
            try e.stream.synchronize();
            const ny = rows * e.c.hidden * 2;
            const yb = try e.gpa.alloc(u8, ny);
            defer e.gpa.free(yb);
            try e.y.download(0, yb);
            var nm2: [32]u8 = undefined;
            const f2 = std.fmt.bufPrint(&nm2, "ky{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial + 4 }) catch unreachable;
            var dir = std.Io.Dir.cwd();
            dir.writeFile(e.io, .{ .sub_path = f2, .data = yb }) catch {};
        }
        if (li == 0 and !e.quiet) try e.b0Print("attn-post-o", e.branch);
    }

    /// The dense MLP on normed -> branch.
    fn denseMlp(e: *Engine, w: *const weights.Dense, rows: usize) !void {
        const inter = e.c.dense_inter;
        try e.g.mm(e.stream, e.normed.ptr, w.gate, e.mlp_tmp.ptr, rows, inter, e.c.hidden, false);
        try e.g.mm(e.stream, e.normed.ptr, w.up, e.mlp_tmp.ptr + rows * inter * 2, rows, inter, e.c.hidden, false);
        try e.ops.act(e.stream, e.mlp_tmp.ptr, e.mlp_tmp.ptr + rows * inter * 2, e.mlp_act.ptr, rows, 1, inter, e.c.swiglu_limit);
        try e.g.mm(e.stream, e.mlp_act.ptr, w.down, e.branch.ptr, rows, e.c.hidden, inter, false);
    }



    /// MLA attention block `mi` on normed -> branch, window at `pos`.
    /// Tensors: q_a[1536,4096] kv_a[512,4096]; indexer wk[128,4096] gate[128,4096]
    /// weights_proj[64,4096] wq_b[8192,1536] q_b[16384,1536]; kv_b[32768,512]:
    /// per head [512,512] - rows 0..256 absorb (k=256 nope), rows 256..512
    /// unabsorb (k=512 latent).
    fn mlaLayer(e: *Engine, mi: usize, w: *const weights.Mla, rows: usize, pos: usize) !void {
        const c = &e.c;
        const cap = e.context;
        const RANK: usize = c.kv_lora;
        const NOPE: usize = c.nope;
        const H: usize = c.mla_heads;
        // x side projections
        try e.g.mm(e.stream, e.normed.ptr, w.q_a, e.mla_xp.ptr, rows, c.q_lora, c.hidden, false); // q_a out [rows, 1536]
        try e.g.mm(e.stream, e.normed.ptr, w.kv_a, e.mla_keys.ptr + (mi * cap + pos) * RANK * 2, rows, RANK, c.hidden, false); // raw latent (norm below)
        try e.g.mm(e.stream, e.normed.ptr, w.wk, e.mla_ik_raw_ptr(mi, pos), rows, c.i_dim, c.hidden, false); // ik raw (LN below)
        try e.g.mm(e.stream, e.normed.ptr, w.igate, e.mla_ig.ptr + (mi * cap + pos) * c.i_dim * 2, rows, c.i_dim, c.hidden, false); // gate logits bf16 (the pool reads bf16)
        try e.g.mm(e.stream, e.normed.ptr, w.wproj, e.mla_iw.ptr, rows, c.i_heads, c.hidden, false); // indexer weights (1/64 below)
        // the norms on the slices: keys = rms(kv_norm) of the latent; ik = LN(k_norm)
        try e.ops.rms(e.stream, e.mla_keys.ptr + (mi * cap + pos) * RANK * 2, w.kv_norm, e.mla_keys.ptr + (mi * cap + pos) * RANK * 2, rows, RANK, RANK, RANK, c.eps);
        try e.ops.layerNorm(e.stream, e.mla_ik_raw_ptr(mi, pos), w.k_norm_w, w.k_norm_b, e.mla_ik.ptr + (mi * cap + pos) * c.i_dim * 2, rows, c.i_dim, c.i_dim, c.i_dim, 1e-6);
        // pool the completed blocks
        const first_blk = pos / c.kpool;
        const last_blk = (pos + rows) / c.kpool;
        if (last_blk > first_blk)
            try e.ops.pool(e.stream, e.mla_ik.ptr + mi * cap * c.i_dim * 2, e.mla_ig.ptr + mi * cap * c.i_dim * 2, w.ape, e.mla_pool.ptr + mi * (cap / 4) * c.i_dim * 2, first_blk, last_blk - first_blk);
        // q side: norm q_a out, project nope (q_b) and the indexer queries (wq_b)
        try e.ops.rms(e.stream, e.mla_xp.ptr, w.q_norm, e.mla_qr.ptr, rows, c.q_lora, c.q_lora, c.q_lora, c.eps);
        // both q_b and the indexer queries write the interleaved [rows, qrProj] layout
        // the absorb and the index scores read (ldc = qrProj, wq_b at column H*NOPE)
        const qst = c.q_lora * 2;
        const qpst = c.qrProj() * 2;
        try e.g.mmStrided(e.stream, e.mla_qr.ptr, qst, w.q_b, qst, e.mla_qp.ptr, qpst, rows, H * NOPE, c.q_lora);
        try e.g.mmStrided(e.stream, e.mla_qr.ptr, qst, w.wq_b, qst, e.mla_qp.ptr + H * NOPE * 2, qpst, rows, c.i_heads * c.i_dim, c.q_lora);
        // absorb: ql[h, latent] = q_nope[h, qk] @ K_h[qk, latent] (kv_b's k rows,
        // heads*qk_dim of them, 512-wide; the W is UNtransposed)
        for (0..H) |h| {
            try e.g.mmNT(e.stream, e.mla_qp.ptr + h * NOPE * 2, e.c.qrProj() * 2, w.kv_b + h * RANK * RANK * 2, RANK * 2, e.mla_ql.ptr + h * RANK * 2, H * RANK * 2, rows, RANK, NOPE);
        }
        // attend
        const key_len = pos + rows;
        if (pos + 1 <= c.i_topk) {
            const idxbuf = try e.gpa.alloc(i32, rows * 2051);
            defer e.gpa.free(idxbuf);
            for (0..rows) |r| for (0..2051) |t| {
                idxbuf[r * 2051 + t] = if (t < key_len and t <= pos + r) @intCast(t) else -1;
            };
            try e.mla_idx.upload(0, std.mem.sliceAsBytes(idxbuf));
        } else {
            const s_str = cap / 4 + 1;
            // the indexer weights carry the 1/64: scale via glm_scale? fold by pre-scaling qp's iq half? the metal scales iw - do a fill-scale pass on mla_iw with a tiny kernel? simplest: scores reads iw directly - prescale here via ops.scale? no kernel; bake 1/64 INTO the index scores by scaling qp iq half is wrong (relu breaks). Use the metal's glm_scale on iw: y=x*s. We lack it - multiply iw during GEMM? mm has no alpha. DO: scale via act? no. Add small scale kernel NOW via gather? No - simplest: scale qp iq half is wrong; scale POOL? no. The clean fix: a glm_scale kernel exists in metal; add to ops later - TEMP: fold 1/64 into the ROUTE... no. For bring-up: skip the 1/64 (wrong magnitudes but exercises the path) - NO, parity matters: add glm_scale.
            try e.scale_iw(rows);
            try e.index.?.scores(e.stream, e.mla_qp.ptr + H * NOPE * 2, c.qrProj(), e.mla_iw.ptr, e.mla_pool.ptr + mi * (cap / 4) * c.i_dim * 2, e.mla_scores.ptr, s_str, rows, pos, c.i_heads);
            try e.index.?.select(e.stream, e.mla_scores.ptr, s_str, e.mla_idx.ptr, 2051, rows, pos, c.i_topk / c.kpool, c.keyWidth());
        }
        {
            const kl = try e.gpa.alloc(i32, 1);
            defer e.gpa.free(kl);
            kl[0] = @intCast(key_len);
            try e.mla_klen.uploadAsync(0, std.mem.sliceAsBytes(kl), e.stream.handle);
        }
        try e.dsa.?.attention(e.stream, e.mla_ql.ptr, e.mla_keys.ptr + mi * cap * RANK * 2, e.mla_idx.ptr, e.mla_scale.ptr, e.mla_klen.ptr, e.mla_att.ptr, rows, H);
        // unabsorb: v[h, vdim] = V_h[vdim, latent] @ att[h, latent] (the v rows at
        // heads*qk_dim; x UNtransposed on the right: out = W @ x^T in col-major)
        for (0..H) |h| {
            // V is stored [v_dim, latent]: the transposed-W form (like mm, with strides)
            try e.g.mmStrided(e.stream, e.mla_att.ptr + h * RANK * 2, H * RANK * 2, w.kv_b + (h * 512 + 256) * RANK * 2, RANK * 2, e.mla_vals.ptr + h * c.v_dim * 2, H * c.v_dim * 2, rows, c.v_dim, RANK);
        }
        try e.g.mm(e.stream, e.mla_vals.ptr, w.o, e.branch.ptr, rows, c.hidden, H * c.v_dim, false);
        if ((e.fwd_count == 1 and !e.quiet) and mi <= 1) {
            try e.stream.synchronize();
            try e.dumpBin("mla_xp.bin", e.mla_xp.ptr, rows * c.q_lora * 2);
            try e.dumpBin("mla_qr.bin", e.mla_qr.ptr, rows * c.q_lora * 2);
            try e.dumpBin("mla_keys.bin", e.mla_keys.ptr + (mi * cap + pos) * RANK * 2, rows * RANK * 2);
            try e.dumpBin("mla_qp.bin", e.mla_qp.ptr, rows * c.qrProj() * 2);
            try e.dumpBin("mla_ql.bin", e.mla_ql.ptr, rows * H * RANK * 2);
            try e.dumpBin("mla_att.bin", e.mla_att.ptr, rows * H * RANK * 2);
            try e.dumpBin("mla_vals.bin", e.mla_vals.ptr, rows * H * c.v_dim * 2);
        }
    }

    fn mla_ik_raw_ptr(e: *Engine, mi: usize, pos: usize) u64 {
        return e.mla_scratch_ikraw.ptr + (mi * e.context + pos) * e.c.i_dim * 2;
    }

    fn mla_klen_ptr(e: *Engine, mi: usize) u64 {
        _ = mi;
        return e.mla_klen.ptr;
    }

    fn scale_iw(e: *Engine, rows: usize) !void {
        // iw *= 1/64 elementwise (bf16); a tiny kernel-free path: use act? no -
        // glm_scale added to ops: scale(x, s)
        try e.ops.scaleBy(e.stream, e.mla_iw.ptr, e.mla_iw.ptr, rows * e.c.i_heads, 1.0 / @sqrt(@as(f32, @floatFromInt(e.c.i_heads)))); // sparse.py wscale
    }

    const MatDesc = extern struct { toff: u64, hoff: u32, k: u32, n: u32, tiles_n: u32, bits: u32 };

    /// One-time #741 setup: repack the per-expert weights into contiguous
    /// stacks + TfExl3Mat descriptors, load the experts module + plan router.
    fn m41Stage(e: *Engine, tag: []const u8) void {
        e.m41_stage_n += 1;
        var nm: [40]u8 = undefined;
        const f = std.fmt.bufPrint(&nm, "m41s{d}.log", .{e.m41_stage_n}) catch return;
        var dir = std.Io.Dir.cwd();
        dir.writeFile(e.io, .{ .sub_path = f, .data = tag }) catch {};
    }

    fn m41Sizes(e: *Engine, ee: usize, im: usize, ut: usize, uh: usize, dt: usize, dh: usize) void {
        var line: [160]u8 = undefined;
        const txt = std.fmt.bufPrint(&line, "E={d} imax={d} up_t={d} up_h={d} dn_t={d} dn_h={d}\n", .{ ee, im, ut, uh, dt, dh }) catch return;
        var dir = std.Io.Dir.cwd();
        dir.writeFile(e.io, .{ .sub_path = "m41sizes.log", .data = txt }) catch {};
    }

    fn m41Err(e: *Engine, stage: []const u8, err: anyerror) void {
        var nm: [96]u8 = undefined;
        const f = std.fmt.bufPrint(&nm, "m41setup.log", .{}) catch return;
        var line: [128]u8 = undefined;
        const txt = std.fmt.bufPrint(&line, "{s} ERR={s}\n", .{ stage, @errorName(err) }) catch return;
        var dir = std.Io.Dir.cwd();
        dir.writeFile(e.io, .{ .sub_path = f, .data = txt }) catch {};
    }

    fn m41DescFor(e: *Engine, bi: usize, w: *const weights.Moe) !u64 {
        if (e.m41_layer_descs) |*m| {
            if (m.get(bi * 2)) |ptr| return ptr;
        }
        const E = w.gate.len;
        const MD = @import("cuda").exl3_experts.MatDesc;
        const ub = try e.gpa.alloc(u8, E * 2 * @sizeOf(MD));
        const db = try e.gpa.alloc(u8, E * @sizeOf(MD));
        const du2: [*]MD = @ptrCast(@alignCast(ub.ptr));
        const dd2: [*]MD = @ptrCast(@alignCast(db.ptr));
        for (w.gate, 0..) |gw, ei| {
            du2[ei * 2 + 0] = .{ .trellis = gw.trellis, .suh = gw.suh, .svh = gw.svh, .k = gw.k, .n = gw.n, .tiles_n = gw.tilesN(), .bits = gw.halves / 2 };
            const uw = w.up[ei];
            du2[ei * 2 + 1] = .{ .trellis = uw.trellis, .suh = uw.suh, .svh = uw.svh, .k = uw.k, .n = uw.n, .tiles_n = uw.tilesN(), .bits = uw.halves / 2 };
            const dw4 = w.down[ei];
            dd2[ei] = .{ .trellis = dw4.trellis, .suh = dw4.suh, .svh = dw4.svh, .k = dw4.k, .n = dw4.n, .tiles_n = dw4.tilesN(), .bits = dw4.halves / 2 };
        }
        const up_dev = try cuda.DeviceBuffer.alloc(e.d, ub.len);
        try up_dev.upload(0, ub);
        const dn_dev = try cuda.DeviceBuffer.alloc(e.d, db.len);
        try dn_dev.upload(0, db);
        e.gpa.free(ub);
        e.gpa.free(db);
        if (e.m41_layer_descs == null) e.m41_layer_descs = .{};
        try e.m41_layer_descs.?.put(e.gpa, bi * 2, up_dev.ptr);
        try e.m41_layer_descs.?.put(e.gpa, bi * 2 + 1, dn_dev.ptr);
        return up_dev.ptr;
    }
    fn moeBlock(e: *Engine, w: *const weights.Moe, rows: usize) !void {
        const om0 = if (op_timing) tns() else 0;
        defer if (op_timing) {
            opt_moe += tns() - om0;
        };
        const hidden = e.c.hidden;
        const inter = e.c.moe_inter;
        // router logits fp32 then the route
        try e.g.mm(e.stream, e.normed.ptr, w.router, e.moe_logits.ptr, rows, e.c.experts, hidden, true);
        try e.ops.route(e.stream, e.moe_logits.ptr, w.bias, e.moe_pick.ptr, e.moe_wts.ptr, rows, e.c.experts, e.c.topk, e.c.routed_scale);
        // the shared expert -> ys
        try e.g.mm(e.stream, e.normed.ptr, w.shared_gate, e.mlp_tmp.ptr, rows, inter, hidden, false);
        try e.g.mm(e.stream, e.normed.ptr, w.shared_up, e.mlp_tmp.ptr + rows * inter * 2, rows, inter, hidden, false);
        try e.ops.act(e.stream, e.mlp_tmp.ptr, e.mlp_tmp.ptr + rows * inter * 2, e.mlp_act.ptr, rows, 1, inter, e.c.swiglu_limit);
        try e.g.mm(e.stream, e.mlp_act.ptr, w.shared_down, e.moe_ys.ptr, rows, hidden, inter, false);
        if (rows == 1 and w.gate.len > 0) {
            const pairs = e.c.topk;
            if (e.m41_xe == null) {
                e.m41_xe = try @import("cuda").exl3_experts.Experts.load(e.d, 148);
                const mod = try cuda.Module.load(e.d, @import("cuda").kernels.experts);
                e.m41_router_mod = mod;
                e.m41_router = try @import("cuda").grouped.Router.resolve(mod);
                e.m41_members = try cuda.DeviceBuffer.alloc(e.d, 8 * 4);
                e.m41_items = try cuda.DeviceBuffer.alloc(e.d, 16 * 3 * 4);
                e.m41_counts = try cuda.DeviceBuffer.alloc(e.d, 4);
                e.m41_rank = try cuda.DeviceBuffer.alloc(e.d, 8 * 4);
                e.m41_hist = try cuda.DeviceBuffer.alloc(e.d, 16 * 4);
                var imax2: usize = 0;
                for (w.gate) |gw| imax2 = @max(imax2, gw.n);
                e.m41_inter_max = imax2;
                e.m41_out = try cuda.DeviceBuffer.alloc(e.d, 2 * e.c.topk * imax2 * 2);
                e.m41_act = try cuda.DeviceBuffer.alloc(e.d, e.c.topk * imax2 * 2);
            }
            const desc_up = try e.m41DescFor(e.cur_bi, w);
            const desc_dn = e.m41_layer_descs.?.get(e.cur_bi * 2 + 1).?;
            const p = @import("cuda").grouped.Plan{
                .members = e.m41_members.ptr,
                .items = e.m41_items.ptr,
                .counts = e.m41_counts.ptr,
                .rank = e.m41_rank.ptr,
                .hist = e.m41_hist.ptr,
            };
            try e.m41_router.?.route(e.stream, e.moe_pick.ptr, pairs, pairs, 16, p);
            if (e.m41_stage_n < 3) {
                e.m41_stage_n += 1;
                try e.stream.synchronize();
                var pb4: [80]u8 = undefined;
                const c2b = @import("cuda");
                try (c2b.DeviceBuffer{ .d = e.d, .ptr = e.m41_counts.ptr, .len = 4, .borrowed = true }).download(0, pb4[0..4]);
                try (c2b.DeviceBuffer{ .d = e.d, .ptr = e.m41_items.ptr, .len = 36, .borrowed = true }).download(0, pb4[4..40]);
                try (c2b.DeviceBuffer{ .d = e.d, .ptr = e.m41_members.ptr, .len = 32, .borrowed = true }).download(0, pb4[40..72]);
                var pk4: [16]u8 = undefined;
                try (c2b.DeviceBuffer{ .d = e.d, .ptr = e.moe_pick.ptr, .len = 32, .borrowed = true }).download(0, pk4[0..32]);
                const cnt = std.mem.readInt(i32, pb4[0..4], .little);
                const it = [_]i32{
                    std.mem.readInt(i32, pb4[4..8], .little),
                    std.mem.readInt(i32, pb4[8..12], .little),
                    std.mem.readInt(i32, pb4[12..16], .little),
                    std.mem.readInt(i32, pb4[16..20], .little),
                    std.mem.readInt(i32, pb4[20..24], .little),
                    std.mem.readInt(i32, pb4[24..28], .little),
                };
                const mm = [_]i32{
                    std.mem.readInt(i32, pb4[40..44], .little),
                    std.mem.readInt(i32, pb4[44..48], .little),
                    std.mem.readInt(i32, pb4[48..52], .little),
                    std.mem.readInt(i32, pb4[52..56], .little),
                };
                const picks4 = [_]i32{
                    std.mem.readInt(i32, pk4[0..4], .little),
                    std.mem.readInt(i32, pk4[4..8], .little),
                    std.mem.readInt(i32, pk4[8..12], .little),
                    std.mem.readInt(i32, pk4[12..16], .little),
                };
                var line3: [256]u8 = undefined;
                const txt = std.fmt.bufPrint(&line3, "cnt={d} it={any} mm={any} pk={any} eng={s}\n", .{ cnt, it, mm, picks4, if (e.cont) "B" else "A" }) catch "";
                var dir3 = std.Io.Dir.cwd();
                dir3.writeFile(e.io, .{ .sub_path = "m41plan.log", .data = txt }) catch {};
            }
            // up: x = normed (1 row, stride hidden), out [topk, inter_max]
            const l_up: @import("cuda").experts.Layer = .{
                .format = .exl3,
                .up = e.m41_up.ptr,
                .down = 0,
                .up_scale = e.m41_up_h.ptr,
                .width = e.m41_inter_max,
                .dims = e.c.hidden,
                .experts = w.gate.len,
                .limit = e.c.swiglu_limit,
                .exl3 = .{ .codebook = .mcg }, // our checkpoint is MCG; widths stay 0 (desc-driven)
            };
            // their .up epi writes swiglu(gate,up) inline: one row per pair at
            // [0..topk) - the down family reads those rows directly
            try e.m41_xe.?.runMixed(e.stream, .up, .{ .x = e.normed.ptr, .stride = e.c.hidden, .slots = e.c.topk }, l_up, p, e.m41_out.ptr, pairs, -1, desc_up);
            var l_dn = l_up;
            l_dn.up = e.m41_down.ptr;
            l_dn.up_scale = e.m41_down_h.ptr;
            try e.m41_xe.?.runMixed(e.stream, .down_bf16, .{ .x = e.m41_out.ptr, .stride = e.m41_inter_max, .slots = 0 }, l_dn, p, e.moe_ye.ptr, pairs, -1, desc_dn);
            try e.ops.combine(e.stream, e.moe_ys.ptr, e.moe_ye.ptr, e.moe_wts.ptr, e.branch.ptr, rows, e.c.topk, hidden);
            return;
        }
        // the picks on host drive the per-slot expert calls (staging allocated
        // once at init: a pinned alloc+free per block costs ~ms each)
        if (e.moe_pick_h.len == 0) {
            e.moe_pick_hb = try cuda.HostBuffer.alloc(e.d, e.rows_max * e.c.topk * 4);
            e.moe_pick_h = try e.gpa.alloc(u8, e.rows_max * e.c.topk * 4);
        }
        const pick_host = e.moe_pick_h[0 .. rows * e.c.topk * 4];
        try e.moe_pick.downloadAsync(0, pick_host, e.stream.handle);
        try e.stream.synchronize();
        const picks = std.mem.bytesAsSlice(i32, @as([]align(1) u8, pick_host));
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            var k: usize = 0;
            while (k < e.c.topk) : (k += 1) {
                const ex = picks[r * e.c.topk + k];
                if (ex < 0) continue;
                const eu: usize = @intCast(ex);
                const x_row = e.normed.ptr + r * hidden * 2;
                try e.ex.project(e.stream, x_row, w.gate[eu], e.moe_gu.ptr, 1);
                try e.ex.project(e.stream, x_row, w.up[eu], e.moe_gu.ptr + inter * 2, 1);
                try e.ops.act(e.stream, e.moe_gu.ptr, e.moe_gu.ptr + inter * 2, e.moe_act.ptr, 1, 1, inter, e.c.swiglu_limit);
                const ye_at = e.moe_ye.ptr + (r * e.c.topk + k) * hidden * 2;
                try e.ex.project(e.stream, e.moe_act.ptr, w.down[eu], ye_at, 1);
            }
        }        try e.ops.combine(e.stream, e.moe_ys.ptr, e.moe_ye.ptr, e.moe_wts.ptr, e.branch.ptr, rows, e.c.topk, hidden);
        if ((e.fwd_count == 1 and !e.quiet) and (e.cur_bi == 3 or e.cur_bi == 11)) {
            try e.stream.synchronize();
            {
                try e.dumpBin("moe_logits.bin", e.moe_logits.ptr, rows * e.c.experts * 4);
                try e.dumpBin("moe_pick.bin", e.moe_pick.ptr, rows * e.c.topk * 4);
                try e.dumpBin("moe_wts.bin", e.moe_wts.ptr, rows * e.c.topk * 4);
                try e.dumpBin("moe_ys.bin", e.moe_ys.ptr, rows * hidden * 2);
                try e.dumpBin("moe_ye.bin", e.moe_ye.ptr, rows * e.c.topk * hidden * 2);
            }
        }
    }

    /// Serving split: the head's engine owns no embedding table, so THIS engine
    /// (block 0's) tucks the window's last-token embedding into the handoff.
    pub fn wantMtpEmb(e: *Engine) !void {
        if (e.mtp_emb.len == 0) e.mtp_emb = try cuda.DeviceBuffer.alloc(e.d, e.c.hidden * 2);
    }

    /// Spec decode: snapshot the conv history before a verify round.
    pub fn backupCs(e: *Engine) !void {
        if (e.kda_cs_bak.len == 0) e.kda_cs_bak = try cuda.DeviceBuffer.alloc(e.d, e.kda_cs.len);
        try e.kda_cs_bak.copyFrom(0, e.kda_cs.ptr, e.kda_cs.len, e.stream.handle);
    }

    /// Undo one forward's recurrent advance (a rejected verify round): the KDA
    /// step never overwrites its input buffer, so flipping the per-layer parity
    /// back restores the pre-round state view; cs comes back from the backup.
    /// MLA/MTP caches need nothing: slots are rewritten before they are read.
    pub fn rollbackRound(e: *Engine) !void {
        for (e.kda_st_cur) |*c| c.* = 1 - c.*;
        try e.kda_cs.copyFrom(0, e.kda_cs_bak.ptr, e.kda_cs.len, e.stream.handle);
        try e.stream.synchronize();
    }

    /// Allocate the prefill snapshot buffers (call once, under this engine's context).
    pub fn wantPrefillCache(e: *Engine) !void {
        if (e.pf_on) return;
        const d = e.d;
        e.pf_st0 = try cuda.DeviceBuffer.alloc(d, e.kda_st0.len);
        e.pf_st1 = try cuda.DeviceBuffer.alloc(d, e.kda_st1.len);
        e.pf_cs = try cuda.DeviceBuffer.alloc(d, e.kda_cs.len);
        e.pf_x0 = try cuda.DeviceBuffer.alloc(d, e.x[0].len);
        e.pf_x1 = try cuda.DeviceBuffer.alloc(d, e.x[1].len);
        e.pf_branch = try cuda.DeviceBuffer.alloc(d, e.branch.len);
        e.pf_post = try cuda.DeviceBuffer.alloc(d, e.post.len);
        e.pf_comb = try cuda.DeviceBuffer.alloc(d, e.comb.len);
        e.pf_y = try cuda.DeviceBuffer.alloc(d, e.y.len);
        if (e.mla_keys.len != 0) {
            e.pf_mla_keys = try cuda.DeviceBuffer.alloc(d, e.mla_keys.len);
            e.pf_mla_ik = try cuda.DeviceBuffer.alloc(d, e.mla_ik.len);
            e.pf_mla_ig = try cuda.DeviceBuffer.alloc(d, e.mla_ig.len);
            e.pf_mla_pool = try cuda.DeviceBuffer.alloc(d, e.mla_pool.len);
        }
        e.pf_on = true;
    }

    fn snapCopy(e: *Engine, dst: cuda.DeviceBuffer, src: cuda.DeviceBuffer) !void {
        if (dst.len == 0 or src.len == 0) return;
        try dst.copyFrom(0, src.ptr, src.len, e.stream.handle);
    }

    /// Snapshot the post-prompt state (call after the prompt forward + tail).
    pub fn snapshotPrefill(e: *Engine) !void {
        if (!e.pf_on) return;
        try e.snapCopy(e.pf_st0, e.kda_st0);
        try e.snapCopy(e.pf_st1, e.kda_st1);
        try e.snapCopy(e.pf_cs, e.kda_cs);
        try e.snapCopy(e.pf_x0, e.x[0]);
        try e.snapCopy(e.pf_x1, e.x[1]);
        try e.snapCopy(e.pf_branch, e.branch);
        try e.snapCopy(e.pf_post, e.post);
        try e.snapCopy(e.pf_comb, e.comb);
        try e.snapCopy(e.pf_y, e.y);
        if (e.mla_keys.len != 0) {
            try e.snapCopy(e.pf_mla_keys, e.mla_keys);
            try e.snapCopy(e.pf_mla_ik, e.mla_ik);
            try e.snapCopy(e.pf_mla_ig, e.mla_ig);
            try e.snapCopy(e.pf_mla_pool, e.mla_pool);
        }
        try e.stream.synchronize();
    }

    /// Restore the snapshotted post-prompt state (kda parity/xi/fwd_count come
    /// from the HOST copies the caller saved alongside).
    pub fn restorePrefill(e: *Engine) !void {
        if (!e.pf_on) return;
        try e.snapCopy(e.kda_st0, e.pf_st0);
        try e.snapCopy(e.kda_st1, e.pf_st1);
        try e.snapCopy(e.kda_cs, e.pf_cs);
        try e.snapCopy(e.x[0], e.pf_x0);
        try e.snapCopy(e.x[1], e.pf_x1);
        try e.snapCopy(e.branch, e.pf_branch);
        try e.snapCopy(e.post, e.pf_post);
        try e.snapCopy(e.comb, e.pf_comb);
        try e.snapCopy(e.y, e.pf_y);
        if (e.mla_keys.len != 0) {
            try e.snapCopy(e.mla_keys, e.pf_mla_keys);
            try e.snapCopy(e.mla_ik, e.pf_mla_ik);
            try e.snapCopy(e.mla_ig, e.pf_mla_ig);
            try e.snapCopy(e.mla_pool, e.pf_mla_pool);
        }
        try e.stream.synchronize();
    }

    /// One window of `rows` tokens: ids -> picks (argmax) + logits on device.
    pub fn forward(e: *Engine, ids: []const u32, rows: usize, pos: usize) !void {
        e.fwd_count += 1;
        e.hc_calls = 0;
        
        const hidden = e.c.hidden;
        if (!e.cont) {
            // embed into the four streams
            try e.ids.uploadAsync(0, std.mem.sliceAsBytes(ids)[0 .. rows * 4], e.stream.handle); // stream-ordered: a null-stream upload races the embed kernel
            for (0..4) |s_i| { if (e.f32_streams) try e.ops.embedF32(e.stream, e.ids.ptr, e.w.embed, e.x[e.xi].ptr + s_i * hidden * 4, rows, hidden, 4 * hidden) else try e.ops.embed(e.stream, e.ids.ptr, e.w.embed, e.x[e.xi].ptr + s_i * hidden * 2, rows, hidden, 4 * hidden); }
            e.xi = 0;
            if (e.mtp_emb.len != 0)
                try e.ops.embed(e.stream, e.ids.ptr + (rows - 1) * 4, e.w.embed, e.mtp_emb.ptr, 1, hidden, hidden);
            if (e.dump_on and rows == 2) {
                    {
                        try e.stream.synchronize();
                        const cb = try e.gpa.alloc(u8, 3 * 3 * 12288 * 2);
                        defer e.gpa.free(cb);
                        try (cuda.DeviceBuffer{ .d = e.d, .ptr = e.kda_cs.ptr, .len = cb.len, .borrowed = true }).download(0, cb);
                        var nmX: [40]u8 = undefined;
                        e.dump_serial += 1;
                        const fX = std.fmt.bufPrint(&nmX, "kse{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                        var dirX = std.Io.Dir.cwd();
                        dirX.writeFile(e.io, .{ .sub_path = fX, .data = cb }) catch {};
                    }
            }
            if (e.dump_on) {
                try e.stream.synchronize();
                const n = rows * 4 * hidden * 4;
                const hb2 = try e.gpa.alloc(u8, n);
                defer e.gpa.free(hb2);
                try e.x[0].download(0, hb2);
                var fn2: [32]u8 = undefined;
                e.dump_serial += 1;
                const fname = std.fmt.bufPrint(&fn2, "xembS{s}S{d}.bin", .{ @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                var dir = std.Io.Dir.cwd();
                dir.writeFile(e.io, .{ .sub_path = fname, .data = hb2 }) catch {};
            }
        }
        var ki: usize = 0;
        var mi_mla: usize = 0;
        for (e.w.layers, e.w.blocks_lo..) |*L, bi| {
            e.cur_bi = @intCast(bi);
            if (e.dump_on and bi <= 21) {
                {
                    try e.stream.synchronize();
                    const cb = try e.gpa.alloc(u8, 147456);
                    defer e.gpa.free(cb);
                    try (cuda.DeviceBuffer{ .d = e.d, .ptr = e.kda_cs.ptr, .len = 147456, .borrowed = true }).download(0, cb);
                    var nmY: [40]u8 = undefined;
                    e.dump_serial += 1;
                    const fY = std.fmt.bufPrint(&nmY, "ksb{d}{s}S{d}.bin", .{ bi, @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                    var dirY = std.Io.Dir.cwd();
                    dirY.writeFile(e.io, .{ .sub_path = fY, .data = cb }) catch {};
                }
                try e.stream.synchronize();
                const n = rows * 4 * e.c.hidden * 4;
                const xb = try e.gpa.alloc(u8, n);
                defer e.gpa.free(xb);
                try e.x[e.xi].download(0, xb);
                var nm9: [40]u8 = undefined;
                e.dump_serial += 1;
                const f9 = std.fmt.bufPrint(&nm9, "kx{d}{s}S{d}.bin", .{ bi, @as([]const u8, if (e.cont) "b" else "a"), e.dump_serial }) catch unreachable;
                var dir9 = std.Io.Dir.cwd();
                dir9.writeFile(e.io, .{ .sub_path = f9, .data = xb }) catch {};
            }
            const hc_a = L.hc_attn orelse break; // the MTP block runs after the tail
            if ((e.fwd_count == 1 and !e.quiet) and bi == 4) {
                try e.stream.synchronize();
                try e.dumpBin("at4_xpre.bin", e.x[e.xi].ptr, rows * 4 * e.c.hidden * 4);
                try e.dumpBin("at4_branch.bin", e.branch.ptr, rows * e.c.hidden * 2);
                try e.dumpBin("at4_afn.bin", hc_a.fnp, 24 * 4 * e.c.hidden * 2);
            }
            try e.hcGates(rows, hc_a, L.in_norm);
            if ((e.fwd_count == 1 and !e.quiet) and bi == 4) {
                try e.stream.synchronize();
                try e.dumpBin("at4_post.bin", e.post.ptr, 16 * 4);
                try e.dumpBin("at4_comb.bin", e.comb.ptr, 16 * 4 * 4);
                try e.dumpBin("at4_xpost.bin", e.x[e.xi].ptr, rows * 4 * e.c.hidden * 4);
                try e.dumpBin("at4_abase.bin", hc_a.base, 24 * 4);
                try e.dumpBin("at4_ascale.bin", hc_a.scale, 3 * 4);
            }
            if (e.attn_sums) |asums| {
                if ((e.fwd_count == 1 and !e.quiet) and bi >= 1) {
                try e.stream.synchronize();
                const hd = e.c.hidden;
                const xb = try e.gpa.alloc(u8, 4 * hd * 4);
                defer e.gpa.free(xb);
                try e.x[e.xi].download(0, xb);
                var acc: f64 = 0;
                if (e.f32_streams) {
                    const fv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, xb));
                    for (0..hd) |d| {
                        var m: f64 = 0;
                        for (0..4) |st| m += fv[st * hd + d];
                        acc += m / 4.0;
                    }
                } else {
                    const hv = std.mem.bytesAsSlice(u16, @as([]align(1) u8, xb));
                    for (0..hd) |d| {
                        var m: f64 = 0;
                        for (0..4) |st| m += @as(f32, @bitCast(@as(u32, hv[st * hd + d]) << 16));
                        acc += m / 4.0;
                    }
                }
                asums[bi - 1] = acc;
                }
            }
            if (((e.fwd_count == 1 or e.dump_on) and !e.quiet) and bi <= 45) try e.dumpPb(bi, "na", e.normed.ptr, rows);
            if (bi == 0 and !e.quiet) {
                try e.b0Print("normed", e.normed);
                const nb = try e.gpa.alloc(u8, e.c.hidden * 2);
                defer e.gpa.free(nb);
                e.normed.download(0, nb) catch {};
                var dir = std.Io.Dir.cwd();
                dir.writeFile(e.io, .{ .sub_path = "normed0.bin", .data = nb }) catch {};
                std.debug.print("  normed row0 dumped 8192B\n", .{});
            }
            const ot0 = if (op_timing and bi == 0) tns() else 0;
            switch (L.attn) {
                .kda => |*w| {
                    const oa0 = if (op_timing and bi == 0) tns() else 0;
                    try e.kdaLayer(ki, w, rows);
                    if (op_timing and bi == 0) opt_gates += tns() - oa0;
                    ki += 1;
                    if (bi == 0 and !e.quiet) try e.b0Print("attn", e.y);
                },
                .mla => {
                    if (!e.skip_mla) try e.mlaLayer(mi_mla, switch (L.attn) {
                        .mla => |*x| x,
                        .kda => unreachable,
                    }, rows, pos) else try e.branch.fill8(0, e.stream.handle);
                    mi_mla += 1;
                },
            }
            if (((e.fwd_count == 1 or e.dump_on) and !e.quiet) and bi <= 45) try e.dumpPb(bi, "ba", e.branch.ptr, rows);
            if ((e.fwd_count == 1 and !e.quiet) and bi <= 45) {
                try e.stream.synchronize();
                var n1: [24]u8 = undefined;
                try e.dumpBin(std.fmt.bufPrint(&n1, "pb{d}_xpre.bin", .{bi}) catch unreachable, e.x[if (e.xi < 2) e.xi else 0].ptr, rows * 4 * e.c.hidden * 4);
                var n2: [24]u8 = undefined;
                try e.dumpBin(std.fmt.bufPrint(&n2, "pb{d}_fnp.bin", .{bi}) catch unreachable, (L.hc_ffn orelse unreachable).fnp, 24 * 4 * e.c.hidden * 2);
                try e.dumpBin("hc_fbase.bin", (L.hc_ffn orelse unreachable).base, 24 * 4);
                try e.dumpBin("hc_fscale.bin", (L.hc_ffn orelse unreachable).scale, 3 * 4);
            }
            try e.hcFold(hc_a.fnp, rows);
            try e.hcGates(rows, L.hc_ffn orelse return error.UnexpectedMtp, L.post_norm);
            if (((e.fwd_count == 1 or e.dump_on) and !e.quiet) and bi <= 45) try e.dumpPb(bi, "nf", e.normed.ptr, rows);
            if ((e.fwd_count == 1 and !e.quiet) and bi <= 45) {
                var n3: [24]u8 = undefined;
                try e.dumpBin(std.fmt.bufPrint(&n3, "pb{d}_post.bin", .{bi}) catch unreachable, e.post.ptr, rows * 4 * 4);
                var n4: [24]u8 = undefined;
                try e.dumpBin(std.fmt.bufPrint(&n4, "pb{d}_comb.bin", .{bi}) catch unreachable, e.comb.ptr, rows * 16 * 4);
                var n5: [24]u8 = undefined;
                try e.dumpBin(std.fmt.bufPrint(&n5, "pb{d}_xpost.bin", .{bi}) catch unreachable, e.x[if (e.xi < 2) e.xi else 0].ptr, rows * 4 * e.c.hidden * 4);
                var n7: [24]u8 = undefined;
                try e.dumpBin(std.fmt.bufPrint(&n7, "pb{d}_part.bin", .{bi}) catch unreachable, e.part.ptr, rows * hc.partBytes(1));
            }
            switch (L.mlp) {
                .dense => |*w| try e.denseMlp(w, rows),
                .moe => |*w| try e.moeBlock(w, rows),
            }
            if (op_timing and bi == 0) opt_gates += tns() - ot0;
            try e.hcFold((L.hc_ffn orelse return error.UnexpectedMtp).fnp, rows);
            if (((e.fwd_count == 1 or e.dump_on) and !e.quiet) and bi <= 45) try e.dumpPb(bi, "bf", e.branch.ptr, rows);
            if (e.block_sums) |sums| {
                // their tap metric: row 0's mean of the 4 streams, summed over D
                const hd = e.c.hidden;
                const b = e.gpa.alloc(u8, 4 * hd * 4) catch |err| return err;
                defer e.gpa.free(b);
                e.stream.synchronize() catch {};
                e.x[e.xi].download(0, b) catch {};
                var acc: f64 = 0;
                if (e.f32_streams) {
                    const fv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, b));
                    for (0..hd) |d| {
                        var m: f64 = 0;
                        for (0..4) |st| m += fv[st * hd + d];
                        acc += m / 4.0;
                    }
                } else {
                    const hv = std.mem.bytesAsSlice(u16, @as([]align(1) u8, b));
                    for (0..hd) |d| {
                        var m: f64 = 0;
                        for (0..4) |st| m += @as(f32, @bitCast(@as(u32, hv[st * hd + d]) << 16));
                        acc += m / 4.0;
                    }
                }
                sums[bi] = acc;
                if (bi == 0 and !e.quiet) {
                    e.stream.synchronize() catch {};
                    var br: f64 = 0;
                    const bb = e.gpa.alloc(u8, e.c.hidden * 2) catch |err| return err;
                    defer e.gpa.free(bb);
                    e.branch.download(0, bb) catch {};
                    for (0..e.c.hidden) |d| {
                        const h: u16 = std.mem.readInt(u16, bb[d * 2 ..][0..2], .little);
                        br += @as(f32, @bitCast(@as(u32, h) << 16));
                    }
                    var yr: f64 = 0;
                    const yb = e.gpa.alloc(u8, e.c.hidden * 2) catch |err| return err;
                    defer e.gpa.free(yb);
                    e.y.download(0, yb) catch {};
                    for (0..e.c.hidden) |d| {
                        const h: u16 = std.mem.readInt(u16, yb[d * 2 ..][0..2], .little);
                        yr += @as(f32, @bitCast(@as(u32, h) << 16));
                    }
                    std.debug.print("  b0 branch sum {d:.2} attn-y sum {d:.2}\n", .{br, yr});
                }
            }
        }
        if (e.stop_after_blocks) return; // the handoff: x, branch, post, comb in place
        // glue's tail: every branch already folded at its own boundary; the
        // mean reads the streams directly (the old extra fn_zero fold re-folded
        // the stale last branch - corruption at every window end)
        if (e.f32_streams) try e.ops.streamMeanF32(e.stream, e.x[e.xi].ptr, e.y.ptr, rows, hidden) else try e.ops.streamMean(e.stream, e.x[e.xi].ptr, e.y.ptr, rows, hidden);
        try e.ops.rms(e.stream, e.y.ptr, e.w.final_norm, e.normed.ptr, rows, hidden, hidden, hidden, e.c.eps);
        if (rows == 4) {
            try e.stream.synchronize();
            try e.dumpBin("tail_x.bin", e.x[e.xi].ptr, rows * 4 * hidden * 2);
            try e.dumpBin("tail_normed.bin", e.normed.ptr, rows * hidden * 2);
        }
        try e.g.mm(e.stream, e.normed.ptr, e.w.head, e.logits.ptr, e.rows_max, e.c.vocab, hidden, false);
        try e.ops.argmax(e.stream, e.logits.ptr, e.picks.ptr, rows, e.c.vocab);
        if (e.mtp_li != null) try e.mtpDraft(rows, pos + rows - 1);
    }

    /// The MTP draft head over the window's LAST row: enorm(embed(next)) | hnorm(h),
    /// eh_proj, a plain MLA + MoE block (own cache plane), shared_head norm, the
    /// shared lm_head. Leaves the draft pick in mtp_pick.
    fn mtpDump1(e: *Engine, what: []const u8, ptr: u64, n: usize) void {
        if (e.mtp_dump_nth == 0 or e.mtp_calls != e.mtp_dump_nth) return;
        const b = e.gpa.alloc(u8, n) catch return;
        defer e.gpa.free(b);
        (cuda.DeviceBuffer{ .d = e.d, .ptr = ptr, .len = n, .borrowed = true }).download(0, b) catch return;
        var nm: [48]u8 = undefined;
        const f = std.fmt.bufPrint(&nm, "/tmp/m1_{d}_{s}.bin", .{ e.mtp_tag, what }) catch return;
        var dir = std.Io.Dir.cwd();
        dir.writeFile(e.io, .{ .sub_path = f, .data = b }) catch {};
    }

    fn mtpDraft(e: *Engine, rows: usize, pos: usize) !void {
        const D = e.c.hidden;
        const L = &e.w.layers[e.mtp_li.?];
        e.mtp_calls += 1;
        e.mtpDump1("emb", e.mtp_emb.ptr, D * 2);
        e.mtpDump1("yin", e.y.ptr + (rows - 1) * D * 2, D * 2);
        const plane_rows = 20;
        e.mtpDump1("plane", e.mla_keys.ptr + @as(usize, e.mtp_mi) * e.context * e.c.kv_lora * 2, plane_rows * e.c.kv_lora * 2);
        try e.ops.rms(e.stream, e.mtp_emb.ptr, L.enorm, e.mtp_eh.ptr, 1, D, D, 2 * D, e.c.eps);
        try e.ops.rms(e.stream, e.y.ptr + (rows - 1) * D * 2, L.hnorm, e.mtp_eh.ptr + D * 2, 1, D, D, 2 * D, e.c.eps);
        try e.g.mm(e.stream, e.mtp_eh.ptr, L.eh_proj, e.mtp_x.ptr, 1, D, 2 * D, false);
        try e.ops.rms(e.stream, e.mtp_x.ptr, L.in_norm, e.normed.ptr, 1, D, D, D, e.c.eps);
        try e.mlaLayer(e.mtp_mi, &L.attn.mla, 1, pos); // -> e.branch
        try e.ops.add(e.stream, e.mtp_x.ptr, e.branch.ptr, e.mtp_out.ptr, D);
        try e.ops.rms(e.stream, e.mtp_out.ptr, L.post_norm, e.normed.ptr, 1, D, D, D, e.c.eps);
        try e.moeBlock(&L.mlp.moe, 1); // -> e.branch
        try e.ops.add(e.stream, e.mtp_out.ptr, e.branch.ptr, e.mtp_x.ptr, D);
        try e.ops.rms(e.stream, e.mtp_x.ptr, L.shared_head_norm, e.normed.ptr, 1, D, D, D, e.c.eps);
        try e.g.mm(e.stream, e.normed.ptr, e.w.head, e.logits.ptr, e.rows_max, e.c.vocab, D, false);
        try e.ops.argmax(e.stream, e.logits.ptr, e.mtp_pick.ptr, 1, e.c.vocab);
        if (e.mtp_dump_nth != 0 and e.mtp_calls == e.mtp_dump_nth) {
            try e.stream.synchronize();
            e.mtpDump1("eh", e.mtp_eh.ptr, 2 * D * 2);
            e.mtpDump1("mx", e.mtp_x.ptr, D * 2);
            e.mtpDump1("normed", e.normed.ptr, D * 2);
            e.mtpDump1("pick", e.mtp_pick.ptr, 4);
        }
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
        inline for (.{ "mtp_eh", "mtp_x", "mtp_out", "mtp_pick", "mtp_emb", "kda_cs_bak" }) |f| {
            if (@field(e, f).len != 0) @field(e, f).free();
        }
        inline for (.{ "ids", "branch", "normed", "part", "post", "comb", "y", "proj", "qkv", "fg", "bv", "kda_st0", "kda_st1", "kda_cs", "kda_cw", "kda_conv_p", "logits", "picks", "scratch" }) |f| {
            if (@field(e, f).len != 0) @field(e, f).free();
        }
        e.x[0].free();
        e.x[1].free();
        e.g.deinit();
        e.kdk.deinit();
        e.ops.unload();
        e.hck.deinit();
        e.ex.deinit();
        e.stream.deinit();
        e.w.deinit();
        gpa.destroy(e.w);
        gpa.destroy(e);
    }
};
