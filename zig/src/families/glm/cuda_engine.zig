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
        const id = key(rows | (xs << 8), n, k, false) ^ (ws_ << 20) ^ (os << 26);
        const gop = try g.map.getOrPut(id);
        if (!gop.found_existing) {
            // build with custom layouts (the shared init hardcodes ld=k)
            var lin: cuda.cublaslt.Linear = undefined;
            lin = try cuda.cublaslt.Linear.initStrided(&g.lt, rows, n, k, .bf16, 64 << 20, xs / 2, ws_ / 2, os / 2);
            gop.value_ptr.* = lin;
        }
        gop.value_ptr.run(x, w, out, g.ws.ptr, g.ws.len, s.handle) catch |err| {
            std.debug.print("mmS FAIL rows={d} n={d} k={d}" ++ "\n", .{ rows, n, k });
            return err;
        };
    }

    /// D[rows, n] = X[rows, k] . W[k, n] (W UNtransposed: row-major x @ W).
    fn mmNT(g: *Gemms, s: cuda.Stream, x: u64, xs: usize, w: u64, ws_: usize, out: u64, os: usize, rows: usize, n: usize, k: usize) !void {
        const id = key(rows | (xs << 8), n, k, false) ^ (ws_ << 20) ^ (os << 26) ^ 0x5A5A;
        const gop = try g.map.getOrPut(id);
        if (!gop.found_existing)
            gop.value_ptr.* = try cuda.cublaslt.Linear.initNT(&g.lt, rows, n, k, .bf16, 64 << 20, xs / 2, ws_ / 2, os / 2);
        gop.value_ptr.run(x, w, out, g.ws.ptr, g.ws.len, s.handle) catch |err| {
            std.debug.print("mmNT FAIL rows={d} n={d} k={d}" ++ "\n", .{ rows, n, k });
            return err;
        };
    }

    /// D[rows, n] bf16 (or fp32) = X[rows, k] . W[n, k]^T.
    fn mm(g: *Gemms, s: cuda.Stream, x: u64, w: u64, out: u64, rows: usize, n: usize, k: usize, f32_out: bool) !void {
        const id = key(rows, n, k, f32_out);
        const gop = try g.map.getOrPut(id);
        if (!gop.found_existing)
            gop.value_ptr.* = try cuda.cublaslt.Linear.init(&g.lt, rows, n, k, if (f32_out) .f32 else .bf16, 64 << 20);
        gop.value_ptr.run(x, w, out, g.ws.ptr, g.ws.len, s.handle) catch |err| {
            std.debug.print("mm FAIL rows={d} n={d} k={d} f32={}" ++ "\n", .{ rows, n, k, f32_out });
            return err;
        };
    }
};

pub const Engine = struct {
    gpa: std.mem.Allocator,
    block_sums: ?[]f64 = null, // set by the test: per-block x-stream fingerprints
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
    kda_st_cur: usize = 0,
    kda_cs: cuda.DeviceBuffer, // [n_kda, TAPS-1, 3W] bf16 history
    kda_cw: cuda.DeviceBuffer, // [TAPS, 3W] bf16, transposed at load
    kda_conv_p: cuda.DeviceBuffer, // [rows, 3W]: the conv reads cs history + p
    logits: cuda.DeviceBuffer, // [rows, vocab] bf16
    picks: cuda.DeviceBuffer, // u32 [rows]
    eps_buf: cuda.DeviceBuffer,
    fn_zero: cuda.DeviceBuffer,
    moe_pick: cuda.DeviceBuffer,
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

    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8, opts: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        const w = try weights.load(gpa, io, d, dir, .{ .lo = opts.blocks_lo, .hi = opts.blocks_hi });
        errdefer {
            w.deinit();
            gpa.destroy(w);
        }
        const rows_max: usize = 16;
        e.* = .{
            .gpa = gpa,
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
        e.x[0] = try cuda.DeviceBuffer.alloc(d, rows * 4 * hidden * 2);
        e.x[1] = try cuda.DeviceBuffer.alloc(d, rows * 4 * hidden * 2);
        e.branch = try cuda.DeviceBuffer.alloc(d, rows * hidden * 2);
        e.normed = try cuda.DeviceBuffer.alloc(d, rows * hidden * 2);
        e.part = try cuda.DeviceBuffer.alloc(d, rows * hc.partBytes(1));
        e.post = try cuda.DeviceBuffer.alloc(d, rows * 4 * 4);
        e.comb = try cuda.DeviceBuffer.alloc(d, rows * 16 * 4);
        e.y = try cuda.DeviceBuffer.alloc(d, rows * hidden * 2);
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
        try e.kda_st0.fill32(0, null);
        try e.kda_st1.fill32(0, null);
        try e.kda_cs.fill8(0, null);
        // the KDA conv weights: [TAPS, 3W] transposed from the per-tensor
        // [W, 1, TAPS] uploads (element (c, j) at c*TAPS + j)
        {
            const Wq: usize = e.c.kdaWidth();
            e.kda_cw = try cuda.DeviceBuffer.alloc(d, 4 * 3 * Wq * 2);
            var ki: usize = 0;
            for (e.w.layers) |*L| {
                const wk = switch (L.attn) {
                    .kda => |*x| x,
                    .mla => continue,
                };
                const srcs = [3]u64{ wk.q_conv, wk.k_conv, wk.v_conv };
                const taps_g: usize = 4;
                for (0..taps_g) |j| for (0..3) |part| {
                    try e.ops.gather(e.stream, srcs[part] + j * 2, e.kda_cw.ptr + (j * 3 + part) * Wq * 2, Wq, 4);
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
            const eps2 = try gpa.alloc(f32, 1);
            defer gpa.free(eps2);
            eps2[0] = e.c.eps;
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
    fn boundary(e: *Engine, rows: usize, h: ?weights.Hc, norm_w: u64, pending: bool) !void {
        const hidden = e.c.hidden;
        const x_old = e.x[e.xi].ptr;
        const x_new = e.x[1 - e.xi].ptr;
        if (h == null) {
            // the final boundary: the expand alone, the streams become the
            // mean's source
            try e.hck.expandMix(e.stream, x_old, e.branch.ptr, e.post.ptr, e.comb.ptr, e.fn_zero.ptr, x_new, e.part.ptr, rows, true);
            e.xi = 1 - e.xi;
            return;
        }
        const layer_h = h.?;
        try e.hck.expandMix(e.stream, x_old, e.branch.ptr, e.post.ptr, e.comb.ptr, layer_h.fnp, x_new, e.part.ptr, rows, pending);
        _ = hidden;
        // without the expand x_new was not written: the collapse reads the old
        try e.hck.split(e.stream, if (pending) x_new else x_old, e.part.ptr, layer_h.scale, layer_h.base, norm_w, e.eps_buf.ptr, e.normed.ptr, e.post.ptr, e.comb.ptr, rows);
        if (pending) e.xi = 1 - e.xi;
    }

    /// KDA attention block `li` (its dense kind index) on normed -> branch.
    fn kdaLayer(e: *Engine, li: usize, w: *const weights.Kda, rows: usize) !void {
        const c = &e.c;
        const Wq: usize = c.kdaWidth();
        const D: usize = c.kda_dim;
        const ps = c.kdaProj();
        const H: usize = c.kda_heads;
        const st_bytes: usize = H * D * D * 4;
        // the stacked projections
        try e.g.mm(e.stream, e.normed.ptr, w.q, e.proj.ptr, rows, Wq, c.hidden, false);
        try e.g.mm(e.stream, e.normed.ptr, w.k, e.proj.ptr + Wq * 2, rows, Wq, c.hidden, false);
        try e.g.mm(e.stream, e.normed.ptr, w.v, e.proj.ptr + 2 * Wq * 2, rows, Wq, c.hidden, false);
        try e.g.mm(e.stream, e.normed.ptr, w.f_a, e.proj.ptr + 3 * Wq * 2, rows, D, c.hidden, false);
        try e.g.mm(e.stream, e.normed.ptr, w.g_a, e.proj.ptr + (3 * Wq + D) * 2, rows, D, c.hidden, false);
        try e.g.mm(e.stream, e.normed.ptr, w.b, e.proj.ptr + (3 * Wq + 2 * D) * 2, rows, H, c.hidden, false);
        // f_b / g_b on the f_a/g_a slices
        try e.g.mm(e.stream, e.proj.ptr + 3 * Wq * 2, w.f_b, e.fg.ptr, rows, Wq, D, false);
        try e.g.mm(e.stream, e.proj.ptr + (3 * Wq + D) * 2, w.g_b, e.fg.ptr + rows * Wq * 2, rows, Wq, D, false);
        // b packed from the proj slice
        try e.ops.copy2d(e.stream, e.proj.ptr + (3 * Wq + 2 * D) * 2, e.bv.ptr, rows, H, ps);
        // conv: history per layer, weights the engine built at load (kda_cw)
        const cs_at = e.kda_cs.ptr + li * 3 * Wq * 2;
        try e.kdk.conv(e.stream, e.proj.ptr, ps, cs_at, e.kda_cw.ptr, e.qkv.ptr, rows);
        // the step, state ping-pong
        const st_in = if (e.kda_st_cur == 0) e.kda_st0.ptr else e.kda_st1.ptr;
        const st_out = if (e.kda_st_cur == 0) e.kda_st1.ptr else e.kda_st0.ptr;
        try e.kdk.step(e.stream, e.qkv.ptr, e.fg.ptr, e.bv.ptr, w.a, w.dt_bias, st_in + li * st_bytes, w.o_norm, c.lower_bound, c.hc_eps, rows, st_out + li * st_bytes, e.y.ptr);
        e.kda_st_cur = 1 - e.kda_st_cur; // TODO: per-layer, not global, once windows chain
        // o_proj -> branch
        try e.g.mm(e.stream, e.y.ptr, w.o, e.branch.ptr, rows, c.hidden, Wq, false);
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
        try e.g.mm(e.stream, e.mla_qr.ptr, w.q_b, e.mla_qp.ptr, rows, H * NOPE, c.q_lora, false);
        try e.g.mm(e.stream, e.mla_qr.ptr, w.wq_b, e.mla_qp.ptr + H * NOPE * 2, rows, c.i_heads * c.i_dim, c.q_lora, false);
        // absorb: ql[h, latent] = q_nope[h, qk] @ K_h[qk, latent] (kv_b's k rows,
        // heads*qk_dim of them, 512-wide; the W is UNtransposed)
        for (0..H) |h| {
            try e.g.mmNT(e.stream, e.mla_qp.ptr + h * NOPE * 2, e.c.qrProj() * 2, w.kv_b + h * NOPE * RANK * 2, RANK * 2, e.mla_ql.ptr + h * RANK * 2, H * RANK * 2, rows, RANK, NOPE);
        }
        // attend
        const key_len = pos + rows;
        if (pos + 1 <= c.i_topk) {
            const idxbuf = try e.gpa.alloc(i32, rows * 2051);
            defer e.gpa.free(idxbuf);
            for (0..rows) |r| for (0..2051) |t| {
                idxbuf[r * 2051 + t] = if (t < key_len) @intCast(t) else -1;
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
            try e.mla_klen.upload(0, std.mem.sliceAsBytes(kl));
        }
        try e.dsa.?.attention(e.stream, e.mla_ql.ptr, e.mla_keys.ptr + mi * cap * RANK * 2, e.mla_idx.ptr, e.mla_scale.ptr, e.mla_klen.ptr, e.mla_att.ptr, rows, H);
        // unabsorb: v[h, vdim] = V_h[vdim, latent] @ att[h, latent] (the v rows at
        // heads*qk_dim; x UNtransposed on the right: out = W @ x^T in col-major)
        for (0..H) |h| {
            try e.g.mmNT(e.stream, e.mla_att.ptr + h * RANK * 2, H * RANK * 2, w.kv_b + (h * 512 + 256) * RANK * 2, RANK * 2, e.mla_vals.ptr + h * c.v_dim * 2, H * c.v_dim * 2, rows, c.v_dim, RANK);
        }
        try e.g.mm(e.stream, e.mla_vals.ptr, w.o, e.branch.ptr, rows, c.hidden, H * c.v_dim, false);
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

    /// The MoE block: router, top-8 route, shared expert, the routed experts
    /// through exl3 per (row, slot) - correctness first, batching later.
    fn moeBlock(e: *Engine, w: *const weights.Moe, rows: usize) !void {
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
        // the picks on host drive the per-slot expert calls
        var pick_bytes = try cuda.HostBuffer.alloc(e.d, rows * e.c.topk * 4);
        defer pick_bytes.free();
        const pick_host = try e.gpa.alloc(u8, rows * e.c.topk * 4);
        defer e.gpa.free(pick_host);
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
        }
        try e.ops.combine(e.stream, e.moe_ys.ptr, e.moe_ye.ptr, e.moe_wts.ptr, e.branch.ptr, rows, e.c.topk, hidden);
    }

    /// One window of `rows` tokens: ids -> picks (argmax) + logits on device.
    pub fn forward(e: *Engine, ids: []const u32, rows: usize) !void {
        const hidden = e.c.hidden;
        if (!e.cont) {
            // embed into the four streams
            try e.ids.upload(0, std.mem.sliceAsBytes(ids)[0 .. rows * 4]);
            for (0..4) |s_i| try e.ops.embed(e.stream, e.ids.ptr, e.w.embed, e.x[e.xi].ptr + s_i * hidden * 2, rows, hidden);
            e.xi = 0;
        }
        var ki: usize = 0;
        var mi_mla: usize = 0;
        const pos: usize = 0;
        for (e.w.layers, e.w.blocks_lo..) |*L, bi| {
            const hc_a = L.hc_attn orelse return error.UnexpectedMtp;
            try e.boundary(rows, hc_a, L.in_norm, ki > 0 or e.w.blocks_lo > 0);
            switch (L.attn) {
                .kda => |*w| try e.kdaLayer(ki, w, rows),
                .mla => {
                    if (!e.skip_mla) try e.mlaLayer(mi_mla, switch (L.attn) {
                        .mla => |*x| x,
                        .kda => unreachable,
                    }, rows, pos) else try e.branch.fill8(0, e.stream.handle);
                    mi_mla += 1;
                },
            }
            ki += 1;
            try e.boundary(rows, L.hc_ffn, L.post_norm, true);
            switch (L.mlp) {
                .dense => |*w| try e.denseMlp(w, rows),
                .moe => |*w| try e.moeBlock(w, rows),
            }
            if (e.block_sums) |sums| {
                // their tap metric: row 0's mean of the 4 streams, summed over D
                const hd = e.c.hidden;
                const b = e.gpa.alloc(u8, 4 * hd * 2) catch |err| return err;
                defer e.gpa.free(b);
                e.x[e.xi].download(0, b) catch {};
                var acc: f64 = 0;
                for (0..hd) |d| {
                    var m: f32 = 0;
                    for (0..4) |st| {
                        const h: u16 = std.mem.readInt(u16, b[(st * hidden + d) * 2 ..][0..2], .little);
                        m += @as(f32, @bitCast(@as(u32, h) << 16));
                    }
                    acc += m / 4.0;
                }
                sums[bi] = acc;
            }
        }
        if (e.stop_after_blocks) return; // the handoff: x, branch, post, comb in place
        // the final boundary folds the last branch, then the tail
        try e.boundary(rows, null, 0, true);
        try e.ops.streamMean(e.stream, e.x[e.xi].ptr, e.y.ptr, rows, hidden);
        try e.ops.rms(e.stream, e.y.ptr, e.w.final_norm, e.normed.ptr, rows, hidden, hidden, hidden, e.c.eps);
        try e.g.mm(e.stream, e.normed.ptr, e.w.head, e.logits.ptr, e.rows_max, e.c.vocab, hidden, false);
        try e.ops.argmax(e.stream, e.logits.ptr, e.picks.ptr, rows, e.c.vocab);
    }

    pub fn deinit(e: *Engine) void {
        const gpa = e.gpa;
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
