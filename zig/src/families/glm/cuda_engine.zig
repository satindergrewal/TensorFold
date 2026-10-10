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
    mlp_tmp: cuda.DeviceBuffer, // [rows, 2*dense_inter] bf16
    mlp_act: cuda.DeviceBuffer, // [rows, dense_inter] bf16
    scratch: cuda.DeviceBuffer, // exl3/experts scratch

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
            .mlp_tmp = undefined,
            .mlp_act = undefined,
            .scratch = undefined,
        };
        e.stream = try cuda.Stream.init(d, true);
        errdefer e.stream.deinit();
        e.ex = try exl3.Kernels.load(d);
        e.hck = try hc.Kernels.load(d);
        e.ops = try glue.Ops.load(d);
        e.kdk = try kda.Kernels.load(d);
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
        const eps_v = try gpa.alloc(f32, 1);
        defer gpa.free(eps_v);
        eps_v[0] = e.c.hc_eps;
        e.eps_buf = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(eps_v));
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

    /// One window of `rows` tokens: ids -> picks (argmax) + logits on device.
    pub fn forward(e: *Engine, ids: []const u32, rows: usize) !void {
        const hidden = e.c.hidden;
        // embed into the four streams
        try e.ids.upload(0, std.mem.sliceAsBytes(ids)[0 .. rows * 4]);
        for (0..4) |s_i| try e.ops.embed(e.stream, e.ids.ptr, e.w.embed, e.x[e.xi].ptr + s_i * hidden * 2, rows, hidden);
        e.xi = 0;
        var ki: usize = 0;
        for (e.w.layers, e.w.blocks_lo..) |*L, bi| {
            const hc_a = L.hc_attn orelse return error.UnexpectedMtp;
            try e.boundary(rows, hc_a, L.in_norm, ki > 0 or e.w.blocks_lo > 0);
            switch (L.attn) {
                .kda => |*w| try e.kdaLayer(ki, w, rows),
                .mla => return error.MlaUnimplemented,
            }
            ki += 1;
            try e.boundary(rows, L.hc_ffn, L.post_norm, true);
            switch (L.mlp) {
                .dense => |*w| try e.denseMlp(w, rows),
                .moe => return error.MoeUnimplemented,
            }
            _ = bi;
        }
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
