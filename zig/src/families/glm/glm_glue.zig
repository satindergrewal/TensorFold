//! The dense ops' host side: glm_ops.cu's launches, plus the cublasLt handle.
//! GEMMs are the caller's Linear (m baked per rows); everything here is
//! shape-free.

const std = @import("std");
const cuda = @import("cuda");

pub const Library = cuda.cublaslt.Library;

fn i(x: usize) c_int {
    return @intCast(x);
}

pub const Ops = struct {
    mod: cuda.Module,
    embed_fn: cuda.Function,
    embed_f32_fn: cuda.Function,
    mean_f32_fn: cuda.Function,
    rms_fn: cuda.Function,
    act_fn: cuda.Function,
    combine_fn: cuda.Function,
    mean_fn: cuda.Function,
    add_fn: cuda.Function,
    argmax_fn: cuda.Function,
    route_fn: cuda.Function,
    copy2d_fn: cuda.Function,
    gather_fn: cuda.Function,
    layer_norm_fn: cuda.Function,
    pool_fn: cuda.Function,
    scale_fn: cuda.Function,
    lt: Library,

    pub fn load(d: *const cuda.Driver) !Ops {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.glm_ops);
        errdefer mod.unload();
        var o: Ops = .{
            .mod = mod,
            .embed_fn = undefined,
            .embed_f32_fn = undefined,
            .mean_f32_fn = undefined,
            .rms_fn = undefined,
            .act_fn = undefined,
            .combine_fn = undefined,
            .mean_fn = undefined,
            .add_fn = undefined,
            .argmax_fn = undefined,
            .route_fn = undefined,
            .copy2d_fn = undefined,
            .gather_fn = undefined,
            .layer_norm_fn = undefined,
            .pool_fn = undefined,
            .scale_fn = undefined,
            .lt = undefined,
        };
        errdefer o.unload();
        o.embed_fn = try mod.function("glm_embed");
        o.embed_f32_fn = try mod.function("glm_embed_f32");
        o.mean_f32_fn = try mod.function("glm_stream_mean_f32");
        o.rms_fn = try mod.function("glm_rms");
        o.act_fn = try mod.function("glm_act");
        o.combine_fn = try mod.function("glm_combine");
        o.mean_fn = try mod.function("glm_stream_mean");
        o.add_fn = try mod.function("glm_add");
        o.argmax_fn = try mod.function("glm_argmax");
        o.route_fn = try mod.function("glm_route");
        o.copy2d_fn = try mod.function("glm_copy2d");
        o.gather_fn = try mod.function("glm_gather");
        o.layer_norm_fn = try mod.function("glm_layer_norm");
        o.pool_fn = try mod.function("glm_pool");
        o.scale_fn = try mod.function("glm_scale");
        o.lt = try Library.open();
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.lt.close();
        o.mod.unload();
    }

    pub fn embed(o: *const Ops, s: cuda.Stream, ids: u64, e: u64, out: u64, rows: usize, d: usize, stride: usize) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(e);
        a.add(out);
        a.add(i(rows));
        a.add(i(d));
        a.add(i(stride));
        const total = rows * d;
        try cuda.launch.launch(o.embed_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Plain RMSNorm, no residual: y = w * bf16(x * inv_rms).
    pub fn rms(o: *const Ops, s: cuda.Stream, x: u64, w: u64, y: u64, rows: usize, width: usize, x_stride: usize, y_stride: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(y);
        a.add(i(rows));
        a.add(i(width));
        a.add(i(x_stride));
        a.add(i(y_stride));
        a.add(eps);
        try cuda.launch.launch(o.rms_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, s, &a);
    }

    /// The swiglu-limit activation over [rows, slots, n]; gate clamps above only.
    pub fn act(o: *const Ops, s: cuda.Stream, g: u64, u: u64, out: u64, rows: usize, slots: usize, n: usize, lim: f32) !void {
        var a: cuda.Args = .{};
        a.add(g);
        a.add(u);
        a.add(out);
        a.add(i(rows));
        a.add(i(slots));
        a.add(i(n));
        a.add(lim);
        const total = rows * slots * n;
        try cuda.launch.launch(o.act_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn combine(o: *const Ops, s: cuda.Stream, ys: u64, ye: u64, wts: u64, out: u64, rows: usize, topk: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(ys);
        a.add(ye);
        a.add(wts);
        a.add(out);
        a.add(i(topk));
        a.add(i(d));
        try cuda.launch.launch(o.combine_fn, .{ .grid = .{ .x = @intCast((d + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn embedF32(o: *const Ops, s: cuda.Stream, ids: u64, e: u64, out: u64, rows: usize, d: usize, stride: usize) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(e);
        a.add(out);
        a.add(i(rows));
        a.add(i(d));
        a.add(i(stride));
        const total = rows * d;
        try cuda.launch.launch(o.embed_f32_fn, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn streamMeanF32(o: *const Ops, s: cuda.Stream, x: u64, y: u64, rows: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(i(d));
        try cuda.launch.launch(o.mean_f32_fn, .{ .grid = .{ .x = @intCast((d + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn streamMean(o: *const Ops, s: cuda.Stream, x: u64, y: u64, rows: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(i(d));
        try cuda.launch.launch(o.mean_fn, .{ .grid = .{ .x = @intCast((d + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn add(o: *const Ops, s: cuda.Stream, x: u64, y: u64, out: u64, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(y);
        a.add(out);
        a.add(i(n));
        try cuda.launch.launch(o.add_fn, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn argmax(o: *const Ops, s: cuda.Stream, logits: u64, out: u64, rows: usize, vocab: usize) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(out);
        a.add(i(vocab));
        try cuda.launch.launch(o.argmax_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }

    pub fn copy2d(o: *const Ops, s: cuda.Stream, src: u64, dst: u64, rows: usize, cols: usize, src_stride: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(i(rows));
        a.add(i(cols));
        a.add(i(src_stride));
        try cuda.launch.launch(o.copy2d_fn, .{ .grid = .{ .x = @intCast((cols + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn gather(o: *const Ops, s: cuda.Stream, src: u64, dst: u64, n: usize, src_stride: usize) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(i(n));
        a.add(i(src_stride));
        try cuda.launch.launch(o.gather_fn, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn layerNorm(o: *const Ops, s: cuda.Stream, x: u64, w: u64, b: u64, y: u64, rows: usize, width: usize, x_stride: usize, y_stride: usize, eps: f32) !void {
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(b);
        a.add(y);
        a.add(i(width));
        a.add(i(x_stride));
        a.add(i(y_stride));
        a.add(eps);
        try cuda.launch.launch(o.layer_norm_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn pool(o: *const Ops, s: cuda.Stream, ik: u64, ig: u64, ape: u64, out: u64, first_blk: usize, n_blocks: usize) !void {
        var a: cuda.Args = .{};
        a.add(ik);
        a.add(ig);
        a.add(ape);
        a.add(out);
        a.add(i(first_blk));
        a.add(i(n_blocks));
        try cuda.launch.launch(o.pool_fn, .{ .grid = .{ .x = 128, .y = @intCast(n_blocks) }, .block = .{ .x = 1 } }, s, &a);
    }

    pub fn scaleBy(o: *const Ops, s: cuda.Stream, src: u64, dst: u64, n: usize, factor: f32) !void {
        var a: cuda.Args = .{};
        a.add(src);
        a.add(dst);
        a.add(i(n));
        a.add(factor);
        try cuda.launch.launch(o.scale_fn, .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn route(o: *const Ops, s: cuda.Stream, logits: u64, bias: u64, pick: u64, wts: u64, rows: usize, experts: usize, topk: usize, scale: f32) !void {
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(bias);
        a.add(pick);
        a.add(wts);
        a.add(i(experts));
        a.add(i(topk));
        a.add(scale);
        try cuda.launch.launch(o.route_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 512 } }, s, &a);
    }
};

test "ops shapes" {
    _ = std.testing.refAllDeclsRecursive;
}
