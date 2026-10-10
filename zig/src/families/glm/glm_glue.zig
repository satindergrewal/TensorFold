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
    rms_fn: cuda.Function,
    act_fn: cuda.Function,
    combine_fn: cuda.Function,
    mean_fn: cuda.Function,
    add_fn: cuda.Function,
    argmax_fn: cuda.Function,
    route_fn: cuda.Function,
    lt: Library,

    pub fn load(d: *const cuda.Driver) !Ops {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.glm_ops);
        errdefer mod.unload();
        var o: Ops = .{
            .mod = mod,
            .embed_fn = undefined,
            .rms_fn = undefined,
            .act_fn = undefined,
            .combine_fn = undefined,
            .mean_fn = undefined,
            .add_fn = undefined,
            .argmax_fn = undefined,
            .route_fn = undefined,
            .lt = undefined,
        };
        errdefer o.unload();
        o.embed_fn = try mod.function("glm_embed");
        o.rms_fn = try mod.function("glm_rms");
        o.act_fn = try mod.function("glm_act");
        o.combine_fn = try mod.function("glm_combine");
        o.mean_fn = try mod.function("glm_stream_mean");
        o.add_fn = try mod.function("glm_add");
        o.argmax_fn = try mod.function("glm_argmax");
        o.route_fn = try mod.function("glm_route");
        o.lt = try Library.open();
        return o;
    }

    pub fn unload(o: *Ops) void {
        o.lt.close();
        o.mod.unload();
    }

    pub fn embed(o: *const Ops, s: cuda.Stream, ids: u64, e: u64, out: u64, rows: usize, d: usize) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(e);
        a.add(out);
        a.add(i(rows));
        a.add(i(d));
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
