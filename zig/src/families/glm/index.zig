//! The indexer's host side: scores and the key-list select.

const std = @import("std");
const cuda = @import("cuda");

pub const Kernels = struct {
    mod: cuda.Module,
    scores_fn: cuda.Function,
    select_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.glm_index);
        errdefer mod.unload();
        var k: Kernels = .{ .mod = mod, .scores_fn = undefined, .select_fn = undefined };
        errdefer k.deinit();
        k.scores_fn = try mod.function("index_scores");
        k.select_fn = try mod.function("index_select");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
    }

    pub fn scores(k: *const Kernels, s: cuda.Stream, q_i: u64, q_stride: usize, iw: u64, pool: u64, out: u64, s_stride: usize, rows: usize, p0: usize, heads: usize) !void {
        var a: cuda.Args = .{};
        a.add(q_i);
        a.add(@as(i32, @intCast(q_stride)));
        a.add(iw);
        a.add(pool);
        a.add(out);
        a.add(@as(i32, @intCast(s_stride)));
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(p0)));
        a.add(@as(i32, @intCast(heads)));
        const blocks = (p0 + rows + 3) / 4; // the last row's, an upper bound
        try cuda.launch.launch(k.scores_fn, .{ .grid = .{ .x = @intCast((blocks + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn select(k: *const Kernels, s: cuda.Stream, sc: u64, s_stride: usize, out: u64, i_stride: usize, rows: usize, p0: usize, top: usize, width: usize) !void {
        var a: cuda.Args = .{};
        a.add(sc);
        a.add(@as(i32, @intCast(s_stride)));
        a.add(out);
        a.add(@as(i32, @intCast(i_stride)));
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(p0)));
        a.add(@as(i32, @intCast(top)));
        a.add(@as(i32, @intCast(width)));
        try cuda.launch.launch(k.select_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }
};
