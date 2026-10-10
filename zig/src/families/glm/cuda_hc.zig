//! The hyper-connection boundary's host side: glm_hc.cu's two launches per
//! boundary, buffers as the engine's walk holds them.

const std = @import("std");
const cuda = @import("cuda");
const launch_ = cuda.launch;

const Function = cuda.Function;

/// The loaded glm_hc image.
pub const Kernels = struct {
    mod: cuda.Module,
    expand_fn: Function,
    split_fn: Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.glm_hc);
        errdefer mod.unload();
        var k: Kernels = .{ .mod = mod, .expand_fn = undefined, .split_fn = undefined };
        errdefer k.deinit();
        k.expand_fn = try mod.function(expand_sym);
        k.split_fn = try mod.function(split_sym);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
    }

    /// The boundary's first launch: the streams' expand (when `expand`) and the
    /// 25 partials. `x_new` untouched when not expanding.
    pub fn expandMix(
        k: *const Kernels,
        s: cuda.Stream,
        x_old: u64,
        branch: u64,
        post: u64,
        comb: u64,
        fn_packed: u64,
        x_new: u64,
        part: u64,
        rows: usize,
        expand: bool,
    ) !void {
        var a: launch_.Args = .{};
        a.add(x_old);
        a.add(branch);
        a.add(post);
        a.add(comb);
        a.add(fn_packed);
        a.add(x_new);
        a.add(part);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intFromBool(expand)));
        try launch_.launch(k.expand_fn, .{ .grid = .{ .x = slices, .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// The second launch: tot, the norm, pre/post, the Sinkhorn comb, the
    /// collapse and its RMSNorm.
    pub fn split(
        k: *const Kernels,
        s: cuda.Stream,
        x: u64,
        part: u64,
        scale: u64,
        base: u64,
        norm_w: u64,
        eps: u64,
        normed: u64,
        post_out: u64,
        comb_out: u64,
        rows: usize,
    ) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(part);
        a.add(scale);
        a.add(base);
        a.add(norm_w);
        a.add(eps);
        a.add(normed);
        a.add(post_out);
        a.add(comb_out);
        a.add(@as(i32, @intCast(rows)));
        try launch_.launch(k.split_fn, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 1024 } }, s, &a);
    }
};

pub const slices = 16; // F / 1024
pub const mix = 24;
pub const parts = mix + 1;

pub fn partBytes(rows: usize) usize {
    return rows * slices * parts * 4;
}

const expand_sym = "hc_expand_mix";
const split_sym = "hc_split";

test "boundary scratch size" {
    try std.testing.expectEqual(@as(usize, 3 * 16 * 25 * 4), partBytes(3));
}
