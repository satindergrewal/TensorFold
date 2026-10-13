//! The KDA kernels' host side: the conv and the recurrent step.

const std = @import("std");
const cuda = @import("cuda");

pub const H = 64;
pub const D = 128;
pub const W = H * D;
pub const taps = 4;

pub const Kernels = struct {
    mod: cuda.Module,
    conv_fn: cuda.Function,
    step_fn: cuda.Function,
    chain_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.glm_kda);
        errdefer mod.unload();
        var k: Kernels = .{ .mod = mod, .conv_fn = undefined, .step_fn = undefined, .chain_fn = undefined };
        errdefer k.deinit();
        k.chain_fn = try mod.function("tf_glm_kda_chain");
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
    }

    /// p is the projections' device pointer at this token window's start (the
    /// caller offsets past the conv history when they share a buffer).
    pub fn conv(k: *const Kernels, s: cuda.Stream, p: u64, ps: usize, cs: u64, cw: u64, out: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(p);
        a.add(@as(i32, @intCast(ps)));
        a.add(cs);
        a.add(cw);
        a.add(out);
        try cuda.launch.launch(k.conv_fn, .{ .grid = .{ .x = @intCast((3 * W + 255) / 256), .y = @intCast(rows) }, .block = .{ .x = 256 } }, s, &a);
    }

    pub fn step(k: *const Kernels, s: cuda.Stream, qkv: u64, fg: u64, bv: u64, a: u64, dtb: u64, st: u64, onw: u64, lb: f32, eps: f32, rows: usize, st_out: u64, y: u64) !void {
        var g: cuda.Args = .{};
        g.add(qkv);
        g.add(fg);
        g.add(bv);
        g.add(a);
        g.add(dtb);
        g.add(st);
        g.add(onw);
        g.add(lb);
        g.add(eps);
        g.add(@as(i32, @intCast(rows)));
        g.add(st_out);
        g.add(y);
        try cuda.launch.launch(k.step_fn, .{ .grid = .{ .x = H }, .block = .{ .x = D } }, s, &g);
    }

    /// Their verified chain: conv + delta-rule step + gated RMSNorm in one launch.
    /// P = fused projections [rows, ps] elements; A/G = the f_b/g_b outputs (f32-strided halves);
    /// cw = f32 channel-major [3W][TAPS]; cs = [TAPS-1, 3W] bf16 history.
    pub fn chain(k: *const Kernels, s: cuda.Stream, p: u64, ps: usize, b_off: usize, a_fg: u64, g_fg: u64, fg_stride: usize, cs: u64, cw: u64, st_in: u64, a_log: u64, dtb: u64, onw: u64, eps: f32, lower: f32, rows: usize, out: u64, st_out: u64) !void {
        var g: cuda.Args = .{};
        g.add(@as(i32, @intCast(H)));
        g.add(p);
        g.add(@as(i32, @intCast(ps)));
        g.add(@as(i32, @intCast(b_off)));
        g.add(a_fg);
        g.add(@as(i32, @intCast(fg_stride)));
        g.add(g_fg);
        g.add(@as(i32, @intCast(fg_stride)));
        g.add(cs);
        g.add(cw);
        g.add(st_in);
        g.add(a_log);
        g.add(dtb);
        g.add(onw);
        g.add(eps);
        g.add(lower);
        g.add(@as(i32, @intCast(rows)));
        g.add(out);
        g.add(st_out);
        g.add(@as(u64, 0));
        g.add(@as(u64, 0));
        g.add(@as(u64, 0));
        g.add(@as(u64, 0));
        try cuda.launch.launch(k.chain_fn, .{ .grid = .{ .x = H }, .block = .{ .x = 1024 } }, s, &g);
    }
};
