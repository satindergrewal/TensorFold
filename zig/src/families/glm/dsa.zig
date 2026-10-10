//! The DSA sparse attention's host side: one launch, grid rows*heads.

const std = @import("std");
const cuda = @import("cuda");

pub const Kernels = struct {
    mod: cuda.Module,
    attn_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.glm_dsa);
        errdefer mod.unload();
        var k: Kernels = .{ .mod = mod, .attn_fn = undefined };
        errdefer k.deinit();
        k.attn_fn = try mod.function("dsa_attention");
        try k.attn_fn.allowDynamicShared(32 * 512 * 4);
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        k.mod.unload();
    }

    pub fn attention(k: *const Kernels, s: cuda.Stream, q: u64, kv: u64, idx: u64, scale: u64, meta: u64, out: u64, rows: usize, heads: usize) !void {
        var a: cuda.Args = .{};
        a.add(q);
        a.add(kv);
        a.add(idx);
        a.add(scale);
        a.add(meta);
        a.add(out);
        a.add(@as(i32, @intCast(heads)));
        try cuda.launch.launch(k.attn_fn, .{ .grid = .{ .y = @intCast(rows * heads) }, .block = .{ .x = 1024 }, .shared = 32 * 512 * 4 }, s, &a);
    }
};
