//! The engine's first real-weights smoke: blocks 0..3 (dense MLP + KDA attn),
//! four tokens, logits checksum + argmax ids printed.

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const check = @import("check.zig");

const Gpu = check.Gpu;

pub fn run(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const nm: usize = if (rest.len > 1) std.fmt.parseInt(usize, rest[1], 10) catch 3 else 3;
    const skip_mla = rest.len > 2 and std.mem.eql(u8, rest[2], "skip-mla");
    const a = gpu.gpa;
    var e = try glm.engine.Engine.init(a, gpu.io, gpu.d, dir, .{ .blocks_lo = 0, .blocks_hi = nm, .skip_mla = skip_mla });
    defer e.deinit();
    const rows: usize = 4;
    const ids = [_]u32{ 100, 2000, 30000, 154879 };
    // probe: embed alone before any block
    try e.ids.upload(0, std.mem.sliceAsBytes(ids[0..rows]));
    for (0..4) |s_i| try e.ops.embed(e.stream, e.ids.ptr, e.w.embed, e.x[0].ptr + s_i * e.c.hidden * 2, rows, e.c.hidden);
    try e.stream.synchronize();
    {
        const xb = try check.download(gpu, e.x[0]);
        defer a.free(xb);
        var nans: usize = 0;
        var first: ?usize = null;
        for (0..rows * 4 * e.c.hidden) |i| {
            const h: u16 = std.mem.readInt(u16, xb[i * 2 ..][0..2], .little);
            const v = @as(f32, @bitCast(@as(u32, h) << 16));
            if (std.math.isNan(v)) {
                nans += 1;
                if (first == null) first = i;
            }
        }
        std.debug.print("  embed stage: {d} NaNs, first {s}" ++ "\n", .{ nans, if (first) |v| std.fmt.allocPrint(a, "{d}", .{v}) catch "?" else "-" });
    }
    try e.forward(&ids, rows);
    try e.stream.synchronize();
    // per-buffer NaN census after the walk
    {
        const bufs = [_]struct { name: []const u8, b: cuda.DeviceBuffer }{
            .{ .name = "x0", .b = e.x[0] }, .{ .name = "x1", .b = e.x[1] },
            .{ .name = "branch", .b = e.branch }, .{ .name = "normed", .b = e.normed },
            .{ .name = "y", .b = e.y }, .{ .name = "qkv", .b = e.qkv },
            .{ .name = "fg", .b = e.fg }, .{ .name = "mlp_act", .b = e.mlp_act },
            .{ .name = "post", .b = e.post }, .{ .name = "comb", .b = e.comb },
            .{ .name = "mla_ql", .b = e.mla_ql }, .{ .name = "mla_att", .b = e.mla_att },
            .{ .name = "mla_keys", .b = e.mla_keys }, .{ .name = "mla_vals", .b = e.mla_vals },
            .{ .name = "mla_qp", .b = e.mla_qp }, .{ .name = "mla_xp", .b = e.mla_xp },
            .{ .name = "mla_ik", .b = e.mla_ik }, .{ .name = "mla_pool", .b = e.mla_pool },
            .{ .name = "mla_iw", .b = e.mla_iw },
        };
        for (bufs) |bb| {
            const bytes = try check.download(gpu, bb.b);
            defer a.free(bytes);
            var nans: usize = 0;
            const n16 = bytes.len / 2;
            for (0..n16) |i| {
                const h: u16 = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
                const v = @as(f32, @bitCast(@as(u32, h) << 16));
                if (std.math.isNan(v)) nans += 1;
            }
            std.debug.print("  {s}: {d}/{d} NaN" ++ "\n", .{ bb.name, nans, n16 });
        }
    }
    const picks = try check.download(gpu, e.picks);
    defer a.free(picks);
    const lg = try check.download(gpu, e.logits);
    defer a.free(lg);
    var sum: f64 = 0;
    var mx: f32 = -1e30;
    var mx_at: usize = 0;
    for (0..rows * e.c.vocab) |i| {
        const h: u16 = std.mem.readInt(u16, lg[i * 2 ..][0..2], .little);
        const v = @as(f32, @bitCast(@as(u32, h) << 16));
        sum += v;
        if (v > mx) {
            mx = v;
            mx_at = i % e.c.vocab;
        }
    }
    const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, picks));
    // where does NaN begin? probe the window's stage buffers via fresh downloads
    {
        const xb = try check.download(gpu, e.x[e.xi]);
        defer a.free(xb);
        var nan_at: ?usize = null;
        for (0..rows * 4 * e.c.hidden) |i| {
            const h: u16 = std.mem.readInt(u16, xb[i * 2 ..][0..2], .little);
            const v = @as(f32, @bitCast(@as(u32, h) << 16));
            if (std.math.isNan(v)) {
                nan_at = i;
                break;
            }
        }
        std.debug.print("  streams first NaN at {s} (len {d})" ++ "\n", .{ if (nan_at) |v| std.fmt.allocPrint(a, "{d}", .{v}) catch "oom" else "none", rows * 4 * e.c.hidden });
    }
    check.pass("glm fwd smoke: logits sum {d:.1}, max {d:.2} at {d}; picks {d},{d},{d},{d}", .{ sum, mx, mx_at, pv[0], pv[1], pv[2], pv[3] });
}
