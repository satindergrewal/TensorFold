//! The engine's first real-weights smoke: blocks 0..3 (dense MLP + KDA attn),
//! four tokens, logits checksum + argmax ids printed.

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const check = @import("check.zig");

const Gpu = check.Gpu;

pub fn run(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const nm: usize = if (rest.len > 1) std.fmt.parseInt(usize, rest[1], 10) catch 3 else 3;
    const split = rest.len > 2 and std.mem.eql(u8, rest[2], "split");
    if (split) return runSplit(gpu, dir, nm);
    const skip_mla = rest.len > 2 and std.mem.eql(u8, rest[2], "skip-mla");
    const a = gpu.gpa;
    var e = try glm.engine.Engine.init(a, gpu.io, gpu.d, dir, .{ .blocks_lo = 0, .blocks_hi = nm, .skip_mla = skip_mla });
    defer e.deinit();
    const rows0: usize = 4;
    // per-block stream fingerprints (first 26 blocks)
    const sums = try a.alloc(f64, 46);
    defer a.free(sums);
    @memset(sums, 0);
    e.block_sums = sums;
    {
        // state zero check: the first 64 floats of kda_st0 right after init
        try e.stream.synchronize();
        const hb = try a.alloc(u8, 256);
        defer a.free(hb);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = e.kda_st0.ptr, .len = 256, .borrowed = true }).download(0, hb);
        const fv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, hb));
        var nz: usize = 0;
        for (fv) |v| if (v != 0) {
            nz += 1;
        };
        std.debug.print("  st0 head nonzero {d}/64 first {d:.4}\n", .{ nz, fv[0] });

        // full-buffer census: every float of kda_st0
        const full = try a.alloc(u8, e.kda_st0.len);
        defer a.free(full);
        try e.kda_st0.download(0, full);
        var nzf: usize = 0;
        const ff = std.mem.bytesAsSlice(f32, @as([]align(1) u8, full));
        for (ff) |x| if (x != 0) {
            nzf += 1;
        };
        std.debug.print("  st0 FULL nonzero {d}/{d}\n", .{ nzf, ff.len });
    }
    var ids_buf: [16]u32 = .{ 100, 2000, 30000, 154800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var n_gen: usize = 0;
    while (n_gen < 12) : (n_gen += 1) {
        const rows_now = rows0 + n_gen;
        if (rows_now > 16) break;
        try e.forward(ids_buf[0..rows_now], rows_now);
        try e.stream.synchronize();
        const pk = try check.download(gpu, e.picks);
        defer a.free(pk);
        const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pk));
        ids_buf[rows_now] = pv[rows_now - 1];
    }
    {
        var out: []u8 = try a.alloc(u8, 160);
        defer a.free(out);
        var w: usize = 0;
        for (0..rows0 + n_gen) |t| {
            if (t == rows0) {
                @memcpy(out[w .. w + 3], " | ");
                w += 3;
            }
            const printed = std.fmt.bufPrint(out[w..], "{d} ", .{ids_buf[t]}) catch unreachable;
            w += printed.len;
        }
        std.debug.print("  greedy[{d}]: {s}" ++ "\n", .{ rows0 + n_gen, out[0..w] });
    }
    {
        var w2: usize = 0;
        const out2 = try a.alloc(u8, 46 * 20);
        defer a.free(out2);
        for (0..@min(nm, 26)) |b| {
            const pr = std.fmt.bufPrint(out2[w2..], "{d}:{d:.4} ", .{ b, sums[b] }) catch unreachable;
            w2 += pr.len;
        }
        std.debug.print("  blocksums {s}" ++ "\n", .{out2[0..w2]});
    }
    const rows: usize = rows0;
    const ids = ids_buf;
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

fn runSplit(gpu: Gpu, dir: []const u8, split_at: usize) !void {
    const a = gpu.gpa;
    // engine A on this context (device 0)
    var ea = try glm.engine.Engine.init(a, gpu.io, gpu.d, dir, .{ .blocks_lo = 0, .blocks_hi = split_at, .stop_after_blocks = true });
    const rows: usize = 4;
    const ids = [_]u32{ 100, 2000, 30000, 154879 };
    try ea.forward(&ids, rows);
    try ea.stream.synchronize();
    // the handoff state through the host: x (both planes), branch, post, comb
    // download the handoff under A's context BEFORE B's context becomes current
    const Handoff = struct { x0: []u8, x1: []u8, branch: []u8, post: []u8, comb: []u8 };
    var ho: Handoff = undefined;
    ho.x0 = try check.download(gpu, ea.x[0]);
    ho.x1 = try check.download(gpu, ea.x[1]);
    ho.branch = try check.download(gpu, ea.branch);
    ho.post = try check.download(gpu, ea.post);
    ho.comb = try check.download(gpu, ea.comb);
    const xi_hand = ea.xi;
    var ctx1 = try cuda.Context.init(gpu.d, 1);
    defer ctx1.deinit();
    var eb = try glm.engine.Engine.init(a, gpu.io, gpu.d, dir, .{ .blocks_lo = split_at, .blocks_hi = 45, .cont = true });
    defer eb.deinit();
    try eb.x[0].upload(0, ho.x0);
    try eb.x[1].upload(0, ho.x1);
    try eb.branch.upload(0, ho.branch);
    try eb.post.upload(0, ho.post);
    try eb.comb.upload(0, ho.comb);
    defer {
        a.free(ho.x0);
        a.free(ho.x1);
        a.free(ho.branch);
        a.free(ho.post);
        a.free(ho.comb);
    }
    eb.xi = xi_hand;
    try eb.forward(&ids, rows);
    try eb.stream.synchronize();
    const lg = try check.download(gpu, eb.logits);
    defer a.free(lg);
    var sum: f64 = 0;
    var mx: f32 = -1e30;
    var mx_at: usize = 0;
    for (0..rows * eb.c.vocab) |i| {
        const h: u16 = std.mem.readInt(u16, lg[i * 2 ..][0..2], .little);
        const v = @as(f32, @bitCast(@as(u32, h) << 16));
        sum += v;
        if (v > mx) {
            mx = v;
            mx_at = i % eb.c.vocab;
        }
    }
    const pk = try check.download(gpu, eb.picks);
    defer a.free(pk);
    const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pk));
    check.pass("glm fwd SPLIT 0..{d}..46: sum {d:.1} max {d:.2} at {d}; picks {d},{d},{d},{d}", .{ split_at, sum, mx, mx_at, pv[0], pv[1], pv[2], pv[3] });
    {
        // deterministic-fingerprint: active rows' first 4 logits + row sums
        var ir: usize = 0;
        while (ir < rows) : (ir += 1) {
            var rs: f64 = 0;
            var jr: usize = 0;
            while (jr < eb.c.vocab) : (jr += 1) {
                const h2: u16 = std.mem.readInt(u16, lg[(ir * eb.c.vocab + jr) * 2 ..][0..2], .little);
                rs += @as(f32, @bitCast(@as(u32, h2) << 16));
            }
            const f0: u16 = std.mem.readInt(u16, lg[(ir * eb.c.vocab) * 2 ..][0..2], .little);
            std.debug.print("  split row{d}: first {d:.3} rowsum {d:.1}" ++ "\n", .{ ir, @as(f32, @bitCast(@as(u32, f0) << 16)), rs });
        }
    }
    _ = ea.deinit();
}
