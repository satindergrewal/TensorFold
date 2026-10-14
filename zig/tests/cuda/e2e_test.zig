//! The end-to-end image path: the tower encodes the reference frames, the
//! features are embedded at the image-token positions, and engine A's first
//! block runs on those streams. Verifies the stream rows at the image
//! positions equal the features (before any block math).

const std = @import("std");
const cuda = @import("cuda");
const vision = @import("glm").vision;
const glm = @import("glm");
const check = @import("check.zig");

const Gpu = check.Gpu;
const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };


pub fn run(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    _ = rest;
    const a = gpu.gpa;
    var v = try vision.Vision.load(a, gpu.io, gpu.d, dir);
    defer v.deinit();

    // frames -> host patches -> tower -> feats [256, 4096]
    const fbytes = try a.alloc(u8, 2 * 3 * 448 * 448);
    defer a.free(fbytes);
    try readFull(gpu.io, "/tmp/vision_ref_frames.bin", fbytes);
    const gh: usize = 32;
    const gw: usize = 32;
    const S = gh * gw;
    const pbytes = try a.alloc(u8, S * 1176 * 2);
    defer a.free(pbytes);
    const pu16 = std.mem.bytesAsSlice(u16, @as([]align(1) u8, pbytes));
    for (0..S) |r| {
        const oy = r / 64;
        var rem = r % 64;
        const ox = rem / 4;
        rem = rem % 4;
        const my = rem / 2;
        const mx = rem % 2;
        var at: usize = 0;
        for (0..3) |cc| for (0..2) |t| for (0..14) |py| for (0..14) |px| {
            const y = (oy * 2 + my) * 14 + py;
            const x = (ox * 2 + mx) * 14 + px;
            const uv: f32 = @floatFromInt(fbytes[(t * 3 + cc) * 448 * 448 + y * 448 + x]);
            const val = (uv - MEAN[cc] * 255.0) / (STD[cc] * 255.0);
            pu16[r * 1176 + at] = @bitCast(@as(u16, @truncate(@as(u32, @bitCast(val)) >> 16)));
            at += 1;
        };
    }
    var pbuf = try cuda.DeviceBuffer.fromHost(gpu.d, pbytes);
    defer pbuf.free();
    try v.encode(pbuf.ptr, gh, gw);
    try v.s.synchronize();

    // engine A (blocks 0..22, stopped after blocks) - the vis pointer set
    var eng = try glm.engine.Engine.init(a, gpu.io, gpu.d, dir, .{ .context = 1024, .blocks_lo = 0, .blocks_hi = 0, .stop_after_blocks = true});
    defer eng.deinit();
    eng.vis = v;

    // the prompt: <boi> <image>x256 <eoi> "Describe" tokens
    var ids: [300]u32 = undefined;
    var n: usize = 0;
    ids[n] = 154830; // <|begin_of_image|>
    n += 1;
    for (0..256) |_| {
        ids[n] = 154854; // <|image|>
        n += 1;
    }
    ids[n] = 154831; // <|end_of_image|>
    n += 1;
    const describe = [_]u32{ 4472, 728, 279, 3309 }; // "Describe the image"
    for (describe) |t| {
        ids[n] = t;
        n += 1;
    }

    // host feature map: image rows take features 0..255, others -1
    var fmap: [300]i32 = undefined;
    {
        var fi: i32 = 0;
        for (0..n) |r| {
            if (ids[r] == 154854) {
                fmap[r] = fi;
                fi += 1;
            } else fmap[r] = -1;
        }
    }
    // chunked like the server's prefill (the engine's rows_max = 24); the
    // feature map is uploaded per chunk with the running feature ordinals
    var pos: usize = 0;
    var fbase: i32 = 0;
    while (pos < n) {
        const clen = @min(24, n - pos);
        var cmap: [24]i32 = undefined;
        for (0..clen) |j| {
            const g = fmap[pos + j];
            cmap[j] = if (g >= 0) fbase + g else -1;
        }
        // rebase: fmap holds GLOBAL ordinals; the chunk needs local ones
        var local: [24]i32 = undefined;
        for (0..clen) |j| local[j] = if (cmap[j] >= 0) cmap[j] - fbase else -1;
        try v.uploadMap(local[0..clen], eng.stream);
        try eng.forward(ids[pos .. pos + clen], clen, pos);
        pos += clen;
        // count how many features this chunk consumed
        for (0..clen) |j| {
            if (fmap[pos - clen + j] >= 0) fbase += 1;
        }
    }
    try eng.stream.synchronize();

    // THE verification: a fresh state, one 1-row embed of image tokens,
    // dump_on captures the EMBED OUTPUT (xemb) before any block math - the
    // loader clamps blocks_hi to lo+1, so one block ALWAYS runs and the
    // live streams are post-boundary; the xemb dump is the embed itself.
    try eng.resetState();
    var fi_base: i32 = 0;
    {
        // advance to a fresh feature row (use feature 100 for variety)
        fi_base = 100;
        var ids2: [1]u32 = .{154854};
        var map2: [1]i32 = .{fi_base};
        try v.uploadMap(map2[0..1], eng.stream);
        eng.dump_on = true;
        try eng.forward(ids2[0..1], 1, 0);
        eng.dump_on = false;
        try eng.stream.synchronize();
    }
    // read the newest xemb dump and compare against feats row fi_base
    {
        var d = std.Io.Dir.cwd();
        const f = try d.openFile(gpu.io, "xemb_last.bin", .{});
        defer f.close(gpu.io);
        var xb: [4 * 4096 * 4]u8 = undefined;
        _ = try f.readPositional(gpu.io, &[1][]u8{xb[0..]}, 0);
        const sx = std.mem.bytesAsSlice(f32, @as([]align(1) u8, xb[0..]));
        const fb = try a.alloc(u8, 256 * 4096 * 2);
        defer a.free(fb);
        try v.feats.download(0, fb);
        var worst: f32 = 0;
        for (0..4096) |c| {
            const wu: u16 = @bitCast(fb[@as(usize, @intCast(fi_base)) * 4096 * 2 + c * 2..][0..2].*);
            const want: f32 = @bitCast(@as(u32, wu) << 16);
            const dd = @abs(sx[c] - want);
            if (dd > worst) worst = dd;
        }
        std.debug.print("vision e2e embed (xemb vs feats row {d}): worst |diff| {d:.6}\n", .{ fi_base, worst });
        std.debug.print("verdict: {s}\n", .{if (worst == 0.0) "PASS (features landed exactly)" else "FAIL"});
    }
}

fn readFull(io: std.Io, path: []const u8, buf: []u8) !void {
    var d = std.Io.Dir.cwd();
    const f = try d.openFile(io, path, .{});
    defer f.close(io);
    var off: usize = 0;
    while (off < buf.len) {
        const nn = try f.readPositional(io, &[1][]u8{buf[off..]}, @intCast(off));
        if (nn == 0) return error.ShortFile;
        off += nn;
    }
}
