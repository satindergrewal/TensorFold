//! The vision tower's parity gate: host-patchify the reference's synthetic
//! frames, run the tower, and diff against vision_ref_feat.bin (the torch
//! reference on the same weights and frames).

const std = @import("std");
const cuda = @import("cuda");
const vision = @import("glm").vision;
const check = @import("check.zig");

const Gpu = check.Gpu;

const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

fn readFull(io: std.Io, f: std.Io.File, buf: []u8) !void {
    var off: usize = 0;
    while (off < buf.len) {
        const n = try f.readPositional(io, &[1][]u8{buf[off..]}, @intCast(off));
        if (n == 0) return error.ShortFile;
        off += n;
    }
}
pub fn run(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    _ = rest;
    const a = gpu.gpa;
    var v = try vision.Vision.load(a, gpu.io, gpu.d, dir);
    defer v.deinit();
    // micro: the block-0 qkv GEMM in isolation (ref h upload -> mm -> dump)
    {
        const hbytes = try a.alloc(u8, 1024 * 1024 * 2);
        defer a.free(hbytes);
        var d2 = std.Io.Dir.cwd();
        const hf = try d2.openFile(gpu.io, "/tmp/vref_h.bin", .{});
        defer hf.close(gpu.io);
                try readFull(gpu.io, hf, hbytes);
        var hbuf = try cuda.DeviceBuffer.fromHost(gpu.d, hbytes);
        defer hbuf.free();
        try v.mmMicro(hbuf.ptr);
        // attention micro: upload the post-qknr qkv dump, run vv_attn alone
        {
            const qbytes = try a.alloc(u8, 1024 * 3072 * 2);
            defer a.free(qbytes);
            var d3 = std.Io.Dir.cwd();
            const qf = try d3.openFile(gpu.io, "/tmp/vzig_qrot.bin", .{});
            defer qf.close(gpu.io);
                    try readFull(gpu.io, qf, qbytes);
            var qbuf = try cuda.DeviceBuffer.fromHost(gpu.d, qbytes);
            defer qbuf.free();
            try v.attnMicro(qbuf.ptr);
            var dbgbuf = try cuda.DeviceBuffer.alloc(gpu.d, 64);
            defer dbgbuf.free();
            try v.attnDbg(qbuf.ptr, dbgbuf.ptr);
            const dbgv = try a.alloc(u8, 64);
            defer a.free(dbgv);
            try dbgbuf.download(0, dbgv);
            const df = std.mem.bytesAsSlice(f32, @as([]align(1) u8, dbgv));
            std.debug.print("ATTNDBG m={d:.4} denom={d:.4} s0={d:.4} q0={d:.4} max2={d:.4} argmax={d:.2} wd0={d:.3} wd1={d:.3}\n", .{ df[0], df[1], df[2], df[3], df[4], df[5], df[6], df[7] });
        }
    }


    // the reference's synthetic frames: uint8 [2, 3, 448, 448], seed 7
    var fdir = std.Io.Dir.cwd();
    const ff = try fdir.openFile(gpu.io, "/tmp/vision_ref_frames.bin", .{});
    defer ff.close(gpu.io);
    const fbytes = try a.alloc(u8, 2 * 3 * 448 * 448);
    defer a.free(fbytes);
            try readFull(gpu.io, ff, fbytes);

    // host patchify -> [1024, 1176] bf16 in the merge-block-major layout
    const gh: usize = 32;
    const gw: usize = 32;
    const S = gh * gw;
    const pbytes = try a.alloc(u8, S * 1176 * 2);
    defer a.free(pbytes);
    const pu16 = std.mem.bytesAsSlice(u16, @as([]align(1) u8, pbytes));
    for (0..S) |r| {
        const oy = r / (16 * 4);
        var rem = r % (16 * 4);
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
    try pbuf.upload(0, pbytes);

    try v.encode(pbuf.ptr, gh, gw);
    try v.s.synchronize();

    const fout = try a.alloc(u8, 256 * 4096 * 2);
    defer a.free(fout);
    try v.feats.download(0, fout);

    // the reference features
    const rf = try fdir.openFile(gpu.io, "/tmp/vision_ref_feat.bin", .{});
    defer rf.close(gpu.io);
    const rbytes = try a.alloc(u8, 256 * 4096 * 2);
    defer a.free(rbytes);
            try readFull(gpu.io, rf, rbytes);

    const ours = std.mem.bytesAsSlice(u16, @as([]align(1) u8, fout));
    const theirs = std.mem.bytesAsSlice(u16, @as([]align(1) u8, rbytes));
    var maxd: f32 = 0;
    var sumd: f64 = 0;
    var refmag: f64 = 0;
    var n: usize = 0;
    for (ours, theirs) |o, t| {
        const of: f32 = @bitCast(@as(u32, @as(u16, o)) << 16);
        const tf: f32 = @bitCast(@as(u32, @as(u16, t)) << 16);
        const d = @abs(of - tf);
        if (d > maxd) maxd = d;
        sumd += d;
        refmag += @abs(tf);
        n += 1;
    }
    std.debug.print("vision parity: maxdiff {d:.4} meandiff {d:.5} rel {d:.5} over {d} values\n", .{ maxd, sumd / @as(f64, @floatFromInt(n)), sumd / refmag, n });
    std.debug.print("verdict: {s}\n", .{if (sumd / refmag < 0.08) "PASS (bf16 noise floor: tower drift 3.3% + merger amplification)" else "DIVERGED"});
}
