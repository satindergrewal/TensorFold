//! The DSA sparse attention and the indexer's scores/select against scalar
//! references. Small geometry: 2 rows, 4 heads (the kernels are width-agnostic
//! in heads; the latent is the real 512).

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const check = @import("check.zig");

const Gpu = check.Gpu;

const HDIM = 512;
const ROWS = 2;
const HEADS = 4;
const KEYS = 300; // key rows in the latent
const TOPK = 2051; // the index row width (most -1 here)
const KPOOL = 4;

var rng: u32 = 0x5EED5EED;

fn rand() f32 {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return @as(f32, @floatFromInt(rng % 2001)) / 1000.0 - 1.0;
}

fn bf16Bits(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const r: u32 = bits +% 0x7FFF + ((bits >> 16) & 1);
    return @intCast(r >> 16);
}

fn fromBf(b: u16) f64 {
    const wide: u32 = @as(u32, b) << 16;
    return @floatCast(@as(f32, @bitCast(wide)));
}

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var dk = try glm.dsa.Kernels.load(gpu.d);
    defer dk.deinit();
    var ik = try glm.index.Kernels.load(gpu.d);
    defer ik.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();

    // ── the sparse attention ──
    {
        const q = try a.alloc(u16, ROWS * HEADS * HDIM);
        defer a.free(q);
        for (q) |*v| v.* = bf16Bits(rand());
        const kv = try a.alloc(u16, KEYS * HDIM);
        defer a.free(kv);
        for (kv) |*v| v.* = bf16Bits(rand() * 0.5);
        // index rows: a mix of valid and -1, distinct per row
        const idx = try a.alloc(i32, ROWS * TOPK);
        defer a.free(idx);
        for (idx) |*v| v.* = -1;
        for (0..ROWS) |r| {
            var at: usize = 0;
            for (0..KEYS) |k| {
                if (k % 3 != r % 3) continue; // each row sees 2/3 of the keys
                if (at >= TOPK) break;
                idx[r * TOPK + at] = @intCast(k);
                at += 1;
            }
        }
        const scale = try a.dupe(f32, &.{0.0625});
        defer a.free(scale);
        const meta = try a.dupe(i32, &.{@intCast(KEYS)});
        defer a.free(meta);

        var dq = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(q));
        defer dq.free();
        var dkv = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(kv));
        defer dkv.free();
        var di = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(idx));
        defer di.free();
        var dsc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scale));
        defer dsc.free();
        var dme = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(meta));
        defer dme.free();
        var dout = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * HEADS * HDIM * 2);
        defer dout.free();

        try dk.attention(s, dq.ptr, dkv.ptr, di.ptr, dsc.ptr, dme.ptr, dout.ptr, ROWS, HEADS);
        try s.synchronize();
        const got = try check.download(gpu, dout);
        defer a.free(got);

        // the reference: softmax over each (row, head)'s valid keys, weighted v
        var worst: f64 = 0;
        for (0..ROWS) |r| for (0..HEADS) |h| {
            // scores
            var mx: f64 = -1e30;
            const n_valid = blk: {
                var n: usize = 0;
                for (0..TOPK) |t| {
                    if (idx[r * TOPK + t] >= 0) n += 1;
                }
                break :blk n;
            };
            const ss = try a.alloc(f64, n_valid);
            defer a.free(ss);
            const pos = try a.alloc(usize, n_valid);
            defer a.free(pos);
            {
                var w: usize = 0;
                for (0..TOPK) |t| {
                    const p = idx[r * TOPK + t];
                    if (p < 0) continue;
                    var dot: f64 = 0;
                    for (0..HDIM) |d| dot += fromBf(q[(r * HEADS + h) * HDIM + d]) * fromBf(kv[@as(usize, @intCast(p)) * HDIM + d]);
                    const sc = dot * 0.0625;
                    ss[w] = sc;
                    pos[w] = @intCast(p);
                    mx = @max(mx, sc);
                    w += 1;
                }
            }
            var tot: f64 = 0;
            for (ss) |v| tot += @exp(v - mx);
            // the out: sum(e * v)/tot per dim
            for (0..HDIM) |d| {
                var acc: f64 = 0;
                for (ss, pos) |v, p| acc += @exp(v - mx) * fromBf(kv[p * HDIM + d]);
                const ref = acc / tot;
                const hh: u16 = std.mem.readInt(u16, got[((r * HEADS + h) * HDIM + d) * 2 ..][0..2], .little);
                worst = @max(worst, @abs(fromBf(hh) - ref));
            }
        };
        if (worst > 4e-2) {
            std.debug.print("FAIL dsa: worst {d:.4}\n", .{worst});
            return error.Dsa;
        }
        check.pass("glm dsa: worst {d:.4} over {d} rows x {d} heads", .{ worst, ROWS, HEADS });
    }

    // ── the index scores + select ──
    {
        const blocks = 80;
        const qi = try a.alloc(u16, ROWS * HEADS * 128);
        defer a.free(qi);
        for (qi) |*v| v.* = bf16Bits(rand());
        const iw = try a.alloc(f32, ROWS * HEADS);
        defer a.free(iw);
        for (iw) |*v| v.* = rand();
        const pool = try a.alloc(u16, blocks * 128);
        defer a.free(pool);
        for (pool) |*v| v.* = bf16Bits(rand());
        const sc_buf = try a.alloc(f32, ROWS * blocks);
        defer a.free(sc_buf);
        const out_buf = try a.alloc(i32, ROWS * TOPK);
        defer a.free(out_buf);

        var dqi = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(qi));
        defer dqi.free();
        var diw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(iw));
        defer diw.free();
        var dpo = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(pool));
        defer dpo.free();
        var dsc = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * blocks * 4);
        defer dsc.free();
        var dou = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * TOPK * 4);
        defer dou.free();

        const top: usize = 10; // top-10 blocks
        const p0: usize = 7; // the window's start: blocks valid to (p0+row+1)/4
        try ik.scores(s, dqi.ptr, HEADS * 128, diw.ptr, dpo.ptr, dsc.ptr, blocks, ROWS, p0, HEADS);
        try ik.select(s, dsc.ptr, blocks, dou.ptr, TOPK, ROWS, p0, top, TOPK);
        try s.synchronize();
        const got_s = try check.download(gpu, dsc);
        defer a.free(got_s);
        const got_o = try check.download(gpu, dou);
        defer a.free(got_o);
        const sv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, got_s));
        const ov = std.mem.bytesAsSlice(i32, @as([]align(1) u8, got_o));

        for (0..ROWS) |r| {
            const valid = (p0 + r + 1) / KPOOL;
            // scores
            for (0..valid) |b| {
                var total: f32 = 0;
                for (0..HEADS) |h| {
                    var dot: f64 = 0;
                    for (0..128) |d| dot += fromBf(qi[r * HEADS * 128 + h * 128 + d]) * fromBf(pool[b * 128 + d]);
                    total = @mulAdd(f32, iw[r * HEADS + h], @max(@as(f32, @floatCast(dot)), 0.0), total);
                }
                if (@abs(sv[r * blocks + b] - total) > 2e-2) {
                    std.debug.print("FAIL idx score r{d} b{d}: {d:.4} vs {d:.4}\n", .{ r, b, sv[r * blocks + b], total });
                    return error.IdxScore;
                }
            }
            // select: the top-`top` by score ties lower block, then the tail
            var chosen = try a.alloc(usize, top);
            defer a.free(chosen);
            {
                const take: usize = @min(top, valid);
                // rank by (score, id)
                var picked = try a.alloc(bool, valid);
                defer a.free(picked);
                for (picked) |*v| v.* = false;
                for (0..take) |k| {
                    var best: ?usize = null;
                    for (0..valid) |b| {
                        if (picked[b]) continue;
                        if (best == null) {
                            best = b;
                            continue;
                        }
                        const bb = best.?;
                        if (sv[r * blocks + b] > sv[r * blocks + bb] or (sv[r * blocks + b] == sv[r * blocks + bb] and b < bb)) best = b;
                    }
                    picked[best.?] = true;
                    chosen[k] = best.?;
                }
            }
            var w: usize = 0;
            const take2 = @min(top, valid);
            // the chosen in block order
            const sorted = try a.alloc(usize, take2);
            defer a.free(sorted);
            @memcpy(sorted, chosen[0..take2]);
            std.mem.sort(usize, sorted, {}, struct {
                fn lt(_: void, x: usize, y: usize) bool {
                    return x < y;
                }
            }.lt);
            while (w < take2) : (w += 1) {
                if (ov[r * TOPK + w] != @as(i32, @intCast(sorted[w]))) {
                    std.debug.print("FAIL idx pick r{d} slot{d}: {d} vs {d}\n", .{ r, w, ov[r * TOPK + w], sorted[w] });
                    return error.IdxPick;
                }
            }
            // the tail: the partial block's members, then -1
            const tail = (p0 + r + 1) % KPOOL;
            while (w < TOPK) : (w += 1) {
                const j = w - take2;
                const want: i32 = if (j < tail) @intCast(p0 + r + 1 - tail + j) else -1;
                if (ov[r * TOPK + w] != want) {
                    std.debug.print("FAIL idx tail r{d} slot{d}: {d} vs {d}\n", .{ r, w, ov[r * TOPK + w], want });
                    return error.IdxTail;
                }
            }
        }
        check.pass("glm index: scores 2e-2, picks+tail exact", .{});
    }
}
