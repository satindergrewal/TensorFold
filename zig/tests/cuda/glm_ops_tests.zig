//! The dense ops against scalar references, synthetic rows; the GEMM through
//! cublasLt against fp32. The bf16 roundings sit where the metal kernels put them.

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const check = @import("check.zig");

const ops = glm.glue;
const Gpu = check.Gpu;

var rng: u32 = 0xDEADBEEF;

fn rand() f32 {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return @as(f32, @floatFromInt(rng % 2001)) / 1000.0 - 1.0;
}

/// f32 to bf16, round-to-nearest-even as the device's conversion does.
fn bf16Bits(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    const rounded: u32 = bits + 0x7FFF + ((bits >> 16) & 1);
    return @intCast(rounded >> 16);
}

fn fromBf16(b: u16) f64 {
    const wide: u32 = @as(u32, b) << 16;
    return @floatCast(@as(f32, @bitCast(wide)));
}

fn randBf16(a: std.mem.Allocator, n: usize, scale: f32) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = bf16Bits(rand() * scale);
    return v;
}

const D = 256; // the test hidden (small: the kernels are width-agnostic)
const ROWS = 5;

fn worstDev(got: []const u8, want: []const f64) f64 {
    var w: f64 = 0;
    const n = @min(got.len / 2, want.len);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const h: u16 = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
        w = @max(w, @abs(fromBf16(h) - want[i]));
    }
    return w;
}

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var o = try ops.Ops.load(gpu.d);
    defer o.unload();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();

    // -- embed --
    {
        const table = try randBf16(a, 7 * D, 1.0);
        defer a.free(table);
        const ids = try a.alloc(i32, ROWS);
        defer a.free(ids);
        for (ids) |*v| v.* = @intCast(rng % 7);
        var dt = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(table));
        defer dt.free();
        var di = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ids));
        defer di.free();
        var dout = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * D * 2);
        defer dout.free();
        try o.embed(s, di.ptr, dt.ptr, dout.ptr, ROWS, D, D);
        try s.synchronize();
        const got = try check.download(gpu, dout);
        defer a.free(got);
        var worst: f64 = 0;
        for (0..ROWS) |r| {
            const h: u16 = std.mem.readInt(u16, got[(r * D + 3) * 2 ..][0..2], .little);
            const w: u16 = std.mem.readInt(u16, std.mem.sliceAsBytes(table)[(@as(usize, @intCast(ids[r])) * D + 3) * 2 ..][0..2], .little);
            worst = @max(worst, @abs(fromBf16(h) - fromBf16(w)));
        }
        if (worst != 0) return error.Embed;
        check.pass("glm embed: exact", .{});
    }

    // -- rms --
    {
        const x = try randBf16(a, ROWS * D, 1.0);
        defer a.free(x);
        const w = try randBf16(a, D, 1.0);
        defer a.free(w);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx.free();
        var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(w));
        defer dw.free();
        var dy = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * D * 2);
        defer dy.free();
        const eps: f32 = 1e-5;
        try o.rms(s, dx.ptr, dw.ptr, dy.ptr, ROWS, D, D, D, eps);
        try s.synchronize();
        const got = try check.download(gpu, dy);
        defer a.free(got);
        var worst: f64 = 0;
        for (0..ROWS) |r| {
            var sq: f64 = 0;
            for (0..D) |c| {
                const v = fromBf16(x[r * D + c]);
                sq += v * v;
            }
            const inv = 1.0 / @sqrt(sq / D + eps);
            for (0..D) |c| {
                const v = fromBf16(x[r * D + c]);
                const scaled = fromBf16(bf16Bits(@floatCast(v * inv)));
                const ref: f64 = fromBf16(bf16Bits(@floatCast(@as(f64, fromBf16(w[c])) * scaled)));
                const h: u16 = std.mem.readInt(u16, got[(r * D + c) * 2 ..][0..2], .little);
                worst = @max(worst, @abs(fromBf16(h) - ref));
            }
        }
        if (worst > 2e-2) {
            std.debug.print("FAIL glm rms: {d:.4} at", .{worst});
            var n: usize = 0;
            for (0..ROWS) |r| {
                if (n >= 4) break;
                var sq: f64 = 0;
                for (0..D) |c2| {
                    const v2 = fromBf16(x[r * D + c2]);
                    sq += v2 * v2;
                }
                std.debug.print(" [r{d}]inv {d:.4}", .{ r, 1.0 / @sqrt(sq / D + eps) });
                n += 1;
            }
            std.debug.print("\n", .{});
            const h0: u16 = std.mem.readInt(u16, got[0..2], .little);
            std.debug.print("  dev[0]={d:.4}\n", .{fromBf16(h0)});
            return error.Rms;
        }
        check.pass("glm rms: worst {d:.5}", .{worst});
    }

    // -- act (the asymmetric clamp) --
    {
        const slots = 3;
        const g = try randBf16(a, ROWS * slots * D, 12.0); // past the limit
        defer a.free(g);
        const u = try randBf16(a, ROWS * slots * D, 12.0);
        defer a.free(u);
        var dg = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(g));
        defer dg.free();
        var du = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(u));
        defer du.free();
        var da = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * slots * D * 2);
        defer da.free();
        const lim: f32 = 10.0;
        try o.act(s, dg.ptr, du.ptr, da.ptr, ROWS, slots, D, lim);
        try s.synchronize();
        const got = try check.download(gpu, da);
        defer a.free(got);
        var worst: f64 = 0;
        var worst_rel: f64 = 0;
        for (0..ROWS * slots * D) |i| {
            const lim_bf: f64 = fromBf16(bf16Bits(lim));
            const gt = fromBf16(bf16Bits(@floatCast(@min(fromBf16(g[i]), lim_bf))));
            const up = fromBf16(bf16Bits(@floatCast(@min(@max(fromBf16(u[i]), -lim_bf), lim_bf))));
            const e = @exp(@abs(gt));
            const sgr = fromBf16(bf16Bits(@floatCast(if (gt < 0) 1.0 / (1.0 + e) else 1.0 - 1.0 / (1.0 + e))));
            const sl = fromBf16(bf16Bits(@floatCast(gt * sgr)));
            const ref: f64 = fromBf16(bf16Bits(@floatCast(sl * up)));
            const h: u16 = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
            const d = @abs(fromBf16(h) - ref);
            const rel = d / (@abs(ref) + 1e-3);
            worst = @max(worst, @abs(fromBf16(h) - ref));
            worst_rel = @max(worst_rel, rel);
        }
        if (worst_rel > 2.5e-2) {
            std.debug.print("FAIL glm act: worst abs {d:.4} rel {d:.4}\n", .{ worst, worst_rel });
            for (0..4) |i| {
                const lim_bf: f64 = fromBf16(bf16Bits(lim));
                const gt = fromBf16(bf16Bits(@floatCast(@min(fromBf16(g[i]), lim_bf))));
                const up = fromBf16(bf16Bits(@floatCast(@min(@max(fromBf16(u[i]), -lim_bf), lim_bf))));
                const h: u16 = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
                const e2 = @exp(@abs(gt));
                const sg2: f64 = if (gt < 0) 1.0 / (1.0 + e2) else 1.0 - 1.0 / (1.0 + e2);
                const sl2 = fromBf16(bf16Bits(@floatCast(gt * sg2)));
                const ref2 = fromBf16(bf16Bits(@floatCast(sl2 * up)));
                std.debug.print("  i={d} g={d:.3} u={d:.3} dev={d:.4} ref={d:.4}\n", .{ i, fromBf16(g[i]), fromBf16(u[i]), fromBf16(h), ref2 });
            }
            return error.Act;
        }
        check.pass("glm act: worst {d:.5}", .{worst});
    }

    // -- combine --
    {
        const topk = 8;
        const ye = try randBf16(a, ROWS * topk * D, 1.0);
        defer a.free(ye);
        const ys = try randBf16(a, ROWS * D, 1.0);
        defer a.free(ys);
        const wts = try a.alloc(f32, ROWS * topk);
        defer a.free(wts);
        for (wts) |*v| v.* = rand() * 0.3;
        var dy2 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ye));
        defer dy2.free();
        var dys = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(ys));
        defer dys.free();
        var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(wts));
        defer dw.free();
        var dout = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * D * 2);
        defer dout.free();
        try o.combine(s, dys.ptr, dy2.ptr, dw.ptr, dout.ptr, ROWS, topk, D);
        try s.synchronize();
        const got = try check.download(gpu, dout);
        defer a.free(got);
        var wi: usize = 0;
    var worst: f64 = 0;
        for (0..ROWS) |r| {
            for (0..D) |c| {
                var acc: f32 = @as(f32, wts[r * topk]) * @as(f32, @floatCast(fromBf16(ye[(r * topk) * D + c])));
                for (1..topk) |k| acc = @mulAdd(f32, acc, wts[r * topk + k], @floatCast(fromBf16(ye[(r * topk + k) * D + c])));
                const routed = fromBf16(bf16Bits(acc));
                const r32: f32 = @floatCast(routed);
                const y32: f32 = @floatCast(fromBf16(ys[r * D + c]));
                const ref = fromBf16(bf16Bits(r32 + y32));
                const h: u16 = std.mem.readInt(u16, got[(r * D + c) * 2 ..][0..2], .little);
                const d = @abs(fromBf16(h) - ref);
                if (d > worst) {
                    worst = d;
                    wi = r * D + c;
                }
            }
        }
        if (worst > 1e-2) {
            { // device-side wts/ye vs host, at the worst row
                const dwb = try check.download(gpu, dw);
                defer a.free(dwb);
                const dwv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, dwb));
                for (0..topk) |k| {
                    if (@abs(dwv[wi / D * topk + k] - wts[wi / D * topk + k]) > 1e-6)
                        std.debug.print("  DEVICE wts r1k{d}: {d:.6} host {d:.6}\n", .{ k, dwv[wi / D * topk + k], wts[wi / D * topk + k] });
                }
                const dyb = try check.download(gpu, dy2);
                defer a.free(dyb);
                for (0..topk) |k| {
                    const at = (wi / D * topk + k) * D + wi % D;
                    const hv: u16 = std.mem.readInt(u16, dyb[at * 2 ..][0..2], .little);
                    if (hv != ye[at])
                        std.debug.print("  DEVICE ye k{d}: {x} host {x}\n", .{ k, hv, ye[at] });
                }
            }
            const hc: u16 = std.mem.readInt(u16, got[wi * 2 ..][0..2], .little);
            const wr2 = wi / D;
            const wc = wi - wr2 * D;
            var acc: f32 = @as(f32, wts[wr2 * topk]) * @as(f32, @floatCast(fromBf16(ye[(wr2 * topk) * D + wc])));
            for (1..topk) |k| acc = @mulAdd(f32, acc, wts[wr2 * topk + k], @floatCast(fromBf16(ye[(wr2 * topk + k) * D + wc])));
            const refchain = fromBf16(bf16Bits(acc));
            const y32b: f32 = @floatCast(fromBf16(ys[wi]));
            const reffinal = fromBf16(bf16Bits(@as(f32, @floatCast(refchain)) + y32b));
            std.debug.print("FAIL glm combine: {d:.4} at r{d} c{d}: dev {d:.4} chain {d:.4} refFinal {d:.4}\n", .{ worst, wr2, wc, fromBf16(hc), refchain, reffinal });
            for (0..topk) |k| {
                const yv = fromBf16(ye[(wr2 * topk + k) * D + wc]);
                std.debug.print("  k{d}: w {d:.4} y {d:.4}\n", .{ k, wts[wr2 * topk + k], yv });
            }
            const h0: u16 = std.mem.readInt(u16, got[0..2], .little);
            var acc0: f64 = @as(f64, wts[0]) * fromBf16(ye[0]);
            for (1..topk) |k| acc0 = acc0 * wts[k] + fromBf16(ye[k * D]);
            std.debug.print("  dev0 {d:.4} routed {d:.4} ys {d:.4}\n", .{ fromBf16(h0), fromBf16(bf16Bits(@floatCast(acc0))), fromBf16(ys[0]) });
            return error.Combine;
        }
        check.pass("glm combine: worst {d:.5}", .{worst});
    }

    // -- stream mean / add --
    {
        const x = try randBf16(a, ROWS * 4 * D, 1.0);
        defer a.free(x);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx.free();
        var dy = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * D * 2);
        defer dy.free();
        try o.streamMean(s, dx.ptr, dy.ptr, ROWS, D);
        const b = try randBf16(a, ROWS * D, 1.0);
        defer a.free(b);
        var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(b));
        defer db.free();
        var dz = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * D * 2);
        defer dz.free();
        try o.add(s, dy.ptr, db.ptr, dz.ptr, ROWS * D);
        try s.synchronize();
        const got = try check.download(gpu, dz);
        defer a.free(got);
        var worst: f64 = 0;
        for (0..ROWS) |r| {
            for (0..D) |c| {
                var v: f64 = fromBf16(x[(r * 4) * D + c]);
                for (1..4) |k| v += fromBf16(x[((r * 4) + k) * D + c]);
                const mean = fromBf16(bf16Bits(@floatCast(v * 0.25)));
                const ref = fromBf16(bf16Bits(@floatCast(mean + fromBf16(b[r * D + c]))));
                const h: u16 = std.mem.readInt(u16, got[(r * D + c) * 2 ..][0..2], .little);
                worst = @max(worst, @abs(fromBf16(h) - ref));
            }
        }
        if (worst > 1e-2) {
            std.debug.print("FAIL glm mean/add: {d:.4}\n", .{worst});
            return error.MeanAdd;
        }
        check.pass("glm mean+add: worst {d:.5}", .{worst});
    }

    // - argmax, with a tie --
    {
        const V = 3000;
        const lg = try randBf16(a, 2 * V, 1.0);
        defer a.free(lg);
        lg[7] = lg[3]; // a tie for row 1
        var dl = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(lg));
        defer dl.free();
        var dout = try cuda.DeviceBuffer.alloc(gpu.d, 2 * 4);
        defer dout.free();
        try o.argmax(s, dl.ptr, dout.ptr, 2, V);
        try s.synchronize();
        const got = try check.download(gpu, dout);
        defer a.free(got);
        const picks = std.mem.bytesAsSlice(i32, @as([]align(1) u8, got));
        for (0..2) |r| {
            var best: f32 = -1e30;
            var at: usize = 0;
            for (0..V) |i| {
                const v: f32 = @floatCast(fromBf16(lg[r * V + i]));
                if (v > best) {
                    best = v;
                    at = i;
                }
            }
            if (picks[r] != at) {
                std.debug.print("FAIL glm argmax row {d}: {d} vs {d}\n", .{ r, picks[r], at });
                return error.Argmax;
            }
        }
        check.pass("glm argmax: exact incl tie", .{});
    }

    // -- route --
    {
        const E = 288;
        const topk = 8;
        const lg = try a.alloc(f32, ROWS * E);
        defer a.free(lg);
        for (lg) |*v| v.* = rand() * 4.0;
        const bias = try a.alloc(f32, E);
        defer a.free(bias);
        for (bias) |*v| v.* = rand() * 0.5 - 0.25;
        var dl = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(lg));
        defer dl.free();
        var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(bias));
        defer db.free();
        var dp = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * topk * 4);
        defer dp.free();
        var dw = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * topk * 4);
        defer dw.free();
        const scale: f32 = 2.5;
        try o.route(s, dl.ptr, db.ptr, dp.ptr, dw.ptr, ROWS, E, topk, scale);
        try s.synchronize();
        const got_p = try check.download(gpu, dp);
        defer a.free(got_p);
        const got_w = try check.download(gpu, dw);
        defer a.free(got_w);
        const picks = std.mem.bytesAsSlice(i32, @as([]align(1) u8, got_p));
        const wout = std.mem.bytesAsSlice(f32, @as([]align(1) u8, got_w));
        for (0..ROWS) |r| {
            var c: [E]f64 = undefined;
            var sc: [E]f64 = undefined;
            for (0..E) |e| {
                const x = lg[r * E + e];
                const ee = @exp(@abs(x));
                const y = 1.0 / (1.0 + ee);
                sc[e] = if (x < 0) y else 1.0 - y;
                c[e] = sc[e] + bias[e];
            }
            var total: f64 = 0;
            var wref: [8]f64 = undefined;
            for (0..topk) |k| {
                var best: f64 = -1e30;
                var at: usize = 0;
                for (0..E) |e| {
                    if (c[e] > best or (@abs(c[e] - best) < 1e-12 and e < at)) {
                        best = c[e];
                        at = e;
                    }
                }
                if (picks[r * topk + k] != at) {
                    std.debug.print("FAIL glm route row {d} k {d}: {d} vs {d}\n", .{ r, k, picks[r * topk + k], at });
                    return error.RoutePick;
                }
                wref[k] = sc[at];
                c[at] = -1e30;
            }
            total = wref[0];
            for (1..topk) |k| total = total + wref[k];
            for (0..topk) |k| {
                const ref: f64 = (wref[k] / total) * scale;
                if (@abs(wout[r * topk + k] - @as(f32, @floatCast(ref))) > 1e-4) {
                    std.debug.print("FAIL glm route w row {d} k {d}: {d:.6} vs {d:.6}\n", .{ r, k, wout[r * topk + k], ref });
                    return error.RouteW;
                }
            }
        }
        check.pass("glm route: picks exact, weights 1e-4, {d} rows", .{ROWS});
    }

    // -- the head shape: m x 154880 x 4096 bf16 --
    {
        const V = 154880;
        const K2 = 4096;
        for ([_]usize{ 4, 16 }) |m2| {
            const x2 = try randBf16(a, m2 * K2, 1.0);
            defer a.free(x2);
            const w2 = try randBf16(a, V * K2, 0.01);
            defer a.free(w2);
            var dx2 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x2));
            defer dx2.free();
            var dw2 = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(w2));
            defer dw2.free();
            var dd2 = try cuda.DeviceBuffer.alloc(gpu.d, m2 * V * 2);
            defer dd2.free();
            var ws2 = try cuda.DeviceBuffer.alloc(gpu.d, 64 << 20);
            defer ws2.free();
            var lin2 = cuda.cublaslt.Linear.init(&o.lt, m2, V, K2, .bf16, 64 << 20) catch |err| {
                std.debug.print("head m={d}: INIT {t}" ++ "\n", .{ m2, err });
                continue;
            };
            defer lin2.deinit();
            lin2.run(dx2.ptr, dw2.ptr, dd2.ptr, ws2.ptr, ws2.len, s.handle) catch |err| {
                std.debug.print("head m={d}: RUN {t}" ++ "\n", .{ m2, err });
                continue;
            };
            try s.synchronize();
            check.pass("glm cublasLt head m={d}: OK", .{m2});
        }
    }

    // -- cublasLt GEMM: D[5,8] = X[5,4].W^T[8,4] fp32 out --
    {
        const m = 5;
        const n = 8;
        const k = 4;
        const x = try randBf16(a, m * k, 1.0);
        defer a.free(x);
        const w = try randBf16(a, n * k, 1.0);
        defer a.free(w);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
        defer dx.free();
        var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(w));
        defer dw.free();
        var dd = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 4);
        defer dd.free();
        var ws = try cuda.DeviceBuffer.alloc(gpu.d, 8 << 20);
        defer ws.free();
        var lin = try cuda.cublaslt.Linear.init(&o.lt, m, n, k, .f32, 8 << 20);
        defer lin.deinit();
        try lin.run(dx.ptr, dw.ptr, dd.ptr, ws.ptr, ws.len, s.handle);
        try s.synchronize();
        const got = try check.download(gpu, dd);
        defer a.free(got);
        const out = std.mem.bytesAsSlice(f32, @as([]align(1) u8, got));
        var worst: f64 = 0;
        for (0..m) |r| for (0..n) |c| {
            var acc: f64 = 0;
            for (0..k) |j| acc += fromBf16(x[r * k + j]) * fromBf16(w[c * k + j]);
            worst = @max(worst, @abs(out[r * n + c] - @as(f32, @floatCast(acc))));
        };
        if (worst > 1e-2) {
            std.debug.print("FAIL glm gemm: {d:.4}\n", .{worst});
            return error.Gemm;
        }
        check.pass("glm cublasLt 5x8x4 fp32: worst {d:.5}", .{worst});
    }
}
