//! The KDA step against a scalar reference of the metal math, one head, a
//! window of tokens; the conv against the same. Small D stand-in? no: the real
//! 64x128 geometry, three tokens.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const glm = @import("glm");

const Gpu = check.Gpu;

const H = 64;
const D = 128;
const W = H * D;
const TAPS = 4;
const ROWS = 3;

var rng: u32 = 0x0BADC0DE;

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

fn bfRound(v: f64) f64 {
    return fromBf(bf16Bits(@floatCast(v)));
}

fn sig(x: f64) f64 {
    const e = @exp(@abs(x));
    const y = 1.0 / (1.0 + e);
    return if (x < 0) y else 1.0 - y;
}

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var k = try glm.kda.Kernels.load(gpu.d);
    defer k.deinit();
    var s = try cuda.Stream.init(gpu.d, true);
    defer s.deinit();

    // inputs
    const p = try a.alloc(u16, ROWS * (3 * W + 2 * W + H)); // q,k,v conved?? no: p is the P projection [rows, PS]
    _ = p;
    // The conv inputs: the P's first 3W are the q/k/v projections; f_b/g_b/b
    // GEMM outputs; the conv weights; the state.
    const conv_in = try a.alloc(u16, (TAPS - 1 + ROWS) * 3 * W);
    defer a.free(conv_in);
    for (conv_in) |*v| v.* = bf16Bits(rand() * 0.5);
    const cw = try a.alloc(u16, TAPS * 3 * W);
    defer a.free(cw);
    for (cw) |*v| v.* = bf16Bits(rand() * 0.5);
    // deterministic probe at a fixed column: cs rows hold 1,2,3; p rows hold 10,11,12; cw all taps 1
    const probe_col = 20000;
    {
        for (0..TAPS - 1) |e| conv_in[e * 3 * W + probe_col] = bf16Bits(@floatFromInt(e + 1));
        for (0..ROWS) |m| conv_in[(TAPS - 1 + m) * 3 * W + probe_col] = bf16Bits(@floatFromInt(10 + m));
        for (0..TAPS) |j| cw[j * 3 * W + probe_col] = bf16Bits(1.0);
    }
    // post-GEMM f_b/g_b [rows, 2W] and b [rows, H]: the kernel takes them flat
    const fg = try a.alloc(u16, ROWS * 2 * W);
    defer a.free(fg);
    for (fg) |*v| v.* = bf16Bits(rand() * 0.5);
    const bv = try a.alloc(u16, ROWS * H);
    defer a.free(bv);
    for (bv) |*v| v.* = bf16Bits(rand());
    // deterministic recurrence probe on head 28: f_b raw = dtb-inverse? simpler: f_b=0, g_b=0; b=+9 (beta~1); q,k,v one-hot-ish via conv_in probe cols already? NO: qkv comes from the conv of conv_in - use NEW probe columns for q/k/v of head 28:
    {
        const ph = 28;
        const cq = ph * D + 5; // part0 col
        const ck = W + ph * D + 5; // part1
        const cv = 2 * W + ph * D + 5; // part2
        // cs rows: q-history taps 1,2,3 -> wait conv window r=0: taps e=0,1,2 (cs) + p row0. Set cw taps [0.25, 0.5, 0.25, 1] wait cw is per (tap,col): set col-specific taps to make qkv deterministic: taps (1,0,0,0) => sl = sig-window of cs[0][col]
        for (0..ROWS) |_| {
            // p rows for r: token r's taps come from cs[r..r+2] and p[r] -- fully deterministic if we set ALL rows' q-col value = vq, etc. Simpler: set every row (cs and p) at cq/ck/cv to a fixed bf16, cw taps [1,0,0,0] at those cols => xb = the fixed value each token.
            const rows_all = (TAPS - 1 + ROWS);
            for (0..rows_all) |e| {
                conv_in[e * 3 * W + cq] = bf16Bits(1.5); // xb=1.5 -> sl = 1.5*sig(1.5)=1.1614
                conv_in[e * 3 * W + ck] = bf16Bits(-1.0); // sl=-0.2689
                conv_in[e * 3 * W + cv] = bf16Bits(0.5); // sl=0.3110
            }
        }
        for (0..TAPS) |j| {
            cw[j * 3 * W + cq] = bf16Bits(if (j == 0) 1.0 else 0.0);
            cw[j * 3 * W + ck] = bf16Bits(if (j == 0) 1.0 else 0.0);
            cw[j * 3 * W + cv] = bf16Bits(if (j == 0) 1.0 else 0.0);
        }
        // f_b=0, g_b=0 for head 28 all rows; b=9 -> beta=1-1/(1+e9)~=1
        for (0..ROWS) |r| {
            const f0 = r * 2 * W + ph * D;
            for (0..D) |d| fg[f0 + d] = bf16Bits(0.0);
            const g0 = r * 2 * W + W + ph * D;
            for (0..D) |d| fg[g0 + d] = bf16Bits(0.0);
            bv[r * H + ph] = bf16Bits(9.0);
        }
        // q=1.1614 uniform over dk (all D cols of q for head28 same? only col 5 set! -> q vector is one-hot at dk=5)
    }
    const av = try a.alloc(f32, H);
    defer a.free(av);
    for (av) |*v| v.* = 0.5 + rand() * 0.5;
    const dtb = try a.alloc(f32, H * D);
    defer a.free(dtb);
    for (dtb) |*v| v.* = rand() * 0.1;
    const st0 = try a.alloc(f32, H * D * D);
    defer a.free(st0);
    for (st0) |*v| v.* = rand() * 0.1;
    const onw = try a.alloc(u16, W);
    defer a.free(onw);
    for (onw) |*v| v.* = bf16Bits(1.0 + rand() * 0.1);

    // device
    var dconv_in = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(conv_in));
    defer dconv_in.free();
    var dcw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(cw));
    defer dcw.free();
    var dfg = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(fg));
    defer dfg.free();
    var dbv = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(bv));
    defer dbv.free();
    var dav = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(av));
    defer dav.free();
    var ddtb = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(dtb));
    defer ddtb.free();
    var dst = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(st0));
    defer dst.free();
    var donw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(onw));
    defer donw.free();
    var dqkv = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * 3 * W * 2);
    defer dqkv.free();
    var dsto = try cuda.DeviceBuffer.alloc(gpu.d, H * D * D * 4);
    defer dsto.free();
    var dy = try cuda.DeviceBuffer.alloc(gpu.d, ROWS * W * 2);
    defer dy.free();

    // the conv's p view: the kernel reads p[(e-(TAPS-1))*ps + i] — the same
    // buffer tail as conv_in's last ROWS rows, ps = 3W
    const p_dev = dconv_in.ptr + (TAPS - 1) * 3 * W * 2;
    try k.conv(s, p_dev, 3 * W, dconv_in.ptr, dcw.ptr, dqkv.ptr, ROWS);
    try k.step(s, dqkv.ptr, dfg.ptr, dbv.ptr, dav.ptr, ddtb.ptr, dst.ptr, donw.ptr, -5.0, 1e-6, ROWS, dsto.ptr, dy.ptr);
    try s.synchronize();

    // ── the reference ──
    const y_ref = try a.alloc(f64, ROWS * W);
    defer a.free(y_ref);
    const st = try a.dupe(f32, st0);
    defer a.free(st);
    const avh = try a.dupe(f32, av);
    defer a.free(avh);
    const lb: f64 = -5.0;
    const eps: f64 = 1e-6;

    for (0..ROWS) |r| {
        // the conv's three parts
        const q = try a.alloc(f64, W);
        defer a.free(q);
        const kk = try a.alloc(f64, W);
        defer a.free(kk);
        const vv = try a.alloc(f64, W);
        defer a.free(vv);
        for (0..3 * W) |i| {
            const part = i / W;
            const rem = i - part * W;
            var acc: f64 = 0;
            for (0..TAPS) |j| {
                const e = r + j;
                const src: f64 = if (e < TAPS - 1)
                    fromBf(conv_in[@as(usize, e) * 3 * W + i])
                else
                    fromBf(conv_in[e * 3 * W + i]); // history then p, contiguous
                acc += src * fromBf(cw[j * 3 * W + i]);
            }
            const xb = bfRound(acc);
            const sl = bfRound(xb * bfRound(sig(xb)));
            switch (part) {
                0 => q[rem] = sl,
                1 => kk[rem] = sl,
                else => vv[rem] = sl,
            }
        }
        const qscale: f32 = 1.0 / @sqrt(@as(f32, D));
        for (0..H) |h| {
            var pq: f32 = 0;
            var pk: f32 = 0;
            for (0..D) |d| {
                pq = @mulAdd(f32, @as(f32, @floatCast(q[h * D + d])), @as(f32, @floatCast(q[h * D + d])), pq);
                pk = @mulAdd(f32, @as(f32, @floatCast(kk[h * D + d])), @as(f32, @floatCast(kk[h * D + d])), pk);
            }
            const rq: f32 = 1.0 / @sqrt(pq + 1e-6);
            const rk: f32 = 1.0 / @sqrt(pk + 1e-6);
            const beta: f32 = @floatCast(sig(fromBf(bv[r * H + h])));
            const gvec = try a.alloc(f32, D);
            defer a.free(gvec);
            const kvec = try a.alloc(f32, D);
            defer a.free(kvec);
            const qvec = try a.alloc(f32, D);
            defer a.free(qvec);
            for (0..D) |d| {
                const raw: f32 = @floatCast(fromBf(fg[r * 2 * W + h * D + d]));
                const avd: f32 = raw + dtb[h * D + d];
                const e2: f32 = @exp(@abs(avd * @as(f32, @floatCast(avh[h]))));
                const y2: f32 = 1.0 / (1.0 + e2);
                const sg2: f32 = if (avd * @as(f32, @floatCast(avh[h])) < 0) y2 else 1.0 - y2;
                gvec[d] = @exp(@as(f32, @floatCast(lb)) * sg2);
                kvec[d] = @floatCast(bfRound(kk[h * D + d] * rk));
                qvec[d] = @floatCast(bfRound((q[h * D + d] * rq) * qscale));
            }
            for (0..D) |dv| {
                const off = (@as(usize, h) * D + dv) * D;
                var kvdot: f32 = 0;
                for (0..D) |dk| {
                    st[off + dk] *= gvec[dk];
                    kvdot = @mulAdd(f32, st[off + dk], kvec[dk], kvdot);
                }
                const delta: f32 = (@as(f32, @floatCast(vv[h * D + dv])) - kvdot) * beta;
                var oo: f32 = 0;
                for (0..D) |dk| {
                    st[off + dk] = @mulAdd(f32, kvec[dk], delta, st[off + dk]);
                    oo = @mulAdd(f32, st[off + dk], qvec[dk], oo);
                }
                y_ref[r * W + h * D + dv] = bfRound(oo);
            }
            var po: f32 = 0;
            for (0..D) |d| po = @mulAdd(f32, @as(f32, @floatCast(y_ref[r * W + h * D + d])), @as(f32, @floatCast(y_ref[r * W + h * D + d])), po);
            const rn: f32 = 1.0 / @sqrt(po / D + @as(f32, @floatCast(eps)));
            for (0..D) |d| {
                const gd: f32 = @floatCast(fromBf(fg[r * 2 * W + W + h * D + d]));
                const ed: f32 = @exp(@abs(gd));
                const gate: f32 = if (gd < 0) 1.0 / (1.0 + ed) else 1.0 - 1.0 / (1.0 + ed);
                var x: f32 = @as(f32, @floatCast(y_ref[r * W + h * D + d])) * rn;
                x = @as(f32, @floatCast(fromBf(onw[h * D + d]))) * x;
                x *= gate;
                y_ref[r * W + h * D + d] = bfRound(x);
            }
        }
    }

    // the probe: expected sl for r = sigmoided sum of taps r..r+3 (values: r<3 mixes cs and p)
    {
        const pb = try check.download(gpu, dqkv);
        defer a.free(pb);
        for (0..ROWS) |r| {
            const h: u16 = std.mem.readInt(u16, pb[(r * 3 * W + probe_col) * 2 ..][0..2], .little);
            var acc: f64 = 0;
            var parts: [4]f64 = undefined;
            for (0..TAPS) |j| {
                const e = r + j;
                const v: f64 = if (e < TAPS - 1) fromBf(conv_in[e * 3 * W + probe_col]) else fromBf(conv_in[e * 3 * W + probe_col]);
                parts[j] = v;
                acc += v * 1.0;
            }
            std.debug.print("  probe r{d}: dev {d:.4} taps {d:.0},{d:.0},{d:.0},{d:.0} (sum {d:.0})\n", .{ r, fromBf(h), parts[0], parts[1], parts[2], parts[3], acc });
        }
    }

    // the conv's out first: isolates conv vs step
    {
        const qkvb = try check.download(gpu, dqkv);
        defer a.free(qkvb);
        var worst_q: f64 = 0;
        var wi_q: usize = 0;
        for (0..ROWS * 3 * W) |i| {
            const part = (i % (3 * W)) / W;
            const rem = i % W;
            const r = i / (3 * W);
            var acc: f64 = 0;
            for (0..TAPS) |j| {
                const e = r + j;
                const src: f64 = fromBf(conv_in[e * 3 * W + (part * W + rem)]);
                acc += src * fromBf(cw[j * 3 * W + (part * W + rem)]);
            }
            const xb = bfRound(acc);
            const sl = bfRound(xb * bfRound(sig(xb)));
            const h16: u16 = std.mem.readInt(u16, qkvb[i * 2 ..][0..2], .little);
            const d2 = @abs(fromBf(h16) - sl);
            if (d2 > worst_q) {
                worst_q = d2;
                wi_q = i;
            }
        }
        check.pass("kda conv: worst {d:.4} at flat {d}", .{ worst_q, wi_q });
        if (worst_q > 4e-2) {
            const r2 = wi_q / (3 * W);
            const col = wi_q - r2 * 3 * W;
            const h16: u16 = std.mem.readInt(u16, qkvb[wi_q * 2 ..][0..2], .little);
            var acc2: f64 = 0;
            for (0..TAPS) |j| {
                const e = r2 + j;
                const src: f64 = fromBf(conv_in[e * 3 * W + col]);
                const w = fromBf(cw[j * 3 * W + col]);
                acc2 += src * w;
                std.debug.print("  tap{d}: src {d:.4} w {d:.4}\n", .{ j, src, w });
            }
            const xb = bfRound(acc2);
            std.debug.print("  flat {d} r{d} col {d}: dev {d:.4} acc {d:.4} xb {d:.4} sl {d:.4}\n", .{ wi_q, r2, col, fromBf(h16), acc2, xb, bfRound(xb * bfRound(sig(xb))) });
            // a shift hunt: does dev[i] match the ref at i + delta?
            const dv: f64 = fromBf(h16);
            var found: ?i64 = null;
            var delta: i64 = -3;
            while (delta <= 3) : (delta += 1) {
                const j: usize = @intCast(@as(i64, @intCast(wi_q)) + delta + 8192 * 3); // wrap-safe probe rows
                const jj = j % (ROWS * 3 * W);
                // recompute ref at jj quickly: brute force one element
                const rr = jj / (3 * W);
                const cc = jj - rr * 3 * W;
                var acc3: f64 = 0;
                for (0..TAPS) |j2| {
                    const e = rr + j2;
                    acc3 += fromBf(conv_in[e * 3 * W + cc]) * fromBf(cw[j2 * 3 * W + cc]);
                }
                const xb3 = bfRound(acc3);
                const sl3 = bfRound(xb3 * bfRound(sig(xb3)));
                if (@abs(sl3 - dv) < 1e-3) {
                    found = delta + 8192 * 3 - (@as(i64, @intCast(wi_q)) - @as(i64, @intCast(wi_q))); // nonsense guard
                    std.debug.print("  dev matches ref at flat {d} (delta {d})\n", .{ jj, @as(i64, @intCast(jj)) - @as(i64, @intCast(wi_q)) });
                }
            }
            if (found == null) std.debug.print("  no +-3 shift match; widening...\n", .{});
            return error.KdaConv;
        }
    }

    // the compare, bf16 grid
    // the state, fp32 exact-ish
    const got_st = try check.download(gpu, dsto);
    defer a.free(got_st);
    const so = std.mem.bytesAsSlice(f32, @as([]align(1) u8, got_st));
    var worst_st: f64 = 0;
    for (so, st) |g, w| worst_st = @max(worst_st, @abs(g - w));
    if (worst_st > 1e-3) {
        std.debug.print("FAIL kda state: worst {d:.5}\n", .{worst_st});
        var wst: f64 = 0;
        var wst_i: usize = 0;
        var idx: usize = 0;
        while (idx < so.len) : (idx += 1) {
            const d3 = @abs(so[idx] - st[idx]);
            if (d3 > wst) {
                wst = d3;
                wst_i = idx;
            }
        }
        const hh2 = wst_i / (D * D);
        const rem2 = wst_i - hh2 * D * D;
        const dv2 = rem2 / D;
        const dk2 = rem2 - dv2 * D;
        std.debug.print("  at h{d} dv{d} dk{d}: dev {d:.5} ref {d:.5}\n", .{ hh2, dv2, dk2, so[wst_i], st[wst_i] });
        // head 28 probe: q one-hot dk=5 (q=1.1614), k=-0.2689 all?? k also one-hot! v=0.311 one-hot
        const ph2 = 28;
        std.debug.print("  probe h28 st[dv5][dk5] dev {d:.5} ref {d:.5} (v=0.311, beta~1, k=-0.2689 one-hot)\n", .{ so[(ph2 * D + 5) * D + 5], st[(ph2 * D + 5) * D + 5] });
        std.debug.print("  probe h28 st[dv0][dk5] dev {d:.5} ref {d:.5}\n", .{ so[(ph2 * D + 0) * D + 5], st[(ph2 * D + 0) * D + 5] });
        return error.KdaSt;
    }
    check.pass("kda state: worst {d:.6}", .{worst_st});

    const got = try check.download(gpu, dy);
    defer a.free(got);
    var worst: f64 = 0;
    for (0..ROWS * W) |i| {
        const h: u16 = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
        worst = @max(worst, @abs(fromBf(h) - y_ref[i]));
    }
    if (worst > 4e-2) {
        std.debug.print("FAIL kda y: worst {d:.4}\n", .{worst});
        var wi: usize = 0;
        var wv: f64 = 0;
        for (0..ROWS * W) |i| {
            const h: u16 = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
            const d = @abs(fromBf(h) - y_ref[i]);
            if (d > wv) {
                wv = d;
                wi = i;
            }
        }
        const r = wi / W;
        const rem = wi - r * W;
        const hh = rem / D;
        const dv = rem - hh * D;
        const hg: u16 = std.mem.readInt(u16, got[wi * 2 ..][0..2], .little);
        std.debug.print("  at r{d} h{d} dv{d}: dev {d:.4} ref {d:.4}\n", .{ r, hh, dv, fromBf(hg), y_ref[wi] });
        // per-token worst, and h20's row snapshot at r2 vs r0
        var worst_r: [ROWS]f64 = .{ 0, 0, 0 };
        for (0..ROWS * W) |i| {
            const h2: u16 = std.mem.readInt(u16, got[i * 2 ..][0..2], .little);
            const rr = i / W;
            worst_r[rr] = @max(worst_r[rr], @abs(fromBf(h2) - y_ref[i]));
        }
        for (0..ROWS) |rr| std.debug.print("  worst r{d}: {d:.4}\n", .{ rr, worst_r[rr] });
        const base = 20 * D + 87;
        for (0..3) |rr| {
            const h0: u16 = std.mem.readInt(u16, got[(rr * W + base) * 2 ..][0..2], .little);
            std.debug.print("  h20dv87 r{d}: dev {d:.4} ref {d:.4}\n", .{ rr, fromBf(h0), y_ref[rr * W + base] });
        }
        return error.KdaY;
    }
    check.pass("kda step: worst {d:.4} over {d} tokens x {d} heads", .{ worst, ROWS, H });

    }
