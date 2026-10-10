//! The hc boundary kernels against a scalar host reference of the same math
//! (the metal kernel's semantics, metal/hc zigs notes in memory), synthetic rows.

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const check = @import("check.zig");

const hc = glm.hc;
const Gpu = check.Gpu;

const S = 4;
const D = 4096;
const F = S * D;
const MIX = 24;
const ROWS = 3;

var rng: u32 = 0x9E3779B9;

fn rand() f32 {
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return @as(f32, @floatFromInt(rng % 2001)) / 1000.0 - 1.0;
}

fn bf16Bits(v: f32) u16 {
    const bits: u32 = @bitCast(v);
    return @intCast(bits >> 16);
}

/// f64 through bf16, the kernel's stream-write rounding.
fn roundBf(v: f64) f64 {
    return fromBf16(bf16Bits(@floatCast(v)));
}

fn fromBf16(b: u16) f32 {
    const wide: u32 = @as(u32, b) << 16;
    return @as(f32, @bitCast(wide));
}

fn randBf16(a: std.mem.Allocator, n: usize) ![]u16 {
    const v = try a.alloc(u16, n);
    for (v) |*x| x.* = bf16Bits(rand() * 0.5);
    return v;
}

fn randF32(a: std.mem.Allocator, n: usize, scale: f32) ![]f32 {
    const v = try a.alloc(f32, n);
    for (v) |*x| x.* = rand() * scale;
    return v;
}

/// The reference, f64; mirrors the metal math including the bf16 roundings the
/// kernel makes (the stream write and the collapse).
fn reference(
    xold: []const u16,
    branch: []const u16,
    post: []const f32,
    comb: []const f32,
    fn_w: []const u16,
    scale: []const f32,
    base: []const f32,
    norm_w: []const u16,
    eps: f32,
    expand: bool,
    x_new: []f32, // out [F] (the streams the split reads)
    normed: []f32, // out [D]
    post_out: []f32, // out [S]
    comb_out: []f32, // out [S*S]
    pre_out: []f32, // out [S] debug
    mix_out: []f64, // out [MIX] debug
) void {
    var mix: [MIX]f64 = undefined;
    var ss: f64 = 0;
    for (0..F) |f| {
        const s = f / D;
        const d = f - s * D;
        var v: f64 = fromBf16(xold[f]);
        if (expand) {
            var mm: f64 = @as(f64, comb[0 * S + s]) * fromBf16(xold[0 * D + d]);
            mm += @as(f64, comb[1 * S + s]) * fromBf16(xold[1 * D + d]);
            mm += @as(f64, comb[2 * S + s]) * fromBf16(xold[2 * D + d]);
            mm += @as(f64, comb[3 * S + s]) * fromBf16(xold[3 * D + d]);
            v = @floatCast(@as(f32, @floatCast(@as(f64, post[s]) * fromBf16(branch[d]) + mm)));
            v = roundBf(v); // the stream write's rounding
        }
        x_new[f] = @floatCast(v);
        ss += v * v;
        for (0..MIX) |m| mix[m] += @as(f64, fromBf16(fn_w[m * F + f])) * v;
    }
    const inv_n = 1.0 / @sqrt(ss / F + eps);
    for (0..MIX) |m| mix[m] *= inv_n; // the metal normalizes before every use
    @memcpy(mix_out, mix[0..]);
    var pre: [S]f64 = undefined;
    for (0..S) |s| {
        const z = mix[s] * scale[0] + base[s];
        pre[s] = 1.0 / (1.0 + @exp(-z)) + eps;
        const zp = mix[S + s] * scale[1] + base[S + s];
        post_out[s] = @floatCast(2.0 / (1.0 + @exp(-zp)));
        pre_out[s] = @floatCast(pre[s]);
    }
    var m4: [S][S]f64 = undefined;
    var rr: [S][S]f64 = undefined;
    for (0..S) |i| for (0..S) |jj| {
        m4[i][jj] = mix[2 * S + i * S + jj] * scale[2] + base[2 * S + i * S + jj];
    };
    for (0..S) |i| { // the row softmax then the Sinkhorn's first col norm
        var mx: f64 = -1e30;
        for (0..S) |jj| mx = @max(mx, m4[i][jj]);
        var sum: f64 = 0;
        for (0..S) |jj| {
            rr[i][jj] = @exp(m4[i][jj] - mx);
            sum += rr[i][jj];
        }
        for (0..S) |jj| rr[i][jj] = rr[i][jj] / (sum + eps) + eps;
    }
    for (0..20) |_| {
        for (0..S) |i| {
            var sum: f64 = 0;
            for (0..S) |jj| sum += rr[i][jj];
            for (0..S) |jj| rr[i][jj] /= sum + eps;
        }
        for (0..S) |jj| {
            var sum: f64 = 0;
            for (0..S) |i| sum += rr[i][jj];
            for (0..S) |i| rr[i][jj] /= sum + eps;
        }
    }
    for (0..S) |i| for (0..S) |jj| {
        comb_out[i * S + jj] = @floatCast(rr[i][jj]);
    };
    var xc: [D]f64 = undefined;
    var sss: f64 = 0;
    for (0..D) |d| {
        var v: f64 = 0;
        for (0..S) |s| v += pre[s] * x_new[s * D + d];
        xc[d] = roundBf(v);
        sss += xc[d] * xc[d];
    }
    const inv = 1.0 / @sqrt(sss / D + eps);
    for (0..D) |d| normed[d] = @floatCast(fromBf16(norm_w[d]) * @as(f32, @floatCast(xc[d] * inv)));
}

pub fn run(gpu: Gpu) !void {
    const a = gpu.gpa;
    var k = try hc.Kernels.load(gpu.d);
    defer k.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    // the persisted pair and the epse
    const post = try randF32(a, S, 0.5);
    defer a.free(post);
    const comb = try randF32(a, S * S, 0.5);
    defer a.free(comb);
    const eps = try a.dupe(f32, &.{1e-6});
    defer a.free(eps);

    for ([_]bool{ true, false }) |expand| {
        const xold = try randBf16(a, F);
        defer a.free(xold);
        const branch = try randBf16(a, D);
        defer a.free(branch);
        const fn_w = try randBf16(a, MIX * F);
        defer a.free(fn_w);
        const scale = try randF32(a, 3, 0.5);
        defer a.free(scale);
        const base = try randF32(a, MIX, 0.5);
        defer a.free(base);
        const norm_w = try randBf16(a, D);
        defer a.free(norm_w);

        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(xold));
        defer dx.free();
        var dbr = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(branch));
        defer dbr.free();
        var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(post));
        defer dp.free();
        var dc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(comb));
        defer dc.free();
        var df = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(fn_w));
        defer df.free();
        var dsc = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scale));
        defer dsc.free();
        var db = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(base));
        defer db.free();
        var dnw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(norm_w));
        defer dnw.free();
        var deps = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(eps));
        defer deps.free();

        var dxn = try cuda.DeviceBuffer.alloc(gpu.d, F * 2);
        defer dxn.free();
        var dpart = try cuda.DeviceBuffer.alloc(gpu.d, hc.partBytes(1));
        defer dpart.free();
        var dn = try cuda.DeviceBuffer.alloc(gpu.d, D * 2);
        defer dn.free();
        var dpo = try cuda.DeviceBuffer.alloc(gpu.d, S * 4);
        defer dpo.free();
        var dco = try cuda.DeviceBuffer.alloc(gpu.d, S * S * 4);
        defer dco.free();

        // the device chain, one row; without the expand the host presets the
        // streams (the engine's ping-pong holds the old there)
        if (!expand) try dxn.upload(0, std.mem.sliceAsBytes(xold));
        try k.expandMix(stream, dx.ptr, dbr.ptr, dp.ptr, dc.ptr, df.ptr, dxn.ptr, dpart.ptr, 1, expand);
        try k.split(stream, dxn.ptr, dpart.ptr, dsc.ptr, db.ptr, dnw.ptr, deps.ptr, dn.ptr, dpo.ptr, dco.ptr, 1);
        try stream.synchronize();

        // the reference
        const x_new = try a.alloc(f32, F);
        defer a.free(x_new);
        const normed = try a.alloc(f32, D);
        defer a.free(normed);
        const post_out = try a.alloc(f32, S);
        defer a.free(post_out);
        const comb_out = try a.alloc(f32, S * S);
        defer a.free(comb_out);
        const xnew_host = try a.alloc(f32, F); // the reference's stream image
        defer a.free(xnew_host);
        const pre_ref = try a.alloc(f32, S);
        defer a.free(pre_ref);
        const mix_ref = try a.alloc(f64, 24);
        defer a.free(mix_ref);
        reference(xold, branch, post, comb, fn_w, scale, base, norm_w, eps[0], expand, xnew_host, normed, post_out, comb_out, pre_ref, mix_ref);

        // the part buffer against the reference's partials, recomputed here
        {
            const pb2 = try check.download(gpu, dpart);
            defer a.free(pb2);
            const dv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, pb2));
            var part_worst: f32 = 0;
            var xn: [F]f32 = undefined;
            // recompute the stream image the way the reference does
            for (0..F) |f| {
                const si = f / D;
                const di = f - si * D;
                var v: f64 = fromBf16(xold[f]);
                if (expand) {
                    var mm: f64 = @as(f64, comb[0 * S + si]) * fromBf16(xold[0 * D + di]);
                    mm += @as(f64, comb[1 * S + si]) * fromBf16(xold[1 * D + di]);
                    mm += @as(f64, comb[2 * S + si]) * fromBf16(xold[2 * D + di]);
                    mm += @as(f64, comb[3 * S + si]) * fromBf16(xold[3 * D + di]);
                    v = roundBf(@as(f64, post[si]) * fromBf16(branch[di]) + mm);
                }
                xn[f] = @floatCast(v);
            }
            for (0..25) |t| {
                var sum: f64 = 0;
                if (t < 24) {
                    for (0..F) |f| sum += @as(f64, fromBf16(fn_w[t * F + f])) * xn[f];
                } else {
                    for (0..F) |f| sum += @as(f64, xn[f]) * xn[f];
                }
                var tot: f64 = 0;
                for (0..16) |j| tot += dv[j * 25 + t];
                part_worst = @max(part_worst, @abs(@as(f32, @floatCast(tot)) - @as(f32, @floatCast(sum))));
            }
            const pob = try check.download(gpu, dpo);
            defer a.free(pob);
            const pol = std.mem.bytesAsSlice(f32, @as([]align(1) u8, pob));
            check.pass("hc part (expand={}): worst {d:.4}", .{ expand, part_worst });
            // post recomputed from the device's own parts: isolates the split
            var tot2: [25]f64 = undefined;
            for (0..25) |t| {
                var acc: f64 = 0;
                for (0..16) |j| acc += dv[j * 25 + t];
                tot2[t] = acc;
            }
            const inv2 = 1.0 / @sqrt(tot2[24] / F + eps[0]);
            var mix_worst: f64 = 0;
            var mt: usize = 0;
            for (0..24) |t| {
                var d: f64 = 0;
                for (0..16) |j| d += dv[j * 25 + t];
                d *= inv2;
                const ref_m: f64 = blk: {
                    var acc: f64 = 0;
                    for (0..F) |f2| acc += @as(f64, fromBf16(fn_w[t * F + f2])) * xn[f2];
                    break :blk acc * (1.0 / @sqrt(tot2[24] / F + eps[0]));
                };
                if (@abs(d - ref_m) > mix_worst) {
                    mix_worst = @abs(d - ref_m);
                    mt = t;
                }
            }
            check.pass("hc mix-from-device (expand={}): worst {d:.4} at t={d}", .{ expand, mix_worst, mt });
            if (expand) {
                for (0..8) |t| {
                    var d: f64 = 0;
                    for (0..16) |j2| d += dv[j2 * 25 + t];
                    d *= inv2;
                    const ref_m: f64 = blk2: {
                        var acc: f64 = 0;
                        for (0..F) |f2| acc += @as(f64, fromBf16(fn_w[t * F + f2])) * xnew_host[f2];
                        break :blk2 acc * (1.0 / @sqrt(tot2[24] / F + eps[0]));
                    };
                    std.debug.print("  mix[{d}]: dev {d:.3} refXn {d:.3}\n", .{ t, d, ref_m });
                }
            }
            var post2: f32 = 0;
            for (0..S) |s2| {
                const z = tot2[4 + s2] * inv2 * scale[1] + base[4 + s2];
                post2 = @max(post2, @abs(@as(f32, @floatCast(2.0 / (1.0 + @exp(-z)))) - pol[s2]));
            }
            check.pass("hc post-from-device-parts (expand={}): worst {d:.5}", .{ expand, post2 });
        }

        // the device x_new (when expanded) against the reference's
        if (expand) {
            const xb = try check.download(gpu, dxn);
            defer a.free(xb);
            const dev = std.mem.bytesAsSlice(u16, @as([]align(1) u8, xb));
            var worst: f32 = 0;
            for (0..F) |f| worst = @max(worst, @abs(fromBf16(dev[f]) - xnew_host[f]));
            if (worst > 2e-2) {
                std.debug.print("FAIL hc x_new: worst {d:.4}\n", .{worst});
                return error.XNew;
            }
            check.pass("hc x_new: worst {d:.5}", .{worst});
        }


        const nb = try check.download(gpu, dn);
        defer a.free(nb);
        const dn16 = std.mem.bytesAsSlice(u16, @as([]align(1) u8, nb));
        var worst: f32 = 0;
        for (0..D) |d| worst = @max(worst, @abs(fromBf16(dn16[d]) - normed[d]));
        if (worst > 3e-2) {
            std.debug.print("FAIL hc normed (expand={}): worst {d:.4} at", .{ expand, worst });
            var shown: usize = 0;
            for (0..D) |d| {
                const dv = @abs(fromBf16(dn16[d]) - normed[d]);
                if (dv > 3e-2 and shown < 8) {
                    std.debug.print(" [{d}]dev {d:.4} ref {d:.4}", .{ d, fromBf16(dn16[d]), normed[d] });
                    shown += 1;
                }
            }
            std.debug.print("\n", .{});
        }
        check.pass("hc normed (expand={}): worst {d:.5}", .{ expand, worst });

        const pb = try check.download(gpu, dpo);
        defer a.free(pb);
        const po = std.mem.bytesAsSlice(f32, @as([]align(1) u8, pb));
        for (0..S) |s| {
            if (@abs(po[s] - post_out[s]) > 1e-2) {
                std.debug.print("FAIL hc post[{d}]: {d:.5} vs {d:.5}\n", .{ s, po[s], post_out[s] });
                return error.Post;
            }
        }
        const cb = try check.download(gpu, dco);
        defer a.free(cb);
        const co = std.mem.bytesAsSlice(f32, @as([]align(1) u8, cb));
        var comb_worst: f32 = 0;
        for (0..S * S) |i| comb_worst = @max(comb_worst, @abs(co[i] - comb_out[i]));
        var post_worst: f32 = 0;
        for (0..S) |i| post_worst = @max(post_worst, @abs(po[i] - post_out[i]));
        check.pass("hc post/comb (expand={}): worst {d:.5} / {d:.5}", .{ expand, post_worst, comb_worst });
    }
}
