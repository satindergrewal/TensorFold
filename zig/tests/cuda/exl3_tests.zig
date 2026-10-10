//! EXL3 kernels against the scalar spec (core/exl3_format) on the mixed checkpoint:
//! the tile decode per stream position, then every projection stage against its
//! host reference, direct and through qlinear, at both rates the checkpoint mixed.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const fmt = core.exl3_format;
const Gpu = check.Gpu;

const rows_decode = 5; // odd, inside one chunk
const rows_prompt = 33; // two chunks and a remainder

var rng: u32 = 0x12345678;

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

fn fromBf16(b: u16) f64 {
    const wide: u32 = @as(u32, b) << 16;
    return @floatCast(@as(f32, @bitCast(wide)));
}

const Case = struct {
    label: [64]u8,
    label_len: usize,
    trellis: []align(1) const u8,
    suh: []align(1) const u8,
    svh: []align(1) const u8,
    halves: u32,
    k: u32,
    n: u32,

    fn name(c: *const Case) []const u8 {
        return c.label[0..c.label_len];
    }
};

fn loadCases(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !std.ArrayList(Case) {
    var cases: std.ArrayList(Case) = .empty;
    errdefer cases.deinit(gpa);
    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    const layer = "model.language_model.layers.3.mlp.experts.";
    var chosen = [2]?usize{ null, null }; // the first K4, the first K3
    for (0..288) |e| {
        var b: [128]u8 = undefined;
        const t = try ck.get(try std.fmt.bufPrint(&b, "{s}{d}.gate_proj.trellis", .{ layer, e }));
        const words = t.dim(2);
        const slot: usize = if (words == 64) 0 else if (words == 48) 1 else continue;
        if (chosen[slot] == null) chosen[slot] = e;
        if (chosen[0] != null and chosen[1] != null) break;
    }
    if (chosen[0] == null or chosen[1] == null) return error.NoMixedPair;
    for (chosen, 0..) |maybe, slot| {
        const e = maybe.?;
        var b: [128]u8 = undefined;
        const base = "{s}{d}.gate_proj";
        const trellis = try ck.get(try std.fmt.bufPrint(&b, base ++ ".trellis", .{ layer, e }));
        const suh = try ck.get(try std.fmt.bufPrint(&b, base ++ ".suh", .{ layer, e }));
        const svh = try ck.get(try std.fmt.bufPrint(&b, base ++ ".svh", .{ layer, e }));
        const words = trellis.dim(2);
        if (words != 64 and words != 48) return error.BadWords;
        // own copies: the checkpoint's mappings die with it
        var c: Case = .{
            .label = undefined,
            .label_len = 0,
            .trellis = try gpa.dupe(u8, trellis.bytes),
            .suh = try gpa.dupe(u8, suh.bytes),
            .svh = try gpa.dupe(u8, svh.bytes),
            .halves = @intCast(words / 8),
            .k = @intCast(trellis.dim(0) * 16),
            .n = @intCast(trellis.dim(1) * 16),
        };
        const printed = try std.fmt.bufPrint(&c.label, "L3 expert {d} gate_proj {s}", .{ e, if (slot == 0) @as([]const u8, "K4") else "K3" });
        c.label_len = printed.len;
        try cases.append(gpa, c);
    }
    return cases;
}

/// An aligned copy of a tensor's bytes as `T`s.
fn asSlice(comptime T: type, a: std.mem.Allocator, bytes: []align(1) const u8, count: usize) ![]T {
    const out = try a.alloc(T, count);
    errdefer a.free(out);
    const dest = std.mem.sliceAsBytes(out);
    @memcpy(dest[0..@min(dest.len, bytes.len)], bytes[0..@min(dest.len, bytes.len)]);
    return out;
}

pub fn run(gpu: Gpu, dir: []const u8) !void {
    var cases = try loadCases(gpu.gpa, gpu.io, dir);
    defer {
        for (cases.items) |*c| {
            gpu.gpa.free(c.trellis);
            gpu.gpa.free(c.suh);
            gpu.gpa.free(c.svh);
        }
        cases.deinit(gpu.gpa);
    }

    var ex = try cuda.exl3.Kernels.load(gpu.d);
    defer ex.deinit();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();

    for (cases.items) |*case| {
        try tileDecode(gpu, stream, &ex, case);
        try stages(gpu, stream, &ex, case);
        try projection(gpu, stream, &ex, case, .qlinear_decode, rows_decode);
        try projection(gpu, stream, &ex, case, .qlinear_prompt, rows_prompt);
    }
}

/// The device tile decode's 256 values against tileStates + codebookValue, three tiles.
fn tileDecode(gpu: Gpu, stream: cuda.Stream, ex: *const cuda.exl3.Kernels, case: *const Case) !void {
    const a = gpu.gpa;
    const words: usize = @as(usize, case.halves) * 8;
    const tiles_k: usize = case.k / 16;
    var dt = try cuda.DeviceBuffer.fromHost(gpu.d, case.trellis);
    defer dt.free();
    var dout = try cuda.DeviceBuffer.alloc(gpu.d, 256 * 2);
    defer dout.free();

    const w: fmt.Width = .{ .halves = @intCast(case.halves) };
    var worst_all: f32 = 0;
    for ([_]usize{ 0, (tiles_k / 2) * case.n / 16, (tiles_k - 1) * case.n / 16 }) |tile_base| {
        const offset = tile_base * words * 2;
        try ex.verifyTile(stream, dt.ptr + offset, case.halves, dout.ptr);
        try stream.synchronize();
        const bytes = try check.download(gpu, dout);
        defer a.free(bytes);

        const tile = try asSlice(i16, a, case.trellis[offset .. offset + words * 2], words);
        defer a.free(tile);
        var states: [256]u32 = undefined;
        fmt.tileStates(w, tile, &states);
        var worst: f32 = 0;
        for (0..256) |p| {
            const h: u16 = std.mem.readInt(u16, bytes[p * 2 ..][0..2], .little);
            const dev: f32 = @floatCast(fromBf16(h));
            const spec: f32 = @floatCast(fmt.codebookValue(.mcg, @intCast(states[p])));
            worst = @max(worst, @abs(dev - spec));
        }
        worst_all = @max(worst_all, worst);
    }
    if (worst_all > 2e-2) {
        std.debug.print("FAIL exl3 tile {s}: worst |device-spec| = {d:.5}\n", .{ case.name(), worst_all });
        return error.TileDecode;
    }
    check.pass("exl3 tile decode {s}: worst |device-spec| {d:.5} over three tiles", .{ case.name(), worst_all });
}

/// Every stage of the direct path against its host reference, at rows_decode.
fn stages(gpu: Gpu, stream: cuda.Stream, ex: *const cuda.exl3.Kernels, case: *const Case) !void {
    const a = gpu.gpa;
    const k: usize = case.k;
    const n: usize = case.n;
    const rows = rows_decode;
    const halves: u32 = case.halves;

    var dt = try cuda.DeviceBuffer.fromHost(gpu.d, case.trellis);
    defer dt.free();
    var dsu = try cuda.DeviceBuffer.fromHost(gpu.d, case.suh);
    defer dsu.free();
    var dsv = try cuda.DeviceBuffer.fromHost(gpu.d, case.svh);
    defer dsv.free();
    const w: cuda.exl3.Weight = .{ .trellis = dt.ptr, .suh = dsu.ptr, .svh = dsv.ptr, .halves = halves, .k = @intCast(k), .n = @intCast(n) };

    // x through bf16
    const x = try a.alloc(u16, rows * k);
    defer a.free(x);
    for (x) |*v| v.* = bf16Bits(rand());

    // host stage references
    const suh = try asSlice(f16, a, case.suh, k);
    defer a.free(suh);
    const svh = try asSlice(f16, a, case.svh, n);
    defer a.free(svh);
    const xrot = try a.alloc(f64, rows * k);
    defer a.free(xrot);
    for (0..rows) |r| {
        for (0..k) |j| xrot[r * k + j] = fromBf16(x[r * k + j]) * @as(f64, suh[j]);
    }
    fmt.rotate(xrot, rows, k, 1);

    const trellis = try asSlice(i16, a, case.trellis, (k / 16) * (n / 16) * halves * 8);
    defer a.free(trellis);
    const wq = try a.alloc(f16, k * n);
    defer a.free(wq);
    fmt.unpack(.{ .halves = @intCast(halves) }, .mcg, trellis, k / 16, n / 16, wq);
    const ypre_ref = try a.alloc(f64, rows * n);
    defer a.free(ypre_ref);
    {
        var r: usize = 0;
        while (r < rows) : (r += 1) {
            var c: usize = 0;
            while (c < n) : (c += 1) {
                var acc: f64 = 0;
                var j: usize = 0;
                while (j < k) : (j += 1) acc += xrot[r * k + j] * @as(f64, wq[j * n + c]);
                ypre_ref[r * n + c] = acc;
            }
        }
    }

    const final_ref = try reference(a, case);
    defer a.free(final_ref);

    // device, stage by stage
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dy = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 2);
    defer dy.free();

    try ex.preRot(stream, dx.ptr, w.suh, ex.scratch.ptr, rows, k);
    try stream.synchronize();
    try against("pre-rot", gpu, ex.scratch, xrot, 2e-2);

    try ex.matmul(stream, ex.scratch.ptr, w, dy.ptr, rows);
    try stream.synchronize();
    try against("y_pre", gpu, dy, ypre_ref, 5e-2);

    try ex.postRot(stream, dy.ptr, w.svh, rows, n);
    try stream.synchronize();
    try againstFinal("final", gpu, dy, final_ref, x, k, n);
}

/// max |device - ref| over the bf16 pairs, FAIL under tol.
fn against(what: []const u8, gpu: Gpu, buf: cuda.DeviceBuffer, ref: []const f64, tol: f64) !void {
    const bytes = try check.download(gpu, buf);
    defer gpu.gpa.free(bytes);
    var worst: f64 = 0;
    var wi: usize = 0;
    const elems = @min(@divTrunc(bytes.len, 2), ref.len);
    for (0..elems) |i| {
        const h: u16 = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
        const d = @abs(fromBf16(h) - ref[i]);
        if (d > worst) {
            worst = d;
            wi = i;
        }
    }
    if (worst > tol) {
        const h: u16 = std.mem.readInt(u16, bytes[wi * 2 ..][0..2], .little);
        std.debug.print("FAIL exl3 {s}: worst at {d}: dev {d:.4} ref {d:.4} (tol {d:.4})\n", .{ what, wi, fromBf16(h), ref[wi], tol });
        return error.Stage;
    }
    check.pass("exl3 {s}: worst {d:.5}", .{ what, worst });
}

/// The post-rotation's result against the f64 dequantize reference.
fn againstFinal(what: []const u8, gpu: Gpu, buf: cuda.DeviceBuffer, ref: []const f64, x: []const u16, k: usize, n: usize) !void {
    const bytes = try check.download(gpu, buf);
    defer gpu.gpa.free(bytes);
    var worst: f64 = 0;
    var sum2: f64 = 0;
    var r: usize = 0;
    while (r < rows_decode) : (r += 1) {
        var c: usize = 0;
        while (c < n) : (c += 1) {
            const h: u16 = std.mem.readInt(u16, bytes[(r * n + c) * 2 ..][0..2], .little);
            var acc: f64 = 0;
            var j: usize = 0;
            while (j < k) : (j += 1) acc += fromBf16(x[r * k + j]) * ref[j * n + c];
            const g = fromBf16(h);
            worst = @max(worst, @abs(g - acc));
            sum2 += acc * acc;
        }
    }
    const rms = @sqrt(sum2 / @as(f64, @floatFromInt(rows_decode * n)));
    if (worst > 0.05 * rms + 1e-2) {
        std.debug.print("FAIL exl3 {s}: worst {d:.4} rms {d:.4}\n", .{ what, worst, rms });
        return error.Final;
    }
    check.pass("exl3 {s}: worst {d:.4}, rms {d:.3}", .{ what, worst, rms });
}

const Mode = enum { qlinear_decode, qlinear_prompt };

fn modeName(mode: Mode) []const u8 {
    return switch (mode) {
        .qlinear_decode => "qlinear decode",
        .qlinear_prompt => "qlinear prompt",
    };
}

/// The whole projection through qlinear against the f64 reference.
fn projection(gpu: Gpu, stream: cuda.Stream, ex: *const cuda.exl3.Kernels, case: *const Case, mode: Mode, rows: usize) !void {
    const a = gpu.gpa;
    const k: usize = case.k;
    const n: usize = case.n;

    var dt = try cuda.DeviceBuffer.fromHost(gpu.d, case.trellis);
    defer dt.free();
    var dsu = try cuda.DeviceBuffer.fromHost(gpu.d, case.suh);
    defer dsu.free();
    var dsv = try cuda.DeviceBuffer.fromHost(gpu.d, case.svh);
    defer dsv.free();
    const w: cuda.exl3.Weight = .{ .trellis = dt.ptr, .suh = dsu.ptr, .svh = dsv.ptr, .halves = case.halves, .k = @intCast(k), .n = @intCast(n) };

    const ref = try reference(a, case);
    defer a.free(ref);

    const x = try a.alloc(u16, rows * k);
    defer a.free(x);
    for (x) |*v| v.* = bf16Bits(rand());
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dy = try cuda.DeviceBuffer.alloc(gpu.d, rows * n * 2);
    defer dy.free();

    const lin: cuda.qlinear.Linear = .{ .exl3 = ex };
    const in: cuda.qlinear.Rows = .{ .x = dx.ptr, .ldx = k };
    const qw: cuda.qlinear.Weight = .{ .exl3 = w };
    if (mode == .qlinear_decode) try lin.decode(stream, qw, in, dy.ptr, rows) else try lin.prompt(stream, qw, in, dy.ptr, rows);
    try stream.synchronize();

    const bytes = try check.download(gpu, dy);
    defer a.free(bytes);
    var worst: f64 = 0;
    var sum2: f64 = 0;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        var c: usize = 0;
        while (c < n) : (c += 1) {
            const h: u16 = std.mem.readInt(u16, bytes[(r * n + c) * 2 ..][0..2], .little);
            var acc: f64 = 0;
            var j: usize = 0;
            while (j < k) : (j += 1) acc += fromBf16(x[r * k + j]) * ref[j * n + c];
            const g = fromBf16(h);
            worst = @max(worst, @abs(g - acc));
            sum2 += acc * acc;
        }
    }
    const rms = @sqrt(sum2 / @as(f64, @floatFromInt(rows * n)));
    if (worst > 0.05 * rms + 1e-2) {
        std.debug.print("FAIL exl3 {s} {s} rows {d}: worst {d:.4} rms {d:.4}\n", .{ case.name(), modeName(mode), rows, worst, rms });
        return error.Projection;
    }
    check.pass("exl3 {s} {s} rows {d}: worst {d:.4}, rms {d:.3}", .{ case.name(), modeName(mode), rows, worst, rms });
}

/// W'[k, n] f64 from the scalar dequantize.
fn reference(a: std.mem.Allocator, case: *const Case) ![]f64 {
    const k: usize = case.k;
    const n: usize = case.n;
    const scratch = try a.alloc(f64, 2 * k * n);
    defer a.free(scratch);
    const out = try a.alloc(f64, k * n);
    errdefer a.free(out);
    const trellis = try asSlice(i16, a, case.trellis, (k / 16) * (n / 16) * case.halves * 8);
    defer a.free(trellis);
    const suh = try asSlice(f16, a, case.suh, k);
    defer a.free(suh);
    const svh = try asSlice(f16, a, case.svh, n);
    defer a.free(svh);
    fmt.dequantize(.{ .halves = @intCast(case.halves) }, .mcg, trellis, k / 16, n / 16, suh, svh, out, scratch);
    return out;
}
