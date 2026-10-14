//! The vision serving path's host preprocessing: a raw RGB(A) image becomes
//! the tower's patch rows. The pipeline: bilinear resize to a 28-aligned
//! canvas (patch x merge whole), CLIP mean/std normalize, the checkpoint's
//! patchify layout (merge-block-major rows, (my, mx, c, t, py, px) values,
//! still image = the frame doubled) - matching vision_test.zig's proven
//! host patchify bit for bit at 448x448.

const std = @import("std");

pub const MEAN = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
pub const STD = [3]f32{ 0.26862954, 0.26130258, 0.27577711 };

/// The canvas: 28-aligned (whole 2x2 merge blocks), aspect-kept, within the
/// token cap (tokens = canvas patches / 4).
pub fn canvas(w: usize, h: usize, max_tokens: usize) struct { cw: usize, ch: usize } {
    const ar: f32 = if (w == 0 or h == 0) 1.0 else @as(f32, @floatFromInt(w)) / @as(f32, @floatFromInt(h));
    var ch: usize = 448;
    // scale the height until (ch/14)*(cw/14)/4 <= max_tokens
            var cw_i: usize = 0;
        cw_i = @intFromFloat(@as(f32, @floatFromInt(ch)) * ar);
        while ((ch / 14) * (cw_i / 14) / 4 > max_tokens and ch > 28) {
            ch -= 28;
            cw_i = @intFromFloat(@as(f32, @floatFromInt(ch)) * ar);
        }
    var ch_a = ch / 28 * 28;
    if (ch_a < 28) ch_a = 28;
    const cw_f2: f32 = @as(f32, @floatFromInt(ch_a)) * ar;
    const cw_a: usize = @max(@as(usize, 28), (@as(usize, @intFromFloat(cw_f2)) / 28 * 28));
    return .{ .cw = cw_a, .ch = ch_a };
}

/// Bilinear resize of channel c of src ([sh x sw x ch] u8) into dst
/// ([dh x dw] f32).
fn resizePlane(dst: []f32, dw: usize, dh: usize, src: []const u8, sw: usize, sh: usize, ch: usize, c: usize) void {
    for (0..dh) |y| {
        const fy = (@as(f32, @floatFromInt(y)) + 0.5) * @as(f32, @floatFromInt(sh)) / @as(f32, @floatFromInt(dh)) - 0.5;
        const y0 = @min(sh - 1, @as(usize, @intFromFloat(@max(0.0, @floor(fy)))));
        const y1 = @min(sh - 1, y0 + 1);
        const wy = fy - @floor(fy);
        for (0..dw) |x| {
            const fx = (@as(f32, @floatFromInt(x)) + 0.5) * @as(f32, @floatFromInt(sw)) / @as(f32, @floatFromInt(dw)) - 0.5;
            const x0 = @min(sw - 1, @as(usize, @intFromFloat(@max(0.0, @floor(fx)))));
            const x1 = @min(sw - 1, x0 + 1);
            const wx = fx - @floor(fx);
            const p00: f32 = @floatFromInt(src[(y0 * sw + x0) * ch + c]);
            const p01: f32 = @floatFromInt(src[(y0 * sw + x1) * ch + c]);
            const p10: f32 = @floatFromInt(src[(y1 * sw + x0) * ch + c]);
            const p11: f32 = @floatFromInt(src[(y1 * sw + x1) * ch + c]);
            dst[y * dw + x] = p00 * (1 - wx) * (1 - wy) + p01 * wx * (1 - wy) + p10 * (1 - wx) * wy + p11 * wx * wy;
        }
    }
}

/// rgb/rgba bytes [h x w x ch] -> bf16 patch rows [S, 1176] (gpa-owned).
pub fn patchify(gpa: std.mem.Allocator, rgb: []const u8, w: usize, h: usize, ch: usize, max_tokens: usize) !struct { bytes: []u8, gh: usize, gw: usize } {
    const cv = canvas(w, h, max_tokens);
    const gh = cv.ch / 14;
    const gw = cv.cw / 14;
    const S = gh * gw;
    const out = try gpa.alloc(u8, S * 1176 * 2);
    errdefer gpa.free(out);
    const plane = try gpa.alloc(f32, cv.cw * cv.ch);
    defer gpa.free(plane);
    for (0..3) |c| {
        resizePlane(plane, cv.cw, cv.ch, rgb, w, h, ch, c);
        for (plane) |*p| p.* = (p.* - MEAN[c] * 255.0) / (STD[c] * 255.0);
        for (0..S) |r| {
            const oy = r / 64;
            var rem = r % 64;
            const ox = rem / 4;
            rem = rem % 4;
            const my = rem / 2;
            const mx = rem % 2;
            const py0 = (oy * 2 + my) * 14;
            const px0 = (ox * 2 + mx) * 14;
            const at0 = c * (2 * 14 * 14);
            for (0..2) |t| for (0..14) |py| for (0..14) |px| {
                const v = plane[(py0 + py) * cv.cw + (px0 + px)];
                const u: u16 = @bitCast(@as(u16, @truncate(@as(u32, @bitCast(v)) >> 16)));
                const off = r * 1176 * 2 + (at0 + t * (14 * 14) + py * 14 + px) * 2;
                out[off] = @truncate(u);
                out[off + 1] = @truncate(u >> 8);
            };
        }
    }
    return .{ .bytes = out, .gh = gh, .gw = gw };
}

/// base64 (standard alphabet) decode into gpa memory.
pub fn b64decode(gpa: std.mem.Allocator, text: []const u8) ![]u8 {
    const dec = std.base64.standard.Decoder;
    const n = try dec.calcSizeForSlice(text);
    const out = try gpa.alloc(u8, n);
    errdefer gpa.free(out);
    try dec.decode(out, text);
    return out;
}
