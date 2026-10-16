//! GLM-5.3-Flash's MLA on the 512-wide latent cache (DSA layers): the Python family's latent on BF16, row by row.

const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");

/// latent.CHUNK, KT, HB, HB_WIDE, PROMPT_ROWS, PROMPT_RB.
pub const chunk = 512;
pub const key_tile = 32;
pub const heads_tile = 16;
pub const heads_tile_wide = 32;
pub const prompt_rows = 64;
pub const prompt_rb = 64;

/// latent.head_block: a rank's 32 heads a program in prompt chunks (each key read once), 16 in decode windows.
pub fn headBlock(rows: usize) usize {
    return if (rows >= 64) heads_tile_wide else heads_tile;
}

/// latent.chunks_for: 512-key chunks to cover `length` keys.
pub fn chunksFor(length: usize) usize {
    return (length + chunk - 1) / chunk;
}

fn cdiv(a: usize, b: usize) u32 {
    return @intCast((a + b - 1) / b);
}

/// latent.absorb_q (BF16 kv_b): out [rows][heads][lw] bf16 = q [rows][heads][d] . wk [heads][d][lw], fp32 sums.
pub fn absorbQ(set: glue.Set, stream: cuda.Stream, q: u64, wk: u64, out: u64, rows: usize, heads: usize, d: usize, lw: usize) !void {
    if (d != 256 or lw != 512) return error.LatentShape;
    try rowBlocks(set.glue.absorb_q, stream, q, wk, out, rows, heads, lw / 32);
}

/// The absorb and expand launches: rows in one block in decode windows, prompt_rb a block in prompt chunks.
fn rowBlocks(f: cuda.Function, stream: cuda.Stream, in: u64, w: u64, out: u64, rows: usize, heads: usize, cols: usize) !void {
    const rb = if (rows >= prompt_rows) prompt_rb else rows;
    var a: cuda.Args = .{};
    inline for (.{ in, w, out, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(rb)) }) |v| a.add(v);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(heads), .y = @intCast(cols), .z = cdiv(rows, rb) }, .block = .{ .x = 128 } }, stream, &a);
}

/// latent.expand_v (BF16 kv_b): out [rows][heads][dv] bf16 = o_lat [rows][heads][lw] . wv [heads][dv][lw].
pub fn expandV(set: glue.Set, stream: cuda.Stream, ol: u64, wv: u64, out: u64, rows: usize, heads: usize, dv: usize, lw: usize) !void {
    if (dv != 256 or lw != 512) return error.LatentShape;
    try rowBlocks(set.glue.expand_v, stream, ol, wv, out, rows, heads, dv / 16);
}

/// latent.latent_write: rows of `lat` (stride `lat_stride`) into slots pos .. pos + rows - 1 (pos an i32 on device).
pub fn latentWrite(set: glue.Set, stream: cuda.Stream, lat: u64, lat_stride: usize, cache: u64, pos: u64, rows: usize, lw: usize) !void {
    var a: cuda.Args = .{};
    inline for (.{ lat, @as(i64, @intCast(lat_stride)), cache, pos, @as(i64, @intCast(lw)) }) |v| a.add(v);
    if (set.glue.lat_write_nvfp4) |fnv| {
        var c: cuda.Args = .{};
        c.add(lat);
        c.add(@as(i64, @intCast(lat_stride)));
        c.add(cache);
        c.add(pos);
        c.add(@as(i64, @intCast(lw)));
        try cuda.launch.launch(fnv, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 32 } }, stream, &c);
        return;
    }
    if (set.glue.lat_write_fp8) |f8| {
        var b: cuda.Args = .{};
        b.add(lat);
        b.add(@as(i64, @intCast(lat_stride)));
        b.add(cache);
        b.add(pos);
        b.add(@as(i64, @intCast(lw)));
        try cuda.launch.launch(f8, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &b);
        return;
    }
    try cuda.launch.launch(set.glue.lat_write, .{ .grid = .{ .x = @intCast(rows) }, .block = .{ .x = 128 } }, stream, &a);
}

/// The chunk partials of one attention call: po fp32 [nch][rows][heads][lw], pm and pl fp32 [nch][rows][heads].
pub const Partials = struct { po: u64, pm: u64, pl: u64 };

/// latent.attention: causal qa over the cache in `nch` 512-key chunks into out bf16; `dummy`: the merge's count i32.
var lat_scratch: ?cuda.DeviceBuffer = null;
var scratch_units: usize = 0;

/// Dequantize the fp8 latent cache's [start, start + count) rows into the bf16 scratch,
/// returning the scratch pointer for the original bf16 attention kernels.
var lat_driver: ?*const cuda.Driver = null;
pub fn setDriver(d: *const cuda.Driver) void { lat_driver = d; }

fn dequantFor(set: glue.Set, stream: cuda.Stream, cache_fp8: u64, start: usize, count: usize, lw: usize) !u64 {
    if (set.glue.lat_dequant_nvfp4) |dnv| {
        if (lat_scratch == null or scratch_units < count * lw) {
            if (lat_scratch) |*b| b.free();
            lat_scratch = try cuda.DeviceBuffer.alloc(lat_driver.?, count * lw * 2);
            scratch_units = count * lw;
        }
        var c: cuda.Args = .{};
        c.add(cache_fp8);
        c.add(lat_scratch.?.ptr);
        c.add(@as(i64, @intCast(start)));
        c.add(@as(i64, @intCast(count)));
        c.add(@as(i64, @intCast(lw)));
        // one block per row (each block dequantizes 512 values from 288 bytes)
        try cuda.launch.launch(dnv, .{ .grid = .{ .x = @intCast((count + 63) / 64) }, .block = .{ .x = 64 } }, stream, &c);
        return lat_scratch.?.ptr;
    }
    const dq = set.glue.lat_dequant orelse return cache_fp8; // bf16: no dequant needed
    if (lat_scratch == null or scratch_units < count * lw) {
        if (lat_scratch) |*b| b.free();
        lat_scratch = try cuda.DeviceBuffer.alloc(lat_driver.?, count * lw * 2);
        scratch_units = count * lw;
    }
    var a: cuda.Args = .{};
    a.add(cache_fp8);
    a.add(lat_scratch.?.ptr);
    a.add(@as(i64, @intCast(start)));
    a.add(@as(i64, @intCast(count)));
    a.add(@as(i64, @intCast(lw)));
    try cuda.launch.launch(dq, .{ .grid = .{ .x = @intCast((count * lw + 255) / 256) }, .block = .{ .x = 256 } }, stream, &a);
    return lat_scratch.?.ptr;
}


/// Launch an NVFP4 chunks kernel (the direct e2m1 attention, no dequant).
fn launchNvfp4(f: cuda.Function, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ qa, cache, pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    const shared = if (hb == heads_tile_wide) glue.chunksSmemNvfp4(heads_tile_wide) else glue.chunksSmemNvfp4(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
}

fn launchBf16(f: cuda.Function, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ qa, cache, pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    const shared = if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
}

pub fn attention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, p: Partials, out: u64, dummy: u64, rows: usize, heads: usize, lw: usize, nch: usize, scale: f32, hb: usize) !void {
    // fp8: dequantize the dense range into a bf16 scratch the original kernel reads
    // direct NVFP4 attention: no dequant, the chunks kernel reads e2m1 + scales inline
    if (rows <= 8) { // a solo decode window: one 32-row tile a block instead of a 16-tile serial chain
        const nch32 = nch * 16; // over-covers the last chunk; empty slices merge as -inf
        if (set.glue.chunks16_nvfp4_s32) |f| {
            try launchNvfp4(f, stream, qa, cache, pos, 0, 0, p, rows, heads, lw, nch32, 0, scale, heads_tile, false);
        } else if (set.glue.chunks_16_s32) |f| {
            try launchBf16(f, stream, qa, cache, pos, 0, 0, p, rows, heads, lw, nch32, 0, scale, heads_tile, false);
        } else return error.NoSlicedKernel;
        try merge(set, stream, p, out, dummy, rows, heads, lw, nch32, false);
        return;
    }
    if (set.glue.chunks16_nvfp4) |f16v| {
        if (hb == heads_tile_wide and set.glue.chunks32_nvfp4 != null) {
            try launchNvfp4(set.glue.chunks32_nvfp4.?, stream, qa, cache, pos, 0, 0, p, rows, heads, lw, nch, 0, scale, hb, false);
        } else {
            try launchNvfp4(f16v, stream, qa, cache, pos, 0, 0, p, rows, heads, lw, nch, 0, scale, heads_tile, false);
        }
        try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
        return;
    }
    var cache_bf16 = cache;
    if (set.glue.lat_dequant != null or set.glue.lat_dequant_nvfp4 != null) cache_bf16 = try dequantFor(set, stream, cache, 0, nch * 128 + rows, lw);
    try chunks(set, stream, qa, cache_bf16, pos, 0, 0, p, rows, heads, lw, nch, 0, scale, hb, false);
    try merge(set, stream, p, out, dummy, rows, heads, lw, nch, false);
}

/// latent._dense_chunks (sparse false: keys through pos + r) and _sparse_chunks (row r's `counts` tokens of `tokens`).
fn chunks(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, pos: u64, tokens: u64, counts: u64, p: Partials, rows: usize, heads: usize, lw: usize, nch: usize, w: usize, scale: f32, hb: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    const f = switch (hb) {
        heads_tile => set.glue.chunks_16,
        heads_tile_wide => set.glue.chunks_32,
        else => return error.HeadBlock,
    };
    var a: cuda.Args = .{};
    inline for (.{ qa, cache, pos, tokens, counts, p.po, p.pm, p.pl, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(i64, @intCast(w)), scale, @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    const shared = if (hb == heads_tile_wide) glue.chunksSmem(heads_tile_wide) else glue.chunksSmem(heads_tile);
    try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(rows), .y = cdiv(heads, hb), .z = @intCast(nch) }, .block = .{ .x = 256 }, .shared = shared }, stream, &a);
}

/// latent.sparse_attention: rows with counts > 0 over their selected tokens (i32 [rows][w]); others left alone.
pub fn sparseAttention(set: glue.Set, stream: cuda.Stream, qa: u64, cache: u64, tokens: u64, counts: u64, w: usize, p: Partials, out: u64, rows: usize, heads: usize, lw: usize, scale: f32) !void {
    const nch = chunksFor(w);
    // fp8: dequantize the whole cache range the sparse tokens may select from
    // direct NVFP4 attention (sparse)
    if (rows <= 8) { // a solo decode window: one 32-row tile a block
        const nch32 = (w + 31) / 32;
        if (set.glue.chunks16_nvfp4_s32) |f| {
            try launchNvfp4(f, stream, qa, cache, 0, tokens, counts, p, rows, heads, lw, nch32, w, scale, heads_tile, true);
        } else if (set.glue.chunks_16_s32) |f| {
            try launchBf16(f, stream, qa, cache, 0, tokens, counts, p, rows, heads, lw, nch32, w, scale, heads_tile, true);
        } else return error.NoSlicedKernel;
        try merge(set, stream, p, out, counts, rows, heads, lw, nch32, true);
        return;
    }
    if (set.glue.chunks16_nvfp4) |f16v| {
        if (headBlock(rows) == heads_tile_wide and set.glue.chunks32_nvfp4 != null) {
            try launchNvfp4(set.glue.chunks32_nvfp4.?, stream, qa, cache, 0, tokens, counts, p, rows, heads, lw, nch, w, scale, headBlock(rows), true);
        } else {
            try launchNvfp4(f16v, stream, qa, cache, 0, tokens, counts, p, rows, heads, lw, nch, w, scale, heads_tile, true);
        }
        try merge(set, stream, p, out, counts, rows, heads, lw, nch, true);
        return;
    }
    var cache_bf16 = cache;
    if (set.glue.lat_dequant != null or set.glue.lat_dequant_nvfp4 != null) cache_bf16 = try dequantFor(set, stream, cache, 0, w, lw);
    try chunks(set, stream, qa, cache_bf16, 0, tokens, counts, p, rows, heads, lw, nch, w, scale, headBlock(rows), true);
    try merge(set, stream, p, out, counts, rows, heads, lw, nch, true);
}

fn merge(set: glue.Set, stream: cuda.Stream, p: Partials, out: u64, counts: u64, rows: usize, heads: usize, lw: usize, nch: usize, sparse: bool) !void {
    if (lw != 512) return error.LatentShape;
    var a: cuda.Args = .{};
    inline for (.{ p.po, p.pm, p.pl, out, counts, @as(i64, @intCast(rows)), @as(i64, @intCast(heads)), @as(c_int, @intCast(nch)), @as(c_int, @intFromBool(sparse)) }) |v| a.add(v);
    try cuda.launch.launch(set.glue.merge, .{ .grid = .{ .x = @intCast(rows), .y = @intCast(heads) }, .block = .{ .x = 128 } }, stream, &a);
}

test "latent.py's tiling rules" {
    try std.testing.expectEqual(@as(usize, 16), headBlock(8));
    try std.testing.expectEqual(@as(usize, 32), headBlock(64));
    try std.testing.expectEqual(@as(usize, 3), chunksFor(1025));
    try std.testing.expectEqual(@as(usize, 1), chunksFor(512));
}
