//! A TP rank's GLM-5.3-Flash weights from an EXL3 checkpoint: matrices cut at 128s, KDA and MoE as EXL3 has them.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const shard = @import("cuda_shard.zig");
const W = @import("cuda_weights.zig");
const quant = @import("cuda_quant.zig");
const b16 = @import("cuda_b16.zig");

/// EXL3 matrix `name`'s share: inputs [klo, khi) (khi 0: all), `spans`' columns; tiles to `tiles`, suh+svh `scales`.
pub fn gather(ck: *core.Checkpoint, gpa: std.mem.Allocator, name: []const u8, klo: usize, khi: usize, spans: []const shard.Span, tiles: *std.ArrayList(u8), scales: *std.ArrayList(u8)) !cuda.exl3_linear.Matrix {
    var nb: [192]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.trellis", .{name}));
    const suh = try ck.get(try std.fmt.bufPrint(&nb, "{s}.suh", .{name}));
    const svh = try ck.get(try std.fmt.bufPrint(&nb, "{s}.svh", .{name}));
    const book: cuda.exl3_linear.Codebook = if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mul1", .{name}))) .mul1 else if (ck.has(try std.fmt.bufPrint(&nb, "{s}.mcg", .{name}))) .mcg else .inst3;
    if (t.dtype != .i16 or t.rank != 3 or t.dim(2) % 16 != 0) return error.UnexpectedTensor;
    const tn = t.dim(1);
    const words = t.dim(2);
    const k_all = t.dim(0) * 16;
    const k_hi = if (khi == 0) k_all else khi;
    if (klo % 128 != 0 or k_hi % 128 != 0 or k_hi > k_all) return error.Exl3RowCut;
    var n: usize = 0;
    for (spans) |c| {
        if (c.lo % 128 != 0 or c.hi % 128 != 0 or c.hi > tn * 16) return error.Exl3ColumnCut;
        n += c.len();
    }
    // trellis tiles: each 16-row group of the rank's inputs, its spans' 16-column tiles in order
    for (klo / 16..k_hi / 16) |r| for (spans) |c| {
        try tiles.appendSlice(gpa, t.bytes[(r * tn + c.lo / 16) * words * 2 ..][0 .. c.len() / 16 * words * 2]);
    };
    try scales.appendSlice(gpa, suh.bytes[klo * 2 .. k_hi * 2]);
    for (spans) |c| try scales.appendSlice(gpa, svh.bytes[c.lo * 2 .. c.hi * 2]);
    return .{ .trellis = 0, .suh = 0, .svh = 0, .k = k_hi - klo, .n = n, .bits = @intCast(words / 16), .codebook = book };
}

/// gather() of one matrix, uploaded as its trellis, suh and svh.
pub fn matrix(L: *W.Loader, ck: *core.Checkpoint, name: []const u8, klo: usize, khi: usize, spans: []const shard.Span) !cuda.qlinear.Weight {
    var scales: std.ArrayList(u8) = .empty;
    defer scales.deinit(L.gpa);
    var m = try gather(ck, L.gpa, name, klo, khi, spans, &L.host, &scales);
    m.trellis = try L.put("{s}.trellis", .{name});
    try L.host.appendSlice(L.gpa, scales.items[0 .. m.k * 2]);
    m.suh = try L.put("{s}.suh", .{name});
    try L.host.appendSlice(L.gpa, scales.items[m.k * 2 ..]);
    m.svh = try L.put("{s}.svh", .{name});
    return .{ .exl3 = m };
}

/// matrix() on the model's tensors, `fmt` naming the prefix of layer `i`.
pub fn layerMatrix(L: *W.Loader, comptime fmt: []const u8, i: u32, klo: usize, khi: usize, spans: []const shard.Span) !cuda.qlinear.Weight {
    var nb: [192]u8 = undefined;
    return matrix(L, L.ck, try std.fmt.bufPrint(&nb, W.prefix ++ fmt, .{i}), klo, khi, spans);
}

/// An EXL3 matrix's every column.
pub fn whole(L: *W.Loader, comptime fmt: []const u8, i: u32) !cuda.qlinear.Weight {
    var nb: [192]u8 = undefined;
    const name = try std.fmt.bufPrint(&nb, W.prefix ++ fmt, .{i});
    var nb2: [192]u8 = undefined;
    const t = try L.ck.get(try std.fmt.bufPrint(&nb2, "{s}.trellis", .{name}));
    return matrix(L, L.ck, name, 0, 0, &.{.{ .lo = 0, .hi = t.dim(1) * 16 }});
}

/// Loader.kda on an EXL3 checkpoint: qkv_proj by heads, [f_a | g_a | b] BF16, conv1d's q, k and v rows, o by inputs.
pub fn kda(L: *W.Loader, i: u32) !@import("cuda_kda_block.zig").Layer {
    const c = &L.w.config;
    const h = L.w.share.kda;
    const dk = c.kda_dim;
    const all = c.kda_heads * dk;
    const lo = h.lo * dk;
    const hi = h.hi * dk;
    const qkv = try layerMatrix(L, "layers.{d}.self_attn.qkv_proj", i, 0, 0, &.{ .{ .lo = lo, .hi = hi }, .{ .lo = all + lo, .hi = all + hi }, .{ .lo = 2 * all + lo, .hi = 2 * all + hi } });
    inline for (.{ "f_a", "g_a" }) |x| try L.whole(try L.get("layers.{d}.self_attn." ++ x ++ "_proj.weight", .{i}), .bf16);
    try L.rows(try L.get("layers.{d}.self_attn.b_proj.weight", .{i}), h.lo, h.hi, c.kda_heads, .bf16);
    const proj = try L.put("layers.{d}.kda.proj", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.f_b_proj.weight", .{i}), lo, hi, all, .bf16);
    const fb = try L.put("layers.{d}.kda.fb", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.g_b_proj.weight", .{i}), lo, hi, all, .bf16);
    const gb = try L.put("layers.{d}.kda.gb", .{i});
    const conv = try L.get("layers.{d}.self_attn.conv1d.weight", .{i});
    for (0..3) |j| try L.rows(conv, j * all + lo, j * all + hi, 3 * all, .f32);
    const conv_w = try L.put("layers.{d}.kda.conv", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.A_log", .{i}), h.lo, h.hi, c.kda_heads, .f32);
    const a_log = try L.put("layers.{d}.kda.a_log", .{i});
    try L.rows(try L.get("layers.{d}.self_attn.dt_bias", .{i}), lo, hi, all, .f32);
    const dt = try L.put("layers.{d}.kda.dt_bias", .{i});
    try L.whole(try L.get("layers.{d}.self_attn.o_norm.weight", .{i}), .bf16);
    const norm = try L.put("layers.{d}.kda.norm", .{i});
    const o = try layerMatrix(L, "layers.{d}.self_attn.o_proj", i, lo, hi, &.{.{ .lo = 0, .hi = c.hidden }});
    return .{ .proj = proj, .qkv = qkv, .fb = fb, .gb = gb, .o = .{ .q = o }, .conv_w = conv_w, .a_log = a_log, .dt_bias = dt, .norm_w = norm, .eps = c.eps, .lower = c.lower_bound, .heads = h.len(), .dims = c.hidden };
}

/// Loader.moeLayer on EXL3: experts as exl3_experts stacks, gate and up by the rank's columns, down by its inputs.
/// Our hybrid checkpoint's MoE: EXL3 routed experts (the x3 gather) with a
/// BF16 shared expert (the nvfp4 branch's rows/cols loads) - the ablit 3.5bpw
/// layout (bf16 attention, bf16 shared, exl3 routed).

/// One BF16 matrix already staged in `rows` as [n][k] host bytes, GPU-quantized to the NVFP4
/// lane layout (the MTP layer's recipe; tf_glm_nvfp4_quantize + tf_pack_dense + packScales).
fn fp4Lane(L: *W.Loader, rows: []const u8, n: usize, k: usize, name: []const u8, scales_name: []const u8) !cuda.qlinear.Weight {
    const qk = L.tools.quant orelse return error.NoQuantizer;
    var wb = try L.scratch(n * k * 2);
    defer wb.free();
    try wb.upload(0, rows);
    var cb = try L.scratch(n * k / 2);
    defer cb.free();
    var sb = try L.scratch(n * k / 16);
    defer sb.free();
    var gb = try L.scratch(4);
    defer gb.free();
    try quant.nvfp4(qk, L.stream, wb.ptr, 1, n, k, cb.ptr, sb.ptr, gb.ptr);
    const npad = cuda.nvfp4.padded(n);
    const kg = k / 64;
    const total: u64 = @as(u64, npad / 64) * kg * 512;
    const words = try cuda.DeviceBuffer.alloc(L.d, total * 4);
    try L.w.buffers.append(L.gpa, words);
    try L.w.named.append(L.gpa, .{ .name = try L.gpa.dupe(u8, name), .ptr = words.ptr, .len = words.len });
    var args: cuda.Args = .{};
    args.add(cb.ptr);
    for ([_]usize{ n, k / 8, kg }) |v| args.add(@as(c_int, @intCast(v)));
    args.add(words.ptr);
    args.add(@as(c_longlong, @intCast(total)));
    try cuda.launch.launch(L.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, L.stream, &args);
    const scales = try L.gpa.alloc(u8, n * k / 16);
    defer L.gpa.free(scales);
    try sb.download(0, scales);
    const packed_scales = try L.gpa.alloc(u8, npad * k / 16);
    defer L.gpa.free(packed_scales);
    cuda.nvfp4.packScales(packed_scales, scales, n, k);
    const bs = try cuda.DeviceBuffer.fromHost(L.d, packed_scales);
    try L.w.buffers.append(L.gpa, bs);
    try L.w.named.append(L.gpa, .{ .name = try L.gpa.dupe(u8, scales_name), .ptr = bs.ptr, .len = bs.len });
    var g4: [4]u8 = undefined;
    try gb.download(0, g4[0..4]);
    const gs: f32 = @bitCast(std.mem.readInt(u32, g4[0..4], .little));
    return .{ .nvfp4 = cuda.nvfp4.weight(words.ptr, bs.ptr, gs, n, k) };
}


const bits_env: ?[]const u8 = std.c.getenv("TF_GLM_DENSE_BITS");

fn bf16Value(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

fn roundEven(x: f32) f32 {
    const f = @floor(x);
    const d = x - f;
    if (d > 0.5) return f + 1;
    if (d < 0.5) return f;
    return if (@mod(f, 2) == 0) f else f + 1;
}

/// Symmetric bits-bit quantization of bf16 rows [n, k] (ported from BobClawblaw/TensorFold
/// pr/glm-flash-cuda @ 3c6ed009 weights.zig: quantize + pack, symmetric groups of 64).
fn affineQuantize(bits: u8, src: []align(1) const u16, n: usize, k: usize, words: []u32, scales: []u16) void {
    const half: i32 = @as(i32, 1) << @intCast(bits - 1);
    const qmax: f32 = @floatFromInt(half - 1);
    const wpc: usize = bits / 2;
    const groups = k / 64;
    for (0..n) |r| {
        const row = src[r * k ..][0..k];
        for (0..groups) |g| {
            var amax: f32 = 0;
            for (row[g * 64 ..][0..64]) |h| amax = @max(amax, @abs(bf16Value(h)));
            const s16: f16 = @floatCast(amax / qmax);
            scales[r * groups + g] = @bitCast(s16);
            const sc: f32 = s16;
            for (0..4) |c| {
                var q: [16]u32 = undefined;
                for (0..16) |i| {
                    const v = bf16Value(row[g * 64 + c * 16 + i]);
                    const t: i32 = if (sc == 0) 0 else @intFromFloat(@max(-qmax, @min(qmax, roundEven(v / sc))));
                    q[i] = @intCast(t + half);
                }
                affinePack(bits, &q, words[((r * groups + g) * 4 + c) * wpc ..][0..wpc]);
            }
        }
    }
}

fn affinePack(bits: u8, q: *const [16]u32, out: []u32) void {
    @memset(out, 0);
    switch (bits) {
        4 => for (0..16) |i| {
            out[i / 8] |= q[i] << @intCast(4 * (i % 8));
        },
        8 => for (0..16) |i| {
            out[i / 4] |= q[i] << @intCast(8 * (i % 4));
        },
        else => {
            for (0..15) |i| out[i / 5] |= q[i] << @intCast(6 * (i % 5));
            for (0..3) |j| out[j] |= ((q[15] >> @intCast(2 * j)) & 3) << 30;
        },
    }
}

/// One bf16 [n][k] host-rows matrix quantized to affine bits and tiled into mma B-fragment order
/// (the glm_mm_tile{N} path; bits from TF_GLM_DENSE_BITS, default 0 = keep bf16).
const QuantPtrs = struct { w: u64, s: u64 };

fn quantizedLane(L: *W.Loader, rows_host: []const u8, n: usize, k: usize, bits: u8, name: []const u8, scales_name: []const u8) !?QuantPtrs {
    if (bits == 0 or bits != 4 and bits != 6 and bits != 8) return null;
    const src: []align(1) const u16 = @as([*]align(1) const u16, @ptrCast(rows_host.ptr))[0 .. rows_host.len / 2];
    const words = try L.gpa.alloc(u32, n * k * bits / 32);
    defer L.gpa.free(words);
    const scales = try L.gpa.alloc(u16, n * k / 64);
    defer L.gpa.free(scales);
    affineQuantize(bits, src, n, k, words, scales);
    // upload codes and scales, tile on the GPU
    var gm = try cuda.Module.load(L.d, cuda.kernels.glm_glue);
    defer gm.unload();
    const tiler = switch (bits) {
        4 => try gm.function("glm_mm_tile4"),
        8 => try gm.function("glm_mm_tile8"),
        else => try gm.function("glm_mm_tile6"),
    };
    var codes_buf = try cuda.DeviceBuffer.fromHost(L.d, std.mem.sliceAsBytes(words));
    defer codes_buf.free();
    var scales_buf = try cuda.DeviceBuffer.fromHost(L.d, std.mem.sliceAsBytes(scales));
    defer scales_buf.free();
    const tiled_words = try cuda.DeviceBuffer.alloc(L.d, words.len * 4);
    try L.w.buffers.append(L.gpa, tiled_words);
    const tiled_scales = try cuda.DeviceBuffer.alloc(L.d, scales.len * 2);
    try L.w.buffers.append(L.gpa, tiled_scales);
    const chunks = (n / 8) * (k / 64) * 32;
    var ta: cuda.Args = .{};
    ta.add(codes_buf.ptr);
    ta.add(scales_buf.ptr);
    ta.add(tiled_words.ptr);
    ta.add(tiled_scales.ptr);
    ta.add(@as(c_int, @intCast(n)));
    ta.add(@as(c_int, @intCast(k)));
    try cuda.launch.launch(tiler, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, L.stream, &ta);
    try L.stream.synchronize();
    try L.w.named.append(L.gpa, .{ .name = try L.gpa.dupe(u8, name), .ptr = tiled_words.ptr, .len = tiled_words.len });
    try L.w.named.append(L.gpa, .{ .name = try L.gpa.dupe(u8, scales_name), .ptr = tiled_scales.ptr, .len = tiled_scales.len });
    // build the Weight: the glm_mm family's "affine" format — codes + scales + a bits field the mm kernel reads
    return .{ .w = tiled_words.ptr, .s = tiled_scales.ptr };
}


pub fn moeHybrid(L: *W.Loader, i: u32) !@import("cuda_moe.zig").Layer {
    const c = &L.w.config;
    const d: usize = c.hidden;
    const cut = L.w.share.moe;
    try L.whole(try L.get("layers.{d}.mlp.gate.weight", .{i}), .bf16);
    const router = try L.put("layers.{d}.moe.router", .{i});
    try L.whole(try L.get("layers.{d}.mlp.gate.e_score_correction_bias", .{i}), .f32);
    const bias = try L.put("layers.{d}.moe.bias", .{i});
    var t_up: std.ArrayList(u8) = .empty;
    defer t_up.deinit(L.gpa);
    var h_up: std.ArrayList(u8) = .empty;
    defer h_up.deinit(L.gpa);
    var t_dn: std.ArrayList(u8) = .empty;
    defer t_dn.deinit(L.gpa);
    var h_dn: std.ArrayList(u8) = .empty;
    defer h_dn.deinit(L.gpa);
    var widths: cuda.exl3_experts.Widths = .{};
    var nb: [192]u8 = undefined;
    // the mma decode pass's per-expert flat scale tables (gate/up: suh hidden + svh span; down: suh span + svh hidden)
    var suh_g: std.ArrayList(u8) = .empty;
    defer suh_g.deinit(L.gpa);
    var suh_u: std.ArrayList(u8) = .empty;
    defer suh_u.deinit(L.gpa);
    var svh_g: std.ArrayList(u8) = .empty;
    defer svh_g.deinit(L.gpa);
    var svh_u: std.ArrayList(u8) = .empty;
    defer svh_u.deinit(L.gpa);
    var suh_d: std.ArrayList(u8) = .empty;
    defer suh_d.deinit(L.gpa);
    var svh_d: std.ArrayList(u8) = .empty;
    defer svh_d.deinit(L.gpa);
    var nb2: [192]u8 = undefined;
    // the mixed-width checkpoint: per-expert MatDesc tables (expert-major, gate then up / down alone)
    const MD = cuda.exl3_experts.MatDesc;
    const up_desc = try L.gpa.alloc(MD, c.experts * 2);
    defer L.gpa.free(up_desc);
    const dn_desc = try L.gpa.alloc(MD, c.experts);
    defer L.gpa.free(dn_desc);
    var up_t: usize = 0;
    var up_h: usize = 0;
    var dn_t: usize = 0;
    var dn_h: usize = 0;
    for (0..c.experts) |e| {
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, m| {
            const name = try std.fmt.bufPrint(&nb, W.prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, e, p });
            const q = if (m < 2) try gather(L.ck, L.gpa, name, 0, 0, &.{cut}, &t_up, &h_up) else try gather(L.ck, L.gpa, name, cut.lo, cut.hi, &.{.{ .lo = 0, .hi = d }}, &t_dn, &h_dn);
            { // the mma tables read the same tensors the gather read, in their flat per-expert order
                const suh_t = try L.ck.get(try std.fmt.bufPrint(&nb2, "{s}.suh", .{name}));
                const svh_t = try L.ck.get(try std.fmt.bufPrint(&nb2, "{s}.svh", .{name}));
                if (m < 2) {
                    const dst_s = if (m == 0) &suh_g else &suh_u;
                    const dst_v = if (m == 0) &svh_g else &svh_u;
                    try dst_s.appendSlice(L.gpa, suh_t.bytes[0..q.k * 2]);
                    try dst_v.appendSlice(L.gpa, svh_t.bytes[cut.lo * 2 .. cut.hi * 2]);
                } else {
                    try suh_d.appendSlice(L.gpa, suh_t.bytes[cut.lo * 2 .. cut.hi * 2]);
                    try svh_d.appendSlice(L.gpa, svh_t.bytes[0..q.n * 2]);
                }
            }
            if (m < 2) widths.up = q.bits else widths.down = q.bits;
            widths.codebook = q.codebook;
            const tbytes = cuda.exl3_linear.Matrix.trellisBytes(q.k, q.n, q.bits);
            if (m < 2) {
                up_desc[e * 2 + m] = .{ .trellis = up_t, .suh = up_h, .svh = up_h + q.k * 2, .k = @intCast(q.k), .n = @intCast(q.n), .tiles_n = @intCast(q.n / 16), .bits = q.bits };
                up_t += tbytes;
                up_h += (q.k + q.n) * 2;
            } else {
                dn_desc[e] = .{ .trellis = dn_t, .suh = dn_h, .svh = dn_h + q.k * 2, .k = @intCast(q.k), .n = @intCast(q.n), .tiles_n = @intCast(q.n / 16), .bits = q.bits };
                dn_t += tbytes;
                dn_h += (q.k + q.n) * 2;
            }
        }
    }
    var ptrs: [4]u64 = undefined;
    for (&ptrs, [_][]const u8{ t_up.items, t_dn.items, h_up.items, h_dn.items }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
        try L.host.appendSlice(L.gpa, bytes);
        p.* = try L.put("layers.{d}.moe.experts.{s}", .{ i, what });
    }
    // absolute device pointers into the four uploads, then the two desc tables
    for (up_desc) |*m| {
        m.trellis += ptrs[0];
        m.suh += ptrs[2];
        m.svh += ptrs[2];
    }
    for (dn_desc) |*m| {
        m.trellis += ptrs[1];
        m.suh += ptrs[3];
        m.svh += ptrs[3];
    }
    try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(up_desc));
    const mats_up = try L.put("layers.{d}.moe.experts.mats_up", .{i});
    try L.host.appendSlice(L.gpa, std.mem.sliceAsBytes(dn_desc));
    const mats_dn = try L.put("layers.{d}.moe.experts.mats_down", .{i});
    var mma_ptrs: [6]u64 = undefined;
    for (&mma_ptrs, [_][]const u8{ suh_g.items, suh_u.items, svh_g.items, svh_u.items, suh_d.items, svh_d.items }, [_][]const u8{ "mma_suh_g", "mma_suh_u", "mma_svh_g", "mma_svh_u", "mma_suh_d", "mma_svh_d" }) |*mp, bytes, what| {
        try L.host.appendSlice(L.gpa, bytes);
        mp.* = try L.put("layers.{d}.moe.experts.{s}", .{ i, what });
    }
    // the shared expert: BF16 rows staged, then either kept raw or GPU-quantized to NVFP4 lanes
    // (TF_GLM_SHARED_FP4=0 keeps BF16); the .q weights ride moe.fill's existing shared path
    const sh = L.w.share.shared;
    var shared_gate: ?cuda.qlinear.Weight = null;
    var shared_up: ?cuda.qlinear.Weight = null;
    var shared_down: b16.Proj = undefined;
    var tiled_down: u64 = 0;
    var shared_down_q: ?@import("cuda_moe.zig").QuantW = null;
    // MEASURED NEGATIVE (session 68zl): fp4-lane shared expert regressed single-stream
    // 156.9 -> 123.1 (the qmmf lane gemv is slower than b16's tuned one-row tiles here);
    // opt-in only, default BF16
    const want_fp4 = blk: {
        const v = std.c.getenv("TF_GLM_SHARED_FP4");
        break :blk v != null and std.mem.eql(u8, std.mem.span(v.?), "1");
    };
    inline for (.{ "gate_proj", "up_proj" }, 0..) |x, m| {
        try L.rows(try L.get("layers.{d}.mlp.shared_experts." ++ x ++ ".weight", .{i}), sh.lo, sh.hi, c.shared_inter, .bf16);
        if (want_fp4) {
            const nn = sh.len();
            var nbb: [96]u8 = undefined;
            const nm = try std.fmt.bufPrint(&nbb, "layers.{d}.moe.shared." ++ x, .{i});
            var nbb2: [96]u8 = undefined;
            const nm2 = try std.fmt.bufPrint(&nbb2, "layers.{d}.moe.shared." ++ x ++ ".scales", .{i});
            const one = try fp4Lane(L, L.host.items, nn, d, nm, nm2);
            if (m == 0) shared_gate = one else shared_up = one;
            L.host.clearRetainingCapacity();
        }
    }
    var gu: u64 = 0;
    if (!want_fp4) gu = try L.put("layers.{d}.moe.shared.gu", .{i});
    try L.cols(try L.get("layers.{d}.mlp.shared_experts.down_proj.weight", .{i}), sh.lo, sh.hi, .bf16);
    if (want_fp4) {
        var nbb3: [96]u8 = undefined;
        const nm3 = try std.fmt.bufPrint(&nbb3, "layers.{d}.moe.shared.down", .{i});
        var nbb4: [96]u8 = undefined;
        const nm4 = try std.fmt.bufPrint(&nbb4, "layers.{d}.moe.shared.down.scales", .{i});
        const one = try fp4Lane(L, L.host.items, d, sh.len(), nm3, nm4);
        shared_down = .{ .q = one };
        L.host.clearRetainingCapacity();
    } else {
        // the mm16 tiling reads the staged rows BEFORE the put consumes L.host (the emptied-list
        // upload was the whole corruption); the raw put remains the TF_GLM_MM16=0 fallback
        const v16 = std.c.getenv("TF_GLM_MM16");
        const want_mm16 = v16 != null and std.mem.eql(u8, std.mem.span(v16.?), "1");
        var src: ?cuda.DeviceBuffer = null;
        defer if (src) |*b| b.free();
        var src16_rows: []const u16 = &.{};
        if (want_mm16) {
            src = try L.scratch(d * sh.len() * 2);
            try src.?.upload(0, L.host.items);
            const rows16 = try L.gpa.alloc(u16, L.host.items.len / 2);
            for (rows16, 0..) |*u16v, ix| u16v.* = std.mem.readInt(u16, L.host.items[ix * 2 ..][0..2], .little);
            src16_rows = rows16;
        }
        defer L.gpa.free(src16_rows);
        // affine quantization of the shared expert's down (TF_GLM_DENSE_BITS): host quantize -> GPU tile
        {
            const benv = std.c.getenv("TF_GLM_DENSE_BITS");
            const qbits: u8 = if (benv) |bv| std.fmt.parseInt(u8, std.mem.span(bv), 10) catch 0 else 0;
            if (qbits == 4 or qbits == 8) {
                if (try quantizedLane(L, L.host.items, d, sh.len(), qbits, "sd.codes", "sd.scales")) |q| {
                    shared_down_q = .{ .w = q.w, .s = q.s, .bits = qbits };
                }
            }
        }
        shared_down = .{ .b16 = try L.put("layers.{d}.moe.shared.down", .{i}) };
        if (want_mm16) {
            var gm = try cuda.Module.load(L.d, cuda.kernels.glm_glue);
            defer gm.unload();
            const t16 = try gm.function("glm_mm_tile16");
            const dst = try cuda.DeviceBuffer.alloc(L.d, d * sh.len() * 2);
            try L.w.buffers.append(L.gpa, dst);
            const chunks = (d / 8) * (sh.len() / 64) * 32;
            var ta: cuda.Args = .{};
            ta.add(src.?.ptr);
            ta.add(dst.ptr);
            ta.add(@as(c_int, @intCast(d)));
            ta.add(@as(c_int, @intCast(sh.len())));
            try cuda.launch.launch(t16, .{ .grid = .{ .x = @intCast((chunks + 255) / 256) }, .block = .{ .x = 256 } }, L.stream, &ta);
            try L.stream.synchronize(); // their lazy tiler syncs too: the engine reads dst on ITS stream
            if (std.c.getenv("TF_GLM_MM16_VERIFY") != null and i == c.dense_layers) {
                // host reference tiling: dst[c*16+v] = src[n*K + g*64 + frag_k(lane,v)] (glm_mm_tile16)
                const got = try L.gpa.alloc(u8, d * sh.len() * 2);
                defer L.gpa.free(got);
                try dst.download(0, got);
                const srcb = std.mem.sliceAsBytes(src16_rows);
                var bad: usize = 0;
                const KG = sh.len() / 64;
                var c2: usize = 0;
                while (c2 < chunks) : (c2 += 1) {
                    const lane = c2 & 31;
                    const tg = c2 >> 5;
                    const g = tg % KG;
                    const tt = tg / KG;
                    const n0 = tt * 8 + (lane >> 2);
                    const t = lane & 3;
                    var v: usize = 0;
                    while (v < 16) : (v += 1) {
                        const kt = v >> 2;
                        const j = v & 3;
                        const fk = kt * 16 + 2 * t + (j & 1) + (j >> 1) * 8;
                        const want_b = srcb[(n0 * sh.len() + g * 64 + fk) * 2 ..][0..2];
                        const have_b = got[c2 * 32 + v * 2 ..][0..2];
                        if (!std.mem.eql(u8, want_b, have_b)) bad += 1;
                    }
                }
                std.debug.print("mm16-verify layer {d}: {d}/{d} chunk-values MISMATCH (0 = perfect)\n", .{ i, bad, chunks * 16 });
            }
            tiled_down = dst.ptr;
        }
    }
    return .{
        .router = router,
        .bias = bias,
        .routed = .{ .format = .exl3, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = cut.len(), .dims = d, .experts = c.experts, .limit = c.swiglu_limit, .exl3 = widths, .mats_up = mats_up, .mats_down = mats_dn },
        .count = c.experts,
        .shared_gu = gu,
        .shared_down = shared_down,
        .shared_down_mm = tiled_down,
        .shared_down_q = shared_down_q,
        .shared_gate = shared_gate,
        .shared_up = shared_up,
        .shared = sh.len(),
        .mma_suh_g = mma_ptrs[0],
        .mma_suh_u = mma_ptrs[1],
        .mma_svh_g = mma_ptrs[2],
        .mma_svh_u = mma_ptrs[3],
        .mma_suh_d = mma_ptrs[4],
        .mma_svh_d = mma_ptrs[5],
        .top_k = c.topk,
        .scale = c.routed_scale,
        .norm = c.norm_topk,
    };
}

pub fn moe(L: *W.Loader, i: u32) !@import("cuda_moe.zig").Layer {
    const c = &L.w.config;
    const d: usize = c.hidden;
    const cut = L.w.share.moe;
    try L.whole(try L.get("layers.{d}.mlp.gate.weight", .{i}), .bf16);
    const router = try L.put("layers.{d}.moe.router", .{i});
    try L.whole(try L.get("layers.{d}.mlp.gate.e_score_correction_bias", .{i}), .f32);
    const bias = try L.put("layers.{d}.moe.bias", .{i});
    var t_up: std.ArrayList(u8) = .empty;
    defer t_up.deinit(L.gpa);
    var h_up: std.ArrayList(u8) = .empty;
    defer h_up.deinit(L.gpa);
    var t_dn: std.ArrayList(u8) = .empty;
    defer t_dn.deinit(L.gpa);
    var h_dn: std.ArrayList(u8) = .empty;
    defer h_dn.deinit(L.gpa);
    var widths: cuda.exl3_experts.Widths = .{};
    var nb: [192]u8 = undefined;
    for (0..c.experts) |e| {
        for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |p, m| {
            const name = try std.fmt.bufPrint(&nb, W.prefix ++ "layers.{d}.mlp.experts.{d}.{s}", .{ i, e, p });
            const q = if (m < 2) try gather(L.ck, L.gpa, name, 0, 0, &.{cut}, &t_up, &h_up) else try gather(L.ck, L.gpa, name, cut.lo, cut.hi, &.{.{ .lo = 0, .hi = d }}, &t_dn, &h_dn);
            if (m < 2) widths.up = q.bits else widths.down = q.bits;
            widths.codebook = q.codebook;
        }
    }
    var ptrs: [4]u64 = undefined;
    for (&ptrs, [_][]const u8{ t_up.items, t_dn.items, h_up.items, h_dn.items }, [_][]const u8{ "up", "down", "up_scale", "down_scale" }) |*p, bytes, what| {
        try L.host.appendSlice(L.gpa, bytes);
        p.* = try L.put("layers.{d}.moe.experts.{s}", .{ i, what });
    }
    const sh = L.w.share.shared;
    return .{
        .router = router,
        .bias = bias,
        .routed = .{ .format = .exl3, .up = ptrs[0], .down = ptrs[1], .up_scale = ptrs[2], .down_scale = ptrs[3], .width = cut.len(), .dims = d, .experts = c.experts, .limit = c.swiglu_limit, .exl3 = widths },
        .count = c.experts,
        .shared_gu = 0,
        .shared_gate = try layerMatrix(L, "layers.{d}.mlp.shared_experts.gate_proj", i, 0, 0, &.{sh}),
        .shared_up = try layerMatrix(L, "layers.{d}.mlp.shared_experts.up_proj", i, 0, 0, &.{sh}),
        .shared_down = .{ .q = try layerMatrix(L, "layers.{d}.mlp.shared_experts.down_proj", i, sh.lo, sh.hi, &.{.{ .lo = 0, .hi = d }}) },
        .shared = sh.len(),
        .top_k = c.topk,
        .scale = c.routed_scale,
        .norm = c.norm_topk,
    };
}
