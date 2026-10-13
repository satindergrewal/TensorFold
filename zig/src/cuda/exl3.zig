//! EXL3 trellis projections: per-tensor 3/3.5/4-bit MCG, as core/exl3_format.zig's
//! dequantize factorizes - signs and the K rotation on x, the raw-trellis GEMM, then
//! the N rotation and signs on y. Geometry from the trellis shape alone; the scalar
//! spec is the oracle and exl3.cu its warp transcription.

const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");

const Stream = stream_.Stream;

/// A trellis weight on the device. `halves` is 2*bits: 6, 7 or 8; the trellis is
/// [k/16, n/16, halves*8] int16 and suh/svh fp16 sign values [k] and [n].
pub const Weight = struct {
    trellis: u64,
    suh: u64,
    svh: u64,
    halves: u32,
    k: u32,
    n: u32,

    pub fn tileWords(w: Weight) u32 {
        return w.halves * 8;
    }

    pub fn tilesK(w: Weight) u32 {
        return (w.k + 15) / 16;
    }

    pub fn tilesN(w: Weight) u32 {
        return (w.n + 15) / 16;
    }
};

/// exl3.cu's exports (cuobjdump -symbols of the fatbin).
pub const symbols = struct {
    pub const pre_rot = "_ZN14tf_exl3_kernel12exl3_pre_rotEPK13__nv_bfloat16PKtPS0_ii";
    pub const post_rot = "_ZN14tf_exl3_kernel13exl3_post_rotEP13__nv_bfloat16PKtii";
    pub const matmul = "_ZN14tf_exl3_kernel18exl3_decode_matmulEPK13__nv_bfloat16PKsPS0_iiiiii";
    pub const verify = "_ZN14tf_exl3_kernel18exl3_verify_decodeEPKsiP13__nv_bfloat16";
    pub const grouped_project = "exl3_grouped_project";
    pub const grouped_pre = "exl3_grouped_pre_rot";
    pub const grouped_post = "exl3_grouped_post_rot";
};

fn i(x: usize) c_int {
    return @intCast(x);
}

fn u(x: usize) u32 {
    return @intCast(x);
}

/// The loaded fatbin's functions and the projection scratch (x_tilde, rows <= chunk).
pub const Kernels = struct {
    mod: module.Module,
    pre_fn: module.Function,
    post_fn: module.Function,
    mm_fn: module.Function,
    verify_fn: module.Function,
    grouped_fn: module.Function,
    grouped_pre_fn: module.Function,
    grouped_post_fn: module.Function,
    desc_table: u64 = 0, // [experts] Exl3Desc; 288 x 48B = ~14KB
    desc_tbl_own: memory.DeviceBuffer = undefined,
    scratch: memory.DeviceBuffer, // x_tilde [chunk_rows, k_max] bf16

    pub const chunk_rows = 16; // the GEMM's staged rows; grid.y tiles more

    /// The largest projection this serves; the scratch is carved once at load.
    pub const k_max = 16384;

    pub fn load(d: *const driver.Driver) !Kernels {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try module.Module.load(d, kernels.exl3);
        errdefer mod.unload();
        var k: Kernels = .{
            .mod = mod,
            .pre_fn = undefined,
            .post_fn = undefined,
            .mm_fn = undefined,
            .verify_fn = undefined,
            .grouped_fn = undefined,
            .grouped_pre_fn = undefined,
            .grouped_post_fn = undefined,
            .scratch = undefined,
        };
        errdefer k.scratch.free();
        k.pre_fn = try mod.function(symbols.pre_rot);
        k.post_fn = try mod.function(symbols.post_rot);
        k.mm_fn = try mod.function(symbols.matmul);
        k.verify_fn = try mod.function(symbols.verify);
        k.grouped_fn = try mod.function(symbols.grouped_project);
        k.grouped_pre_fn = try mod.function(symbols.grouped_pre);
        k.grouped_post_fn = try mod.function(symbols.grouped_post);
        k.scratch = try memory.DeviceBuffer.alloc(d, chunk_rows * k_max * 2);
        const tbl = try memory.DeviceBuffer.alloc(d, 512 * @sizeOf(Desc));
        k.desc_table = tbl.ptr;
        k.desc_tbl_own = tbl;
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        if (k.desc_table != 0) k.desc_tbl_own.free();
        k.scratch.free();
        k.mod.unload();
    }

    /// x_tilde [m, k] = rot128(x [m, k] .* suh): the GEMM's input stage.
    pub fn preRot(k: *const Kernels, s: Stream, x: u64, suh: u64, out: u64, m: usize, dim: usize) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(suh);
        a.add(out);
        a.add(i(m));
        a.add(i(dim));
        try launch_.launch(k.pre_fn, .{ .grid = .{ .x = u((dim + 127) / 128), .y = u(m) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// y [m, n] .^ svh then rot128, in place: the GEMM's output stage.
    pub fn postRot(k: *const Kernels, s: Stream, y: u64, svh: u64, m: usize, dim: usize) !void {
        var a: launch_.Args = .{};
        a.add(y);
        a.add(svh);
        a.add(i(m));
        a.add(i(dim));
        try launch_.launch(k.post_fn, .{ .grid = .{ .x = u((dim + 127) / 128), .y = u(m) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// y [rows, n] bf16 = x_tilde [rows, k] W_q; any row count, 16-row blocks on grid.y.
    pub fn matmul(k: *const Kernels, s: Stream, x_tilde: u64, w: Weight, y: u64, rows: usize) !void {
        var a: launch_.Args = .{};
        a.add(x_tilde);
        a.add(w.trellis);
        a.add(y);
        a.add(i(rows));
        a.add(i(w.k));
        a.add(i(w.n));
        a.add(i(w.tilesK()));
        a.add(i(w.tilesN()));
        a.add(i(w.halves));
        const row_chunks = (rows + chunk_rows - 1) / chunk_rows;
        try launch_.launch(k.mm_fn, .{ .grid = .{ .x = u(w.tilesN()), .y = u(row_chunks) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// One call: pre, GEMM (x_tilde in the shared scratch) and post. Single stream -
    /// the scratch is the instance's.
    pub const Desc = extern struct {
        trellis: u64,
        suh: u64,
        svh: u64,
        k: i32,
        n: i32,
        tiles_k: i32,
        tiles_n: i32,
        halves: i32,
    };

    /// Upload the per-expert descriptor table once (weights are static post-load).
    pub fn tableFrom(k: *const Kernels, s: Stream, ws: []const Weight) !u64 {
        _ = s;
        const n = ws.len;
        if (k.desc_table == 0) return error.NoTable;
        const host = try std.heap.page_allocator.alloc(u8, n * @sizeOf(Desc));
        defer std.heap.page_allocator.free(host);
        const ds: [*]Desc = @ptrCast(@alignCast(host.ptr));
        for (ws, 0..) |w, di| {
            ds[di] = .{
                .trellis = w.trellis,
                .suh = w.suh,
                .svh = w.svh,
                .k = @intCast(w.k),
                .n = @intCast(w.n),
                .tiles_k = @intCast(w.tilesK()),
                .tiles_n = @intCast(w.tilesN()),
                .halves = @intCast(w.halves),
            };
        }
        try k.desc_tbl_own.upload(0, host[0 .. n * @sizeOf(Desc)]);
        return k.desc_table;
    }

    /// All slots in ONE launch family: grid (tiles_n_max, topk) - the row is
    /// picked on DEVICE. rows must be 1 (serving decode). y: [topk, n_max].
    pub fn projectGrouped(k: *const Kernels, s: Stream, x: u64, x_stride: usize, table: u64, picks: u64, y: u64, y_stride: usize, topk: usize, tiles_n_max: usize) !void {
        var a: launch_.Args = .{};
        a.add(x);
        a.add(table);
        a.add(picks);
        a.add(y);
        a.add(i(topk));
        a.add(i(x_stride));
        a.add(i(y_stride));
        try launch_.launch(k.grouped_fn, .{ .grid = .{ .x = u(tiles_n_max), .y = u(topk) }, .block = .{ .x = 128 } }, s, &a);
    }

    /// pre-rot + grouped matmul + post-rot for one weight family; x is the
    /// [1, k] row, y lands at [topk, n]. xt = per-slot scratch [topk, k_max].
    pub fn projectGroupedFull(k: *const Kernels, s: Stream, x: u64, x_stride: usize, table: u64, picks: u64, xt: u64, y: u64, y_stride: usize, topk: usize, k_blocks: usize, tiles_n_max: usize) !void {
        {
            var a: launch_.Args = .{};
            a.add(x);
            a.add(table);
            a.add(picks);
            a.add(xt);
            a.add(i(topk));
            a.add(i(x_stride));
            try launch_.launch(k.grouped_pre_fn, .{ .grid = .{ .x = u(k_blocks), .y = u(topk) }, .block = .{ .x = 128 } }, s, &a);
        }
        // xt rows are x_stride apart (the caller's slot stride); the matmul
        // must read the same layout the pre-rot wrote
        try k.projectGrouped(s, xt, x_stride, table, picks, y, y_stride, topk, tiles_n_max);
        {
            var a: launch_.Args = .{};
            a.add(y);
            a.add(table);
            a.add(picks);
            a.add(i(topk));
            a.add(i(y_stride));
            try launch_.launch(k.grouped_post_fn, .{ .grid = .{ .x = u(tiles_n_max * 16 / 128 + 1), .y = u(topk) }, .block = .{ .x = 128 } }, s, &a);
        }
    }

    pub fn project(k: *const Kernels, s: Stream, x: u64, w: Weight, y: u64, rows: usize) !void {
        if (rows > chunk_rows or w.k > k_max) return error.WindowTooWide;
        try k.preRot(s, x, w.suh, k.scratch.ptr, rows, w.k);
        try k.matmul(s, k.scratch.ptr, w, y, rows);
        try k.postRot(s, y, w.svh, rows, w.n);
    }

    /// Any row count, 16-row chunks through the shared scratch (the prompt path);
    /// the stages of a chunk are ordering-visible on the one stream.
    pub fn projectWide(k: *const Kernels, s: Stream, x: u64, w: Weight, y: u64, rows: usize) !void {
        if (w.k > k_max) return error.WindowTooWide;
        var done: usize = 0;
        while (done < rows) : (done += chunk_rows) {
            const here = @min(chunk_rows, rows - done);
            try k.preRot(s, x + done * w.k * 2, w.suh, k.scratch.ptr, here, w.k);
            try k.matmul(s, k.scratch.ptr, w, y + done * w.n * 2, here);
            try k.postRot(s, y + done * w.n * 2, w.svh, here, w.n);
        }
    }

    /// One tile's 256 stream values through the device decode, for the host tests.
    pub fn verifyTile(k: *const Kernels, s: Stream, trellis: u64, halves: u32, out: u64) !void {
        var a: launch_.Args = .{};
        a.add(trellis);
        a.add(i(halves));
        a.add(out);
        try launch_.launch(k.verify_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 256 } }, s, &a);
    }
};

test "weight geometry from the trellis shape" {
    const w: Weight = .{ .trellis = 0, .suh = 0, .svh = 0, .halves = 7, .k = 4096, .n = 2048 };
    try std.testing.expectEqual(@as(u32, 56), w.tileWords());
    try std.testing.expectEqual(@as(u32, 256), w.tilesK());
    try std.testing.expectEqual(@as(u32, 128), w.tilesN());
}
