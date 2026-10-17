//! exl3_experts.cu's EXL3 routed experts on grouped.zig's plan: gate and up under SwiGLU, down to fp32 or bf16 rows.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const grouped = @import("grouped.zig");
const experts = @import("experts.zig");
const exl3 = @import("exl3_linear.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// exl3_experts.cu's TfExl3Experts, passed by value.
pub const Geometry = extern struct { k: u32, n: u32, bits: u32, tiles_n: u32, mats: u32, epi: u32, book: u32, pad: u32 = 0, stride: u64 };

/// Per-matrix mixed-width descriptor (our fork): full device pointers.
pub const MatDesc = extern struct { trellis: u64, suh: u64, svh: u64, k: u32, n: u32, tiles_n: u32, bits: u32 };

/// A layer's EXL3 widths and codebook; every expert of a layer shares them (a family stacks its shared ones apart).
pub const Widths = struct { up: u32 = 0, down: u32 = 0, codebook: exl3.Codebook = .mul1 };

/// Halves an expert matrix carries: suh (k) then svh (n).
pub fn halves(k: usize, n: usize) usize {
    return k + n;
}

/// The device stacks a layer view points at: up [E][2] (gate, up) and down [E] trellises, each with its halves.
pub const Stacks = struct {
    pub fn upBytes(count: usize, dims: usize, width: usize, bits: u32) usize {
        return count * 2 * exl3.Matrix.trellisBytes(dims, width, bits);
    }
    pub fn downBytes(count: usize, dims: usize, width: usize, bits: u32) usize {
        return count * exl3.Matrix.trellisBytes(width, dims, bits);
    }
    pub fn upHalves(count: usize, dims: usize, width: usize) usize {
        return count * 2 * halves(dims, width);
    }
    pub fn downHalves(count: usize, dims: usize, width: usize) usize {
        return count * halves(width, dims);
    }
};

pub const Experts = struct {
    mod: Module,
    decode_fn: Function, // every width and codebook
    mma_rot_in_fn: Function = undefined, // the mixed-width mma decode pass (ported skeleton)
    mma_grouped16_fn: Function = undefined,
    mma_grouped64_fn: Function = undefined,
    mma_grouped32_fn: Function = undefined,
    mma_gateup_fn: Function = undefined,
    mma_down_fn: Function = undefined,
    prompt_fn: Function, // prompt rows on the tensor units
    resident: [2]usize, // blocks the grid may hold, decode and prompt: per SM times SMs

    pub fn load(d: *const Driver, sms: usize) !Experts {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.exl3_experts);
        errdefer mod.unload();
        const decode = try mod.function("tf_exl3_experts");
        const prompt = try mod.function("tf_exl3_experts_prompt");
        // the mixed-width mma decode pass (the same fatbin; tf_exl3m_* in exl3_experts.cu)
        const mma: struct { rot: Function, g: Function, g64: Function, g32: Function, gu: Function, dn: Function } = .{
            .rot = try mod.function("tf_exl3m_rot_in"),
            .g = try mod.function("tf_exl3m_grouped16"),
            .g64 = try mod.function("tf_exl3m_grouped64"),
            .g32 = try mod.function("tf_exl3m_grouped32"),
            .gu = try mod.function("tf_exl3m_gateup"),
            .dn = try mod.function("tf_exl3m_down"),
        };
        const e: Experts = .{
            .mod = mod,
            .decode_fn = decode,
            .prompt_fn = prompt,
            .resident = .{ @max(1, try decode.occupancy(256, 0)) * sms, @max(1, try prompt.occupancy(256, 0)) * sms },
            .mma_rot_in_fn = mma.rot,
            .mma_grouped16_fn = mma.g,
            .mma_grouped64_fn = mma.g64,
            .mma_grouped32_fn = mma.g32,
            .mma_gateup_fn = mma.gu,
            .mma_down_fn = mma.dn,
        };
        return e;
    }

    pub fn unload(e: *Experts) void {
        e.mod.unload();
    }

    /// The geometry of one call: up takes D inputs to NI outputs (gate and up), down NI to D.
    /// The uniform Matrix.check is SKIPPED for mixed-width layers (our fork): the
    /// per-expert desc drives k/n/bits/tiles inside the kernel; g here only sets
    /// the grid bound (n_max) and the epilogue.
    pub fn geometry(epi: experts.Epi, l: experts.Layer) !Geometry {
        const up = epi == .up;
        const k = if (up) l.dims else l.width;
        const n = if (up) l.width else l.dims;
        const bits = if (up) l.exl3.up else l.exl3.down;
        if (l.exl3.up != 0) try (exl3.Matrix{ .trellis = 0, .suh = 0, .svh = 0, .k = k, .n = n, .bits = bits, .codebook = l.exl3.codebook }).check();
        return .{
            .k = @intCast(k),
            .n = @intCast(n),
            .bits = bits,
            .tiles_n = @intCast(n / 16),
            .mats = if (up) 2 else 1,
            .epi = switch (epi) {
                .up => 2,
                .down_f32 => 0,
                .down_bf16 => 3,
            },
            .book = @backingInt(l.exl3.codebook),
            .stride = exl3.Matrix.trellisBytes(k, n, bits),
        };
    }

    /// Per-matrix mixed-width descriptors (our fork); 0 = uniform (their behavior).
    pub fn runMixed(e: *const Experts, s: Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, skip: i32, mats_desc: u64) !void {
        return e.runInner(s, epi, in, l, p, out, items, skip, false, mats_desc);
    }

    /// The mma pass's item shape: 16 pairs (the default), 32 or 64 (TF_GLM_MMA_MT).
pub const MmaTile = enum { mt16, mt32, mt64 };

/// The mixed-width mma decode pass (the skeleton of pr/glm-flash-cuda @ 3c6ed009's glm/exl3.cu, our tiles):
    /// X rows -> the pairs' gate and up rotations, both mma passes, the gate/up epilogue with the down input
    /// rotation, and the down epilogue into the ey slot rows. SK K-splits a pass.
    pub fn mmaMixed(e: *const Experts, s: Stream, x: u64, x_stride: usize, rows: usize, slots: usize, pick: u64, suh_g: u64, suh_u: u64, svh_g: u64, svh_u: u64, suh_d: u64, svh_d: u64, mats_up: u64, mats_down: u64, plan: grouped.Plan, items: usize, z: u64, xg: u64, xu: u64, xd: u64, ey: u64, width: usize, dims: usize, limit: f32, book: c_uint, skip: i32, wide: MmaTile) !void {
                // SK=8 measured best at ALL stream counts (adaptive-SK to 2 at multi-stream dropped
        // aggregate 40-47%: the expert kernel needs the split parallelism even at high rows)
        // measured end-to-end on the 2x RTX PRO 6000 box: gate/up SK=2 with the down pass whole (SK=1)
        // runs ~101 tok/s vs 90 at the old SK=8-for-both (the python engine's split direction; our
        // kernel shape prefers 2 for gate/up) - TF_GLM_MMA_SK still overrides
        const sk: usize = if (std.c.getenv("TF_GLM_MMA_SK") != null) blk: {
            const v = std.c.getenv("TF_GLM_MMA_SK").?;
            break :blk std.fmt.parseInt(usize, std.mem.span(v), 10) catch 2;
        } else 2;
        // the python engine splits the gate/up pass but runs the down pass whole (its SK=1):
        // TF_GLM_MMA_SK_DOWN overrides the down pass alone (default: the shared SK)
        const sk_down: usize = if (std.c.getenv("TF_GLM_MMA_SK_DOWN") != null) blk: {
            const v = std.c.getenv("TF_GLM_MMA_SK_DOWN").?;
            break :blk std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1;
        } else 1;
        const p = rows * slots;
        {
            var a: launch_.Args = .{};
            a.add(x);
            a.add(@as(i32, @intCast(x_stride)));
            a.add(@as(i32, @intCast(slots)));
            a.add(pick);
            a.add(suh_g);
            a.add(suh_u);
            a.add(xg);
            a.add(xu);
            a.add(@as(i32, @intCast(dims)));
            a.add(skip);
            try launch_.launch(e.mma_rot_in_fn, .{ .grid = .{ .x = @intCast(p), .y = @intCast(dims / 128), .z = 2 }, .block = .{ .x = 32 } }, s, &a);
        }
        {
            var a: launch_.Args = .{};
            a.add(xg);
            a.add(xu);
            a.add(mats_up);
            a.add(plan.items);
            a.add(plan.counts);
            a.add(plan.members);
            a.add(z);
            a.add(@as(i32, @intCast(width)));
            a.add(@as(i32, @intCast(p)));
            a.add(@as(c_int, @intCast(sk)));
            a.add(skip);
            a.add(@as(c_uint, book));
            a.add(@as(c_int, 2));
            if (wide == .mt32) try launch_.launch(e.mma_grouped32_fn, .{ .grid = .{ .x = @intCast(items), .y = @intCast(width / 32), .z = @intCast(2 * sk) }, .block = .{ .x = 128 }, .shared = 32 * 4 + 4 * 32 * 32 * 4 }, s, &a) else if (wide == .mt64) try launch_.launch(e.mma_grouped64_fn, .{ .grid = .{ .x = @intCast(items), .y = @intCast(width / 16), .z = @intCast(2 * sk) }, .block = .{ .x = 128 }, .shared = 64 * 4 + 4 * 64 * 16 * 4 }, s, &a) else try launch_.launch(e.mma_grouped16_fn, .{ .grid = .{ .x = @intCast(items), .y = @intCast(width / 128), .z = @intCast(2 * sk) }, .block = .{ .x = 128 }, .shared = 16 * 4 + 4 * 16 * 128 * 4 }, s, &a);
        }
        {
            var a: launch_.Args = .{};
            a.add(z);
            a.add(pick);
            a.add(svh_g);
            a.add(svh_u);
            a.add(suh_d);
            a.add(xd);
            a.add(@as(i32, @intCast(p)));
            a.add(@as(i32, @intCast(width)));
            a.add(@as(c_int, @intCast(sk)));
            a.add(limit);
            a.add(skip);
            try launch_.launch(e.mma_gateup_fn, .{ .grid = .{ .x = @intCast(p), .y = @intCast(width / 128) }, .block = .{ .x = 32 } }, s, &a);
        }
        {
            var a: launch_.Args = .{};
            a.add(xd);
            a.add(xd);
            a.add(mats_down);
            a.add(plan.items);
            a.add(plan.counts);
            a.add(plan.members);
            a.add(z);
            a.add(@as(i32, @intCast(dims)));
            a.add(@as(i32, @intCast(p)));
            a.add(@as(c_int, @intCast(sk_down)));
            a.add(skip);
            a.add(@as(c_uint, book));
            a.add(@as(c_int, 1));
            if (wide == .mt32) try launch_.launch(e.mma_grouped32_fn, .{ .grid = .{ .x = @intCast(items), .y = @intCast(dims / 32), .z = @intCast(sk_down) }, .block = .{ .x = 128 }, .shared = 32 * 4 + 4 * 32 * 32 * 4 }, s, &a) else if (wide == .mt64) try launch_.launch(e.mma_grouped64_fn, .{ .grid = .{ .x = @intCast(items), .y = @intCast(dims / 16), .z = @intCast(sk_down) }, .block = .{ .x = 128 }, .shared = 64 * 4 + 4 * 64 * 16 * 4 }, s, &a) else try launch_.launch(e.mma_grouped16_fn, .{ .grid = .{ .x = @intCast(items), .y = @intCast(dims / 128), .z = @intCast(sk_down) }, .block = .{ .x = 128 }, .shared = 16 * 4 + 4 * 16 * 128 * 4 }, s, &a);
        }
        {
            var a: launch_.Args = .{};
            a.add(z);
            a.add(pick);
            a.add(svh_d);
            a.add(ey);
            a.add(@as(i32, @intCast(p)));
            a.add(@as(i32, @intCast(dims)));
            a.add(@as(c_int, @intCast(sk_down)));
            a.add(@as(i32, @intCast(slots)));
            a.add(@as(i32, @intCast(slots)));
            a.add(skip);
            try launch_.launch(e.mma_down_fn, .{ .grid = .{ .x = @intCast(p), .y = @intCast(dims / 128) }, .block = .{ .x = 32 } }, s, &a);
        }
    }

    pub fn run(e: *const Experts, s: Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, skip: i32, prompt: bool) !void {
        return e.runInner(s, epi, in, l, p, out, items, skip, prompt, 0);
    }

    fn runInner(e: *const Experts, s: Stream, epi: experts.Epi, in: experts.Rows, l: experts.Layer, p: grouped.Plan, out: u64, items: usize, skip: i32, prompt: bool, mats_desc: u64) !void {
        const g = try geometry(epi, l);
        const pi: usize = @intFromBool(prompt);
        const grid = @min(items * (g.n / 128), e.resident[pi]);
        if (grid < 1) return;
        const up = epi == .up;
        var a: launch_.Args = .{};
        a.add(in.x);
        a.add(@as(i32, @intCast(in.stride)));
        a.add(@as(i32, @intCast(if (up) in.slots else 0)));
        a.add(if (up) l.up else l.down);
        a.add(if (up) l.up_scale else l.down_scale);
        for ([_]u64{ p.items, p.counts, p.members, out }) |v| a.add(v);
        a.add(if (up) l.limit else @as(f32, 0.0));
        a.add(skip);
        a.add(g);
        a.add(mats_desc);
        try launch_.launch(if (prompt) e.prompt_fn else e.decode_fn, .{ .grid = .{ .x = @intCast(grid), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }
};

comptime {
    std.debug.assert(@sizeOf(Geometry) == 40);
    // CUDA TfExl3Mat: 3 pointers (8B each) then 4 u32 = 32B; field order must match
    std.debug.assert(@sizeOf(MatDesc) == 40); // 3 ptrs + 4 u32, 8-align
}

test "an EXL3 layer's geometry follows the epilogue, and a shape exl3.cu cannot take is refused" {
    const l: experts.Layer = .{ .format = .exl3, .up = 0, .down = 0, .width = 2048, .dims = 4096, .experts = 288, .exl3 = .{ .up = 4, .down = 4 } };
    const g = try Experts.geometry(.up, l);
    try std.testing.expectEqual(@as(u32, 2), g.mats);
    try std.testing.expectEqual(@as(u64, 256 * 128 * 64 * 2), g.stride); // a gate_proj: trellis [256, 128, 64] int16
    const d = try Experts.geometry(.down_bf16, l);
    try std.testing.expectEqual(@as(u32, 2048), d.k);
    try std.testing.expectEqual(@as(u32, 3), d.epi);
    var bad = l;
    bad.width = 2000;
    try std.testing.expectError(error.InvalidExl3Matrix, Experts.geometry(.up, bad));
    try std.testing.expectEqual(@as(usize, 288 * 2 * (4096 + 2048)), Stacks.upHalves(288, 4096, 2048));
}
