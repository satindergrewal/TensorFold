//! GLM-5.3-Flash's vision tower on CUDA: model.visual.* weights, the 24-block
//! 2-D-rope tower, the 2x2 downsample (an exact matmul over merge-block rows)
//! and the SwiGLU merger into the LM's 4,096 columns. Parity target:
//! vision_ref_feat.bin (the torch reference on the same synthetic frames).
//!
//! The checkpoint's patch layout is MERGE-BLOCK-MAJOR: row r decodes as
//! (oy = r/64, ox = (r%64)/4, my = (r%4)/2, mx = r%2), patch (py, px) =
//! (oy*2+my, ox*2+mx), and the row's 1,176 values are ordered
//! (my, mx, c, t, py_in, px_in). The 2x2 merge therefore reads CONSECUTIVE
//! rows 4g..4g+3 into one [4, D] group.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");

const Driver = cuda.Driver;
const DeviceBuffer = cuda.DeviceBuffer;

pub const Kernels = struct {
    mod: cuda.Module,
    ops_mod: cuda.Module,
    rms_fn: cuda.Function,
    act_fn: cuda.Function,
    bias_fn: cuda.Function,
    attn_fn: cuda.Function,
    attndbg_fn: cuda.Function,
    gelu_fn: cuda.Function,
    ln_fn: cuda.Function,
    qknr_fn: cuda.Function,
    add_fn: cuda.Function,
    gather_fn: cuda.Function,
    embed_fn: cuda.Function,
    embedf32_fn: cuda.Function,

    pub fn load(d: *const Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var mod = try cuda.Module.load(d, cuda.kernels.glm_vision);
        errdefer mod.unload();
        var ops = try cuda.Module.load(d, cuda.kernels.glm_ops);
        errdefer ops.unload();
        return .{
            .mod = mod,
            .ops_mod = ops,
            .rms_fn = try ops.function("glm_rms"),
            .act_fn = try ops.function("glm_act"),
            .bias_fn = try mod.function("vv_bias"),
            .attn_fn = try mod.function("vv_attn"),
            .attndbg_fn = try mod.function("vv_attn_dbg"),
            .gelu_fn = try mod.function("vv_gelu"),
            .ln_fn = try mod.function("vv_layernorm"),
            .qknr_fn = try mod.function("vv_qk_norm_rope"),
            .add_fn = try mod.function("vv_add"),
            .gather_fn = try mod.function("vv_gather_merge"),
            .embed_fn = try mod.function("vv_embed"),
            .embedf32_fn = try mod.function("vv_embed_f32"),
        };
    }

    pub fn deinit(k: *Kernels) void {
        k.ops_mod.unload();
        k.mod.unload();
    }
};

fn go(k: cuda.Function, s: cuda.Stream, grid: cuda.Dim3, block: u32, smem: u32, a: *cuda.Args) !void {
    try cuda.launch.launch(k, .{ .grid = grid, .block = .{ .x = block }, .shared = smem }, s, a);
}

const Lin = struct { w: u64, b: u64, n: usize, k: usize };
const GPlan = struct { m: u64, n: u64, kk: u64, lin: cuda.cublaslt.Linear };
const Block = struct { norm1: u64, qkv: Lin, qn: u64, kn: u64, proj: Lin, norm2: u64, gate: Lin, up: Lin, down: Lin };

pub const Config = struct {
    hidden: usize = 1024,
    depth: usize = 24,
    heads: usize = 16,
    hd: usize = 64,
    inter: usize = 4096,
    out: usize = 4096,
    minter: usize = 10240,
    eps: f32 = 1e-5,
    limit: f32 = 10.0,
    merge: usize = 2,
};

pub const Vision = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const Driver,
    s: cuda.Stream,
    k: Kernels,
    lt: cuda.cublaslt.Library,
    ws: DeviceBuffer,
    bufs: std.ArrayList(DeviceBuffer),
    blocks: []Block,
    patch: Lin,
    post_ln: u64,
    down: Lin, // [out, 4*hidden] as the exact-matmul form
    m_proj: Lin,
    m_ln_w: u64,
    m_ln_b: u64,
    m_gate: Lin,
    m_up: Lin,
    m_down: Lin,
    c: Config,
    // work (S = 1024)
    x: DeviceBuffer,
    h: DeviceBuffer,
    qkv: DeviceBuffer,
    o: DeviceBuffer,
    g: DeviceBuffer,
    u: DeviceBuffer,
    cos: DeviceBuffer,
    sin: DeviceBuffer,
    groups: DeviceBuffer, // [S/4, 4*hidden]
    feats: DeviceBuffer, // [S/4, out]
    feat_map: DeviceBuffer, // [rows_max] i32: -1 or the feature row (host-built)

    plans: std.ArrayList(GPlan) = .empty, // lazy shape cache

    fn zeros(v: *Vision, n: usize) !u64 {
        const b = try v.gpa.alloc(u8, n * 2);
        defer v.gpa.free(b);
        @memset(b, 0);
        return v.upT(b);
    }
    fn upT(v: *Vision, bytes: []const u8) !u64 {
        const b = try DeviceBuffer.fromHost(v.d, bytes);
        const p = b.ptr;
        try v.bufs.append(v.gpa, b);
        return p;
    }

    fn linT(v: *Vision, ck: *core.checkpoint.Checkpoint, base: []const u8, n: usize, kk: usize) !Lin {
        var nb: [256]u8 = undefined;
        var kb: [256]u8 = undefined;
        const wn = try std.fmt.bufPrint(&nb, "{s}.weight", .{base});
        const bn = try std.fmt.bufPrint(&kb, "{s}.bias", .{base});
        const w = try v.upT((try ck.get(wn)).bytes);
        const b = if (ck.get(bn)) |t| try v.upT(t.bytes) else |_| try v.zeros(n);
        return .{ .w = w, .b = b, .n = n, .k = kk };
    }

    fn wtT(v: *Vision, ck: *core.checkpoint.Checkpoint, name: []const u8) !u64 {
        return v.upT((try ck.get(name)).bytes);
    }

    pub fn deinit(v: *Vision) void {
        for (v.bufs.items) |*b| b.free();
        v.bufs.deinit(v.gpa);
        inline for (.{ "x", "h", "qkv", "o", "g", "u", "cos", "sin", "groups", "feats", "feat_map", "ws" }) |f| @field(v, f).free();
        v.gpa.free(v.blocks);
        v.lt.close();
        v.k.deinit();
        v.s.deinit();
        v.* = undefined;
    }

    pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8) !*Vision {
        const cfgpath = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
        defer gpa.free(cfgpath);
        const text = try std.Io.Dir.cwd().readFileAlloc(io, cfgpath, gpa, .limited(1 << 20));
        defer gpa.free(text);
        const vc = @import("std").json.parseFromSlice(std.json.Value, gpa, text, .{}) catch return error.BadConfig;
        defer vc.deinit();
        const root = vc.value.object.get("vision_config") orelse return error.NoVisionConfig;
        var c: Config = .{};
        if (root.object.get("depth")) |x| c.depth = @intCast(x.integer);
        if (root.object.get("hidden_size")) |x| c.hidden = @intCast(x.integer);
        if (root.object.get("num_heads")) |x| c.heads = @intCast(x.integer);
        c.hd = c.hidden / c.heads;
        if (root.object.get("intermediate_size")) |x| c.inter = @intCast(x.integer);
        if (root.object.get("out_hidden_size")) |x| c.out = @intCast(x.integer);
        if (root.object.get("projection_intermediate_size")) |x| c.minter = @intCast(x.integer);
        if (root.object.get("swiglu_limit")) |x| c.limit = @floatCast(x.float);
        if (root.object.get("rms_norm_eps")) |x| c.eps = @floatCast(x.float);

        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        v.* = .{
            .gpa = gpa, .io = io,
            .d = d,
            .s = try cuda.Stream.init(d, true),
            .k = try Kernels.load(d),
            .lt = try cuda.cublaslt.Library.open(),
            .ws = try DeviceBuffer.alloc(d, 16 << 20),
            .bufs = .empty,
            .blocks = &.{},
            .patch = undefined,
            .post_ln = undefined,
            .down = undefined,
            .m_proj = undefined,
            .m_ln_w = undefined,
            .m_ln_b = undefined,
            .m_gate = undefined,
            .m_up = undefined,
            .m_down = undefined,
            .c = c,
            .x = undefined,
            .h = undefined,
            .qkv = undefined,
            .o = undefined,
            .g = undefined,
            .u = undefined,
            .cos = undefined,
            .sin = undefined,
            .groups = undefined,
            .feats = undefined,
            .feat_map = undefined,
        };
        errdefer {
            for (v.bufs.items) |*b| b.free();
            v.bufs.deinit(gpa);
        }
        var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        const P = "model.visual.";

        v.patch = .{ .w = try v.wtT(&ck, P ++ "patch_embed.proj.weight"), .b = try v.wtT(&ck, P ++ "patch_embed.proj.bias"), .n = c.hidden, .k = 3 * 2 * 14 * 14 };
        v.post_ln = try v.wtT(&ck, P ++ "post_layernorm.weight");
        // downsample [out, hidden, 2, 2] -> exact matmul [out, 4*hidden] in (dy,dx,c) order
        {
            const t = try ck.get(P ++ "downsample.weight");
            const bt = try ck.get(P ++ "downsample.bias");
            if (t.dim(0) != c.out or t.dim(1) != c.hidden or bt.dim(0) != c.out) return error.BadDownsample;
            const src = std.mem.bytesAsSlice(u16, @as([]align(1) const u8, t.bytes));
            const dst = try gpa.alloc(u16, c.out * 4 * c.hidden);
            defer gpa.free(dst);
            // W[o, c, dy, dx] -> M[o, (dy*2+dx)*hidden + c]
            for (0..c.out) |o| for (0..c.hidden) |cc| {
                dst[o * 4 * c.hidden + cc] = src[o * c.hidden * 4 + cc * 4 + 0];
                dst[o * 4 * c.hidden + c.hidden + cc] = src[o * c.hidden * 4 + cc * 4 + 1];
                dst[o * 4 * c.hidden + 2 * c.hidden + cc] = src[o * c.hidden * 4 + cc * 4 + 2];
                dst[o * 4 * c.hidden + 3 * c.hidden + cc] = src[o * c.hidden * 4 + cc * 4 + 3];
            };
            v.down = .{ .w = try v.upT(std.mem.sliceAsBytes(dst)), .b = try v.upT(bt.bytes), .n = c.out, .k = 4 * c.hidden };
        }
        v.m_proj = try v.linT(&ck, P ++ "merger.proj", c.out, c.out);
        v.m_ln_w = try v.wtT(&ck, P ++ "merger.post_projection_norm.weight");
        v.m_ln_b = try v.wtT(&ck, P ++ "merger.post_projection_norm.bias");
        v.m_gate = try v.linT(&ck, P ++ "merger.gate_proj", c.minter, c.out);
        v.m_up = try v.linT(&ck, P ++ "merger.up_proj", c.minter, c.out);
        v.m_down = try v.linT(&ck, P ++ "merger.down_proj", c.out, c.minter);

        const blocks = try gpa.alloc(Block, c.depth);
        for (0..c.depth) |i| {
            var bb: [256]u8 = undefined;
            const g = try std.fmt.bufPrint(&bb, "{s}blocks.{d}.", .{ P, i });
            var n2: [256]u8 = undefined;
            blocks[i] = .{
                .norm1 = try v.wtT(&ck, try std.fmt.bufPrint(&n2, "{s}norm1.weight", .{g})),
                .qkv = try v.linT(&ck, try std.fmt.bufPrint(&n2, "{s}attn.qkv", .{g}), 3 * c.heads * c.hd, c.hidden),
                .qn = try v.wtT(&ck, try std.fmt.bufPrint(&n2, "{s}attn.q_norm.weight", .{g})),
                .kn = try v.wtT(&ck, try std.fmt.bufPrint(&n2, "{s}attn.k_norm.weight", .{g})),
                .proj = try v.linT(&ck, try std.fmt.bufPrint(&n2, "{s}attn.proj", .{g}), c.hidden, c.hidden),
                .norm2 = try v.wtT(&ck, try std.fmt.bufPrint(&n2, "{s}norm2.weight", .{g})),
                .gate = try v.linT(&ck, try std.fmt.bufPrint(&n2, "{s}mlp.gate_proj", .{g}), c.inter, c.hidden),
                .up = try v.linT(&ck, try std.fmt.bufPrint(&n2, "{s}mlp.up_proj", .{g}), c.inter, c.hidden),
                .down = try v.linT(&ck, try std.fmt.bufPrint(&n2, "{s}mlp.down_proj", .{g}), c.hidden, c.inter),
            };
        }
        v.blocks = blocks;

        const S: usize = 32 * 32; // the 448-image patch grid
        v.x = try DeviceBuffer.alloc(d, S * c.hidden * 2);
        v.h = try DeviceBuffer.alloc(d, S * c.hidden * 2);
        v.qkv = try DeviceBuffer.alloc(d, S * 3 * c.heads * c.hd * 2);
        v.o = try DeviceBuffer.alloc(d, S * c.hidden * 2);
        v.g = try DeviceBuffer.alloc(d, S * c.inter * 2);
        v.u = try DeviceBuffer.alloc(d, S * c.inter * 2);
        v.cos = try DeviceBuffer.alloc(d, S * c.hd * 4);
        v.sin = try DeviceBuffer.alloc(d, S * c.hd * 4);
        v.groups = try DeviceBuffer.alloc(d, (S / 4) * 4 * c.hidden * 2);
        v.feats = try DeviceBuffer.alloc(d, (S / 4) * c.out * 2 * 4); // x4: multi-image requests concatenate features here
        v.feat_map = try DeviceBuffer.alloc(d, 24 * 4);
        { // the table path until a request uploads a map: garbage here would route rows to features
            const neg: [24]i32 = @splat(-1);
            try v.feat_map.upload(0, std.mem.sliceAsBytes(&neg));
        }
        return v;
    }



    fn mm(v: *Vision, x: u64, w: u64, out: u64, rows: usize, n: usize, kk: usize) !void {
        // a tiny shape cache: at most ~10 shapes here
        
        for (v.plans.items) |*p| {
            if (p.m == rows and p.n == n and p.kk == kk) {
                try p.lin.run(x, w, out, v.ws.ptr, v.ws.len, v.s.handle);
                return;
            }
        }
        var lin = try cuda.cublaslt.Linear.init(&v.lt, rows, n, kk, .bf16, v.ws.len);
        try lin.run(x, w, out, v.ws.ptr, v.ws.len, v.s.handle);
        try v.plans.append(v.gpa, .{ .m = rows, .n = n, .kk = kk, .lin = lin });
    }

    fn rmsW(v: *Vision, w: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        a.add(v.x.ptr);
        a.add(w);
        a.add(v.h.ptr);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(v.c.hidden)));
        a.add(@as(i32, @intCast(v.c.hidden)));
        a.add(@as(i32, @intCast(v.c.hidden)));
        a.add(v.c.eps);
        try go(v.k.rms_fn, v.s, .{ .x = @intCast(rows), .y = 1, .z = 1 }, 512, 0, &a);
    }

    fn dumpX(v: *Vision, name: []const u8, S: usize) !void {
        try v.s.synchronize();
        const b = try v.gpa.alloc(u8, S * v.c.hidden * 2);
        defer v.gpa.free(b);
        try (DeviceBuffer{ .d = v.d, .ptr = v.x.ptr, .len = b.len, .borrowed = true }).download(0, b);
        var dirv = std.Io.Dir.cwd();
        dirv.writeFile(v.io, .{ .sub_path = name, .data = b }) catch {};
    }
    fn dumpBuf(v: *Vision, name: []const u8, ptr: u64, n: usize) !void {
        try v.s.synchronize();
        const b = try v.gpa.alloc(u8, n);
        defer v.gpa.free(b);
        try (DeviceBuffer{ .d = v.d, .ptr = ptr, .len = n, .borrowed = true }).download(0, b);
        var dirv = std.Io.Dir.cwd();
        dirv.writeFile(v.io, .{ .sub_path = name, .data = b }) catch {};
    }
    /// The block-0 qkv GEMM alone: mm(h, Wqkv) pre-bias -> /tmp/vzig_mm.bin.
    pub fn mmMicro(v: *Vision, h: u64) !void {
        try v.mm(h, v.blocks[0].qkv.w, v.qkv.ptr, 1024, v.blocks[0].qkv.n, v.c.hidden);
        try v.s.synchronize();
        const b = try v.gpa.alloc(u8, 1024 * v.blocks[0].qkv.n * 2);
        defer v.gpa.free(b);
        try v.qkv.download(0, b);
        var dirv = std.Io.Dir.cwd();
        dirv.writeFile(v.io, .{ .sub_path = "/tmp/vzig_mm.bin", .data = b }) catch {};
    }
    /// vv_attn alone on uploaded post-qknr qkv -> /tmp/vzig_attnmicro.bin.

    pub fn attnDbg(v: *Vision, qkv: u64, dbg: u64) !void {
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(qkv + v.c.heads * v.c.hd * 2);
        a.add(qkv + 2 * v.c.heads * v.c.hd * 2);
        a.add(dbg);
        a.add(@as(i32, 1024));
        a.add(@as(i32, @intCast(v.c.heads)));
        a.add(@as(i32, @intCast(v.c.hd)));
        a.add(@as(f32, 0.125));
        try go(v.k.attndbg_fn, v.s, .{ .x = @as(u32, @intCast(1024 * v.c.heads)), .y = 1, .z = 1 }, @as(u32, @intCast(v.c.hd)), @as(u32, @intCast((v.c.hd / 32 * 2) * 4)), &a);
        try v.s.synchronize();
    }
    pub fn attnMicro(v: *Vision, qkv: u64) !void {
        const c = v.c;
        const S: usize = 1024;
        var a: cuda.Args = .{};
        a.add(qkv);
        a.add(@as(i32, 0));
        a.add(@as(i32, @intCast(c.heads * c.hd)));
        a.add(@as(i32, @intCast(2 * c.heads * c.hd)));
        a.add(@as(i32, @intCast(3 * c.heads * c.hd)));
        a.add(v.o.ptr);
        a.add(@as(i32, @intCast(S)));
        a.add(@as(i32, @intCast(c.heads)));
        a.add(@as(i32, @intCast(c.hd)));
        a.add(@as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(c.hd))));
        try go(v.k.attn_fn, v.s, .{ .x = @as(u32, @intCast(S * c.heads)), .y = 1, .z = 1 }, @as(u32, @intCast(c.hd)), @as(u32, @intCast((c.hd / 32 * 2 + c.hd) * 4)), &a);
        try v.s.synchronize();
        const b = try v.gpa.alloc(u8, S * c.hidden * 2);
        defer v.gpa.free(b);
        try v.o.download(0, b);
        var dirv = std.Io.Dir.cwd();
        dirv.writeFile(v.io, .{ .sub_path = "/tmp/vzig_attnmicro.bin", .data = b }) catch {};
    }
    /// The serving embed: rows with a >=0 feature ordinal read the vision
    /// features; the rest the embedding table. feat_map: [rows] i32 (the
    /// host builds it - it knows the image positions). streams in bf16 or
    /// f32 to match the engine's stream dtype.
    pub fn embedRows(v: *Vision, s: cuda.Stream, ids: u64, table: u64, feats: u64, feat_map: u64, rows: usize, d: usize, out: u64, stride: usize, f32_streams: bool) !void {
        var a: cuda.Args = .{};
        a.add(ids);
        a.add(table);
        a.add(feats);
        a.add(feat_map);
        a.add(@as(i32, 154854));
        a.add(out);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(d)));
        a.add(@as(i32, @intCast(stride)));
        const total = rows * d;
        const fn_ = if (f32_streams) v.k.embedf32_fn else v.k.embed_fn;
        try go(fn_, s, .{ .x = @as(u32, @intCast((total + 255) / 256)), .y = 1, .z = 1 }, 256, 0, &a);
    }

    /// Upload the host-built feature map (i32 per prompt row: -1 = table).
    /// MUST be stream-ordered: a null-stream upload races the embed kernel
    /// (the ids-race bug class).
    pub fn uploadMap(v: *Vision, map: []const i32, stream: ?cuda.Stream) !void {
        if (stream) |st| {
            try (DeviceBuffer{ .d = v.d, .ptr = v.feat_map.ptr, .len = map.len * 4, .borrowed = true }).uploadAsync(0, std.mem.sliceAsBytes(map), st.handle);
        } else {
            try (DeviceBuffer{ .d = v.d, .ptr = v.feat_map.ptr, .len = map.len * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(map));
        }
    }

    fn biasW(v: *Vision, y: u64, b: u64, rows: usize, n: usize) !void {
        var a: cuda.Args = .{};
        a.add(y);
        a.add(b);
        a.add(@as(i32, @intCast(rows)));
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(n)));
        try go(v.k.bias_fn, v.s, .{ .x = @as(u32, @intCast(rows)), .y = 1, .z = 1 }, 256, 0, &a);
    }

    /// patches: device [S, 1176] bf16; writes feats [S/4, out] bf16.
    pub fn encode(v: *Vision, patches: u64, gh: usize, gw: usize) !void {
        const S = gh * gw;
        const c = v.c;
        try v.mm(patches, v.patch.w, v.x.ptr, S, c.hidden, 3 * 2 * 14 * 14);
        try v.biasW(v.x.ptr, v.patch.b, S, c.hidden);
        try v.dumpX("/tmp/vzig_x0.bin", S);

        // rope tables in the checkpoint's MERGE-BLOCK-MAJOR row order
        const cb = try v.gpa.alloc(f32, S * c.hd);
        defer v.gpa.free(cb);
        const sb = try v.gpa.alloc(f32, S * c.hd);
        defer v.gpa.free(sb);
        buildRopeMerge(cb, sb, gh, gw, c.hd);
        try (DeviceBuffer{ .d = v.d, .ptr = v.cos.ptr, .len = S * c.hd * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(cb));
        try (DeviceBuffer{ .d = v.d, .ptr = v.sin.ptr, .len = S * c.hd * 4, .borrowed = true }).upload(0, std.mem.sliceAsBytes(sb));

        for (v.blocks, 0..) |*B, bi0| {
            try v.rmsW(B.norm1, S);
            try v.mm(v.h.ptr, B.qkv.w, v.qkv.ptr, S, 3 * c.heads * c.hd, c.hidden);
            try v.biasW(v.qkv.ptr, B.qkv.b, S, 3 * c.heads * c.hd);
            if (bi0 == 0) try v.dumpBuf("/tmp/vzig_qkvin.bin", v.qkv.ptr, S * 3 * c.heads * c.hd * 2);
            {
                var a: cuda.Args = .{};
                a.add(v.qkv.ptr);
                a.add(@as(i32, 0));
                a.add(@as(i32, @intCast(c.heads * c.hd)));
                a.add(B.qn);
                a.add(B.kn);
                a.add(v.cos.ptr);
                a.add(v.sin.ptr);
                a.add(@as(i32, @intCast(S)));
                a.add(@as(i32, @intCast(c.heads)));
                a.add(@as(i32, @intCast(c.hd)));
                a.add(@as(i32, @intCast(3 * c.heads * c.hd)));
                a.add(c.eps);
                try go(v.k.qknr_fn, v.s, .{ .x = @intCast(S * c.heads), .y = 1, .z = 1 }, @intCast(c.hd), 0, &a);
                if (bi0 == 0) try v.dumpBuf("/tmp/vzig_qrot.bin", v.qkv.ptr, S * 3 * v.c.heads * v.c.hd * 2);
            }
            {
                var a: cuda.Args = .{};
                a.add(v.qkv.ptr);
                a.add(@as(i32, 0));
                a.add(@as(i32, @intCast(c.heads * c.hd)));
                a.add(@as(i32, @intCast(2 * c.heads * c.hd)));
                a.add(@as(i32, @intCast(3 * c.heads * c.hd)));
                a.add(v.o.ptr);
                a.add(@as(i32, @intCast(S)));
                a.add(@as(i32, @intCast(c.heads)));
                a.add(@as(i32, @intCast(c.hd)));
                a.add(@as(f32, 1.0) / @sqrt(@as(f32, @floatFromInt(c.hd))));
                try go(v.k.attn_fn, v.s, .{ .x = @as(u32, @intCast(S * c.heads)), .y = 1, .z = 1 }, @as(u32, @intCast(c.hd)), @as(u32, @intCast((c.hd / 32 * 2 + c.hd) * 4)), &a);
            }
            if (bi0 == 0) try v.dumpBuf("/tmp/vzig_attnout.bin", v.o.ptr, S * v.c.hidden * 2);
            try v.mm(v.o.ptr, B.proj.w, v.o.ptr, S, c.hidden, c.hidden);
            try v.biasW(v.o.ptr, B.proj.b, S, c.hidden);
            {
                var a: cuda.Args = .{};
                a.add(v.x.ptr);
                a.add(v.o.ptr);
                a.add(@as(i64, @intCast(S * c.hidden)));
                try go(v.k.add_fn, v.s, .{ .x = @intCast((S * c.hidden + 255) / 256), .y = 1, .z = 1 }, 256, 0, &a);
            }
            if (bi0 == 0 or bi0 == 7 or bi0 == 15 or bi0 == 23) {
                var nmb: [28]u8 = undefined;
                const nmx = std.fmt.bufPrint(&nmb, "/tmp/vzig_xb{d}.bin", .{bi0}) catch return;
                try v.dumpX(nmx, S);
            }
            if (bi0 == 0) {
                try v.dumpBuf("/tmp/vzig_h.bin", v.h.ptr, S * v.c.hidden * 2);

            }
            try v.rmsW(B.norm2, S);
            try v.mm(v.h.ptr, B.gate.w, v.g.ptr, S, c.inter, c.hidden);
            try v.biasW(v.g.ptr, B.gate.b, S, c.inter);
            try v.mm(v.h.ptr, B.up.w, v.u.ptr, S, c.inter, c.hidden);
            try v.biasW(v.u.ptr, B.up.b, S, c.inter);
            {
                var a: cuda.Args = .{};
                a.add(v.g.ptr);
                a.add(v.u.ptr);
                a.add(v.g.ptr);
                a.add(@as(i32, @intCast(S)));
                a.add(@as(i32, 1));
                a.add(@as(i32, @intCast(c.inter)));
                a.add(c.limit);
                try go(v.k.act_fn, v.s, .{ .x = @intCast((S * c.inter + 255) / 256), .y = 1, .z = 1 }, 256, 0, &a);
            }
            try v.mm(v.g.ptr, B.down.w, v.o.ptr, S, c.hidden, c.inter);
            try v.biasW(v.o.ptr, B.down.b, S, c.hidden);
            {
                var a: cuda.Args = .{};
                a.add(v.x.ptr);
                a.add(v.o.ptr);
                a.add(@as(i64, @intCast(S * c.hidden)));
                try go(v.k.add_fn, v.s, .{ .x = @intCast((S * c.hidden + 255) / 256), .y = 1, .z = 1 }, 256, 0, &a);
            }
        }
        // post rms (h = rms(x)) then the merge gather: consecutive 4-row groups
        try v.rmsW(v.post_ln, S);
        {
            var a: cuda.Args = .{};
            a.add(v.h.ptr);
            a.add(v.groups.ptr);
            a.add(@as(i32, @intCast(gh)));
            a.add(@as(i32, @intCast(gw)));
            a.add(@as(i32, @intCast(c.hidden)));
            try go(v.k.gather_fn, v.s, .{ .x = @intCast(S / 4), .y = 1, .z = 1 }, @intCast(c.hidden), 0, &a);
        }
        try v.dumpBuf("/tmp/vzig_hpostln.bin", v.h.ptr, S * c.hidden * 2);
        const G4 = S / 4;
        try v.mm(v.groups.ptr, v.down.w, v.feats.ptr, G4, c.out, 4 * c.hidden);
        try v.biasW(v.feats.ptr, v.down.b, G4, c.out);
        try v.dumpBuf("/tmp/vzig_downout.bin", v.feats.ptr, (S / 4) * c.out * 2);
        // merger: proj -> gelu -> LN -> swiglu -> down  (feats reused in place where safe)
        try v.mm(v.feats.ptr, v.m_proj.w, v.groups.ptr, G4, c.out, c.out); // groups := proj(feats)
        {
            var a: cuda.Args = .{};
            a.add(v.groups.ptr);
            a.add(v.m_ln_w);
            a.add(v.m_ln_b);
            a.add(v.feats.ptr);
            a.add(@as(i32, @intCast(G4)));
            a.add(@as(i32, @intCast(c.out)));
            a.add(@as(i32, @intCast(c.out)));
            a.add(@as(i32, @intCast(c.out)));
            a.add(c.eps);
            try go(v.k.ln_fn, v.s, .{ .x = @intCast(G4), .y = 1, .z = 1 }, 256, 0, &a);
        }
        {
            var a: cuda.Args = .{};
            a.add(v.feats.ptr);
            a.add(@as(i64, @intCast(G4 * c.out)));
            try go(v.k.gelu_fn, v.s, .{ .x = @intCast((G4 * c.out + 255) / 256), .y = 1, .z = 1 }, 256, 0, &a);
        }
        try v.dumpBuf("/tmp/vzig_mln.bin", v.feats.ptr, (S / 4) * c.out * 2);
        try v.mm(v.feats.ptr, v.m_gate.w, v.g.ptr, G4, c.minter, c.out);
        try v.mm(v.feats.ptr, v.m_up.w, v.u.ptr, G4, c.minter, c.out);
        {
            var a: cuda.Args = .{};
            a.add(v.g.ptr);
            a.add(v.u.ptr);
            a.add(v.g.ptr);
            a.add(@as(i32, @intCast(G4)));
            a.add(@as(i32, 1));
            a.add(@as(i32, @intCast(c.minter)));
            a.add(c.limit);
            try go(v.k.act_fn, v.s, .{ .x = @intCast((G4 * c.minter + 255) / 256), .y = 1, .z = 1 }, 256, 0, &a);
        }
        try v.mm(v.g.ptr, v.m_down.w, v.feats.ptr, G4, c.out, c.minter);
    }
};

fn buildRopeMerge(cos: []f32, sin: []f32, gh: usize, gw: usize, hd: usize) void {
    const spatial = hd / 2; // 32
    const m: usize = 2;
    _ = gh / m;
    const og_w = gw / m;
    for (0..gh * gw) |r| {
        const oy = r / (og_w * m * m);
        var rem = r % (og_w * m * m);
        const ox = rem / (m * m);
        rem = rem % (m * m);
        const my = rem / m;
        const mx = rem % m;
        const py = oy * m + my;
        const px = ox * m + mx;
        for (0..spatial / 2) |fi| {
            const theta_i: f32 = 10000.0;
            const freq = 1.0 / std.math.pow(f32, theta_i, @as(f32, @floatFromInt(2 * fi)) / @as(f32, @floatFromInt(spatial)));
            const hv: f32 = @floatFromInt(py);
            const wv: f32 = @floatFromInt(px);
            const a1 = hv * freq;
            const a2 = wv * freq;
            cos[r * hd + fi] = @cos(a1);
            sin[r * hd + fi] = @sin(a1);
            cos[r * hd + spatial / 2 + fi] = @cos(a2);
            sin[r * hd + spatial / 2 + fi] = @sin(a2);
        }
        for (0..spatial) |i| {
            cos[r * hd + spatial + i] = cos[r * hd + i];
            sin[r * hd + spatial + i] = sin[r * hd + i];
        }
    }
}
