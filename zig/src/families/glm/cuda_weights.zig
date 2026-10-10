//! GLM-5.3-Flash's weights on the GPU: the routed projections as EXL3 trellis
//! weights (rate from the trellis shape, never config), everything else bf16 as
//! stored. A load takes a device and a block range [lo, hi) of `layers + 1`
//! blocks (the MTP block trails) — the full model is 146G, past one card, so
//! the box serves it split; block 0's device takes the embedding, the last the
//! head.

const std = @import("std");
const cuda = @import("cuda");
const exl3 = cuda.exl3;
const core = @import("core");
const Config = @import("config.zig").Config;
const Driver = cuda.Driver;

const prefix = "model.language_model.";
const McgMul: u32 = 0xCBAC1FED;

pub const Hc = struct { fnp: u64, base: u64, scale: u64 };

pub const Kda = struct {
    q: u64,
    k: u64,
    v: u64,
    q_conv: u64,
    k_conv: u64,
    v_conv: u64,
    f_a: u64,
    f_b: u64,
    g_a: u64,
    g_b: u64,
    b: u64,
    o: u64,
    o_norm: u64,
    a_log: u64,
    a: u64, // exp(a_log), host-computed fp32
    dt_bias: u64,
};

pub const Mla = struct {
    q_a: u64,
    q_b: u64,
    kv_a: u64,
    kv_b: u64,
    o: u64,
    q_norm: u64,
    kv_norm: u64,
    wq_b: u64, // the indexer's queries
    wk: u64,
    wproj: u64, // indexer.weights_proj
    ape: u64,
    igate: u64, // index_kpool_compress_gate
    k_norm_w: u64,
    k_norm_b: u64,
};

pub const Dense = struct { gate: u64, up: u64, down: u64 };

pub const Moe = struct {
    router: u64, // [experts, hidden] bf16
    bias: u64, // [experts] f32
    gate: []const exl3.Weight,
    up: []const exl3.Weight,
    down: []const exl3.Weight,
    shared_gate: u64,
    shared_up: u64,
    shared_down: u64,
};

pub const Attn = union(enum) { kda: Kda, mla: Mla };
pub const Mlp = union(enum) { dense: Dense, moe: Moe };

pub const Layer = struct {
    in_norm: u64,
    post_norm: u64,
    hc_attn: ?Hc, // absent on the MTP block
    hc_ffn: ?Hc,
    attn: Attn,
    mlp: Mlp,
    eh_proj: u64 = 0, // the MTP block's extras
    enorm: u64 = 0,
    hnorm: u64 = 0,
    shared_head_norm: u64 = 0,
};

pub const Weights = struct {
    gpa: std.mem.Allocator,
    config: Config,
    embed: u64 = 0,
    final_norm: u64 = 0,
    head: u64 = 0,
    blocks_lo: usize = 0,
    blocks_hi: usize = 0,
    layers: []Layer,
    buffers: std.ArrayList(cuda.DeviceBuffer) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.buffers.items) |*b| b.free();
        w.buffers.deinit(w.gpa);
        for (w.layers) |l| {
            switch (l.mlp) {
                .moe => |m| {
                    w.gpa.free(m.gate);
                    w.gpa.free(m.up);
                    w.gpa.free(m.down);
                },
                .dense => {},
            }
        }
        if (w.layers.len > 0) w.gpa.free(w.layers);
        w.* = undefined;
    }

    pub fn moeBlocks(w: *const Weights) usize {
        var n: usize = 0;
        for (w.layers) |l| n += @intFromBool(l.mlp == .moe);
        return n;
    }
};

const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const Driver,
    c: *const Config,
    w: *Weights,
    ck: *core.checkpoint.Checkpoint,
    pbuf: [160]u8 = undefined, // the block prefix, stable across its fields
    nb: [4][192]u8 = undefined, // name scratch by nesting depth; a name lives until the next at its level

    fn pref(L: *Loader, blk: usize) []const u8 {
        return std.fmt.bufPrint(&L.pbuf, "{s}layers.{d}", .{ prefix, blk }) catch unreachable;
    }

    fn n(L: *Loader, depth: usize, comptime f: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&L.nb[depth], f, args) catch unreachable;
    }

    fn buf(L: *Loader, bytes: []const u8) !u64 {
        var b = try cuda.DeviceBuffer.fromHost(L.d, bytes);
        errdefer b.free();
        const ptr = b.ptr;
        try L.w.buffers.append(L.gpa, b);
        return ptr;
    }

    fn bf16(L: *Loader, name: []const u8) !u64 {
        const t = try L.ck.get(name);
        if (t.dtype != .bf16) {
            std.log.err("{s}: expected bf16, got {t}", .{ name, t.dtype });
            return error.UnexpectedTensor;
        }
        return L.buf(t.bytes);
    }

    /// Any dtype as stored.
    fn raw(L: *Loader, name: []const u8) !u64 {
        return L.buf((try L.ck.get(name)).bytes);
    }

    fn f32v(L: *Loader, name: []const u8) !u64 {
        const t = try L.ck.get(name);
        if (t.dtype != .f32) return error.UnexpectedTensor;
        return L.buf(t.bytes);
    }

    /// One routed projection's EXL3 quad; the rate is the trellis's own.
    /// `name` lives at depth 1; the quad's names build at depth 2.
    fn routed(L: *Loader, name: []const u8) !exl3.Weight {
        const trellis = try L.ck.get(L.n(2, "{s}.trellis", .{name}));
        if (trellis.dtype != .i16 or trellis.rank != 3) return error.BadTrellis;
        const words = trellis.dim(2);
        if (words % 8 != 0 or words > 128) return error.BadTrellis;
        const suh = try L.ck.get(L.n(2, "{s}.suh", .{name}));
        const svh = try L.ck.get(L.n(2, "{s}.svh", .{name}));
        const mcg = try L.ck.get(L.n(2, "{s}.mcg", .{name}));
        if (suh.dtype != .f16 or svh.dtype != .f16) return error.BadSigns;
        if (mcg.dtype == .i32 and mcg.dim(0) == 1) {
            var m: u32 = 0;
            @memcpy(std.mem.asBytes(&m), mcg.bytes[0..4]);
            if (m != McgMul) return error.BadCodebook;
        }
        const halves: u32 = @intCast(words / 8);
        const k: u32 = @intCast(trellis.dim(0) * 16);
        const nn: u32 = @intCast(trellis.dim(1) * 16);
        if (suh.dim(0) != k or svh.dim(0) != nn) return error.SignShape;
        return .{
            .trellis = try L.buf(trellis.bytes),
            .suh = try L.buf(suh.bytes),
            .svh = try L.buf(svh.bytes),
            .halves = halves,
            .k = k,
            .n = nn,
        };
    }

    /// The hyper-connection triple as stored (fp32 base/scale, bf16 or fp32 fn).
    fn hc(L: *Loader, base: []const u8) !Hc {
        const t = try L.ck.get(L.n(1, "{s}_fn", .{base}));
        return .{
            .fnp = try L.buf(t.bytes),
            .base = try L.buf((try L.ck.get(L.n(1, "{s}_base", .{base}))).bytes),
            .scale = try L.buf((try L.ck.get(L.n(1, "{s}_scale", .{base}))).bytes),
        };
    }

    /// exp(A_log) elementwise fp32, as the state step reads it; bf16 or f32 as stored.
    fn expA(L: *Loader, name: []const u8) !u64 {
        const t = try L.ck.get(name);
        const count = t.numel();
        const out = try L.gpa.alloc(f32, count);
        defer L.gpa.free(out);
        switch (t.dtype) {
            .bf16 => {
                const src = try L.gpa.alloc(u16, count);
                defer L.gpa.free(src);
                @memcpy(std.mem.sliceAsBytes(src), t.bytes[0 .. count * 2]);
                for (src, out) |bits, *o| o.* = @exp(@as(f32, @floatCast(@as(f16, @bitCast(bits)))));
            },
            .f32 => {
                const src = try L.gpa.alloc(f32, count);
                defer L.gpa.free(src);
                @memcpy(std.mem.sliceAsBytes(src), t.bytes[0 .. count * 4]);
                for (src, out) |v, *o| o.* = @exp(v);
            },
            else => return error.UnexpectedTensor,
        }
        return L.buf(std.mem.sliceAsBytes(out));
    }

    fn kda(L: *Loader, pre: []const u8) !Kda {
        const A = "{s}.self_attn.";
        return .{
            .q = try L.bf16(L.n(0, A ++ "q_proj.weight", .{pre})),
            .k = try L.bf16(L.n(0, A ++ "k_proj.weight", .{pre})),
            .v = try L.bf16(L.n(0, A ++ "v_proj.weight", .{pre})),
            .q_conv = try L.bf16(L.n(0, A ++ "q_conv1d.weight", .{pre})),
            .k_conv = try L.bf16(L.n(0, A ++ "k_conv1d.weight", .{pre})),
            .v_conv = try L.bf16(L.n(0, A ++ "v_conv1d.weight", .{pre})),
            .f_a = try L.bf16(L.n(0, A ++ "f_a_proj.weight", .{pre})),
            .f_b = try L.bf16(L.n(0, A ++ "f_b_proj.weight", .{pre})),
            .g_a = try L.bf16(L.n(0, A ++ "g_a_proj.weight", .{pre})),
            .g_b = try L.bf16(L.n(0, A ++ "g_b_proj.weight", .{pre})),
            .b = try L.bf16(L.n(0, A ++ "b_proj.weight", .{pre})),
            .o = try L.bf16(L.n(0, A ++ "o_proj.weight", .{pre})),
            .o_norm = try L.bf16(L.n(0, A ++ "o_norm.weight", .{pre})),
            .a_log = try L.raw(L.n(0, A ++ "A_log", .{pre})),
            .a = try L.expA(L.n(0, A ++ "A_log", .{pre})),
            .dt_bias = try L.raw(L.n(0, A ++ "dt_bias", .{pre})),
        };
    }

    fn mla(L: *Loader, pre: []const u8) !Mla {
        const A = "{s}.self_attn.";
        const I = A ++ "indexer.";
        return .{
            .q_a = try L.bf16(L.n(0, A ++ "q_a_proj.weight", .{pre})),
            .q_b = try L.bf16(L.n(0, A ++ "q_b_proj.weight", .{pre})),
            .kv_a = try L.bf16(L.n(0, A ++ "kv_a_proj_with_mqa.weight", .{pre})),
            .kv_b = try L.bf16(L.n(0, A ++ "kv_b_proj.weight", .{pre})),
            .o = try L.bf16(L.n(0, A ++ "o_proj.weight", .{pre})),
            .q_norm = try L.bf16(L.n(0, A ++ "q_a_layernorm.weight", .{pre})),
            .kv_norm = try L.bf16(L.n(0, A ++ "kv_a_layernorm.weight", .{pre})),
            .wq_b = try L.bf16(L.n(0, I ++ "wq_b.weight", .{pre})),
            .wk = try L.bf16(L.n(0, I ++ "wk.weight", .{pre})),
            .wproj = try L.bf16(L.n(0, I ++ "weights_proj.weight", .{pre})),
            .ape = try L.bf16(L.n(0, I ++ "index_kpool_compress_ape", .{pre})),
            .igate = try L.bf16(L.n(0, I ++ "index_kpool_compress_gate", .{pre})),
            .k_norm_w = try L.bf16(L.n(0, I ++ "k_norm.weight", .{pre})),
            .k_norm_b = try L.bf16(L.n(0, I ++ "k_norm.bias", .{pre})),
        };
    }

    fn mlp(L: *Loader, blk: usize, pre: []const u8) !Mlp {
        const P = "{s}.mlp.";
        if (blk < L.c.dense_layers) {
            return .{ .dense = .{
                .gate = try L.bf16(L.n(0, P ++ "gate_proj.weight", .{pre})),
                .up = try L.bf16(L.n(0, P ++ "up_proj.weight", .{pre})),
                .down = try L.bf16(L.n(0, P ++ "down_proj.weight", .{pre})),
            } };
        }
        const e = L.c.experts;
        const gate = try L.gpa.alloc(exl3.Weight, e);
        errdefer L.gpa.free(gate);
        const up = try L.gpa.alloc(exl3.Weight, e);
        errdefer L.gpa.free(up);
        const down = try L.gpa.alloc(exl3.Weight, e);
        errdefer L.gpa.free(down);
        for (0..e) |ei| {
            const epre = L.n(1, "{s}.mlp.experts.{d}.", .{ pre, ei });
            // depth 0 held the last leaf; the expert prefix at 1, quads at 2
            gate[ei] = try L.routed(L.n(0, "{s}gate_proj", .{epre}));
            up[ei] = try L.routed(L.n(0, "{s}up_proj", .{epre}));
            down[ei] = try L.routed(L.n(0, "{s}down_proj", .{epre}));
        }
        return .{ .moe = .{
            .router = try L.bf16(L.n(0, P ++ "gate.weight", .{pre})),
            .bias = try L.f32v(L.n(0, P ++ "gate.e_score_correction_bias", .{pre})),
            .gate = gate,
            .up = up,
            .down = down,
            .shared_gate = try L.bf16(L.n(0, P ++ "shared_experts.gate_proj.weight", .{pre})),
            .shared_up = try L.bf16(L.n(0, P ++ "shared_experts.up_proj.weight", .{pre})),
            .shared_down = try L.bf16(L.n(0, P ++ "shared_experts.down_proj.weight", .{pre})),
        } };
    }

    fn block(L: *Loader, bi: usize) !Layer {
        const pre = L.pref(bi);
        const mtp = bi == L.c.layers; // the MTP block trails the backbone
        var l: Layer = .{
            .in_norm = try L.bf16(L.n(0, "{s}.input_layernorm.weight", .{pre})),
            .post_norm = try L.bf16(L.n(0, "{s}.post_attention_layernorm.weight", .{pre})),
            .hc_attn = null,
            .hc_ffn = null,
            .attn = undefined,
            .mlp = undefined,
        };
        if (!mtp) {
            l.hc_attn = try L.hc(L.n(0, "{s}.hc_attn", .{pre}));
            l.hc_ffn = try L.hc(L.n(0, "{s}.hc_ffn", .{pre}));
        } else {
            l.eh_proj = try L.bf16(L.n(0, "{s}.eh_proj.weight", .{pre}));
            l.enorm = try L.bf16(L.n(0, "{s}.enorm.weight", .{pre}));
            l.hnorm = try L.bf16(L.n(0, "{s}.hnorm.weight", .{pre}));
            l.shared_head_norm = try L.bf16(L.n(0, "{s}.shared_head.norm.weight", .{pre}));
        }
        l.attn = if (L.c.kind(@intCast(bi)) == .kda) .{ .kda = try L.kda(pre) } else .{ .mla = try L.mla(pre) };
        l.mlp = try L.mlp(bi, pre);
        return l;
    }
};

/// A half-open range of the `layers + 1` blocks; `hi` past the end clamps.
pub const Range = struct { lo: usize = 0, hi: usize = 1 << 16 };

/// Loads the embedding (with block 0), final norm and head (with the last), and
/// blocks [lo, hi) onto the calling context's device.
pub fn load(gpa: std.mem.Allocator, io: std.Io, d: *const Driver, dir: []const u8, range: Range) !*Weights {
    const path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const c = try @import("config.zig").parse(gpa, text);

    const w = try gpa.create(Weights);
    errdefer {
        w.deinit();
        gpa.destroy(w);
    }
    w.* = .{ .gpa = gpa, .config = c, .layers = &.{} };

    var ck = try core.checkpoint.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    var L: Loader = .{ .gpa = gpa, .d = d, .c = &w.config, .w = w, .ck = &ck };

    const blocks = c.layers + 1;
    const lo = @min(range.lo, blocks);
    const hi = @min(@max(range.hi, lo + 1), blocks);
    w.blocks_lo = lo;
    w.blocks_hi = hi;

    if (lo == 0) {
        const t = try ck.get(prefix ++ "embed_tokens.weight");
        if (t.dim(0) != c.vocab or t.dim(1) != c.hidden) return error.BadEmbed;
        w.embed = try L.bf16(prefix ++ "embed_tokens.weight");
    }
    if (hi == blocks) {
        w.final_norm = try L.bf16(prefix ++ "norm.weight");
        const ht = try ck.get("lm_head.weight");
        if (ht.dim(1) != c.hidden) return error.BadHead;
        w.head = try L.bf16("lm_head.weight");
    }

    const layers = try gpa.alloc(Layer, hi - lo);
    for (lo..hi) |bi| layers[bi - lo] = try L.block(bi);
    w.layers = layers;
    return w;
}
