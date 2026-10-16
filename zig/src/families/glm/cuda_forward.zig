//! GLM-5.3-Flash's TP forward on CUDA (the Python family's forward.compute and commit), partials over NCCL or copied.

const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const weights = @import("cuda_weights.zig");
const glue = @import("cuda_glue.zig");
const kda = @import("cuda_kda.zig");
const b16 = @import("cuda_b16.zig");
const latent = @import("cuda_latent.zig");
const sparse = @import("cuda_sparse.zig");
const experts = @import("cuda_experts.zig");
const kda_block = @import("cuda_kda_block.zig");
const dsa_block = @import("cuda_dsa_block.zig");
const mlp = @import("cuda_mlp.zig");
const moe = @import("cuda_moe.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");
const state = @import("cuda_state.zig");
const shared = @import("cuda_shared.zig");

pub const prefill_rows = state.prefill_rows;
pub const window_rows = state.window_rows;
pub const State = state.State;
pub const Buffers = state.Buffers;

/// The modules a forward launches from: our fatbins.
pub const Modules = struct {
    glue: glue.Kernels,
    mods: [7]cuda.Module,
    kda: kda.Kernels,
    lane: cuda.qmmf.Lane, // the NVFP4 linears' decode rows (qlinear)
    prompt: cuda.nvfp4.Prompt, // and their prompt rows
    exl3_mod: cuda.Module,
    exl3: cuda.exl3_linear.Kernels, // EXL3 checkpoints' projections (qlinear)
    experts: experts.Kernels,
    copy_rows: cuda.Function,
    put_slot: cuda.Function,
    pack_dense: cuda.Function,
    quant: quant.Kernels,
    q4: q4.Kernels,

    pub fn load(d: *const cuda.Driver, ctx: *const cuda.Context) !Modules {
        const k = cuda.kernels;
        var m: Modules = undefined;
        var n: usize = 0;
        errdefer for (m.mods[0..n]) |*x| x.unload();
        for (&m.mods, [_][]const u8{ k.glm_kda, k.glm_ops, k.experts, k.affine4_pack, k.glm_quant, k.qmm_group, k.glm_glue }) |*x, image| {
            x.* = try cuda.Module.load(d, image);
            n += 1;
        }
        m.lane = try cuda.qmmf.Lane.load(d, @intCast(try ctx.attribute(.compute_capability_major)));
        errdefer m.lane.unload();
        m.prompt = try cuda.nvfp4.Prompt.load(d);
        errdefer m.prompt.unload();
        m.exl3_mod = try cuda.Module.load(d, k.exl3);
        errdefer m.exl3_mod.unload();
        m.exl3 = try cuda.exl3_linear.Kernels.resolve(d, m.exl3_mod, 16384);
        errdefer m.exl3.deinit();
        m.kda = try kda.Kernels.load(m.mods[0]);
        m.experts = try experts.Kernels.load(d, m.mods[2], @intCast(try ctx.attribute(.multiprocessor_count)));
        errdefer m.experts.unload();
        m.copy_rows = try m.mods[1].function("tf_glm_copy_rows");
        m.put_slot = try m.mods[1].function("tf_glm_put_slot");
        m.pack_dense = try m.mods[3].function("tf_pack_dense");
        m.quant = try quant.Kernels.load(m.mods[4]);
        m.q4 = try q4.Kernels.load(m.mods[5], m.mods[3], ctx);
        m.glue = try glue.Kernels.load(m.mods[6]);
        return m;
    }

    /// What weights.load runs: the dense packing, and for the MTP head the quantizers and the 4-bit tiling.
    pub fn tools(m: *const Modules) weights.Tools {
        return .{ .pack_dense = m.pack_dense, .quant = &m.quant, .q4 = &m.q4 };
    }

    pub fn ops(m: *const Modules) cuda.Module {
        return m.mods[1];
    }

    pub fn deinit(m: *Modules) void {
        for (&m.mods) |*x| x.unload();
        m.lane.unload();
        m.prompt.unload();
        m.exl3.deinit();
        m.exl3_mod.unload();
        m.experts.unload();
        m.* = undefined;
    }
};

/// overlap.ROW_STEP and PIECES_MAX: piece boundaries on the BF16 prompt matmul's 128-row blocks, at most 16 pieces.
pub const row_step = 128;
pub const max_pieces = 16;

/// overlap.ranges: [lo, hi) row pieces of a chunk of `rows`, about equal on row_step boundaries (fewer when short).
pub fn ranges(rows: usize, pieces: usize, out: *[max_pieces][2]usize) usize {
    const per = (rows + pieces - 1) / pieces;
    const size = (per + row_step - 1) / row_step * row_step;
    var n: usize = 0;
    var lo: usize = 0;
    while (lo < rows) : (lo += size) {
        out[n] = .{ lo, @min(rows, lo + size) };
        n += 1;
    }
    return n;
}

/// A prompt chunk's pieced exchanges (overlap.Overlap): second stream and events; `count`: TF_GLM_OVERLAP_PIECES (3).
pub const Pieces = struct {
    count: usize = 4,
    side: cuda.Stream,
    filled: [max_pieces]cuda.Event,
    gathered: [max_pieces]cuda.Event,
    glues: [max_pieces]cuda.Event, // split: a piece's own rows glued
    shared: [max_pieces]cuda.Event, // split: every rank's glued rows of a piece in

    pub fn init(d: *const cuda.Driver, count: usize) !Pieces {
        if (count == 0 or count > max_pieces) return error.BadPieces;
        var p: Pieces = .{ .count = count, .side = try cuda.Stream.init(d, false), .filled = undefined, .gathered = undefined, .glues = undefined, .shared = undefined };
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.* = try cuda.Event.init(d, false);
            b.* = try cuda.Event.init(d, false);
            g.* = try cuda.Event.init(d, false);
            s.* = try cuda.Event.init(d, false);
        }
        return p;
    }

    pub fn deinit(p: *Pieces) void {
        for (&p.filled, &p.gathered, &p.glues, &p.shared) |*a, *b, *g, *s| {
            a.deinit();
            b.deinit();
            g.deinit();
            s.deinit();
        }
        p.side.deinit();
        p.* = undefined;
    }
};

/// How the ranks' partials meet: every rank in-process copying into the others' slots, or one rank over NCCL.
pub const Exchange = union(enum) {
    local,
    nccl: struct { lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm },
    /// liumorrisclaw's host-staged one-shot gather (fork branch shm-allgather): one /dev/shm
    /// segment both ranks mmap and cudaHostRegister(Mapped); decode-sized payloads take the
    /// kernel, larger ones stay on NCCL
    shm: Shm,
};

/// one stream's max window rows (7 drafts + 1): shm only activates ABOVE this (multi-stream)
const rows_one_stream: usize = 8 * 4096;

pub const Shm = struct {
    base: u64, // the device view of the shared, host-registered arena
    seq: u64, // per-block generation counters (device u32[4], the kernel advances them)
    gather_fn: cuda.Function,
    rank: u32,
    slot_units: usize, // one rank's stage slot in fp32 units
    lib: *const cuda.nccl.Library, // the NCCL fallback above the slot
    comm: cuda.nccl.Comm,
};

/// The decode wavefront's shared pieces (docs/WAVEFRONT.md): the group-exchange scratch (two
/// [2][rows][D] fp32 regions), group B's pos word, the side stream, and the four events
/// choreographing mixer-done / exchange-done per group.
pub const Wave = struct {
    scratch: u64,
    pos2: u64,
    split: usize, // TF_GLM_WAVE_SPLIT: group A's rows (0: R/2); R skips group B entirely (bisect)
    sync: bool = false, // TF_GLM_WAVE_SYNC=1: gathers on the main stream, no events (bisect the dance)
    preserve: u64, // the round's pre-round KDA ring half, saved for commit's partial-keep replay
    preserve_len: usize,
    conv: u64, // group B's conv taps: [kda layers][2 rows] refilled from A's staging each layer
    side: cuda.Stream,
    ev: [4]cuda.Event,
};

/// One rank of the forward: its weights, its sequence's caches, the window's buffers.
pub const Rank = struct {
    w: *const weights.Weights,
    st: *State,
    b: *Buffers,
    /// A shared window (Buffers.initShared): each stream's rows in row order; empty: every row on `st`.
    segs: []const Seg = &.{},
};

/// One stream's rows of a shared window: rows row0..row0 + rows, on sequence `st` from its position.
pub const Seg = struct { st: *State, row0: usize, rows: usize };

pub const Forward = struct {
    d: *const cuda.Driver,
    m: *const Modules,
    stream: cuda.Stream,
    ranks: []const Rank, // every rank of a local check, or this process's one
    exchange: Exchange,
    kda_wide: bool = true, // TF_GLM_KDA_DECODE_WIDE (the Python family's default): decode windows on KDA's three-kernel chain
    wave: ?*const Wave = null, // TF_GLM_WAVE=1: decode windows run the two-group exchange wavefront
    pieces: ?*const Pieces = null, // a prompt chunk's exchanges in row pieces on a second stream (null: whole)
    split: bool = true, // TF_GLM_PREFILL_REDUCE=split (the default with send/recv): each rank glues its own rows

    pub fn set(f: *const Forward) glue.Set {
        return .{ .glue = &f.m.glue, .linear = f.linear(), .copy_rows = f.m.copy_rows };
    }

    /// qlinear over every format a GLM checkpoint brings: NVFP4 (lane and prompt GEMM) and EXL3.
    pub fn linear(f: *const Forward) cuda.qlinear.Linear {
        return .{ .lane = &f.m.lane, .nvfp4_prompt = &f.m.prompt, .exl3 = &f.m.exl3 };
    }

    /// reduce.sums' gather: every rank's partial of `rows` rows into every rank's gathered buffer, rank order.
    fn gather(f: *const Forward, rows: usize) !void {
        const c = &f.ranks[0].w.config;
        const bytes = rows * c.hidden * 4;
        switch (f.exchange) {
            .local => for (f.ranks) |to| for (f.ranks) |from| {
                if (to.w.share.rank == from.w.share.rank) continue;
                const off = from.w.share.rank * bytes;
                try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(to.b.gathered + off, from.b.gathered + off, bytes, f.stream.handle), "cuMemcpyDtoDAsync");
            },
            .nccl => |n| {
                const r = f.ranks[0];
                const count = rows * c.hidden;
                try n.lib.check(n.lib.api.ncclAllGather(r.b.gathered + r.w.share.rank * bytes, r.b.gathered, count, .f32, n.comm, f.stream.handle), "ncclAllGather");
            },
            .shm => |sh| {
                const r = f.ranks[0];
                const count = rows * c.hidden;
                if (count <= sh.slot_units) { // any decode-sized payload: the old solo-corruption was the split-exchange bug (fixed), not this kernel
                    // multi-stream only: the engine's single-stream shm path corrupts (deterministic;
                    // the standalone kernel + graph test passes, so it's the engine integration)
                    var a: cuda.Args = .{};
                    a.add(r.b.gathered + r.w.share.rank * bytes);
                    a.add(r.b.gathered);
                    a.add(sh.base);
                    a.add(sh.seq);
                    a.add(@as(c_int, @intCast(sh.rank)));
                    a.add(@as(i64, @intCast(count)));
                    a.add(@as(i64, @intCast(sh.slot_units)));
                    a.add(@as(c_longlong, 2_000_000_000));
                    try cuda.launch.launch(sh.gather_fn, .{ .grid = .{ .x = 4 }, .block = .{ .x = 256 } }, f.stream, &a);
                } else {
                    try sh.lib.check(sh.lib.api.ncclAllGather(r.b.gathered + r.w.share.rank * bytes, r.b.gathered, count, .f32, sh.comm, f.stream.handle), "ncclAllGather");
                }
            },
        }
    }

    fn slot(r: Rank, rows: usize) u64 {
        return r.b.gathered + r.w.share.rank * rows * r.w.config.hidden * 4;
    }

    /// forward._pre: hc_pre of every row at a site into b.normed, xs, post and comb.
    fn pre(f: *const Forward, r: Rank, hc: weights.Hc, norm: u64, rows: usize) !void {
        const c = &r.w.config;
        const b = r.b;
        try glue.hcPre(f.set(), f.stream, b.x, hc.fn_w, hc.base, hc.scale, norm, b.normed, b.xs, b.post, b.comb, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, rows, c.hidden, c.hc);
    }

    fn post(f: *const Forward, r: Rank, rows: usize) !void {
        const c = &r.w.config;
        try glue.hcPost(f.set(), f.stream, r.b.x, r.b.x, r.b.gathered, r.b.post, r.b.comb, rows, c.hidden, c.hc, r.w.share.world);
    }

    /// forward._mixer: the layer's KDA or DSA block on b.normed; `li`/`di`: its index among loaded KDA / DSA layers.
    fn mixer(f: *const Forward, r: Rank, l: weights.Layer, li: usize, di: usize, rows: usize, nch: usize, out: ?u64) !void {
        const c = &r.w.config;
        const b = r.b;
        const st = r.st;
        switch (l.mixer) {
            .kda => |k| {
                const cur = st.cur[li];
                var s = b.kda;
                s.p = if (b.prefill) b.kproj else st.proj + li * st.proj_layer * 2;
                s.saves.state_out = st.recAt(1 - cur, li);
                if (!b.prefill) { // a window's saved rows stay with the layer until its commit
                    const kv = li * window_rows * st.heads * kda.dk;
                    s.saves.k = st.save_k + kv * 4;
                    s.saves.v = st.save_v + kv * 2;
                    s.saves.g = st.save_g + kv * 4;
                    s.saves.b = st.save_b + li * window_rows * st.heads * 4;
                }
                const conv = st.conv + li * 3 * st.conv_c * 2;
                try kda_block.body(f.set(), &f.m.kda, f.stream, k, s, b.normed, c.hidden, rows, conv, st.recAt(cur, li), !b.prefill and f.kda_wide, b.prefill);
                if (out) |o| try kda_block.fill(f.set(), f.stream, k, s, 0, rows, o);
                if (b.prefill) st.cur[li] = 1 - cur; // a prompt chunk keeps every row: the layer commits now
            },
            .mla => |a| {
                try dsa_block.body(.{ .set = f.set(), .ops = f.m.ops() }, f.stream, a, st.caches[di], b.dsa, b.normed, c.hidden, rows, st.pos_dev, st.pos, nch, b.part_rows);
                if (out) |o| try dsa_block.fill(f.set(), f.stream, a, b.dsa, 0, rows, o);
            },
        }
    }

    /// forward._ffn: the dense MLP or MoE block on b.normed into this rank's slot.
    fn ffn(f: *const Forward, r: Rank, l: weights.Layer, rows: usize, out: ?u64) !void {
        const c = &r.w.config;
        const b = r.b;
        switch (l.ffn) {
            .dense => |m| {
                try mlp.body(f.mlpKernels(), f.stream, m, b.dense, b.normed, c.hidden, rows, b.prefill);
                if (out) |o| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, 0, rows, b.prefill, o);
            },
            .moe => |m| {
                try moe.body(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, rows, b.prefill);
                if (out) |o| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, 0, rows, b.prefill, o);
            },
        }
    }

    fn mlpKernels(f: *const Forward) mlp.Kernels {
        return .{ .set = f.set(), .linear = f.linear(), .copy_rows = f.m.copy_rows };
    }

    fn moeKernels(f: *const Forward) moe.Kernels {
        return .{ .set = f.set(), .experts = &f.m.experts, .put_slot = f.m.put_slot };
    }

    const Site = enum { mixer, ffn };

    /// forward.out_proj's fill of rows lo..hi at a site of layer `l` into `out` (this rank's rows of a piece).
    fn fillRows(f: *const Forward, r: Rank, l: weights.Layer, site: Site, lo: usize, hi: usize, out: u64) !void {
        const c = &r.w.config;
        const b = r.b;
        switch (site) {
            .mixer => switch (l.mixer) {
                .kda => |k| try kda_block.fill(f.set(), f.stream, k, b.kda, lo, hi, out),
                .mla => |a| try dsa_block.fill(f.set(), f.stream, a, b.dsa, lo, hi, out),
            },
            .ffn => switch (l.ffn) {
                .dense => |m| try mlp.fill(f.mlpKernels(), f.stream, m, b.dense, lo, hi, b.prefill, out),
                .moe => |m| try moe.fill(f.moeKernels(), f.stream, m, b.moe, b.normed, c.hidden, lo, hi, b.prefill, out),
            },
        }
    }

    /// Piece [lo, hi)'s room in a rank's gathered buffer (pieces never share bytes): [world][hi - lo][dims] fp32.
    fn room(r: Rank, lo: usize) u64 {
        return r.b.gathered + r.w.share.world * lo * r.w.config.hidden * 4;
    }

    /// reduce.shares: [lo, hi) of the rows of a piece of `n` each rank owns: about equal, the first n % world one more.
    fn share(n: usize, world: usize, rank: usize) [2]usize {
        const base = n / world;
        const extra = n % world;
        const lo = rank * base + @min(rank, extra);
        return .{ lo, lo + base + @intFromBool(rank < extra) };
    }

    /// Whether piece [lo, lo + n) splits (TF_GLM_PREFILL_REDUCE=split): at least one row for every rank.
    fn splits(f: *const Forward, n: usize) bool {
        return f.split and n >= f.ranks[0].w.share.world;
    }

    fn copy(f: *const Forward, stream: cuda.Stream, dst: u64, src: u64, bytes: usize) !void {
        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, stream.handle), "cuMemcpyDtoDAsync");
    }

    /// overlap.Overlap.partials: each piece filled on the main stream, exchanged on the second (gathered or split).
    fn exchangePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
        const D = f.ranks[0].w.config.hidden;
        const world = f.ranks[0].w.share.world;
        for (cut, 0..) |piece, q| {
            const lo = piece[0];
            const n = piece[1] - lo;
            const split = f.splits(n);
            for (f.ranks) |r| try f.fillRows(r, r.w.layers[j], site, lo, piece[1], if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4);
            try p.filled[q].record(f.stream);
            try p.side.wait(p.filled[q]);
            const bytes = n * D * 4;
            if (!split) switch (f.exchange) {
                .shm => |x| {
                    const r = f.ranks[0];
                    try x.lib.check(x.lib.api.ncclAllGather(room(r, lo) + r.w.share.rank * bytes, room(r, lo), n * D, .f32, x.comm, p.side.handle), "ncclAllGather");
                },
                .local => for (f.ranks) |to| for (f.ranks) |from| {
                    if (to.w.share.rank == from.w.share.rank) continue;
                    const off = from.w.share.rank * bytes;
                    try f.copy(p.side, room(to, lo) + off, room(from, lo) + off, bytes);
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    try x.lib.check(x.lib.api.ncclAllGather(room(r, lo) + r.w.share.rank * bytes, room(r, lo), n * D, .f32, x.comm, p.side.handle), "ncclAllGather");
                },
            } else switch (f.exchange) {
                .shm => |x| { // the split exchange is send/recv halves, never an allgather (the allgather
                    // here corrupted every pieced prefill: rank halves never crossed)
                    const r = f.ranks[0];
                    const me = r.w.share.rank;
                    const mine = share(n, world, me);
                    const own = mine[1] - mine[0];
                    try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                    for (0..world) |peer| if (peer != me) {
                        const theirs = share(n, world, peer);
                        try x.lib.check(x.lib.api.ncclSend(r.b.part + (lo + theirs[0]) * D * 4, (theirs[1] - theirs[0]) * D, .f32, @intCast(peer), x.comm, p.side.handle), "ncclSend");
                        try x.lib.check(x.lib.api.ncclRecv(room(r, lo) + peer * own * D * 4, own * D, .f32, @intCast(peer), x.comm, p.side.handle), "ncclRecv");
                    };
                    try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
                    try f.copy(p.side, room(r, lo) + me * own * D * 4, r.b.part + (lo + mine[0]) * D * 4, own * D * 4);
                },
                .local => for (f.ranks) |to| {
                    const mine = share(n, world, to.w.share.rank);
                    const own = mine[1] - mine[0];
                    for (f.ranks) |from| try f.copy(p.side, room(to, lo) + from.w.share.rank * own * D * 4, from.b.part + (lo + mine[0]) * D * 4, own * D * 4);
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    const me = r.w.share.rank;
                    const mine = share(n, world, me);
                    const own = mine[1] - mine[0];
                    try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                    for (0..world) |peer| if (peer != me) {
                        const theirs = share(n, world, peer);
                        try x.lib.check(x.lib.api.ncclSend(r.b.part + (lo + theirs[0]) * D * 4, (theirs[1] - theirs[0]) * D, .f32, @intCast(peer), x.comm, p.side.handle), "ncclSend");
                        try x.lib.check(x.lib.api.ncclRecv(room(r, lo) + peer * own * D * 4, own * D, .f32, @intCast(peer), x.comm, p.side.handle), "ncclRecv");
                    };
                    try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
                    try f.copy(p.side, room(r, lo) + me * own * D * 4, r.b.part + (lo + mine[0]) * D * 4, own * D * 4);
                },
            }
            try p.gathered[q].record(p.side);
        }
    }

    /// hc_post of rows lo..hi from partials `g` [world][hi - lo][D] (a gathered room or a split one's own rows).
    fn postRows(f: *const Forward, r: Rank, lo: usize, hi: usize, g: u64) !void {
        const c = &r.w.config;
        const b = r.b;
        const S: usize = c.hc;
        const D: usize = c.hidden;
        const x = b.x + lo * S * D * 2;
        try glue.hcPost(f.set(), f.stream, x, x, g, b.post + lo * S * 4, b.comb + lo * S * S * 4, hi - lo, D, S, r.w.share.world);
    }

    /// forward._pre of rows lo..hi.
    fn preRows(f: *const Forward, r: Rank, hc: weights.Hc, norm: u64, lo: usize, hi: usize) !void {
        const c = &r.w.config;
        const b = r.b;
        const S: usize = c.hc;
        const D: usize = c.hidden;
        try glue.hcPre(f.set(), f.stream, b.x + lo * S * D * 2, hc.fn_w, hc.base, hc.scale, norm, b.normed + lo * D * 2, b.xs + lo * (D / 64) * 4, b.post + lo * S * 4, b.comb + lo * S * S * 4, b.hcpart, c.eps, c.hc_eps, c.sinkhorn, hi - lo, D, S);
    }

    /// One piece's glue after a site: hc_post, then the next layer's hc_pre or (after the last layer) the stream mean.
    fn glueRows(f: *const Forward, r: Rank, j: usize, site: Site, lo: usize, hi: usize, g: u64) !void {
        const c = &r.w.config;
        const layers = r.w.layers;
        try f.postRows(r, lo, hi, g);
        if (site == .mixer) return f.preRows(r, layers[j].ffn_hc, layers[j].post_norm, lo, hi);
        if (j + 1 < layers.len) return f.preRows(r, layers[j + 1].attn_hc, layers[j + 1].in_norm, lo, hi);
        try glue.streamMean(f.set(), f.stream, r.b.x + lo * c.hc * c.hidden * 2, r.b.hidden + lo * c.hidden * 2, hi - lo, c.hidden, c.hc);
    }

    /// overlap.Overlap.glue: each piece's glue once its exchange is in, its rows then shared on the second stream.
    fn gluePieces(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
        const c = &f.ranks[0].w.config;
        const D: usize = c.hidden;
        const world = f.ranks[0].w.share.world;
        const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
        var sent: usize = 0;
        for (cut, 0..) |piece, q| {
            const lo = piece[0];
            const n = piece[1] - lo;
            try f.stream.wait(p.gathered[q]);
            if (!f.splits(n)) {
                for (f.ranks) |r| try f.glueRows(r, j, site, lo, piece[1], room(r, lo));
                continue;
            }
            for (f.ranks) |r| {
                const mine = share(n, world, r.w.share.rank);
                try f.glueRows(r, j, site, lo + mine[0], lo + mine[1], room(r, lo));
            }
            try p.glues[q].record(f.stream);
            try p.side.wait(p.glues[q]);
            switch (f.exchange) {
                .shm => |x| { // send/recv halves as .nccl below: the old allgather here dropped the
                    // peer's span entirely, corrupting every pieced prefill
                    const r = f.ranks[0];
                    const me = r.w.share.rank;
                    const rows_of = if (last) r.b.hidden else r.b.normed;
                    const mine = share(n, world, me);
                    try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                    for (0..world) |peer| if (peer != me) {
                        const theirs = share(n, world, peer);
                        try x.lib.check(x.lib.api.ncclSend(rows_of + (lo + mine[0]) * D * 2, (mine[1] - mine[0]) * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclSend");
                        try x.lib.check(x.lib.api.ncclRecv(rows_of + (lo + theirs[0]) * D * 2, (theirs[1] - theirs[0]) * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclRecv");
                    };
                    try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
                },
                .local => for (f.ranks) |from| {
                    const mine = share(n, world, from.w.share.rank);
                    const bytes = (mine[1] - mine[0]) * D * 2;
                    for (f.ranks) |to| if (to.w.share.rank != from.w.share.rank) {
                        const at = (lo + mine[0]) * D * 2;
                        try f.copy(p.side, (if (last) to.b.hidden else to.b.normed) + at, (if (last) from.b.hidden else from.b.normed) + at, bytes);
                    };
                },
                .nccl => |x| {
                    const r = f.ranks[0];
                    const me = r.w.share.rank;
                    const rows_of = if (last) r.b.hidden else r.b.normed;
                    const mine = share(n, world, me);
                    try x.lib.check(x.lib.api.ncclGroupStart(), "ncclGroupStart");
                    for (0..world) |peer| if (peer != me) {
                        const theirs = share(n, world, peer);
                        try x.lib.check(x.lib.api.ncclSend(rows_of + (lo + mine[0]) * D * 2, (mine[1] - mine[0]) * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclSend");
                        try x.lib.check(x.lib.api.ncclRecv(rows_of + (lo + theirs[0]) * D * 2, (theirs[1] - theirs[0]) * D, .bf16, @intCast(peer), x.comm, p.side.handle), "ncclRecv");
                    };
                    try x.lib.check(x.lib.api.ncclGroupEnd(), "ncclGroupEnd");
                },
            }
            try p.shared[q].record(p.side);
            sent |= @as(usize, 1) << @intCast(q);
        }
        for (0..cut.len) |q| if (sent >> @intCast(q) & 1 == 1) try f.stream.wait(p.shared[q]);
    }

    /// forward.pieced_layers: a prompt chunk's layers with exchanges in row pieces; same kernels and bits as unpieced.
    fn pieced(f: *const Forward, p: *const Pieces, R: usize, cut: []const [2]usize) !void {
        const layers = f.ranks[0].w.layers;
        for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
        var li: usize = 0;
        var di: usize = 0;
        for (layers, 0..) |_, j| {
            for (f.ranks) |r| try f.mixer(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null);
            try f.exchangePieces(p, j, .mixer, cut);
            try f.gluePieces(p, j, .mixer, cut);
            for (f.ranks) |r| try f.ffn(r, r.w.layers[j], R, null);
            try f.exchangePieces(p, j, .ffn, cut);
            try f.gluePieces(p, j, .ffn, cut);
            switch (layers[j].mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
        }
    }

    /// forward.forward of `ids`: logits in b.logits; a prompt chunk's KDA layers committed, a window's at `commit`.
    pub fn run(f: *const Forward, ids: []const i32) !void {
        const R = ids.len;
        for (f.ranks) |r| {
            if (R == 0 or R > r.b.rows) return error.BadWindow;
            if (r.segs.len == 0) {
                if (r.st.pos + R > r.st.capacity) return error.PastCapacity;
                r.st.pending = R;
                r.st.prompt = r.b.prefill;
            } else try shared.begin(r, R);
            try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(r.b.ids, ids.ptr, R * 4, f.stream.handle), "cuMemcpyHtoDAsync");
        }
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
        var cut: [max_pieces][2]usize = undefined;
        const pieces = if (f.pieces) |p| (if (f.ranks[0].b.prefill) ranges(R, p.count, &cut) else 0) else 0;
        f.ranks[0].st.last_wave = false;
        if (std.c.getenv("TF_GLM_WAVE_DUMP") != null and !f.ranks[0].b.prefill) dumpBytesHost(f, @ptrCast(ids.ptr), R * 4, "ids");
        if (f.wave != null and pieces == 0 and !f.ranks[0].b.prefill and f.ranks[0].segs.len == 0 and R >= 4) {
            try f.wholeWave(f.wave.?, R);
        } else if (pieces > 1 or (pieces == 1 and f.splits(R))) try f.pieced(f.pieces.?, R, cut[0..pieces]) else try f.whole(R);
        try f.finish(R);
        if (std.c.getenv("TF_GLM_WAVE_DUMP") != null and !f.ranks[0].b.prefill) {
            dumpBytes(f, f.ranks[0].b.logits, R * f.ranks[0].w.share.vocab.len() * 2, "logits");
            dumpBytes(f, f.ranks[0].b.fnormed, R * c.hidden * 2, "fnormed");
        }
    }

    /// The two-group decode wavefront (docs/WAVEFRONT.md): clones of the rank with offset
    /// pointers carry each group's rows, so every kernel runs unchanged. Group A's exchange
    /// overlaps group B's mixer compute on the side stream; B's exchange overlaps A's post.
    /// The ffn site stays whole in v1. The KDA ring recurrence stays ordered by launching
    /// A before B on the main stream.
    fn wholeWave(f: *const Forward, w: *const Wave, R: usize) !void {
        if (std.c.getenv("TF_GLM_WAVE_QUIET") == null) std.log.info("wave round: {d} rows", .{R});
        const c = &f.ranks[0].w.config;
        const r0 = f.ranks[0];
        const D = c.hidden;
        var a = R / 2;
        if (w.split != 0) a = @min(w.split, R);
        const br = R - a;
        // a memset, never an async copy from a stack word: the copy could outlive this frame
        if (R - a < R) {
            const p2v: u32 = @bitCast(@as(i32, @intCast(r0.st.pos + a)));
            try f.d.check(f.d.api.cuMemsetD32Async(w.pos2, p2v, 1, f.stream.handle), "Memset pos2");
        } // A-only bisect: no group B, so pos2 stays untouched - the entry ops are then no-ops
        // the wave's second launch overwrites BOTH ring halves: keep the pre-round half for the
        // partial-keep replay commit may need (rejections are common). A single-group wave writes
        // the ring exactly like eager, so the preserve and the wave parity only apply to two groups.
        if (R - a < R) {
            const half_bytes = r0.st.kda_layers * r0.st.rec_layer;
            if (half_bytes > w.preserve_len) {
                std.log.err("wave: preserve buffer {d} < half {d} (layers {d}, rec_layer {d})", .{ w.preserve_len, half_bytes, r0.st.kda_layers, r0.st.rec_layer });
                return error.WavePreserveTooSmall;
            }
            try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(w.preserve, r0.st.recAt(r0.st.cur[0], 0), half_bytes, f.stream.handle), "preserve ring");
            r0.st.wave_preserve = w.preserve;
        }
        var flipped: [128]u1 = undefined;
        for (r0.st.cur, 0..) |v, i| { if (i < flipped.len) flipped[i] = 1 - v; }
        const nch_a = @min(latent.chunksFor(r0.st.pos + a), r0.b.nch);
        const nch_b = @min(latent.chunksFor(r0.st.pos + a + br), r0.b.nch);
        // the partials are [nch][rows][mla heads][lw]: the row stride is the HEAD count (the
        // rank's mla span), not the mla LAYER count - caches.len() is the layer count
        const mh = r0.w.share.mla.len();
        const part_b = a * nch_b * mh * c.kv_lora * 4; // bytes: group B's partials past A's
        var bA = r0.b.*;
        if (R - a < R) bA.gathered = w.scratch; // a lone full-window group keeps the real gathered:
        // bisect - if A-only still differs from eager with the real buffer, the bug is outside wholeWave
        var bB = groupBuffers(f, r0.b, a, w.scratch + 2 * a * D * 4, part_b, a * nch_b * mh * 4);
        var stA = r0.st.*;
        var stB = groupState(r0.st, a, w.pos2, &flipped);
        if (br > 0) stB.conv = w.conv; // B's prep reads its first taps here, not the pre-window state
        var ranks_a = [1]Rank{.{ .w = r0.w, .st = &stA, .b = &bA }};
        var ranks_b = [1]Rank{.{ .w = r0.w, .st = &stB, .b = &bB }};
        var fA = f.*;
        fA.ranks = &ranks_a;
        fA.pieces = null;
        fA.wave = null;
        var fB = f.*;
        fB.ranks = &ranks_b;
        fB.pieces = null;
        fB.wave = null;
        var li: usize = 0;
        var di: usize = 0;
        if (std.c.getenv("TF_GLM_WAVE_DUMP") != null) {
            dump_layer = 0;
            dump_rounds += 1;
            std.log.info("LAYERDUMP round {d} wave R={d} a={d} br={d}", .{ dump_rounds, R, a, br });
        }
        for (r0.w.layers, 0..) |_, j| {
            const l = r0.w.layers[j];
            try fA.pre(ranks_a[0], l.attn_hc, l.in_norm, a);
            if (a > 0) try fA.mixer(ranks_a[0], l, li, di, a, nch_a, slot(ranks_a[0], a));
            if (br > 0 and a >= 2) {
                switch (l.mixer) {
                // B's prep sources its first TAPS-1 conv taps from the staging rows just before it
                // (window rows a-2, a-1 - A's rows); its conv-state slots hold the PRE-WINDOW rows
                .kda => {
                    const taps = 2; // TAPS - 1
                    const c_dim = 3 * r0.st.heads * 128; // C: q|k|v bf16 a row
                    const src = r0.st.proj + li * r0.st.proj_layer * 2 + (a - taps) * r0.st.proj_row * 2;
                    const dst = w.conv + li * 3 * r0.st.conv_c * 2;
                    var t_i: usize = 0;
                    while (t_i < taps) : (t_i += 1) {
                        try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(dst + t_i * c_dim * 2, src + t_i * r0.st.proj_row * 2, c_dim * 2, f.stream.handle), "wave taps");
                    }
                },
                else => {},
                }
            }
            if (std.c.getenv("TF_GLM_WAVE_DUMP") != null and dump_rounds == 1 and j == 0 and a > 0) kdaDump(f, &ranks_a[0], li, "A");
            if (w.sync) {
                // bisect: identical ops to the danced path, but ordered on one stream - if this
                // matches eager the bug is the event choreography; if not, the scratch or clones
                if (a > 0) {
                    try waveGather(f.exchange, f.stream, bA.gathered + r0.w.share.rank * a * D * 4, bA.gathered, a * D);
                    try fA.post(ranks_a[0], a);
                }
                if (br > 0) {
                    try fB.pre(ranks_b[0], l.attn_hc, l.in_norm, br);
                    try fB.mixer(ranks_b[0], l, li, di, br, nch_b, slot(ranks_b[0], br));
                    try waveGather(f.exchange, f.stream, w.scratch + 2 * a * D * 4 + r0.w.share.rank * br * D * 4, w.scratch + 2 * a * D * 4, br * D);
                    try fB.post(ranks_b[0], br);
                }
            } else {
                try w.ev[0].record(f.stream);
                if (br > 0) {
                    try fB.pre(ranks_b[0], l.attn_hc, l.in_norm, br);
                    try fB.mixer(ranks_b[0], l, li, di, br, nch_b, slot(ranks_b[0], br));
                }
                try w.ev[2].record(f.stream);
                try w.side.wait(w.ev[0]);
                if (a > 0) {
                    try waveGather(f.exchange, w.side, bA.gathered + r0.w.share.rank * a * D * 4, bA.gathered, a * D);
                    try w.ev[1].record(w.side);
                }
                try w.side.wait(w.ev[2]);
                if (br > 0) {
                    try waveGather(f.exchange, w.side, w.scratch + 2 * a * D * 4 + r0.w.share.rank * br * D * 4, w.scratch + 2 * a * D * 4, br * D);
                    try w.ev[3].record(w.side);
                }
                if (a > 0) {
                    try f.stream.wait(w.ev[1]);
                    try fA.post(ranks_a[0], a);
                }
                if (br > 0) {
                    try f.stream.wait(w.ev[3]);
                    try fB.post(ranks_b[0], br);
                }
            }
            if (std.c.getenv("TF_GLM_WAVE_DUMP") != null and dump_rounds == 1 and j == 0 and br > 0) kdaDump(f, &ranks_b[0], li, "B");
            try f.pre(r0, l.ffn_hc, l.post_norm, R);
            try f.ffn(r0, l, R, slot(r0, R));
            try f.gather(R);
            try f.post(r0, R);
            switch (l.mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
            if (std.c.getenv("TF_GLM_WAVE_DUMP") != null and dump_rounds <= 60) {
                dump(f, r0.b.x, R, "x");
                dump_layer += 1;
            }
        }
        try glue.streamMean(f.set(), f.stream, r0.b.x, r0.b.hidden, R, c.hidden, c.hc);
        r0.st.last_wave = a < R; // both groups ran; a lone full-window group (a == R) writes the ring like eager
        if (std.c.getenv("TF_GLM_WAVE_DUMP") != null)
            std.log.info("LAYERDUMP waveend r{d} R={d} a={d} lw={} stptr={x}", .{ dump_rounds, R, a, r0.st.last_wave, @intFromPtr(r0.st) });
    }

    /// TF_GLM_WAVE_DUMP=1: xxhash of rows x [0..rows) of bf16 [rows][S*D] at `ptr` (a layer's
    /// output or a gather site), logged per layer for the first few rounds. Diff the eager and
    /// wave tables: the first differing (round, layer) names the diverging site.
    var dump_rounds: usize = 0;
    var dump_layer: usize = 0;
    fn dump(f: *const Forward, ptr: u64, rows: usize, what: []const u8) void {
        if (std.c.getenv("TF_GLM_WAVE_DUMP") == null) return;
        if (dump_rounds >= 60) return;
        const bytes = rows * 4096 * 4 * 2; // S=4 hc streams x D bf16
        const host = std.c.malloc(bytes) orelse return;
        defer std.c.free(host);
        const st2 = f.d.api.cuMemcpyDtoHAsync_v2(host, ptr, bytes, f.stream.handle);
        if (st2 != 0) return;
        f.stream.synchronize() catch {};
        const h = std.hash.XxHash64.hash(0, @as([*]const u8, @ptrCast(host))[0..bytes]);
        std.log.info("LAYERDUMP r{d} l{d: >2} {s} {x: >16}", .{ dump_rounds, dump_layer, what, h });
    }

    fn dumpBytesHost(f: *const Forward, host_ptr: [*]const u8, bytes: usize, what: []const u8) void {
        _ = f;
        if (std.c.getenv("TF_GLM_WAVE_DUMP") == null) return;
        if (dump_rounds > 60) return;
        const h = std.hash.XxHash64.hash(0, host_ptr[0..bytes]);
        std.log.info("LAYERDUMP r{d} -- {s} {x: >16}", .{ dump_rounds + 1, what, h });
    }

    fn dumpBytes(f: *const Forward, dev: u64, bytes: usize, what: []const u8) void {
        if (std.c.getenv("TF_GLM_WAVE_DUMP") == null) return;
        if (dump_rounds > 60) return;
        const host = std.c.malloc(bytes) orelse return;
        defer std.c.free(host);
        if (f.d.api.cuMemcpyDtoHAsync_v2(host, dev, bytes, f.stream.handle) != 0) return;
        f.stream.synchronize() catch {};
        const h = std.hash.XxHash64.hash(0, @as([*]const u8, @ptrCast(host))[0..bytes]);
        std.log.info("LAYERDUMP r{d} -- {s} {x: >16}", .{ dump_rounds, what, h });
    }

    /// Layer-0 KDA fingerprint after a group's mixer: saves, conv state, both ring halves.
    fn kdaDump(f: *const Forward, r: *const Rank, li: usize, tag: []const u8) void {
        const st = r.st;
        const h = st.heads;
        const dk = 128; // kda.dk
        const kv = li * window_rows * h * dk;
        var buf: [16]u8 = undefined;
        const t = std.fmt.bufPrint(&buf, "kda{s}", .{tag}) catch return;
        dumpBytes(f, st.save_k + kv * 4, window_rows * h * dk * 4, t);
        dumpBytes(f, st.save_v + kv * 2, window_rows * h * dk * 2, t);
        dumpBytes(f, st.save_g + kv * 4, window_rows * h * dk * 4, t);
        dumpBytes(f, st.save_b + li * window_rows * h * 4, window_rows * h * 4, t);
        dumpBytes(f, st.conv + li * 3 * st.conv_c * 2, 3 * st.conv_c * 2, t);
        dumpBytes(f, st.recAt(0, li), st.rec_layer, t);
        dumpBytes(f, st.recAt(1, li), st.rec_layer, t);
        std.log.info("LAYERDUMP kdastate {s} li={d} cur={} rec_layer={d}", .{ tag, li, st.cur[li], st.rec_layer });
    }

    fn waveGather(ex: Exchange, stream: cuda.Stream, send: u64, recv: u64, count: usize) !void {
        switch (ex) {
            .local => return error.WaveNeedsExchange,
            .nccl => |x| try x.lib.check(x.lib.api.ncclAllGather(send, recv, count, .f32, x.comm, stream.handle), "waveGather"),
            .shm => |x| try x.lib.check(x.lib.api.ncclAllGather(send, recv, count, .f32, x.comm, stream.handle), "waveGather"),
        }
    }

    /// A copy of `src`'s Buffers whose row-sized pointers advance `off` rows (group B);
    /// `gathered` redirects to the group's exchange scratch, the partials past `part_off` bytes.
    fn groupBuffers(f: *const Forward, src: *const Buffers, off: usize, gathered: u64, part_off: usize, part_off_m: usize) Buffers {
        const c = &f.ranks[0].w.config;
        const s = &f.ranks[0].w.share;
        const D = c.hidden;
        const S = c.hc;
        const n = s.kda.len() * c.kda_dim;
        var mla_heads: usize = 0;
        var mla_qk: usize = 0;
        var mla_dv: usize = 0;
        for (f.ranks[0].w.layers) |l| switch (l.mixer) {
            .mla => |m| {
                mla_heads = m.heads;
                mla_qk = m.qk;
                mla_dv = m.dv;
                break;
            },
            else => {},
        };
        var b = src.*;
        b.x = src.x + off * S * D * 2;
        b.normed = src.normed + off * D * 2;
        b.xs = src.xs + off * (D / 64) * 4;
        b.post = src.post + off * S * 4;
        b.comb = src.comb + off * S * S * 4;
        b.hcpart = src.hcpart + off * glue.hcPartBytes(1);
        b.gathered = gathered;
        b.kda.ka = src.kda.ka + off * n * 2;
        b.kda.kg = src.kda.kg + off * n * 2;
        b.kda.saves.out = src.kda.saves.out + off * n * 2;
        b.kda.wide.q = src.kda.wide.q + off * n * 4;
        b.kda.wide.y = src.kda.wide.y + off * n * 2;
        const lw = c.kv_lora;
        const q_lora = c.q_lora;
        b.dsa.dp = src.dsa.dp + off * (q_lora + lw) * 2;
        b.dsa.qr = src.dsa.qr + off * q_lora * 2;
        b.dsa.xs_qr = src.dsa.xs_qr + off * (q_lora / 64) * 4;
        b.dsa.lat = src.dsa.lat + off * lw * 2;
        b.dsa.xs_lat = src.dsa.xs_lat + off * (lw / 64) * 4;
        b.dsa.q = src.dsa.q + off * mla_heads * mla_qk * 2;
        b.dsa.ikr = src.dsa.ikr + off * (c.i_dim + c.i_heads) * 2;
        b.dsa.igr = src.dsa.igr + off * c.i_dim * 4;
        b.dsa.router_part = src.dsa.router_part + off * @max(c.experts, c.i_dim) * 4; // the row stride is the expert count (288), not i_dim
        b.dsa.qa = src.dsa.qa + off * mla_heads * lw * 2;
        b.dsa.ol = src.dsa.ol + off * mla_heads * lw * 2;
        b.dsa.vn = src.dsa.vn + off * mla_heads * mla_dv * 2;
        b.dsa.qi = src.dsa.qi + off * c.i_heads * c.i_dim * 2;
        b.dsa.select.scores = src.dsa.select.scores + off * (f.ranks[0].st.capacity / sparse.pool) * 4;
        b.dsa.select.ls = src.dsa.select.ls + off * sparse.select_list * 4;
        b.dsa.select.li = src.dsa.select.li + off * sparse.select_list * 4;
        b.dsa.select.pools = src.dsa.select.pools + off * sparse.topk_pools * 8;
        b.dsa.tokens = src.dsa.tokens + off * sparse.width * 4;
        b.dsa.counts = src.dsa.counts + off * 4;
        b.dsa.partials.po = src.dsa.partials.po + part_off;
        b.dsa.partials.pm = src.dsa.partials.pm + part_off_m;
        b.dsa.partials.pl = src.dsa.partials.pl + part_off_m;
        return b;
    }

    /// A copy of `src`'s State for group B: positions shift `off` rows, the pos word swaps to
    /// `pos2`, the KDA save/proj staging advances, and cur reads a flipped copy.
    fn groupState(src: *const State, off: usize, pos2: u64, flipped: []const u1) State {
        var st = src.*;
        const row = st.heads * kda.dk;
        st.save_k = src.save_k + off * row * 4;
        st.save_v = src.save_v + off * row * 2;
        st.save_g = src.save_g + off * row * 4;
        st.save_b = src.save_b + off * st.heads * 4;
        st.proj = src.proj + off * st.proj_row * 2;
        st.pos = src.pos + off;
        st.pos_dev = pos2;
        st.cur = @constCast(flipped);
        return st;
    }

    /// The unpieced loop over the layers (and the stream mean): each site's partials whole, then the glue.
    fn whole(f: *const Forward, R: usize) !void {
        if (std.c.getenv("TF_GLM_WAVE_DUMP") != null) {
            dump_layer = 0;
            dump_rounds += 1;
            std.log.info("LAYERDUMP round {d} eager R={d}", .{ dump_rounds, R });
        }
        const c = &f.ranks[0].w.config;
        var li: usize = 0;
        var di: usize = 0;
        for (f.ranks[0].w.layers, 0..) |_, j| {
            for (f.ranks) |r| {
                const l = r.w.layers[j];
                try f.pre(r, l.attn_hc, l.in_norm, R);
                if (r.segs.len > 0) try shared.mixer(f, r, l, li, di, R, slot(r, R)) else try f.mixer(r, l, li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), slot(r, R));
                if (std.c.getenv("TF_GLM_WAVE_DUMP") != null and dump_rounds == 1 and j == 0) kdaDump(f, &r, li, "E");
            }
            try f.gather(R);
            for (f.ranks) |r| {
                const l = r.w.layers[j];
                try f.post(r, R);
                try f.pre(r, l.ffn_hc, l.post_norm, R);
                try f.ffn(r, l, R, slot(r, R));
            }
            try f.gather(R);
            for (f.ranks) |r| try f.post(r, R);
            switch (f.ranks[0].w.layers[j].mixer) {
                .kda => li += 1,
                .mla => di += 1,
            }
            if (std.c.getenv("TF_GLM_WAVE_DUMP") != null and dump_rounds <= 60) {
                dump(f, f.ranks[0].b.x, R, "x");
                dump_layer += 1;
            }
        }
        for (f.ranks) |r| try glue.streamMean(f.set(), f.stream, r.b.x, r.b.hidden, R, c.hidden, c.hc);
    }

    /// The final norm and this rank's rows of the head (a prompt buffer's last row only).
    fn finish(f: *const Forward, R: usize) !void {
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| {
            const b = r.b;
            try glue.rmsnorm(f.set(), f.stream, b.hidden, c.hidden, r.w.norm, b.fnormed, c.hidden, b.fxs, c.eps, R, c.hidden);
            const head_rows: usize = if (b.prefill) 1 else R;
            const first = if (b.prefill) R - 1 else 0;
            try b16.proj(f.set(), f.stream, b.fnormed + first * c.hidden * 2, c.hidden, r.w.head, b.logits, b.kda.mm_part, head_rows, r.w.share.vocab.len(), c.hidden, false);
        }
    }

    /// mtp.mtp_forward: caches from mtp_len, stream in b.mx, a window's draft logits in b.logits (prompt: no head).
    pub fn mtp(f: *const Forward, ids: []const i32, hidden: []const u64) !void {
        const n = ids.len;
        const c = &f.ranks[0].w.config;
        const D: usize = c.hidden;
        for (f.ranks, hidden) |r, h| {
            if (r.w.mtp == null) return error.NoMtpHead;
            if (n == 0 or n > r.b.rows) return error.BadWindow;
            if (r.st.mtp_len + n > r.st.capacity) return error.PastCapacity;
            try f.d.check(f.d.api.cuMemcpyHtoDAsync_v2(r.b.ids, ids.ptr, n * 4, f.stream.handle), "cuMemcpyHtoDAsync");
            if (std.c.getenv("TF_GLM_WAVE_DUMP") != null) dumpBytes(f, h, n * D * 2, "mtp_src");
            if (h != r.b.hin) try f.d.check(f.d.api.cuMemcpyDtoDAsync_v2(r.b.hin, h, n * D * 2, f.stream.handle), "cuMemcpyDtoDAsync");
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.embedB16(f.set(), f.stream, b.ids, r.w.embed, b.me, n, D, 1);
            if (r.st.mtp_len == 0) try f.d.check(f.d.api.cuMemsetD8Async(b.me, 0, D * 2, f.stream.handle), "cuMemsetD8Async"); // position 0's embedding is zeroed
            try glue.rmsnorm(f.set(), f.stream, b.me, D, m.enorm, b.mcat, 2 * D, null, c.eps, n, D);
            try glue.rmsnorm(f.set(), f.stream, b.hin, D, m.hnorm, b.mcat + D * 2, 2 * D, null, c.eps, n, D);
            try b16.proj(f.set(), f.stream, b.mcat, 2 * D, m.eh, b.mx, b.kda.mm_part, n, D, 2 * D, false);
            try glue.rmsnorm(f.set(), f.stream, b.mx, D, m.layer.in_norm, b.normed, D, b.xs, c.eps, n, D);
            const nch = @min(latent.chunksFor(r.st.mtp_len + n), b.nch);
            try dsa_block.block(.{ .set = f.set(), .ops = f.m.ops() }, f.stream, m.layer.mixer.mla, r.st.mtp_caches, b.dsa, b.normed, D, n, r.st.mtp_pos_dev, r.st.mtp_len, nch, b.part_rows, slot(r, n));
        }
        try f.gather(n);
        for (f.ranks) |r| {
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.residualAdd(f.set(), f.stream, b.mx, b.mx, b.gathered, n, D, r.w.share.world);
            try glue.rmsnorm(f.set(), f.stream, b.mx, D, m.layer.post_norm, b.normed, D, b.xs, c.eps, n, D);
            try f.ffn(r, m.layer, n, slot(r, n));
        }
        try f.gather(n);
        for (f.ranks) |r| {
            const m = r.w.mtp.?;
            const b = r.b;
            try glue.residualAdd(f.set(), f.stream, b.mx, b.mx, b.gathered, n, D, r.w.share.world);
            try glue.rmsnorm(f.set(), f.stream, b.mx + (n - 1) * D * 2, D, m.norm, b.fnormed, D, b.fxs, c.eps, 1, D);
            if (b.prefill) continue;
            if (m.draft) |dq| try q4.matmul(&f.m.q4, f.stream, b.fnormed, D, b.fxs, dq, b.logits, 1) else try b16.proj(f.set(), f.stream, b.fnormed, D, r.w.head, b.logits, b.kda.mm_part, 1, r.w.share.vocab.len(), D, false);
            if (std.c.getenv("TF_GLM_WAVE_DUMP") != null) {
                dumpBytes(f, b.hin, n * D * 2, "mtp_hin");
                dumpBytes(f, b.logits, r.w.share.vocab.len() * 2, "mtp_logits");
            }
        }
    }

    /// forward.commit: keep the first `keep` rows; window KDA layers take new states (replayed if fewer), conv shifted.
    pub fn commit(f: *const Forward, keep: usize) !void {
        for (f.ranks) |r| {
            const st = r.st;
            if (std.c.getenv("TF_GLM_WAVE_DUMP") != null)
                std.log.info("LAYERDUMP commit keep={d} pending={d} last_wave={} pos={d} stptr={x}", .{ keep, st.pending, st.last_wave, st.pos, @intFromPtr(st) });
            if (keep == 0 or keep > st.pending or (st.prompt and keep != st.pending)) return error.BadCommit;
            if (!st.prompt and st.kda_layers > 0) {
                const cur = st.cur[0];
                for (st.cur) |x| if (x != cur) return error.KdaLayersOutOfStep;
                // the wave's two-launch ping-pong leaves the newest ring half at cur; eager at 1-cur
                const new_half: u1 = if (st.last_wave) cur else @as(u1, 1) - cur;
                if (keep < st.pending) {
                    const h = st.heads;
                    var a: cuda.Args = .{};
                    a.add(@as(c_int, @intCast(h)));
                    a.add(if (st.last_wave) st.wave_preserve else st.recAt(cur, 0));
                    a.add(@as(usize, st.rec_layer / 4));
                    for ([_]u64{ st.save_k, st.save_v, st.save_g, st.save_b }) |x| a.add(x);
                    a.add(@as(usize, window_rows * h * kda.dk));
                    a.add(@as(usize, window_rows * h));
                    a.add(@as(c_int, @intCast(keep)));
                    a.add(st.recAt(new_half, 0));
                    try cuda.launch.launch(f.m.kda.replay_layers, .{ .grid = .{ .x = @intCast(st.kda_layers * h) }, .block = .{ .x = 1024 } }, f.stream, &a);
                }
                if (!st.last_wave) { for (st.cur) |*x| x.* = 1 - x.*; }
                if (st.kda_layers > 0) try glue.convShift(f.set(), f.stream, st.conv, st.proj, keep, st.proj_layer, st.proj_row, st.kda_layers, st.conv_c);
            }
            st.pending = 0;
            try st.setPos(f.stream, st.pos + keep);
        }
    }
};
