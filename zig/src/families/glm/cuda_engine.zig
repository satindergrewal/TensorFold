//! One TP rank's GLM-5.3-Flash CUDA engine (the Python family's decode.Engine); all ranks draw the same tokens.

const std = @import("std");
const cuda = @import("cuda");
const latent = @import("cuda_latent.zig");
const lanes = @import("lanes");
const config = @import("config.zig");
const shard = @import("cuda_shard.zig");
const weights = @import("cuda_weights.zig");
const fw = @import("cuda_forward.zig");
const ranks = @import("cuda_ranks.zig");
const sampler = @import("cuda_sampler.zig");
const sparse = @import("cuda_sparse.zig");
const heat = @import("heat");

pub const Options = struct {
    capacity: usize, // tokens a sequence holds
    mtp: bool, // load the MTP head (drafts)
    settings: ranks.Settings,
};

/// Rank 0's own word on its request (cancelled or not); the other ranks have none and hear rank 0's.
pub const Cancel = struct { ptr: ?*anyopaque, check: *const fn (?*anyopaque) bool };

/// The most drafts a chain holds (decode.DepthPolicy's top depth).
/// The draft chain depth: round time barely grows with window rows (weights dominate), so
/// deeper chains amortize the weight reads. 7 default (was 3).
pub const max_drafts = 7;

/// A prompt pass's chunk rows (decode.prefill's).
pub const prompt_rows = fw.prefill_rows;

/// A shared verify's max rows and streams: under hc_pre's 64-row block change, 8 streams of 4 rows.
pub const shared_rows = 128; // 8+ streams x 8+ rows: the cap was splitting multi-stream windows
pub const max_shared = 8;

/// One stream's window of a shared verify: its slot, its ids, and its drawn tokens.
pub const Part = struct { slot: u32, ids: []const u32, out: []u32 };

/// A kept prompt state as the prompt cache holds it: the id every rank keeps its own copy under, and its position.
pub const Kept = struct { id: u64, at: u32 };

/// A stream's own sequence: state, sampling rule, MTP drafts; buffers are shared, one stream's round at a time.
const Seq = struct {
    st: fw.State,
    sampling: ?sampler.Sampling = null, // null: greedy
    in_head: usize = 0, // drafted rows the MTP caches hold past the kept ones (st.mtp_drafted)
    rows_at: u64 = 0, // its last verify's final-norm rows (bf16 [rows][hidden]), which the next draft reads
};

const GraphKey = struct { slot: u32, rows: u8, parity: u8, bucket: u8 = 0 };
const VerifyGraph = struct { exec: cuda.graph.Exec, nch: usize };

/// One rank's copy of a kept prompt state: the state arena's bytes and the host fields beside them.
const Snap = struct { buf: cuda.DeviceBuffer, pos: usize, mtp_len: usize, cur: []u1 };

/// Where a prompt pass resumes: a kept state (every rank holds `id`) at `at`, a multiple of the chunk rows.
pub const Resume = struct { id: u64, at: usize };

/// Rank 0's marks in a prompt pass: `call` may keep the state at a chunk start; other ranks copy at that chunk's gate.
pub const Marks = struct { at: []const u32, ctx: *anyopaque, call: *const fn (ctx: *anyopaque, at: u32) void };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    c: config.Config,
    share: shard.Share,
    spans: []shard.Span, // every rank's vocabulary span
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    shm: ?fw.Shm = null, // liumorrisclaw's host-staged gather arena (TF_GLM_SHM_GATHER=1)
    shm_seq: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    settings: ranks.Settings,
    m: fw.Modules,
    stream: cuda.Stream,
    pieces: fw.Pieces,
    w: weights.Weights,
    bp: fw.Buffers, // prompt chunks
    bd: fw.Buffers, // verify windows
    bm: fw.Buffers, // the MTP head's rows
    bs: ?fw.Buffers = null, // shared verify windows (with the MTP head, as the lane core shares only then)
    hold: cuda.DeviceBuffer, // the prompt's last row's hidden (decode's e.last_hidden), the first draft's source
    pairs: cuda.DeviceBuffer, // every rank's (value, id) of a greedy draw (the send slices)
    pairs_all: cuda.DeviceBuffer, // the all-gather's destination: a separate buffer, out-of-place on every rank
    smp: sampler.Sampler,
    seqs: std.ArrayList(?*Seq) = .empty, // every stream's sequence by slot (the lane backend's), slot 0 made at init
    cur: *Seq = undefined, // the bound slot's: every step below runs on it
    // Solo verify rounds as CUDA graphs (TF_GLM_NO_GRAPHS=1 keeps them eager): keyed by slot and
    // row count, recaptured when the dense attention's chunk count grows past what a graph covers
    // (a replay at a LARGER nch over-covers safely: the extra slices write -inf partials the merge
    // skips, and pos is read on the device). ids_stage is the stable host source whose bytes the
    // captured H2D memcpy re-reads at every replay.
    graphs: std.AutoHashMapUnmanaged(GraphKey, VerifyGraph) = .empty,
    // TF_GLM_WAVE=1: the decode wavefront's scratch, pos2 word and events (docs/WAVEFRONT.md)
    wave: ?fw.Wave = null,
    wave_scratch: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    wave_pos2: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    wave_keep: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    wave_conv: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    wave_ev: [4]cuda.Event = undefined,
    l2pf: ?fw.L2pf = null, // TF_GLM_L2PF=1: the decode-round weight prefetcher
    l2pf_table: cuda.DeviceBuffer = .{ .d = undefined, .ptr = 0, .len = 0, .borrowed = true },
    l2pf_sites: std.ArrayList([2]u32) = .empty,
    l2pf_side: cuda.Stream = .{ .d = undefined, .handle = null },
    l2pf_ev: cuda.Event = undefined,
    ids_stage: cuda.HostBuffer = undefined, // pinned: a captured H2D from pageable memory snapshots
    // the driver staging at capture time (every replay would copy the first round's ids)
    slot: u32 = 0,
    capacity: usize,
    mtp: bool,
    heat_gate: heat.Gate, // upstream's bands (off unless set), read before each prompt chunk on the gathered maximum
    shared_buf: cuda.DeviceBuffer, // one f64 every rank all-reduces to its maximum
    snaps: std.AutoHashMapUnmanaged(u64, Snap) = .empty, // kept prompt states by id (the same ids on every rank)
    next_snap: u64 = 1, // rank 0: the next id
    announce: u64 = 0, // rank 0: a state kept at this chunk's mark, for the other ranks to copy at its gate
    dropped: std.ArrayList(u32) = .empty, // rank 0: ids dropped since the other ranks last heard (cuda_follow)

    /// Opens rank `rank` of `world` over `comm`; the ranks agree on settings (cuda_ranks) before the weights load.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, ctx: *const cuda.Context, checkpoint: []const u8, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, rank: usize, world: usize, o: Options) !*Engine {
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var arena = std.heap.ArenaAllocator.init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cfg_path = try std.fs.path.join(a, &.{ checkpoint, "config.json" });
        const c = try config.parse(gpa, try std.Io.Dir.cwd().readFileAlloc(io, cfg_path, a, .limited(1 << 24)));
        var stream = try cuda.Stream.init(d, false);
        errdefer stream.deinit();
        const fp = ranks.Fingerprint.of(world, rank, o.settings, o.capacity, try ranks.hashFile(a, io, cfg_path), try ranks.hashFile(a, io, "/proc/self/exe"), checkpoint);
        try ranks.agree(d, lib, comm, stream, fp, try a.alloc(ranks.Fingerprint, world));
        const share = try shard.Share.of(&c, rank, world);
        const spans = try gpa.alloc(shard.Span, world);
        errdefer gpa.free(spans);
        for (spans, 0..) |*sp, q| sp.* = (try shard.Share.of(&c, q, world)).vocab;
        var m = try fw.Modules.load(d, ctx);
        errdefer m.deinit();
        var pieces = try fw.Pieces.init(d, o.settings.pieces);
        errdefer pieces.deinit();
        if (o.settings.layers > c.layers) return error.BadLayers;
        const layers = try a.alloc(u32, if (o.settings.layers > 0) o.settings.layers else c.layers);
        for (layers, 0..) |*l, i| l.* = @intCast(i);
        e.* = .{ .gpa = gpa, .io = io, .heat_gate = .{ .bands = o.settings.heat, .root = heatRoot() }, .shared_buf = undefined, .d = d, .c = c, .share = share, .spans = spans, .lib = lib, .comm = comm, .settings = o.settings, .m = m, .stream = stream, .pieces = pieces, .w = undefined, .capacity = o.capacity, .bp = undefined, .bd = undefined, .bm = undefined, .hold = undefined, .pairs = undefined, .pairs_all = undefined, .smp = undefined, .mtp = o.mtp };
        e.w = try weights.load(gpa, io, d, stream, e.m.tools(), checkpoint, c, share, layers, o.mtp);
        errdefer e.w.deinit();
        try e.bind(0);
        errdefer {
            e.free(0);
            e.seqs.deinit(gpa);
        }
        e.bp = try fw.Buffers.init(d, &e.w, fw.prefill_rows, true, o.capacity);
        errdefer e.bp.deinit();
        e.bd = try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bd.deinit();
        e.bm = try fw.Buffers.init(d, &e.w, fw.window_rows, false, o.capacity);
        errdefer e.bm.deinit();
        if (o.mtp) e.bs = try fw.Buffers.initShared(d, &e.w, shared_rows, o.capacity);
        errdefer if (e.bs) |*b| b.deinit();
        e.hold = try cuda.DeviceBuffer.alloc(d, c.hidden * 2);
        errdefer e.hold.free();
        e.pairs = try cuda.DeviceBuffer.alloc(d, 8 * world * fw.window_rows);
        e.pairs_all = try cuda.DeviceBuffer.alloc(d, 8 * world * fw.window_rows);
        errdefer e.pairs.free();
        e.smp = try sampler.Sampler.init(d, c.vocab, fw.window_rows);
        @import("cuda_latent.zig").setDriver(d);
        errdefer e.smp.deinit();
        e.shared_buf = try cuda.DeviceBuffer.alloc(d, 8);
        if (std.c.getenv("TF_GLM_L2PF") != null) {
            try e.initL2pf();
        }
        if (std.c.getenv("TF_GLM_WAVE") != null) {
            e.wave_scratch = try cuda.DeviceBuffer.alloc(d, 4 * fw.window_rows * c.hidden * 4);
            e.wave_pos2 = try cuda.DeviceBuffer.alloc(d, 4);
            // one ring half is kda_layers * rec_layer: the layer count from w.layers' kda mixers,
            // rec_layer from share.kda (they differ: the share list excludes some layers)
            var nk: usize = 0;
            for (e.w.layers) |l| switch (l.mixer) {
                .kda => nk += 1,
                else => {},
            };
            const rec_layer = e.w.share.kda.len() * 128 * 128 * 4; // kda.dk * kda.dv * 4
            e.wave_keep = try cuda.DeviceBuffer.alloc(d, nk * rec_layer);
            for (&e.wave_ev) |*ev| ev.* = try cuda.Event.init(d, false);
            var split: usize = 0;
            if (std.c.getenv("TF_GLM_WAVE_SPLIT")) |v| split = std.fmt.parseInt(usize, std.mem.span(v), 10) catch 0;
            // group B's conv taps: the conv state's own size (kda layers x 3 x conv_c bf16)
            const conv_c = 3 * e.w.share.kda.len() * e.w.config.kda_dim;
            e.wave_conv = try cuda.DeviceBuffer.alloc(d, nk * 3 * conv_c * 2);
            e.wave = .{ .scratch = e.wave_scratch.ptr, .pos2 = e.wave_pos2.ptr, .preserve = e.wave_keep.ptr, .preserve_len = e.wave_keep.len, .conv = e.wave_conv.ptr, .split = split, .sync = std.c.getenv("TF_GLM_WAVE_SYNC") != null, .side = e.pieces.side, .ev = e.wave_ev };
        }
        e.ids_stage = try cuda.HostBuffer.alloc(d, fw.window_rows * 4);
        errdefer e.ids_stage.free();
        try e.stream.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.stream.synchronize() catch {};
        if (e.wave_scratch.ptr != 0) e.wave_scratch.free();
        if (e.wave_pos2.ptr != 0) e.wave_pos2.free();
        if (e.wave_keep.ptr != 0) e.wave_keep.free();
        if (e.wave_conv.ptr != 0) e.wave_conv.free();
        if (e.wave != null) for (&e.wave_ev) |*ev| ev.deinit();
        if (e.l2pf != null) {
            e.l2pf_ev.deinit();
            e.l2pf_side.deinit();
            e.l2pf_sites.deinit(e.gpa);
            e.l2pf_table.free();
        }
        var git = e.graphs.valueIterator();
        while (git.next()) |g| g.exec.deinit();
        e.graphs.deinit(e.gpa);
        e.ids_stage.free();
        var it = e.snaps.valueIterator();
        while (it.next()) |s| e.freeSnap(s);
        e.snaps.deinit(e.gpa);
        e.dropped.deinit(e.gpa);
        e.shared_buf.free();
        e.smp.deinit();
        e.pairs.free();
        e.pairs_all.free();
        e.hold.free();
        if (e.bs) |*b| b.deinit();
        e.bm.deinit();
        e.bd.deinit();
        e.bp.deinit();
        for (0..e.seqs.items.len) |i| e.free(i);
        e.seqs.deinit(e.gpa);
        e.w.deinit();
        e.pieces.deinit();
        e.m.deinit();
        e.stream.deinit();
        e.gpa.free(e.spans);
        const gpa = e.gpa;
        gpa.destroy(e);
    }

    fn exchange(e: *const Engine) fw.Exchange {
        if (e.shm != null and std.c.getenv("TF_GLM_SHM_NCCL") != null) return .{ .nccl = .{ .lib = e.lib, .comm = e.comm } }; // bisect: arena up, exchange over NCCL
        if (e.shm) |s| return .{ .shm = .{
            .base = s.base,
            .seq = s.seq,
            .gather_fn = s.gather_fn,
            .rank = @intCast(e.share.rank),
            .slot_units = s.slot_units,
            .lib = e.lib,
            .comm = e.comm,
        } };
        return .{ .nccl = .{ .lib = e.lib, .comm = e.comm } };
    }

    /// The host-staged gather arena (liumorrisclaw's protocol; credit: fork shm-allgather):
    /// one /dev/shm segment, both ranks mmap + cudaHostRegister(Mapped). Call after the comm
    /// is up on every rank (the Link exchange acts as the rendezvous).
    pub fn initShm(e: *Engine) !void {
        const cid = std.c.getenv("NCCL_COMM_ID") orelse return error.NoCommId;
        var h: u64 = 5381;
        for (std.mem.span(cid)) |ch| h = (h *% 33) +% ch;
        var nb: [64]u8 = undefined;
        const path = try std.fmt.bufPrint(&nb, "/dev/shm/tf_glm_shm_{x}\x00", .{h});
        const slot_units: usize = 64 * e.c.hidden; // 8 streams x 8 rows: the multi-stream payload
        const arena_len: usize = 256 + 2 * 2 * slot_units * 4; // ShmLayout (256B) + 2 stages x 2 ranks
        const fd = std.c.open(@ptrCast(path.ptr), .{ .ACCMODE = .RDWR, .CREAT = true }, @as(c_uint, 0o600));
        if (fd < 0) return error.ShmOpenFailed;
        defer _ = std.c.close(fd);
        _ = std.c.ftruncate(fd, @intCast(arena_len));
        const mapped = try std.posix.mmap(null, arena_len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0);
        var dev_ptr: cuda.abi.DevicePtr = 0;
        const flags: c_uint = 0x1 | 0x2; // cudaHostRegisterPortable | Mapped
        try e.d.check(e.d.api.cuMemHostRegister_v2(mapped.ptr, arena_len, flags), "cuMemHostRegister");
        errdefer _ = e.d.api.cuMemHostUnregister(mapped.ptr);
        try e.d.check(e.d.api.cuMemHostGetDevicePointer_v2(&dev_ptr, mapped.ptr, 0), "cuMemHostGetDevicePointer");
        var seq = try cuda.DeviceBuffer.alloc(e.d, 16);
        errdefer seq.free();
        var z: [16]u8 = @splat(0);
        try seq.upload(0, &z);
        const fnp = try e.m.mods[6].function("tf_shm_gather_f32");
        e.shm = .{
            .base = dev_ptr,
            .seq = seq.ptr,
            .gather_fn = fnp,
            .rank = @intCast(e.share.rank),
            .slot_units = slot_units,
            .lib = e.lib,
            .comm = e.comm,
        };
        e.shm_seq = seq;
        std.log.info("glm: shm gather arena ready (slot {d} units, dev {x})", .{ slot_units, dev_ptr });
    }


    /// TF_GLM_L2PF: the pieces table (i64 [n][2]) and per-(layer, site) ranges. Site 'a' (the mixer gather)
    /// prefetches this layer's FFN side: hc weights, post norm, router, bias, the shared expert's gate/up and
    /// down. Site 'f' (the ffn gather) prefetches the NEXT layer's attention side (or, past the last layer,
    /// the final norm and the head). Small tensors whole, big ones' leading bytes, 256KiB pieces, 8MiB a site.
    fn initL2pf(e: *Engine) !void {
        const gpa = e.gpa;
        const w = &e.w;
        const budget: usize = 8 << 20;
        const piece: usize = 256 << 10;
        const cap: usize = 2 << 20; // one buffer's leading bytes at most
        var rows: std.ArrayList(i64) = .empty;
        defer rows.deinit(gpa);
        var sites: std.ArrayList([2]u32) = .empty;
        errdefer sites.deinit(gpa);
        const lenOf = struct {
            fn f(w2: *const weights.Weights, p: u64) usize {
                if (p == 0) return 0;
                for (w2.named.items) |n| if (n.ptr == p) return n.len;
                return 0;
            }
        }.f;
        const wOf = struct {
            fn q(qq: cuda.qlinear.Weight) u64 {
                return switch (qq) {
                    .bf16 => |m| m.codes,
                    .nvfp4 => |m| m.codes,
                    .fp8 => |m| m.codes,
                    .fp8g => |m| m.codes,
                    .affine4 => 0,
                    .exl3 => 0,
                };
            }
            fn p(pr: @import("cuda_b16.zig").Proj) u64 {
                return switch (pr) {
                    .b16 => |v| v,
                    .q => |qq| q(qq),
                };
            }
        };
        const push = struct {
            fn f(gpa2: std.mem.Allocator, rows2: *std.ArrayList(i64), lo: *usize, a: u64, n0: usize, cap2: usize, piece2: usize) void {
                if (a == 0 or n0 == 0 or lo.* < piece2) return;
                const n = @min(n0, cap2);
                var off: usize = 0;
                while (off < n and lo.* >= piece2) : (off += piece2) {
                    const take = @min(piece2, n - off);
                    rows2.append(gpa2, @intCast(a + off)) catch return;
                    rows2.append(gpa2, @intCast(take)) catch return;
                    lo.* -= take;
                }
            }
        }.f;
        var total_pieces: u32 = 0;
        for (w.layers, 0..) |l, j| {
            var site: usize = 0;
            while (site < 2) : (site += 1) {
                var left = budget;
                const before = rows.items.len;
                if (site == 0) {
                    push(gpa, &rows, &left, l.ffn_hc.fn_w, lenOf(w, l.ffn_hc.fn_w), cap, piece);
                    push(gpa, &rows, &left, l.ffn_hc.base, lenOf(w, l.ffn_hc.base), cap, piece);
                    push(gpa, &rows, &left, l.ffn_hc.scale, lenOf(w, l.ffn_hc.scale), cap, piece);
                    push(gpa, &rows, &left, l.post_norm, lenOf(w, l.post_norm), cap, piece);
                    switch (l.ffn) {
                        .moe => |m| {
                            push(gpa, &rows, &left, m.router, lenOf(w, m.router), cap, piece);
                            push(gpa, &rows, &left, m.bias, lenOf(w, m.bias), cap, piece);
                            push(gpa, &rows, &left, m.shared_gu, lenOf(w, m.shared_gu), cap, piece);
                            push(gpa, &rows, &left, wOf.p(m.shared_down), lenOf(w, wOf.p(m.shared_down)), cap, piece);
                        },
                        .dense => |m| {
                            push(gpa, &rows, &left, wOf.q(m.gate), lenOf(w, wOf.q(m.gate)), cap, piece);
                            push(gpa, &rows, &left, wOf.q(m.up), lenOf(w, wOf.q(m.up)), cap, piece);
                            if (m.down) |d| push(gpa, &rows, &left, wOf.q(d), lenOf(w, wOf.q(d)), cap, piece);
                        },
                    }
                } else {
                    const nxt_opt = if (j + 1 < w.layers.len) w.layers[j + 1] else null;
                    if (nxt_opt) |nx| {
                        push(gpa, &rows, &left, nx.attn_hc.fn_w, lenOf(w, nx.attn_hc.fn_w), cap, piece);
                        push(gpa, &rows, &left, nx.attn_hc.base, lenOf(w, nx.attn_hc.base), cap, piece);
                        push(gpa, &rows, &left, nx.attn_hc.scale, lenOf(w, nx.attn_hc.scale), cap, piece);
                        push(gpa, &rows, &left, nx.in_norm, lenOf(w, nx.in_norm), cap, piece);
                        switch (nx.mixer) {
                            .kda => |k| {
                                push(gpa, &rows, &left, k.a_log, lenOf(w, k.a_log), cap, piece);
                                push(gpa, &rows, &left, k.dt_bias, lenOf(w, k.dt_bias), cap, piece);
                                push(gpa, &rows, &left, k.norm_w, lenOf(w, k.norm_w), cap, piece);
                                push(gpa, &rows, &left, k.conv_w, lenOf(w, k.conv_w), cap, piece);
                                push(gpa, &rows, &left, k.fb, lenOf(w, k.fb), cap, piece);
                                push(gpa, &rows, &left, k.gb, lenOf(w, k.gb), cap, piece);
                                push(gpa, &rows, &left, k.proj, lenOf(w, k.proj), cap, piece);
                            },
                            .mla => |a| {
                                push(gpa, &rows, &left, a.q_norm, lenOf(w, a.q_norm), cap, piece);
                                push(gpa, &rows, &left, a.kv_norm, lenOf(w, a.kv_norm), cap, piece);
                                push(gpa, &rows, &left, a.wk, lenOf(w, a.wk), cap, piece);
                                push(gpa, &rows, &left, a.wv, lenOf(w, a.wv), cap, piece);
                                push(gpa, &rows, &left, a.proj, lenOf(w, a.proj), cap, piece);
                                push(gpa, &rows, &left, wOf.p(a.q_b), lenOf(w, wOf.p(a.q_b)), cap, piece);
                            },
                        }
                    } else {
                        push(gpa, &rows, &left, w.norm, lenOf(w, w.norm), cap, piece);
                        push(gpa, &rows, &left, wOf.p(w.head), lenOf(w, wOf.p(w.head)), cap, piece);
                    }
                }
                const count: u32 = @intCast((rows.items.len - before) / 2);
                sites.append(e.gpa, .{ total_pieces, count }) catch {};
                total_pieces += count;
            }
        }
        if (rows.items.len == 0) return; // nothing resolvable: the prefetcher stays off
        e.l2pf_table = try cuda.DeviceBuffer.alloc(e.d, rows.items.len * 8);
        try e.l2pf_table.upload(0, std.mem.sliceAsBytes(rows.items));
        e.l2pf_side = try cuda.Stream.init(e.d, false);
        e.l2pf_ev = try cuda.Event.init(e.d, false);
        e.l2pf_sites = sites;
        e.l2pf = .{ .fn_table = e.l2pf_table.ptr, .sites = e.l2pf_sites.items, .side = e.l2pf_side, .ev = e.l2pf_ev };
        std.log.info("glm: l2pf on ({d} pieces over {d} sites, up to 8 MiB a site)", .{ total_pieces, e.l2pf_sites.items.len });
    }

    fn view(e: *Engine, b: *fw.Buffers, out: *[1]fw.Rank) fw.Forward {
        out.* = .{.{ .w = &e.w, .st = &e.cur.st, .b = b }};
        return .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = out, .exchange = e.exchange(), .pieces = if (b == &e.bp) &e.pieces else null, .split = e.settings.split, .kda_wide = e.settings.kda_wide, .wave = if (b == &e.bd and e.wave != null) &e.wave.? else null, .l2pf = if (b == &e.bd and e.l2pf != null) &e.l2pf.? else null };
    }

    /// Binds slot `slot`'s sequence for the steps that follow, making it (a zeroed state) on first use.
    pub fn bind(e: *Engine, slot: u32) !void {
        if (slot >= e.seqs.items.len) try e.seqs.appendNTimes(e.gpa, null, slot + 1 - e.seqs.items.len);
        if (e.seqs.items[slot] == null) {
            const q = try e.gpa.create(Seq);
            errdefer e.gpa.destroy(q);
            q.* = .{ .st = try fw.State.init(e.gpa, e.d, &e.w, e.capacity) };
            e.seqs.items[slot] = q;
        }
        e.cur = e.seqs.items[slot].?;
        e.slot = slot;
    }

    /// Frees slot `slot`'s sequence (slot 0 stays: the first stream's, made at init) and binds slot 0.
    pub fn release(e: *Engine, slot: u32) void {
        if (slot != 0) e.free(slot);
        e.cur = e.seqs.items[0].?;
        e.slot = 0;
    }

    fn free(e: *Engine, slot: usize) void {
        const q = e.seqs.items[slot] orelse return;
        e.stream.synchronize() catch {};
        q.st.deinit();
        e.gpa.destroy(q);
        e.seqs.items[slot] = null;
    }

    /// Bytes one more stream's sequence takes on this rank (its state arena).
    pub fn seqBytes(e: *const Engine) u64 {
        return e.seqs.items[0].?.st.arena.buf.len;
    }

    /// The committed length: the pending token sits at this position, a window's row r draws position pos + 1 + r.
    pub fn pos(e: *const Engine) usize {
        return e.cur.st.pos;
    }

    /// A fresh sequence with `sampling` (null or temperature 0: greedy).
    pub fn begin(e: *Engine, sampling: ?sampler.Sampling) !void {
        e.cur.sampling = sampler.check(sampling);
        e.cur.in_head = 0;
        try e.cur.st.reset(e.stream);
    }

    /// decode.prefill: `ids` in 2,048-row chunks, then the first token from the last row; `from` restores a kept state.
    pub fn prefill(e: *Engine, ids: []const u32, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !u32 {
        var start = try e.prefillStart(ids.len, from);
        while (true) : (start += fw.prefill_rows) {
            if (try e.prefillChunk(ids[start..@min(ids.len, start + fw.prefill_rows + 1)], start, ids.len, null, cancel, marks)) |first| return first;
        }
    }

    /// Where a prompt pass of `len` ids starts: 0, or `from`'s position with its kept state restored.
    fn prefillStart(e: *Engine, len: usize, from: ?Resume) !usize {
        if (len == 0 or len + fw.window_rows >= e.cur.st.capacity) return error.PromptTooLong;
        const r = from orelse return 0;
        if (r.at == 0 or r.at >= len or r.at % fw.prefill_rows != 0) return error.BadResume;
        try e.restoreState(r.id);
        return r.at;
    }

    /// One chunk of a `len`-id prompt pass: `part` from `start` plus the next id; first token at the end, else null.
    pub fn prefillChunk(e: *Engine, part: []const u32, start: usize, len: usize, from: ?Resume, cancel: ?Cancel, marks: ?Marks) !?u32 {
        if ((from != null or start == 0) and try e.prefillStart(len, from) != start) return error.BadChunk;
        if (start >= len or start % fw.prefill_rows != 0) return error.BadChunk;
        const rows = @min(len - start, fw.prefill_rows);
        if (part.len != @min(len - start, rows + 1)) return error.BadChunk;
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bp, &one);
        const as_i32: []const i32 = @ptrCast(part);
        if (marks) |m| if (start > 0 and std.mem.indexOfScalar(u32, m.at, @intCast(start)) != null) m.call(m.ctx, @intCast(start));
        try e.gate(cancel);
        try f.run(as_i32[0..rows]);
        const last = start + rows == len;
        var first: ?u32 = null;
        if (last) {
            var t: [1]u32 = undefined;
            try e.draw(e.bp.logits, 1, len, &t);
            first = t[0];
        }
        if (e.mtp) {
            if (last) try e.copy(e.hold.ptr, e.bp.fnormed + (rows - 1) * e.c.hidden * 2, e.c.hidden * 2);
            const nxt = as_i32[1..];
            if (nxt.len > 0) { // decode._absorb_rows
                try f.mtp(nxt, &.{e.bp.fnormed});
                try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + nxt.len);
            }
        }
        try f.commit(rows);
        return first;
    }

    /// The chunk gate on every rank: a kept state at the mark, the heat wait, rank 0's cancel; the id exchanged first.
    fn gate(e: *Engine, cancel: ?Cancel) !void {
        const kept = try e.shared(@floatFromInt(e.announce));
        e.announce = 0;
        if (kept > 0 and e.share.rank != 0) try e.keepAs(@intFromFloat(kept));
        if (e.heat_gate.bands != null) {
            var w: Wait = .{ .e = e, .cancel = cancel };
            const before = e.heat_gate.waited_s;
            e.heat_gate.waitUntil(Wait.read, Wait.sleep, &w, .{ .ptr = &w, .check = Wait.stop }) catch |err| return w.failed orelse err;
            if (e.heat_gate.waited_s > before) heat.note(e.heat_gate.waited_s - before);
        }
        if (try e.shared(if (cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0) > 0) return error.Cancelled;
    }

    /// Bytes one kept state takes on this rank: the whole state arena (its caches are sized for the window, not `at`).
    pub fn stateBytes(e: *const Engine) u64 {
        return e.cur.st.arena.buf.len;
    }

    /// Rank 0, at a mark in a prompt pass: a copy of the state under a new id, named to the other ranks at the gate.
    pub fn keepState(e: *Engine) !u64 {
        const id = e.next_snap;
        try e.keepAs(id);
        e.next_snap += 1;
        e.announce = id;
        return id;
    }

    fn keepAs(e: *Engine, id: u64) !void {
        if (e.snaps.contains(id)) return error.StateKept;
        var buf = try cuda.DeviceBuffer.alloc(e.d, e.cur.st.arena.buf.len);
        errdefer buf.free();
        try e.copy(buf.ptr, e.cur.st.arena.buf.ptr, buf.len);
        try e.stream.synchronize();
        const cur = try e.gpa.dupe(u1, e.cur.st.cur);
        errdefer e.gpa.free(cur);
        try e.snaps.put(e.gpa, id, .{ .buf = buf, .pos = e.cur.st.pos, .mtp_len = e.cur.st.mtp_len, .cur = cur });
    }

    /// The live state made kept state `id`'s; a rank without it refuses rather than prefill from another position.
    fn restoreState(e: *Engine, id: u64) !void {
        const s = e.snaps.getPtr(id) orelse return error.NoKeptState;
        try e.copy(e.cur.st.arena.buf.ptr, s.buf.ptr, s.buf.len);
        try e.cur.st.restoreHost(e.stream, s.pos, s.mtp_len, s.cur);
        e.cur.in_head = 0;
        try e.stream.synchronize();
    }

    pub fn holds(e: *const Engine, id: u64) bool {
        return e.snaps.contains(id);
    }

    /// Frees kept state `id`; rank 0 also notes it for the other ranks (cuda_follow sends it before its next step).
    pub fn dropState(e: *Engine, id: u64) void {
        var kv = e.snaps.fetchRemove(id) orelse return;
        e.freeSnap(&kv.value);
        if (e.share.rank == 0 and e.share.world > 1) e.dropped.append(e.gpa, @intCast(id)) catch
            std.log.warn("kept state {d}: the other ranks keep their copy until they exit (no memory for the note)", .{id});
    }

    fn freeSnap(e: *Engine, s: *Snap) void {
        s.buf.free();
        e.gpa.free(s.cur);
    }

    /// The ranks' maximum of `v` (an all-reduce on the engine's stream).
    fn shared(e: *Engine, v: f64) !f64 {
        var x = v;
        try e.shared_buf.uploadAsync(0, std.mem.asBytes(&x), e.stream.handle);
        try e.lib.check(e.lib.api.ncclAllReduce(e.shared_buf.ptr, e.shared_buf.ptr, 1, .f64, .max, e.comm, e.stream.handle), "ncclAllReduce");
        try e.stream.synchronize();
        try e.shared_buf.download(0, std.mem.asBytes(&x));
        return x;
    }

    /// heat.Gate.waitUntil's hooks over the ranks: gathered max reading, rank 0's stop: all pause together.
    const Wait = struct {
        e: *Engine,
        cancel: ?Cancel,
        failed: ?anyerror = null,

        fn read(ptr: *anyopaque) anyerror!f64 {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            const local: f64 = blk: {
                var dir = std.Io.Dir.cwd().openDir(w.e.io, w.e.heat_gate.root, .{ .iterate = true }) catch break :blk -std.math.inf(f64);
                defer dir.close(w.e.io);
                break :blk heat.hottest(w.e.io, dir) catch -std.math.inf(f64);
            };
            return w.e.shared(local);
        }

        fn sleep(ptr: *anyopaque) void {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            std.Io.sleep(w.e.io, .fromMilliseconds(2000), .awake) catch {};
        }

        fn stop(ptr: *anyopaque) bool {
            const w: *Wait = @ptrCast(@alignCast(ptr));
            const mine: f64 = if (w.cancel) |c| @floatFromInt(@intFromBool(c.check(c.ptr))) else 0;
            const any = w.e.shared(mine) catch |err| {
                w.failed = err;
                return true;
            };
            return any > 0;
        }
    };

    /// A verify window: `ids` (pending token, then drafts) at the committed position; `out[r]` drawn at pos + 1 + r.
    pub fn verify(e: *Engine, ids: []const u32, out: []u32) !void {
        if (ids.len == 0 or ids.len > fw.window_rows) return error.WindowTooWide;
        const nch_now = @min(latent.chunksFor(e.cur.st.pos + ids.len), e.bd.nch);
        // graphs cover the dense window (pos + rows below the key width, static grids) and the
        // sparse phase in power-of-two pool buckets: inside a bucket np_max is constant, the
        // scores grid over-covers (past-visible blocks only write -inf), and the sparse kernels
        // read the position from pos_dev, so a capture replays across the whole bucket. A window
        // straddling the dense limit runs eager (its host branch is not bakeable).
        const dense_phase = e.cur.st.pos + ids.len < e.c.keyWidth();
        const sparse_phase = e.cur.st.pos >= e.c.keyWidth();
        const np_bucket: usize = if (sparse_phase) sparse.poolBucket(e.cur.st.pos, ids.len, e.cur.st.capacity / sparse.pool) else 0;
        const key: GraphKey = .{
            .slot = e.slot,
            .rows = @intCast(ids.len),
            .parity = @as(u8, e.cur.st.cur[0]),
            .bucket = if (sparse_phase) @intCast(1 + std.math.log2_int(usize, np_bucket / sparse.pool_min)) else 0,
        };
        // Opt-in while the replay still diverges from eager by one input (deterministic since the
        // pinned staging + exact-nch keying; 161.8 tok/s vs 91.6 eager when running - the remaining
        // baked input is the next session's instrumented eager-vs-graph logits diff).
        if (std.c.getenv("TF_GLM_GRAPHS") != null and (dense_phase or sparse_phase) and e.wave == null) {
            if (e.graphs.getPtr(key)) |g| {
                if (g.nch == nch_now) { // exact: the indexer score path reads garbage for chunks past
                    // its own range, so a replay may only cover exactly the nch this round needs
                    @memcpy(@as([*]u32, @ptrCast(e.ids_stage.bytes.ptr))[0..ids.len], ids);
                    e.cur.st.pending = ids.len; // the host writes run() does, mirrored on replay rounds
                    e.cur.st.prompt = e.bd.prefill;
                    try g.exec.launchOn(e.stream);
                    e.cur.rows_at = e.bd.fnormed;
                    try e.draw(e.bd.logits, ids.len, e.cur.st.pos + 1, out);
                    return;
                }
                g.exec.deinit();
                _ = e.graphs.remove(key);
                // the dense nch grew a bucket: fall through and recapture (a few times a session)
            }
            @memcpy(@as([*]u32, @ptrCast(e.ids_stage.bytes.ptr))[0..ids.len], ids);
            try cuda.graph.beginCapture(e.stream, .relaxed);
            const ran = blk: {
                var one: [1]fw.Rank = undefined;
                const f = e.view(&e.bd, &one);
                const stage_ids: []const i32 = @as([*]i32, @ptrCast(e.ids_stage.bytes.ptr))[0..fw.window_rows];
                f.run(stage_ids[0..@intCast(ids.len)]) catch break :blk false;
                break :blk true;
            };
            if (!ran) {
                _ = cuda.graph.endCapture(e.stream) catch {}; // restore the stream, then fail the step
                return error.GraphCaptureFailed;
            }
            const g = try cuda.graph.endCapture(e.stream);
            defer { var gm = g; gm.deinit(); }
            var exec = try g.instantiate();
            errdefer exec.deinit();
            try exec.upload(e.stream);
            try e.graphs.put(e.gpa, key, .{ .exec = exec, .nch = nch_now });
            try exec.launchOn(e.stream); // capture records only: this first round runs here
            e.cur.rows_at = e.bd.fnormed;
            try e.draw(e.bd.logits, ids.len, e.cur.st.pos + 1, out);
            return;
        }
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bd, &one);
        const at = e.cur.st.pos;
        try f.run(@ptrCast(ids));
        e.cur.rows_at = e.bd.fnormed;
        try e.draw(e.bd.logits, ids.len, at + 1, out);
    }

    /// Several streams' windows in one forward: shared row ops, per-part attention and KDA.
    pub fn verifyShared(e: *Engine, parts: []const Part) !void {
        const b = if (e.bs) |*x| x else return error.NoSharedWindows;
        if (parts.len == 0 or parts.len > max_shared) return error.WindowTooWide;
        var segs: [max_shared]fw.Seg = undefined;
        var ids: [shared_rows]u32 = undefined;
        var rows: usize = 0;
        for (parts, 0..) |p, i| {
            if (p.ids.len == 0 or p.ids.len > fw.window_rows or rows + p.ids.len > shared_rows) return error.WindowTooWide;
            for (parts[0..i]) |q| if (q.slot == p.slot) return error.SlotTwice;
            try e.bind(p.slot);
            segs[i] = .{ .st = &e.cur.st, .row0 = rows, .rows = p.ids.len };
            @memcpy(ids[rows..][0..p.ids.len], p.ids);
            rows += p.ids.len;
        }
        var one: [1]fw.Rank = .{.{ .w = &e.w, .st = segs[0].st, .b = b, .segs = segs[0..parts.len] }};
        const f: fw.Forward = .{ .d = e.d, .m = &e.m, .stream = e.stream, .ranks = &one, .exchange = e.exchange(), .pieces = null, .split = e.settings.split, .kda_wide = e.settings.kda_wide };
        try f.run(@ptrCast(ids[0..rows]));
        for (parts, segs[0..parts.len]) |p, g| {
            try e.bind(p.slot);
            e.cur.rows_at = b.fnormed + g.row0 * e.c.hidden * 2;
            try e.draw(b.logits + g.row0 * e.share.vocab.len() * 2, g.rows, g.st.pos + 1, p.out);
        }
    }

    /// forward.commit of the last window's first `n` rows (the KDA replay when fewer than it ran).
    pub fn keep(e: *Engine, n: usize) !void {
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bd, &one);
        try f.commit(n);
    }

    /// decode.draft: the MTP head absorbs the kept rows (committed first) with `follow`, then chains `depth` drafts.
    pub fn draft(e: *Engine, from_prompt: bool, follow: []const u32, depth: usize, out: []u32) !void {
        if (!e.mtp) return error.NoDraftHead;
        if (depth > max_drafts) return error.TooDeep;
        var one: [1]fw.Rank = undefined;
        const f = e.view(&e.bm, &one);
        const D = e.c.hidden;
        const hidden = if (from_prompt) e.hold.ptr else e.cur.rows_at;
        try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len - e.cur.in_head);
        e.cur.in_head = 0;
        const next: []const i32 = @ptrCast(follow);
        var s0: usize = 0;
        while (s0 < next.len) : (s0 += fw.window_rows) {
            const part = next[s0..@min(next.len, s0 + fw.window_rows)];
            try f.mtp(part, &.{hidden + s0 * D * 2});
            try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + part.len);
        }
        for (0..depth) |j| {
            try e.draw(e.bm.logits, 1, e.cur.st.pos + 1 + j, out[j..][0..1]);
            if (j + 1 < depth) {
                try f.mtp(@ptrCast(out[j..][0..1]), &.{e.bm.fnormed});
                try e.cur.st.setMtpLen(e.stream, e.cur.st.mtp_len + 1);
                e.cur.in_head += 1;
            }
        }
    }

    /// `rows` logits rows of this rank's vocabulary -> the tokens at positions first, first + 1, ...
    fn draw(e: *Engine, logits: u64, rows: usize, first: u64, out: []u32) !void {
        if (e.cur.sampling) |s| return e.smp.draw(e.stream, e.exchange(), e.spans, &.{.{ .rank = e.share.rank, .logits = logits }}, rows, first, sampler.rule(s), out);
        try e.greedyRows(logits, rows, out);
    }

    /// decode.sample_rows' greedy pick for a whole window: the per-row top-1 launches stay, but the
    /// count-2 all-gathers run as one NCCL group with ONE sync and ONE download for the whole window
    /// (TF_GLM_GREEDY_ROWS=0 keeps a full round trip a row). Bit-exact vs the per-row path
    /// (shas c6b7f32e/f5ac5dd91226 on both). Credit: BobClawblaw's TensorFold PR #978 launch surgery.
    fn greedyRows(e: *Engine, logits: u64, rows: usize, out: []u32) !void {
        const world = e.share.world;
        if (std.c.getenv("TF_GLM_GREEDY_ROWS")) |v| {
            if (std.mem.eql(u8, std.mem.span(v), "0")) {
                for (0..rows) |r| out[r] = try e.greedyRow(logits + r * e.share.vocab.len() * 2);
                return;
            }
        }
        for (0..rows) |r| {
            var a: cuda.Args = .{};
            a.add(logits + r * e.share.vocab.len() * 2);
            a.add(@as(i64, @intCast(e.share.vocab.len())));
            a.add(@as(u32, @intCast(e.share.vocab.lo)));
            a.add(e.pairs.ptr + e.share.rank * 8);
            try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
            // one count-2 all-gather a row (the proven shape), grouped: one sync covers the whole window.
            // Row r's gather lands at pairs_all + r*world*8, a separate destination (out-of-place on both ranks).
            try e.lib.check(e.lib.api.ncclGroupStart(), "ncclGroupStart");
            try e.lib.check(e.lib.api.ncclAllGather(e.pairs.ptr + e.share.rank * 8, e.pairs_all.ptr + r * world * 8, 2, .u32, e.comm, e.stream.handle), "ncclAllGather");
            try e.lib.check(e.lib.api.ncclGroupEnd(), "ncclGroupEnd");
        }
        try e.stream.synchronize();
        var all: [fw.window_rows * 16][2]u32 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.pairs_all.ptr, world * rows * 8), "cuMemcpyDtoH");
        for (0..rows) |r| {
            var pick = all[r * world];
            for (all[r * world + 1 .. r * world + world]) |p| {
                const v: f32 = @bitCast(p[0]);
                const b: f32 = @bitCast(pick[0]);
                if (v > b or (v == b and p[1] < pick[1])) pick = p;
            }
            out[r] = pick[1];
        }
    }

    /// The original single-row greedy: per-rank top-1 (tf_glm_top1), all-gathered, largest value then lowest id.
    fn greedyRow(e: *Engine, logits: u64) !u32 {
        const world = e.share.world;
        var a: cuda.Args = .{};
        a.add(logits);
        a.add(@as(i64, @intCast(e.share.vocab.len())));
        a.add(@as(u32, @intCast(e.share.vocab.lo)));
        a.add(e.pairs.ptr + e.share.rank * 8);
        try cuda.launch.launch(e.m.glue.top1, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, e.stream, &a);
        try e.lib.check(e.lib.api.ncclAllGather(e.pairs.ptr + e.share.rank * 8, e.pairs.ptr, 2, .u32, e.comm, e.stream.handle), "ncclAllGather");
        try e.stream.synchronize();
        var all: [16][2]u32 = undefined;
        try e.d.check(e.d.api.cuMemcpyDtoH_v2(&all, e.pairs.ptr, world * 8), "cuMemcpyDtoH");
        var pick = all[0];
        for (all[1..world]) |p| {
            const v: f32 = @bitCast(p[0]);
            const b: f32 = @bitCast(pick[0]);
            if (v > b or (v == b and p[1] < pick[1])) pick = p;
        }
        return pick[1];
    }

    fn heatRoot() []const u8 {
        const p = std.c.getenv("TF_HEAT_ROOT") orelse return heat.default_root;
        const s = std.mem.span(p);
        return if (s.len == 0) heat.default_root else s;
    }

    fn copy(e: *const Engine, dst: u64, src: u64, bytes: usize) !void {
        try e.d.check(e.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, e.stream.handle), "cuMemcpyDtoDAsync");
    }
};
