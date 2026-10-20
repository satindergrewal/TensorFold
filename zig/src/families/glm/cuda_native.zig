//! GLM-5.3-Flash NVFP4 native CUDA at TP over processes (TF_GLM_RANK, NCCL_COMM_ID): rank 0 serves, others follow.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const glm_lanes = @import("cuda_lanes.zig");
const ranks = @import("cuda_ranks.zig");
const fw = @import("cuda_forward.zig");
const api = @import("engine_api");
const images_mod = @import("cuda_images.zig");

pub const model_type = "glm5_next";
pub const formats: []const []const u8 = &.{"modelopt-nvfp4"};
// cc-defer: a fixed window; the Python family sizes it from free memory (capacity.admit), port that when serving needs
pub const default_context: i64 = 8192;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = fw.prefill_rows;
/// Prompt kernels change arithmetic with a chunk's rows: states keep and resume only on chunk starts.
pub const snap_grid: u32 = prompt_rows;
/// `explain` words this family's open refusals too (where to set the rank and the id).
pub const explains_open = true;

/// `slide` stays false: the server refuses --slide for a family without a learner (no `slide` here).
pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false, vision: ?images_mod.Limits = null };

pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 1,
    learn_bytes: usize = 0, // no learner: the streams keep all the room
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null,
    target: *engine.Engine, // the prompt cache keeps this engine's states (snap_*)
    vision: ?*images_mod.Images = null, // rank 0's tower for the server's image offer
};

/// This rank's place: TF_GLM_RANK of TF_GLM_WORLD (both, or neither for a refusal that names them).
pub const Place = struct { rank: usize, world: usize };

pub fn place() !Place {
    const r = std.c.getenv("TF_GLM_RANK") orelse return error.NoRank;
    const w = std.c.getenv("TF_GLM_WORLD") orelse return error.NoRank;
    const p: Place = .{ .rank = std.fmt.parseInt(usize, std.mem.span(r), 10) catch return error.NoRank, .world = std.fmt.parseInt(usize, std.mem.span(w), 10) catch return error.NoRank };
    if (p.world < 1 or p.rank >= p.world) return error.NoRank;
    return p;
}

/// Bytes this rank's weights take: the checkpoint's share (its tensors split by world), for the server's admission.
pub fn weightShare(total: u64) u64 {
    const p = place() catch return total;
    return total / p.world;
}

/// Slots for `context` plus a verify window, a multiple of 64 so the sparse pool keeps the AOT set.
pub fn capacity(context: usize) usize {
    return std.mem.alignForward(usize, context + fw.window_rows, 64);
}

const Owned = struct {
    gpa: std.mem.Allocator,
    lib: cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    e: *engine.Engine,
    link: follow.Link,
    lanes: glm_lanes.Lanes,
};

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    _ = kernels; // our own kernels: a captured set is not used
    const p = try place();
    if (p.world > 1 and std.c.getenv("NCCL_COMM_ID") == null) return error.NoCommId;
    var problem: []const u8 = "";
    const set = ranks.Settings.read(&ranks.environment, &problem) catch |err| {
        std.log.err("{s}", .{problem});
        return err;
    };
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.gpa = gpa;
    own.lib = try cuda.nccl.Library.open();
    errdefer own.lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    try own.lib.check(own.lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId"); // NCCL_COMM_ID: the same id on every rank
    own.comm = null;
    try own.lib.check(own.lib.api.ncclCommInitRank(&own.comm, @intCast(p.world), id, @intCast(p.rank)), "ncclCommInitRank");
    errdefer _ = own.lib.api.ncclCommDestroy(own.comm);
    const cap = capacity(o.context);
    own.e = try engine.Engine.init(gpa, io, ctx.d, ctx, dir, &own.lib, own.comm, p.rank, p.world, .{ .capacity = cap, .mtp = o.drafts, .settings = set });
    errdefer own.e.deinit();
    if (o.vision) |lim| { // before the rooms: the tower's bytes count against rank 0's
        var vproblem: []const u8 = "";
        own.e.images = (try images_mod.Images.init(gpa, io, ctx.d, own.e.stream, dir, lim, fw.prefill_rows + 1, p.rank == 0, &vproblem)) orelse {
            std.log.err("{s}", .{vproblem});
            return error.NoVisionTower;
        };
    }
    own.link = try follow.Link.init(gpa, ctx.d, &own.lib, own.comm, own.e.stream, cap, own.e.m.glue.bell_wait);
    errdefer own.link.deinit();
    if (std.c.getenv("TF_GLM_SHM_GATHER") != null and p.world > 1) own.e.initShm() catch |err| {
        std.log.info("glm: shm gather off ({s}); NCCL carries the exchanges", .{@errorName(err)});
    };
    own.e.setTreeTest() catch |err| std.log.info("glm: tree test mask off ({s})", .{@errorName(err)});
    if (p.rank != 0) { // a follower never serves: it runs rank 0's steps, then leaves the process
        follow.follow(own.e, &own.link) catch |err| {
            std.log.err("rank {d}: following rank 0 stopped ({s})", .{ p.rank, @errorName(err) });
            std.process.exit(1);
        };
        std.log.info("rank {d}: rank 0 ended; leaving", .{p.rank});
        std.process.exit(0);
    }
    // Rank 0 alone reads these: TF_GLM_PROMPT_STEPS=0 passes prompts whole, TF_GLM_SHARED=0 runs windows alone.
    const off = struct {
        fn get(name: [*:0]const u8) bool {
            return if (std.c.getenv(name)) |v| std.mem.eql(u8, std.mem.span(v), "0") else false;
        }
    }.get;
    own.lanes = .{ .x = .{ .e = own.e, .link = &own.link }, .whole = off("TF_GLM_PROMPT_STEPS"), .alone = off("TF_GLM_SHARED") };
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = glm_lanes.max_rows,
        .stream_bytes = own.e.seqBytes() + (if (own.e.images) |v| v.streamBytes() else 0), // each stream past the first makes its own sequence on every rank
        .free_streams = 1, // the first is the sequence the engine made
        .ctx = own,
        .deinit = close,
        .target = own.e,
        .vision = own.e.images,
    };
}

// Each rank keeps a copy under rank 0's id. cc-defer: rank 0's budget, others unchecked; trigger: step 5's admission.

/// Bytes a state takes at any position: the whole state arena on each rank.
pub fn snap_bytes(ptr: *anyopaque, at: u32) u64 {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    return e.stateBytes() + e.kvBytes(at);
}

/// The prompt pass stands at `at` (a mark at a chunk start): copy the state there under a new id.
pub fn snap_save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    if (e.pos() != at) return error.NotAtMark;
    const k = try e.gpa.create(engine.Kept);
    errdefer e.gpa.destroy(k);
    k.* = .{ .id = try e.keepState(), .at = at };
    return k;
}

/// A restore outside a prompt pass (the store's `begin`) would leave the other ranks behind, so it refuses.
pub fn snap_restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = ptr;
    _ = owner;
    _ = saved;
    return error.RestoredInPromptPass;
}

pub fn snap_drop(ptr: *anyopaque, saved: *anyopaque) void {
    const e: *engine.Engine = @ptrCast(@alignCast(ptr));
    const k: *engine.Kept = @ptrCast(@alignCast(saved));
    e.dropState(k.id);
    e.gpa.destroy(k);
}

fn close(ptr: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(ptr));
    own.lanes.x.end() catch {};
    own.lanes.deinit();
    own.link.deinit();
    if (own.e.images) |v| v.deinit();
    own.e.deinit();
    _ = own.lib.api.ncclCommDestroy(own.comm);
    own.lib.close();
    own.gpa.destroy(own);
}

/// The image tower's bytes to tower input behind the server's offer (its limits are from --vision).
fn prepareImage(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, max_tokens: u32) anyerror!api.PreparedImage {
    const t: *images_mod.Images = @ptrCast(@alignCast(ctx));
    const p = try t.prepare(a, bytes, max_tokens);
    return .{ .pixels = p.pixels, .gh = p.gh, .gw = p.gw, .tokens = p.tokens, .hash = p.hash };
}

/// Info.vision for this rank 0's tower: GLM's template writes each image's markers itself.
pub fn visionOffer(v: ?*images_mod.Images) ?api.Vision {
    const t = v orelse return null;
    const wrap: ?[2]u32 = if (images_mod.Images.template_wraps) null else .{ t.marks.open, t.marks.close };
    return .{ .ctx = t, .prepare = prepareImage, .image_token = t.marks.image, .markers = &t.mark_ids, .wrap = wrap, .image_tokens = t.limits.image_tokens, .max_images = t.limits.max_images, .explain = images_mod.Images.explain };
}

/// The server's words for this family's refusals.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.NoRank => "GLM-5.3-Flash runs at TP over hosts: set TF_GLM_RANK and TF_GLM_WORLD on every host (WORLD 1: one GPU, with TF_GLM_LAYERS for a thin model)",
        error.NoCommId => "GLM-5.3-Flash's ranks meet over NCCL: with TF_GLM_WORLD 2 or more, set NCCL_COMM_ID to rank 0's address:port on every host",
        error.NoVisionTower => "this checkpoint's vision tower did not load: the log names the file or the vision_config field it refused",
        error.RankMismatch => "the ranks' settings differ (each rank's log names the one): start them alike",
        else => null,
    };
}

test "capacity keeps a verify window past the context and a multiple of 64" {
    try std.testing.expectEqual(@as(usize, 8256), capacity(8192));
    try std.testing.expectEqual(@as(usize, 330560), capacity(330529));
    try std.testing.expect(capacity(330529) % 64 == 0);
}
