//! GLM-5.3-Flash's CUDA lane backend (as nemotron/cuda_lanes.zig): a slot per stream, shared verify, chunked prompts.

const std = @import("std");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const follow = @import("cuda_follow.zig");
const fw = @import("cuda_forward.zig");

const be = lanes.backend;

var tree_logs: u32 = 0; // TF_GLM_TREE diagnostics: the first rounds' shapes and kept paths

/// First tokens a handle names.
const ring = 1024;

/// The widest verify window: the pending token and a full draft chain (glm-decode-check's, the AOT row variants).
pub const max_rows = 1 + engine.max_drafts;

/// This engine's verify round by width, ms: one row ~17, each further row ~0.4 (weights-bound at TP2 over
/// NCCL SHM). Depth-rule seeds only - observeCost replaces them with measured round times as depths are tried.
const window_costs: [max_rows]lanes.config.Cost = blk: {
    var c: [max_rows]lanes.config.Cost = undefined;
    for (&c, 0..) |*x, i| x.* = .{ .width = i + 1, .ms = 17.0 + 0.12 * @as(f64, @floatFromInt(i)) }; // rows are near-free (weight-bound round): 0.4 priced shared drafts to ~1.6 a stream
    break :blk c;
};

/// A stream's place in the backend: its sequence's slot, the rows its last verify left uncommitted, its held drafts.
const Lane = struct {
    slot: u32,
    pending_rows: ?usize = null,
    held: [engine.max_drafts]u32 = undefined,
    held_n: usize = 0,
    fill: ?usize = null, // a prompt pass going a chunk at a time: where its next chunk starts
    tree_tokens: [3]u32 = undefined, // a draft tree's rows after the pending token (p0, s, p1)
    tree_parents: [3]i32 = .{ 0, 0, 1 }, // row 0 is the pending token: p0 and s are its children, p1 is p0's
    tree_n: u8 = 0,
};

pub const Lanes = struct {
    x: follow.Leader,
    whole: bool = false, // prompt passes run whole (TF_GLM_PROMPT_STEPS=0): no other stream's round between chunks
    alone: bool = false, // each window in its own forward (TF_GLM_SHARED=0)
    shared: u64 = 0, // forwards that held several streams' windows (checks read it)
    lanes: std.AutoHashMapUnmanaged(*const lanes.Stream, Lane) = .empty,
    drawn: [ring]u32 = undefined,
    next: u64 = 0,

    pub fn deinit(self: *Lanes) void {
        self.lanes.deinit(self.x.e.gpa);
    }

    pub fn backend(self: *Lanes) be.Backend {
        return .{ .ptr = self, .vtable = if (self.whole) &whole_vt else &steps_vt };
    }

    const whole_vt: be.Backend.VTable = .{
        .prefill = prefillFn,
        .first = firstFn,
        .queue = queueFn,
        .read = readFn,
        .verify = verifyFn,
        .keep = keepFn,
        .draft = draftFn,
        .tree = treeFn,
        .release = releaseFn,
    };
    const steps_vt: be.Backend.VTable = vt: {
        var v = whole_vt;
        v.prefill_step = prefillStepFn;
        break :vt v;
    };

    /// Round loop facts: windows up to max_rows, drafts with the head loaded, shared windows unless TF_GLM_SHARED=0.
    /// Window costs seed the depth rule from this engine's own 2x RTX PRO 6000 measurements (a verify round is
    /// weights-bound: 17 ms at one row, ~0.4 ms a row more); the rule replaces them with each round's measured
    /// wall time as it probes, but without seeds it cannot price an untried depth and never drafts past one.
    pub fn facts(self: *const Lanes) lanes.Model {
        const drafting = self.x.e.mtp;
        const sharing = drafting and !self.alone;
        return .{
            .exact_width = if (drafting) max_rows else 1,
            .gpu_tokens = false,
            .mtp = drafting,
            .speculate = drafting,
            .speculate_early = false,
            .drafts = engine.max_drafts,
            .hidden_rows = sharing,
            .batch_rows = if (sharing) engine.shared_rows else 0,
            .max_streams = if (sharing) engine.max_shared else 1,
            .plain_guard = drafting, // a draft row rides the same weights-bound round a plain row does
            // the acceptance the per-8 NVFP4 cache actually shows (solo: 6.9 tokens a round at depth 7 ~ 0.93 a
            // step); the default prior (0.85, 0.75, 0.70...) priced shared drafts to ~1.7 a stream and the
            // streams never drafted deep enough to measure it
            .draft_prior = &.{ 0.95, 0.93, 0.90, 0.87, 0.84, 0.80, 0.75 },
            .head_trees = std.c.getenv("TF_GLM_TREE") != null,
            .head_lanes = if (std.c.getenv("TF_GLM_LANES")) |v| (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0) else 0,
            .lane_costs = if (drafting) &window_costs else &.{},
            .window_costs = if (drafting) &window_costs else &.{},
            .shared_costs = if (drafting) &window_costs else &.{},
            .mtp_step_ms = if (drafting) 0.6 else 0.0,
        };
    }

    fn of(ptr: *anyopaque) *Lanes {
        return @ptrCast(@alignCast(ptr));
    }

    fn take(self: *Lanes, token: u32) u64 {
        const h = self.next;
        self.drawn[h % ring] = token;
        self.next += 1;
        return h;
    }

    fn value(self: *const Lanes, feed: be.Feed) u32 {
        return switch (feed) {
            .handle => |h| self.drawn[h % ring],
            .value => |v| v,
        };
    }

    /// The lowest slot no stream holds.
    fn freeSlot(self: *const Lanes) u32 {
        var slot: u32 = 0;
        while (true) : (slot += 1) {
            var it = self.lanes.valueIterator();
            const taken = while (it.next()) |l| {
                if (l.slot == slot) break true;
            } else false;
            if (!taken) return slot;
        }
    }

    /// The stream's sequence bound, its last verify committed (`kept` rows, else all of them).
    fn bind(self: *Lanes, s: *const lanes.Stream, kept: ?usize) !*Lane {
        const l = self.lanes.getPtr(s) orelse return error.UnknownStream;
        try self.x.bind(l.slot);
        if (l.pending_rows) |rows| try self.x.keep(kept orelse rows);
        l.pending_rows = null;
        return l;
    }

    fn cancelled(ptr: ?*anyopaque) bool {
        return @as(*lanes.Stream, @ptrCast(@alignCast(ptr.?))).isCancelled();
    }

    // -- the vtable ---------------------------------------------------------------------------------------------

    fn prefillFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!void {
        const self = of(ptr);
        const from = try self.opening(s);
        _ = self.take(try self.x.prefill(s.prompt(), from, .{ .ptr = s, .check = cancelled }, marksOf(s)));
    }

    /// One chunk of the stream's prompt pass (fillOne runs other rounds between); true once the first token is drawn.
    fn prefillStepFn(ptr: *anyopaque, s: *lanes.Stream) anyerror!bool {
        const self = of(ptr);
        const going: ?usize = if (self.lanes.getPtr(s)) |l| l.fill else null;
        const from = if (going == null) try self.opening(s) else null;
        const l = self.lanes.getPtr(s).?;
        if (going != null) try self.x.bind(l.slot);
        const at = going orelse if (from) |r| r.at else 0;
        errdefer l.fill = null;
        if (try self.x.chunk(s.prompt(), at, from, .{ .ptr = s, .check = cancelled }, marksOf(s))) |first| {
            _ = self.take(first);
            l.fill = null;
            return true;
        }
        l.fill = at + engine.prompt_rows;
        return false;
    }

    /// A prompt pass begins: the stream's slot bound and its sequence fresh; where the pass resumes (a kept state).
    fn opening(self: *Lanes, s: *lanes.Stream) !?engine.Resume {
        const ids = s.prompt();
        if (ids.len + s.max_new + max_rows >= self.x.e.capacity) return error.PromptTooLong;
        const gop = try self.lanes.getOrPut(self.x.e.gpa, s);
        if (!gop.found_existing) gop.value_ptr.* = .{ .slot = self.freeSlot() };
        gop.value_ptr.pending_rows = null;
        gop.value_ptr.held_n = 0;
        gop.value_ptr.fill = null;
        if (self.x.e.pool != null) {
            // the stream leases prompt + reply + one verify window from the shared pool
            const r = self.x.e.reserveRegion(ids.len + s.max_new + max_rows) catch |err| switch (err) {
                error.PoolFull => return error.ContextFull,
                else => return err,
            };
            try self.x.bindRegion(gop.value_ptr.slot, r.start, r.blocks);
        } else try self.x.bind(gop.value_ptr.slot);
        try self.x.begin(s.sampling);
        s.cached = 0;
        return self.resumeFrom(s, ids.len);
    }

    fn marksOf(s: *lanes.Stream) ?engine.Marks {
        return if (s.reuse.marks.len == 0) null else .{ .at = s.reuse.marks, .ctx = s, .call = keptAt };
    }

    /// The stream's kept state, where its prompt pass starts (null: none, off the chunk grid, or no longer held).
    fn resumeFrom(self: *Lanes, s: *lanes.Stream, len: usize) ?engine.Resume {
        const saved = s.reuse.saved orelse return null;
        const k: *const engine.Kept = @ptrCast(@alignCast(saved));
        if (k.at != s.reuse.at or k.at == 0 or k.at >= len or k.at % fw.prefill_rows != 0 or !self.x.e.holds(k.id)) {
            s.reuse_failed = true;
            return null;
        }
        s.cached = k.at;
        return .{ .id = k.id, .at = k.at };
    }

    /// The prompt pass stands at a mark: the lane host keeps the state there (cuda_native's snap_save).
    fn keptAt(ctx: *anyopaque, at: u32) void {
        const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
        if (s.reuse.hook) |k| k.at(k.ptr, s, at);
    }

    fn firstFn(ptr: *anyopaque, s: *lanes.Stream, position: u64) anyerror!u64 {
        const self = of(ptr);
        if (position != s.prompt_len) return error.PositionMismatch;
        return self.next - 1;
    }

    fn queueFn(ptr: *anyopaque, s: *lanes.Stream, feed: be.Feed, position: u64) anyerror!u64 {
        _ = ptr;
        _ = s;
        _ = feed;
        _ = position;
        return error.NotPipelined;
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        const self = of(ptr);
        if (handle >= self.next or self.next - handle > ring) return error.NoSuchToken;
        return self.drawn[handle % ring];
    }

    /// Each window in its own forward on its stream's sequence (the core sends one when no forward is shared).
    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const self = of(ptr);
        if (windows.len > 1) return self.verifyShared(windows, out);
        for (windows, out) |w, *o| {
            const l = try self.bind(w.stream, null);
            // a tree round's lanes are HELD in tree order (draftFn filled them lane for lane)
            if (w.parents != null and w.held > l.tree_n) return error.TreesNotBuilt;
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            if (w.parents == null) for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            var ids: [max_rows]u32 = undefined;
            ids[0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[1..][0..drafts.len], drafts);
            var sampled: [max_rows]u32 = undefined;
            try self.x.e.setTreeRound(w.parents);
            try self.x.verify(ids[0..rows], sampled[0..rows]);
            @memcpy(o.sampled[0..rows], sampled[0..rows]);
            @memcpy(o.drafts[0..drafts.len], drafts);
            l.pending_rows = rows;
        }
    }

    /// A round's windows in one forward (Engine.verifyShared), each stream's rows on its own sequence.
    fn verifyShared(self: *Lanes, windows: []const be.Window, out: []be.Verified) !void {
        if (windows.len > engine.max_shared) return error.WindowTooWide;
        var parts: [engine.max_shared]engine.Part = undefined;
        var ids: [engine.max_shared][max_rows]u32 = undefined;
        var sampled: [engine.max_shared][max_rows]u32 = undefined;
        var lane: [engine.max_shared]*Lane = undefined;
        for (windows, 0..) |w, k| {
            if (w.parents != null) return error.TreesNotBuilt;
            const l = try self.bind(w.stream, null);
            const rows = w.rows();
            if (rows > max_rows) return error.WindowTooWide;
            if (w.held > l.held_n) return error.NoHeldDrafts;
            const pos = self.x.e.pos();
            for (w.positions, 0..) |p, r| if (p != pos + 1 + r) return error.PositionMismatch;
            ids[k][0] = w.pending;
            const drafts = if (w.held > 0) l.held[0..w.held] else w.tokens;
            @memcpy(ids[k][1..][0..drafts.len], drafts);
            parts[k] = .{ .slot = l.slot, .ids = ids[k][0..rows], .out = sampled[k][0..rows] };
            lane[k] = l;
        }
        try self.x.verifyShared(parts[0..windows.len]);
        self.shared += 1;
        for (out[0..windows.len], parts[0..windows.len], lane[0..windows.len]) |*o, p, l| {
            const rows = p.ids.len;
            @memcpy(o.sampled[0..rows], p.out);
            @memcpy(o.drafts[0 .. rows - 1], p.ids[1..rows]);
            l.pending_rows = rows;
        }
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const self = of(ptr);
        for (windows, paths) |w, path| {
            if (path.len == 0) return error.EmptyPath;
            const l = self.lanes.getPtr(w.stream) orelse return error.UnknownStream;
            try self.x.bind(l.slot);
            // only a window still uncommitted settles here; a shared round may already have done it
            if (l.pending_rows != null) {
                var prefix = true;
                for (path, 0..) |r, i| if (r != i) {
                    prefix = false;
                    break;
                };
                if (prefix) {
                    try self.x.keep(path.len);
                } else {
                    if (tree_logs < 12) {
                        tree_logs += 1;
                        std.log.info("glm tree: keep path {any} of rows {d}", .{ path, w.rows() });
                    }
                    try self.x.keepPath(path);
                }
            }
            l.pending_rows = null;
        }
    }

    /// A tree head's held drafts as host tokens and parents; null when the lane has none.
    fn treeFn(ptr: *anyopaque, s: *lanes.Stream, gpa: std.mem.Allocator) anyerror!?be.Held {
        const self = of(ptr);
        const l = self.lanes.getPtr(s) orelse return null;
        if (l.tree_n == 0) return null;
        const n: usize = l.tree_n;
        const tokens = try gpa.dupe(u32, l.tree_tokens[0..n]);
        errdefer gpa.free(tokens);
        const parents = try gpa.dupe(i32, l.tree_parents[0..n]);
        return .{ .count = @intCast(n), .tokens = tokens, .parents = parents };
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const self = of(ptr);
        if (!self.x.e.mtp) return error.NoDraftHead;
        for (requests) |r| {
            const l = try self.bind(r.stream, if (r.rows) |rows| rows.len else null);
            if (r.position != self.x.e.pos() + 1) return error.PositionMismatch;
            const tree = r.lanes != null;
            var need: usize = 0;
            if (tree) {
                for (r.lanes.?.depths) |d| need = @max(need, @as(usize, d) + 1);
            }
            const depth = if (tree) @min(@max(2, @max(@as(usize, r.depth), need)), engine.max_drafts) else @min(r.depth, engine.max_drafts);
            if (r.rows) |rows| {
                for (rows, 0..) |row, i| if (row != i) return error.TreesNotBuilt;
                try self.x.draft(false, r.follow[0..rows.len], depth, &l.held);
            } else {
                const first = self.value(r.first orelse return error.NoFirstToken);
                try self.x.draft(true, &.{first}, depth, &l.held);
            }
            l.held_n = depth;
            if (tree) {
                const sh = r.lanes.?;
                const n = sh.parents.len;
                if (n == 0 or n > engine.max_drafts) {
                    std.log.err("glm tree: lanes {d} outside 1..{d}", .{ n, engine.max_drafts });
                    return error.TreesNotBuilt;
                }
                var sib: [engine.max_drafts]i32 = undefined;
                self.x.e.draftAlts(depth, sib[0..depth]) catch {
                    std.log.err("glm tree: no runner-up buffer for depth {d}", .{depth});
                    return error.TreesNotBuilt;
                };
                // a lane's rank-0 token is the chain's at its depth; rank 1 is that step's runner-up
                // (under the chain lane one level up). Lanes under a rank-1 parent are not producible.
                var hold: [engine.max_drafts]u32 = undefined;
                for (0..n) |i| {
                    const d: usize = sh.depths[i];
                    const par = sh.parents[i];
                    if (d >= depth) {
                        std.log.err("glm tree: lane {d} depth {d} beyond the chain {d}", .{ i, d, depth });
                        return error.TreesNotBuilt;
                    }
                    if (par >= 0) {
                        const up: usize = @intCast(par);
                        if (up >= i or sh.ranks[up] != 0) {
                            std.log.err("glm tree: lane {d} under parent {d} (rank {d})", .{ i, par, sh.ranks[up] });
                            return error.TreesNotBuilt;
                        }
                    }
                    hold[i] = switch (sh.ranks[i]) {
                        0 => l.held[d],
                        1 => if (sib[d] >= 0) @intCast(sib[d]) else {
                            std.log.err("glm tree: lane {d} rank 1 at depth {d} has no runner-up", .{ i, d });
                            return error.TreesNotBuilt;
                        },
                        else => {
                            std.log.err("glm tree: lane {d} rank {d} unsupported", .{ i, sh.ranks[i] });
                            return error.TreesNotBuilt;
                        },
                    };
                }
                @memcpy(l.held[0..n], hold[0..n]);
                l.held_n = n;
                // the backend's own tree view (used only when the core asks without a shape)
                l.tree_tokens = .{ hold[0], if (n > 1) hold[1] else hold[0], if (n > 2) hold[2] else hold[0] };
                l.tree_parents = .{ 0, 0, 0 };
                l.tree_n = @intCast(n);
                if (tree_logs < 6) {
                    tree_logs += 1;
                    std.log.info("glm tree: lanes={d} depth={d} ranks={any} depths={any} parents={any} hold={any}", .{ n, depth, sh.ranks, sh.depths, sh.parents, hold[0..n] });
                }
            } else l.tree_n = 0;
        }
    }

    /// The stream left the rounds: its sequence is freed on every rank (slot 0 stays for the next stream).
    fn releaseFn(ptr: *anyopaque, s: *lanes.Stream) void {
        const self = of(ptr);
        const kv = self.lanes.fetchRemove(s) orelse return;
        self.x.release(kv.value.slot) catch |err| std.log.err("releasing slot {d}: {s}", .{ kv.value.slot, @errorName(err) });
    }
};

comptime {
    std.debug.assert(max_rows <= fw.window_rows);
}
