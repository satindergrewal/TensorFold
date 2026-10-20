//! TP over processes until #552: rank 0's Leader sends each step (header, ids, slot) first; other ranks `follow` it.

const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const sampler = @import("cuda_sampler.zig");
const Engine = engine.Engine;

pub const Kind = enum(u32) { begin = 1, prefill, verify, keep, draft, end, drop, release, chunk, shared, bind };

var step_logs: u32 = 0; // TF_GLM_STEP_LOG: the first steps each side handles (deadlock diagnosis)

/// One step as it crosses: 64 bytes, then `n` u32 ids.
pub const Header = extern struct {
    kind: u32,
    n: u32 = 0, // ids that follow
    depth: u32 = 0, // draft: drafts to chain; chunk: where it starts; shared: its parts
    flags: u32 = 0, // begin: sampled; draft: from the prompt
    a: u64 = 0, // keep: rows; prefill, chunk: the kept state it resumes (0: none)
    seed: u64 = 0, // begin: the sampling seed; prefill, chunk: the kept state's position
    temperature: f64 = 0,
    top_p: f64 = 0,
    min_p: f64 = 0,
    top_k: u32 = 0, // begin: top_k; chunk: the prompt's length
    slot: u32 = 0, // the stream's sequence the step runs on (Engine.bind)

    comptime {
        std.debug.assert(@sizeOf(Header) == 64);
    }
};

/// tf_glm_put_words' argument: a step's header (16 words) and up to `put_ids` ids.
pub const Words = extern struct { w: [96]u32 };
pub const put_ids = 96 - 16;

/// The step link lives in host memory: one /dev/shm segment (named from NCCL_COMM_ID, zero at
/// creation) mapped by both processes. A one-slot mailbox with backpressure: rank 0 waits until the
/// followers consumed the previous step (ack), writes the header and ids with CPU stores, then
/// release-stores the sequence; followers wait for seq == seen + 1, copy, and release-store ack.
/// No NCCL and no cudaIpc imports on the step path - the per-step broadcasts were the trace's
/// ~10-12ms stalls (NCCL's own cuMemImportFromShareableHandle churn mid-decode).
pub const LinkLayout = struct {
    seq: u32, // rank 0: release-stored after a step's bytes are in place
    ack: u32, // followers: release-stored after copying a step out
    n: u32, // ids count
    _pad: u32,
    hdr: [64]u8, // the Header bytes
    ids: [2049]u32, // a prefill chunk (2,048 rows) plus the chunk command lookahead; 1024 refused every prompt past ~1k tokens (TooManyIds)
};

const Arena = struct {
    lay: *LinkLayout,
    dev: u64, // the device view (registered for completeness; the link no longer runs kernels)
};

fn initArena(d: *const cuda.Driver, max_ids: usize) ?Arena {
    if (std.c.getenv("TF_GLM_LINK_HOST")) |v| {
        if (std.mem.eql(u8, std.mem.span(v), "0")) return null;
    }
    const cid = std.c.getenv("NCCL_COMM_ID") orelse return null;
    var h: u64 = 5381;
    for (std.mem.span(cid)) |ch| h = (h *% 33) +% ch;
    var nb: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&nb, "/dev/shm/tf_glm_link_{x}\x00", .{h}) catch return null;
    const fd = std.c.open(@ptrCast(path.ptr), .{ .ACCMODE = .RDWR, .CREAT = true }, @as(c_uint, 0o600));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    const len = @sizeOf(LinkLayout) + 64;
    _ = std.c.ftruncate(fd, @intCast(len));
    const m = std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0) catch return null;
    _ = max_ids;
    const flags: c_uint = 0x1 | 0x2; // cudaHostRegisterPortable | Mapped
    _ = d.api.cuMemHostRegister_v2(m.ptr, len, flags);
    var dev: cuda.abi.DevicePtr = 0;
    _ = d.api.cuMemHostGetDevicePointer_v2(&dev, m.ptr, 0);
    return .{ .lay = @ptrCast(@alignCast(m.ptr)), .dev = dev };
}

/// The most ids one step carries (a prompt at the engine's capacity).
pub const Link = struct {
    d: *const cuda.Driver,
    lib: *const cuda.nccl.Library,
    comm: cuda.nccl.Comm,
    stream: cuda.Stream,
    buf: cuda.DeviceBuffer, // header, then ids
    ids: []u32, // host copy of the last step's ids
    gpa: std.mem.Allocator,
    wait: bool, // TF_GLM_LINK_WAIT=1: every step uploaded and waited for, as before the put
    arena: ?Arena, // the host-memory step link (see initArena): CPU stores a step, CPU loads it
    sent: u32, // rank 0: the last step's sequence written
    seen: u32, // followers: the last sequence consumed

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, max_ids: usize, bell_wait: cuda.Function) !Link {
        _ = bell_wait;
        const buf = try cuda.DeviceBuffer.alloc(d, 64 + max_ids * 4);
        const wait = if (std.c.getenv("TF_GLM_LINK_WAIT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        var l: Link = .{ .d = d, .lib = lib, .comm = comm, .stream = stream, .buf = buf, .ids = try gpa.alloc(u32, max_ids), .gpa = gpa, .wait = wait, .arena = initArena(d, max_ids), .sent = 0, .seen = 0 };
        if (l.arena) |a| {
            // the /dev/shm segment outlives a server: continue from its current sequence instead of
            // racing a fresh zero (a restarted follower on a stale seq would wait forever). Rank 0
            // also resets ack to its baseline so its first send is not blocked by a stale ack.
            const base = @atomicLoad(u32, &a.lay.seq, .acquire);
            l.sent = base;
            l.seen = base;
            @atomicStore(u32, &a.lay.ack, base, .release);
        }
        return l;
    }

    pub fn deinit(l: *Link) void {
        l.gpa.free(l.ids);
        l.buf.free();
    }

    fn bcast(l: *Link, bytes: usize) !void {
        try l.post(bytes);
        try l.stream.synchronize();
    }

    fn post(l: *Link, bytes: usize) !void {
        try l.lib.check(l.lib.api.ncclBroadcast(l.buf.ptr, l.buf.ptr, bytes, .u8, 0, l.comm, l.stream.handle), "ncclBroadcast");
    }

    /// Rank 0: the step to every rank - CPU stores into the host arena (or the NCCL fallback).
    pub fn send(l: *Link, put: cuda.Function, h: Header, ids: []const u32) !void {
        if (ids.len > l.ids.len) return error.TooManyIds;
        if (l.arena) |a| {
            const lay = a.lay;
            if (ids.len > lay.ids.len) return error.TooManyIds;
            while (@atomicLoad(u32, &lay.ack, .acquire) != l.sent) {
                std.atomic.spinLoopHint();
            }
            lay.n = @intCast(ids.len);
            const hb = std.mem.asBytes(&h);
            @memcpy(lay.hdr[0..hb.len], hb);
            @memcpy(lay.ids[0..ids.len], ids);
            l.sent +%= 1;
            @atomicStore(u32, &lay.seq, l.sent, .release);
            return;
        }
        if (!l.wait and ids.len <= put_ids) {
            var w: Words = undefined;
            @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&h)));
            @memcpy(w.w[16..][0..ids.len], ids);
            var a2: cuda.Args = .{};
            a2.add(l.buf.ptr);
            a2.add(@as(u32, @intCast(16 + ids.len)));
            a2.add(w);
            try cuda.launch.launch(put, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, l.stream, &a2);
            try l.post(64);
            if (ids.len > 0) try l.post(64 + ids.len * 4);
            return;
        }
        try l.buf.uploadAsync(0, std.mem.asBytes(&h), l.stream.handle);
        try l.post(64);
        if (ids.len > 0) {
            try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
            try l.post(64 + ids.len * 4);
        }
        if (l.wait) try l.stream.synchronize();
    }

    /// Other ranks: the next step and its ids (valid until the next receive).
    pub fn receive(l: *Link) !struct { h: Header, ids: []const u32 } {
        if (l.arena) |a| {
            const lay = a.lay;
            const want = l.seen +% 1;
            var ts = std.c.timespec{ .sec = 0, .nsec = 100 };
            while (@atomicLoad(u32, &lay.seq, .acquire) != want) {
                std.atomic.spinLoopHint();
                std.atomic.spinLoopHint();
                _ = std.c.nanosleep(&ts, null);
            }
            const n = lay.n;
            if (n > l.ids.len or n > lay.ids.len) return error.TooManyIds;
            var h: Header = undefined;
            const hb = std.mem.asBytes(&h);
            @memcpy(hb, lay.hdr[0..hb.len]);
            @memcpy(l.ids[0..n], lay.ids[0..n]);
            l.seen = want;
            @atomicStore(u32, &lay.ack, want, .release);
            return .{ .h = h, .ids = l.ids[0..n] };
        }
        try l.bcast(64);
        var h: Header = undefined;
        try l.buf.download(0, std.mem.asBytes(&h));
        if (h.n > l.ids.len) return error.TooManyIds;
        if (h.n > 0) {
            try l.bcast(64 + h.n * 4);
            try l.buf.download(64, std.mem.sliceAsBytes(l.ids[0..h.n]));
        }
        return .{ .h = h, .ids = l.ids[0..h.n] };
    }
};

/// Rank 0's engine: every step goes to the other ranks first.
pub const Leader = struct {
    e: *Engine,
    link: *Link,

    /// The step on the bound stream's sequence (its slot) to every rank.
    fn send(x: Leader, h: Header, ids: []const u32) !void {
        var hh = h;
        hh.slot = x.e.slot;
        if (std.c.getenv("TF_GLM_STEP_LOG") != null and step_logs < 400) {
            step_logs += 1;
            std.debug.print("STEP> kind={d} slot={d} n={d} seq={d}\n", .{ hh.kind, hh.slot, ids.len, x.link.sent + 1 });
        }
        try x.link.send(x.e.m.glue.put_words, hh, ids);
    }

    /// The stream's sequence for the steps that follow (each step names it, so the other ranks bind it too).
    pub fn bind(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
    }

    /// The pool's bind: rank 0's lease travels so every rank addresses the same region.
    pub fn bindRegion(x: Leader, slot: u32, start: u32, blocks: u32) !void {
        // bind locally first: send() tags every step with the engine's CURRENT slot, so a bind sent
        // first went out tagged with the previous stream's slot (the 8-stream deadlock)
        try x.e.bindRegion(slot, .{ .start = start, .blocks = blocks });
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.bind), .a = start, .depth = blocks }, &.{});
    }

    /// Frees a stream's sequence on every rank.
    pub fn release(x: Leader, slot: u32) !void {
        try x.flush();
        try x.e.bind(slot);
        try x.send(.{ .kind = @backingInt(Kind.release) }, &.{});
        x.e.release(slot);
    }

    /// States rank 0 dropped since the last step, so the other ranks free theirs (ids that fit one step at a time).
    fn flush(x: Leader) !void {
        const d = &x.e.dropped;
        while (d.items.len > 0) {
            const n = @min(d.items.len, x.link.ids.len);
            try x.send(.{ .kind = @backingInt(Kind.drop) }, d.items[0..n]);
            d.replaceRangeAssumeCapacity(0, n, &.{});
        }
    }

    pub fn begin(x: Leader, s: ?sampler.Sampling) !void {
        try x.flush();
        const c = sampler.check(s);
        var h: Header = .{ .kind = @backingInt(Kind.begin) };
        if (c) |v| {
            h.flags = 1;
            h.seed = v.seed;
            h.temperature = v.temperature;
            h.top_p = v.top_p;
            h.min_p = v.min_p;
            h.top_k = v.top_k;
        }
        try x.send(h, &.{});
        try x.e.begin(c);
    }

    /// `cancel` (rank 0's), asked before each chunk, decides for every rank; `from` every rank holds; `marks` rank 0's.
    pub fn prefill(x: Leader, ids: []const u32, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks) !u32 {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.prefill), .a = if (from) |r| r.id else 0, .seed = if (from) |r| r.at else 0 }, ids);
        return x.e.prefill(ids, from, cancel, marks);
    }

    /// One chunk of a prompt pass on every rank (Engine.prefillChunk); the first token once the chunk ends the prompt.
    pub fn chunk(x: Leader, ids: []const u32, start: usize, from: ?engine.Resume, cancel: ?engine.Cancel, marks: ?engine.Marks) !?u32 {
        if (start >= ids.len) return error.BadChunk;
        try x.flush();
        const part = ids[start..@min(ids.len, start + engine.prompt_rows + 1)];
        try x.send(.{ .kind = @backingInt(Kind.chunk), .a = if (from) |r| r.id else 0, .seed = if (from) |r| r.at else 0, .depth = @intCast(start), .top_k = @intCast(ids.len) }, part);
        return x.e.prefillChunk(part, start, ids.len, from, cancel, marks);
    }

    pub fn verify(x: Leader, ids: []const u32, out: []u32) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.verify) }, ids);
        try x.e.verify(ids, out);
    }

    /// Several streams' windows in one forward on every rank (Engine.verifyShared): parts' slots and rows, then ids.
    pub fn verifyShared(x: Leader, parts: []const engine.Part) !void {
        if (parts.len == 0 or parts.len > engine.max_shared) return error.WindowTooWide;
        try x.flush();
        var buf: [2 * engine.max_shared + engine.shared_rows]u32 = undefined;
        var n: usize = 2 * parts.len;
        for (parts, 0..) |p, i| {
            if (n + p.ids.len > buf.len) return error.WindowTooWide;
            buf[2 * i] = p.slot;
            buf[2 * i + 1] = @intCast(p.ids.len);
            @memcpy(buf[n..][0..p.ids.len], p.ids);
            n += p.ids.len;
        }
        try x.send(.{ .kind = @backingInt(Kind.shared), .depth = @intCast(parts.len) }, buf[0..n]);
        try x.e.verifyShared(parts);
    }

    pub fn keep(x: Leader, n: usize) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.keep), .a = n }, &.{});
        try x.e.keep(n);
    }

    /// A tree round's accepted path: both ranks move the same rows into slots 0..n-1.
    pub fn keepPath(x: Leader, path: []const u32) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.keep), .a = path.len }, path);
        try x.e.keepPath(path);
    }

    pub fn draft(x: Leader, from_prompt: bool, follow_ids: []const u32, depth: usize, out: []u32) !void {
        try x.flush();
        try x.send(.{ .kind = @backingInt(Kind.draft), .depth = @intCast(depth), .flags = @intFromBool(from_prompt) }, follow_ids);
        try x.e.draft(from_prompt, follow_ids, depth, out);
    }

    /// The other ranks leave their loops.
    pub fn end(x: Leader) !void {
        try x.send(.{ .kind = @backingInt(Kind.end) }, &.{});
    }
};

/// A rank other than 0 runs rank 0's steps until it ends; a cancelled prefill is error.Cancelled and the loop goes on.
pub fn follow(e: *Engine, link: *Link) !void {
    var drafts: [engine.max_drafts]u32 = undefined;
    var sampled: [16]u32 = undefined;
    while (true) {
        const step = try link.receive();
        const h = step.h;
        const kind: Kind = @fromBackingInt(@intCast(h.kind));
        if (std.c.getenv("TF_GLM_STEP_LOG") != null and step_logs < 400) {
            step_logs += 1;
            std.debug.print("STEP< kind={d} slot={d} n={d} seq={d}\n", .{ h.kind, h.slot, h.n, link.seen });
        }
        if (kind != .end and kind != .drop and kind != .shared and kind != .bind) try e.bind(h.slot);
        switch (kind) {
            .begin => try e.begin(if (h.flags & 1 != 0) .{ .seed = h.seed, .temperature = h.temperature, .top_p = h.top_p, .min_p = h.min_p, .top_k = h.top_k } else null),
            .prefill => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                const from: ?engine.Resume = if (h.a != 0) .{ .id = h.a, .at = h.seed } else null;
                _ = e.prefill(ids, from, null, null) catch |err| switch (err) {
                    error.Cancelled => {},
                    else => return err,
                };
            },
            .chunk => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                const from: ?engine.Resume = if (h.a != 0) .{ .id = h.a, .at = h.seed } else null;
                _ = e.prefillChunk(ids, h.depth, h.top_k, from, null, null) catch |err| switch (err) {
                    error.Cancelled => {},
                    else => return err,
                };
            },
            .verify => try e.verify(step.ids, sampled[0..step.ids.len]),
            .shared => {
                const k = h.depth;
                if (k == 0 or k > engine.max_shared or step.ids.len < 2 * k) return error.BadStep;
                var parts: [engine.max_shared]engine.Part = undefined;
                var outs: [engine.shared_rows]u32 = undefined;
                var at: usize = 2 * k;
                var row: usize = 0;
                for (parts[0..k], 0..) |*p, i| {
                    const n = step.ids[2 * i + 1];
                    if (at + n > step.ids.len or row + n > outs.len) return error.BadStep;
                    p.* = .{ .slot = step.ids[2 * i], .ids = step.ids[at..][0..n], .out = outs[row..][0..n] };
                    at += n;
                    row += n;
                }
                try e.verifyShared(parts[0..k]);
            },
            .keep => if (step.ids.len > 0) try e.keepPath(step.ids) else try e.keep(@intCast(h.a)),
            .draft => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                try e.draft(h.flags & 1 != 0, ids, h.depth, &drafts);
            },
            .drop => for (step.ids) |id| e.dropState(id),
            .bind => try e.bindRegion(h.slot, .{ .start = @intCast(h.a), .blocks = h.depth }),
            .release => e.release(h.slot),
            .end => return,
        }
    }
}

test "a step's header is 64 bytes and keeps its fields" {
    const h: Header = .{ .kind = @backingInt(Kind.draft), .n = 3, .depth = 2, .flags = 1, .a = 7, .seed = 1 << 62, .temperature = 0.6, .top_p = 0.95, .top_k = 20 };
    const bytes = std.mem.asBytes(&h);
    var back: Header = undefined;
    @memcpy(std.mem.asBytes(&back), bytes);
    try std.testing.expectEqual(h, back);
    try std.testing.expectEqual(Kind.draft, @as(Kind, @fromBackingInt(@intCast(back.kind))));
}
