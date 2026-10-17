//! TP over processes until #552: rank 0's Leader sends each step (header, ids, slot) first; other ranks `follow` it.

const std = @import("std");
const cuda = @import("cuda");
const engine = @import("cuda_engine.zig");
const sampler = @import("cuda_sampler.zig");
const Engine = engine.Engine;

pub const Kind = enum(u32) { begin = 1, prefill, verify, keep, draft, end, drop, release, chunk, shared };

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

/// The follower doorbell: one 64-byte /dev/shm word named from NCCL_COMM_ID (unique a launch, like
/// initShm's arena), zero at creation. TF_GLM_LINK_BELL=0 keeps the old GPU-side spin.
fn openBell() ?[*]volatile u64 {
    // opt-in WIP: the bell kills the follower's idle GPU spin (GPU1 99% -> 0%) but deadlocks the
    // step protocol as-is - the leader's flow assumes the follower's receive broadcasts are always
    // enqueued. The working design (next pass): cuStreamWaitValue32 on a cudaHostRegister-mapped
    // bell - the broadcast stays enqueued in protocol order, its EXECUTION hardware-gated on the
    // flag (no SM spin, no order change). TF_GLM_LINK_BELL=1 enables this CPU-poll variant.
    const on = std.c.getenv("TF_GLM_LINK_BELL") orelse return null;
    if (!std.mem.eql(u8, std.mem.span(on), "1")) return null;
    const cid = std.c.getenv("NCCL_COMM_ID") orelse return null;
    var h: u64 = 5381;
    for (std.mem.span(cid)) |ch| h = (h *% 33) +% ch;
    var nb: [64]u8 = undefined;
    const path = std.fmt.bufPrint(&nb, "/dev/shm/tf_glm_bell_{x}\x00", .{h}) catch return null;
    const fd = std.c.open(@ptrCast(path.ptr), .{ .ACCMODE = .RDWR, .CREAT = true }, @as(c_uint, 0o600));
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    _ = std.c.ftruncate(fd, 64);
    const m = std.posix.mmap(null, 64, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0) catch return null;
    return @ptrCast(@alignCast(m.ptr));
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
    bell: ?[*]volatile u64, // /dev/shm doorbell: rank 0's host stores a step counter once its
    // broadcasts are enqueued, so a follower never holds a spinning NCCL kernel on its GPU while
    // rank 0 is idle or computing (the GPU1 99%-idle spin; credit: jayleaton patch 0630's class)
    bell_seq: u64,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, lib: *const cuda.nccl.Library, comm: cuda.nccl.Comm, stream: cuda.Stream, max_ids: usize) !Link {
        const buf = try cuda.DeviceBuffer.alloc(d, 64 + max_ids * 4);
        const wait = if (std.c.getenv("TF_GLM_LINK_WAIT")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        return .{ .d = d, .lib = lib, .comm = comm, .stream = stream, .buf = buf, .ids = try gpa.alloc(u32, max_ids), .gpa = gpa, .wait = wait, .bell = openBell(), .bell_seq = 0 };
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

    /// Rank 0: the step and ids to every rank; up to `put_ids` ids go by `put` unwaited, longer ones upload and wait.
    /// The doorbell rings FIRST: the follower wakes and enqueues its receive broadcasts before this
    /// rank's own are enqueued, so no post can sit unpaired behind one of this rank's later syncs.
    pub fn send(l: *Link, put: cuda.Function, h: Header, ids: []const u32) !void {
        if (ids.len > l.ids.len) return error.TooManyIds;
        l.ring();
        var hh = h;
        hh.n = @intCast(ids.len);
        if (!l.wait and ids.len <= put_ids) {
            var w: Words = undefined;
            @memcpy(w.w[0..16], std.mem.bytesAsSlice(u32, std.mem.asBytes(&hh)));
            @memcpy(w.w[16..][0..ids.len], ids);
            var a: cuda.Args = .{};
            a.add(l.buf.ptr);
            a.add(@as(u32, @intCast(16 + ids.len)));
            a.add(w);
            try cuda.launch.launch(put, .{ .grid = .{ .x = 1 }, .block = .{ .x = 128 } }, l.stream, &a);
            try l.post(64);
            if (ids.len > 0) try l.post(64 + ids.len * 4);
            return;
        }
        try l.buf.uploadAsync(0, std.mem.asBytes(&hh), l.stream.handle);
        try l.bcast(64);
        if (ids.len > 0) {
            try l.buf.uploadAsync(64, std.mem.sliceAsBytes(ids), l.stream.handle);
            try l.bcast(64 + ids.len * 4);
        }
    }

    /// Rank 0 after enqueueing a step's broadcasts: the followers may wake and enqueue their side.
    fn ring(l: *Link) void {
        if (l.bell) |b| {
            l.bell_seq += 1;
            b[0] = l.bell_seq;
        }
    }

    /// A follower before enqueueing its receive: wait on the CPU doorbell, never a spinning GPU kernel.
    fn awaitBell(l: *Link) void {
        const b = l.bell orelse return;
        var ts = std.c.timespec{ .sec = 0, .nsec = 100 };
        while (b[0] == l.bell_seq) {
            std.atomic.spinLoopHint();
            std.atomic.spinLoopHint();
            _ = std.c.nanosleep(&ts, null);
        }
        l.bell_seq = b[0];
    }

    /// Other ranks: the next step and its ids (valid until the next receive).
    pub fn receive(l: *Link) !struct { h: Header, ids: []const u32 } {
        l.awaitBell();
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
        try x.link.send(x.e.m.glue.put_words, hh, ids);
    }

    /// The stream's sequence for the steps that follow (each step names it, so the other ranks bind it too).
    pub fn bind(x: Leader, slot: u32) !void {
        try x.e.bind(slot);
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
        if (kind != .end and kind != .drop and kind != .shared) try e.bind(h.slot);
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
            .keep => try e.keep(@intCast(h.a)),
            .draft => {
                const ids = try e.gpa.dupe(u32, step.ids);
                defer e.gpa.free(ids);
                try e.draft(h.flags & 1 != 0, ids, h.depth, &drafts);
            },
            .drop => for (step.ids) |id| e.dropState(id),
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
