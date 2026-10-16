//! cuBLASLt and NCCL through dlopen: one bf16 GEMM against a host reference, one-rank collectives.

const std = @import("std");
const cuda = @import("cuda");
const graph = @import("cuda").graph;
const check = @import("check.zig");
const Gpu = check.Gpu;
const expect = check.expect;

/// Round-to-nearest-even fp32 to bf16 bits (inputs here are finite).
fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn widen(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

pub fn cublaslt(gpu: Gpu) !void {
    const d = gpu.d;
    var lt = try cuda.cublaslt.Library.open();
    defer lt.close();
    const m: usize = 16;
    const n: usize = 256;
    const k: usize = 512;
    var rng = std.Random.DefaultPrng.init(1234);
    const r = rng.random();
    const x = try gpu.gpa.alloc(u16, m * k);
    defer gpu.gpa.free(x);
    const w = try gpu.gpa.alloc(u16, n * k);
    defer gpu.gpa.free(w);
    for (x) |*v| v.* = bf16(r.floatNorm(f32));
    for (w) |*v| v.* = bf16(r.floatNorm(f32) * 0.05);

    var xd = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x));
    defer xd.free();
    var wd = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(w));
    defer wd.free();
    var out = try cuda.DeviceBuffer.alloc(d, m * n * 4);
    defer out.free();
    const workspace_len: usize = 32 << 20;
    var ws = try cuda.DeviceBuffer.alloc(d, workspace_len);
    defer ws.free();
    var linear = try cuda.cublaslt.Linear.init(&lt, m, n, k, .f32, workspace_len);
    defer linear.deinit();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try linear.run(xd.ptr, wd.ptr, out.ptr, ws.ptr, workspace_len, stream.handle);
    try stream.synchronize();
    const first = try check.download(gpu, out);
    defer gpu.gpa.free(first);
    try out.fill8(0, null);
    try linear.run(xd.ptr, wd.ptr, out.ptr, ws.ptr, workspace_len, stream.handle);
    try stream.synchronize();
    const second = try check.download(gpu, out);
    defer gpu.gpa.free(second);
    try check.sameBytes("cuBLASLt rerun", second, first);

    var worst: f64 = 0;
    const got = std.mem.bytesAsSlice(f32, first);
    for (0..m) |i| for (0..n) |j| {
        var acc: f64 = 0;
        for (0..k) |t| acc += @as(f64, widen(x[i * k + t])) * @as(f64, widen(w[j * k + t]));
        worst = @max(worst, @abs(@as(f64, got[i * n + j]) - acc));
    };
    try expect(worst < 1e-3, "cuBLASLt max abs error {e} vs fp64 reference", .{worst});
    check.pass("cuBLASLt {d}: bf16 x bf16 -> fp32 ({d}x{d}x{d}), reruns bit-equal, max abs error {e} vs fp64, workspace {d} B", .{ lt.api.cublasLtGetVersion(), m, n, k, worst, linear.workspace_need });
}

/// `nccl-pair RANK WORLD ID`: one rank of a multi-host communicator (id as hex in file ID): send/recv, sums, gathers.

/// `ipc-pair`: two ranks exchange cuIpcMemHandle tokens through files, map each other's
/// buffers, and run the gather protocol bare: a cross-GPU DtoD slice copy each way, a
/// stream-ordered flag write, and a cuStreamWaitValue32 on the peer's flag. Verifies both
/// slices landed - the probe that must pass on a box before the engine's gathers go IPC.
pub fn ipcPair(gpu: Gpu, rank: c_int, world: c_int, id_path: []const u8) !void {
    _ = world;
    _ = id_path;
    const d = gpu.d;
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    const s = stream.handle;

    const bytes: usize = 16 << 10; // one decode gather's size: rows x hidden fp32
    const half = bytes / 2;
    var data = try cuda.DeviceBuffer.alloc(d, bytes);
    defer data.free();
    var flag = try cuda.DeviceBuffer.alloc(d, 4);
    defer flag.free();
    var seqbuf = try cuda.DeviceBuffer.alloc(d, 4);
    defer seqbuf.free();

    // the flags start at 0 (the peer waits >= 1; garbage could pass the wait early)
    try d.check(d.api.cuMemsetD32Async(flag.ptr, 0, 1, s), "flag zero");
    try d.check(d.api.cuMemsetD32Async(data.ptr, 0, bytes / 4, s), "data zero");
    try stream.synchronize();

    var pair: [2]cuda.abi.IpcMemHandle = .{ .{ .data = @splat(0xAA) }, .{ .data = @splat(0xAA) } };
    std.debug.print("rank {d}: data.ptr={x} flag.ptr={x}\n", .{ rank, data.ptr, flag.ptr });
    try d.check(d.api.cuIpcGetMemHandle(&pair[0], data.ptr), "cuIpcGetMemHandle");
    std.debug.print("rank {d}: h0 after first get: {x}\n", .{ rank, std.mem.readInt(u64, pair[0].data[0..8], .little) });
    try d.check(d.api.cuIpcGetMemHandle(&pair[1], flag.ptr), "cuIpcGetMemHandle(flag)");
    std.debug.print("rank {d}: after get: h0={x} h1={x} (0xAAAA.. = get wrote nothing)\n", .{ rank, std.mem.readInt(u64, pair[0].data[0..8], .little), std.mem.readInt(u64, pair[1].data[0..8], .little) });

    // handle exchange through files (the probe needs no comm): mine out, the peer's in
    var tokens: [2][2]cuda.abi.IpcMemHandle = @splat(@splat(undefined));
    tokens[@intCast(rank)] = pair;
    {
        var nb: [64]u8 = undefined;
        const mine_path = try std.fmt.bufPrint(&nb, "ipc_tk{d}.bin", .{rank});
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = mine_path, .data = std.mem.sliceAsBytes(&pair) });
        const peer_path = try std.fmt.bufPrint(&nb, "ipc_tk{d}.bin", .{1 - rank});
        var tries: usize = 0;
        while (tries < 200) : (tries += 1) {
            if (std.Io.Dir.cwd().readFile(gpu.io, peer_path, std.mem.sliceAsBytes(&tokens[@intCast(1 - rank)]))) |_| {
                break;
            } else |_| {
                var ts: std.posix.timespec = .{ .sec = 0, .nsec = 50_000_000 };
                _ = std.c.nanosleep(&ts, null);
            }
        }
        if (tries == 200) return error.PeerHandlesMissing;
    }
    std.debug.print("rank {d}: tok0={x} tok1={x}\n", .{ rank, std.mem.readInt(u64, tokens[0][0].data[0..8], .little), std.mem.readInt(u64, tokens[1][0].data[0..8], .little) });

    const peer: usize = @intCast(1 - rank);
    const vis0 = std.c.getenv("CUDA_VISIBLE_DEVICES");
    const phys1 = vis0 != null and std.mem.eql(u8, std.mem.span(vis0.?), "1");
    var peer_data: cuda.abi.DevicePtr = 0;
    var peer_flag: cuda.abi.DevicePtr = 0;
    // ONLY THE MOVER OPENS: this box's fabric maps GPU0's memory from GPU1, never the reverse
    // (the reverse open ERRORS or HANGS - the waiter must never try it). The waiter needs no
    // mapping at all: it fills its slice locally and spins on its own flag.
    if (phys1) {
        try d.check(d.api.cuIpcOpenMemHandle_v2(&peer_data, tokens[peer][0], 1), "cuIpcOpenMemHandle");
        try d.check(d.api.cuIpcOpenMemHandle_v2(&peer_flag, tokens[peer][1], 1), "cuIpcOpenMemHandle(flag)");
    } else {
        std.debug.print("rank {d}: waiter - no open (one-directional fabric)\n", .{rank});
    }

    // THE ONE-DIRECTIONAL PROTOCOL (this box: the GPU1-side maps GPU0, never the reverse).
    // The mover (physical GPU1) does ALL cross-fabric work; the waiter (physical GPU0)
    // fills its slice locally and spins on its own flag.
    std.debug.print("rank {d}: step-1 pre-module\n", .{rank});
    var spin_mod = try cuda.Module.load(d, cuda.kernels.probe);
    std.debug.print("rank {d}: step-2 module loaded\n", .{rank});
    defer spin_mod.unload();
    const spin = try spin_mod.function("tf_ipc_spin");

    const pattern: u32 = 0x5A000000 | @as(u32, @intCast(rank));
    std.debug.print("rank {d}: step-3 pre-memset\n", .{rank});
    try d.check(d.api.cuMemsetD32Async(data.ptr + @as(usize, @intCast(rank)) * half, pattern, half / 4, s), "memset my slice");
    try stream.synchronize();
    std.debug.print("rank {d}: step-4 slice set\n", .{rank});

    var t_move: i128 = 0;
    var t_wait: i128 = 0;
    const rounds: usize = 1000;
    for (0..rounds) |iter| {
        const seq: c_uint = @intCast(iter + 1);
        if (phys1) {
            const t0 = std.Io.Clock.awake.now(gpu.io).toNanoseconds();
            // mover: my slice into the peer's buffer; the peer's slice into mine; raise its flag
            try d.check(d.api.cuMemcpyDtoDAsync_v2(peer_data + @as(usize, @intCast(rank)) * half, data.ptr + @as(usize, @intCast(rank)) * half, half, s), "mover: my slice to peer");
            try d.check(d.api.cuMemcpyDtoDAsync_v2(data.ptr + (1 - @as(usize, @intCast(rank))) * half, peer_data + (1 - @as(usize, @intCast(rank))) * half, half, s), "mover: peer slice to me");
            { // memset on remote-mapped memory silently no-ops on this box; a 4-byte copy lands
                var sb: [4]u8 = undefined;
                std.mem.writeInt(u32, &sb, seq, .little);
                try seqbuf.upload(0, &sb);
                try d.check(d.api.cuMemcpyDtoDAsync_v2(peer_flag, seqbuf.ptr, 4, s), "mover: raise peer flag");
            }
            try stream.synchronize();
            t_move += std.Io.Clock.awake.now(gpu.io).toNanoseconds() - t0;
        } else {
            const t0 = std.Io.Clock.awake.now(gpu.io).toNanoseconds();
            var a: cuda.Args = .{};
            a.add(flag.ptr);
            a.add(@as(c_int, @intCast(seq)));
            try cuda.launch.launch(spin, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } }, stream, &a);
            try stream.synchronize();
            t_wait += std.Io.Clock.awake.now(gpu.io).toNanoseconds() - t0;
        }
    }

    var host: [4096]u32 = undefined;
    try data.download(0, std.mem.sliceAsBytes(&host));
    var ok = true;
    for (host[0 .. half / 4]) |v| if (v != 0x5A000000) {
        ok = false;
    };
    for (host[half / 4 ..]) |v| if (v != 0x5A000001) {
        ok = false;
    };
    std.debug.print("rank {d}: slot0[0]={x} slot1[0]={x} slot0[mid]={x} slot1[mid]={x}\n", .{ rank, host[0], host[half / 4], host[half / 8], host[half / 4 + half / 8] });
    try expect(ok, "both ranks' slices landed over the one-directional protocol", .{});
    std.debug.print("rank {d} ({s}): total {d}ms over {d} rounds = {d:.1}us/gather-exchange\n", .{ rank, if (phys1) "mover" else "waiter", @divTrunc(t_move, std.time.ns_per_ms), rounds, @as(f64, @floatFromInt(if (phys1) t_move else t_wait)) / @as(f64, @floatFromInt(rounds)) / 1000.0 });
    check.pass("IPC pair: rank {d} one-directional protocol works (copy + spin sync)", .{rank});
    if (phys1) {
        _ = d.api.cuMemFree_v2(peer_data);
        _ = d.api.cuMemFree_v2(peer_flag);
    }
    check.pass("IPC pair: rank {d} saw both slices (cross-GPU DtoD + wait-value work on this box)", .{rank});
}


/// `mm16-check <n> <k> <m>`: the ported glm_mm16 (bf16 tiled) against a float64 host matmul on
/// random data - the standalone gate for the dense-linear path before it runs in the engine.
pub fn mm16Check(gpu: Gpu, n_: usize, k_: usize, m_: usize) !void {
    const d = gpu.d;
    var g = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = g.random();
    const a = gpu.gpa;
    const wh = try a.alloc(u16, n_ * k_);
    defer a.free(wh);
    const xh = try a.alloc(u16, m_ * k_);
    defer a.free(xh);
    for (wh) |*v| v.* = bf16bits(rnd.float(f32) * 2 - 1);
    for (xh) |*v| v.* = bf16bits(rnd.float(f32) * 2 - 1);
    var wraw = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(wh));
    defer wraw.free();
    var xbuf = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(xh));
    defer xbuf.free();
    var tiled = try cuda.DeviceBuffer.alloc(d, n_ * k_ * 2);
    defer tiled.free();
    var out = try cuda.DeviceBuffer.alloc(d, m_ * n_ * 4);
    defer out.free();
    var part = try cuda.DeviceBuffer.alloc(d, m_ * n_ * 4);
    defer part.free();

    const glm_mod = @import("glm_cuda");
    var mod = try cuda.Module.load(d, cuda.kernels.glm_glue);
    defer mod.unload();
    const mm = try glm_mod.glue.Mm16.load(mod);
    var st = try cuda.Stream.init(d, true);
    defer st.deinit();
    try mm.tile(st, wraw.ptr, tiled.ptr, n_, k_);
    if (std.c.getenv("MM16_GRAPH") != null) {
        // the engine runs the fill inside captured decode-window graphs: replicate
        try graph.beginCapture(st, .thread_local);
        try mm.run(st, xbuf.ptr, k_, m_, n_, k_, tiled.ptr, 0, 16, out.ptr, part.ptr, true);
        const gr = try graph.endCapture(st);
        var ex = try gr.instantiate();
        defer ex.deinit();
        try ex.launchOn(st);
    } else {
        try mm.run(st, xbuf.ptr, k_, m_, n_, k_, tiled.ptr, 0, 16, out.ptr, part.ptr, true);
    }
    try st.synchronize();

    const got = try a.alloc(f32, m_ * n_);
    defer a.free(got);
    try out.download(0, std.mem.sliceAsBytes(got));
    var worst: f64 = 0;
    var at: usize = 0;
    for (0..m_) |r| for (0..n_) |c| {
        var acc: f64 = 0;
        for (0..k_) |kk| acc += bf16f(wh[c * k_ + kk]) * bf16f(xh[r * k_ + kk]);
        const err = @abs(@as(f64, got[r * n_ + c]) - acc);
        if (err > worst) {
            worst = err;
            at = r * n_ + c;
        }
    };
    try expect(worst < 0.05, "glm_mm16 bf16 vs float64: worst abs error {d:.4} at {d} (m {d} n {d} k {d})", .{ worst, at, m_, n_, k_ });
    check.pass("mm16-check: worst abs error {d:.4} over m={d} n={d} k={d}", .{ worst, m_, n_, k_ });
}

fn bf16bits(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf16f(bits: u16) f64 {
    const v: f32 = @bitCast(@as(u32, bits) << 16);
    return v;
}

pub fn ncclPair(gpu: Gpu, rank: c_int, world: c_int, id_path: []const u8) !void {
    const d = gpu.d;
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    var id: cuda.nccl.UniqueId = undefined;
    if (rank == 0) {
        try lib.check(lib.api.ncclGetUniqueId(&id), "ncclGetUniqueId");
        const hex = std.fmt.bytesToHex(id.internal, .lower);
        try std.Io.Dir.cwd().writeFile(gpu.io, .{ .sub_path = id_path, .data = &hex });
    } else {
        var hex: [256]u8 = undefined;
        const got = try std.Io.Dir.cwd().readFile(gpu.io, id_path, &hex);
        try expect(got.len == 256, "unique id file {s}: {d} hex digits", .{ id_path, got.len });
        _ = try std.fmt.hexToBytes(&id.internal, &hex);
    }
    var comm: cuda.nccl.Comm = null;
    try lib.check(lib.api.ncclCommInitRank(&comm, world, id, rank), "ncclCommInitRank");
    defer _ = lib.api.ncclCommDestroy(comm);
    const count: usize = 1 << 20;
    const W: usize = @intCast(world);
    const R: usize = @intCast(rank);
    const mine = try gpu.gpa.alloc(f32, count);
    defer gpu.gpa.free(mine);
    for (mine, 0..) |*v, i| v.* = @floatFromInt(R * 1000 + i % 977);
    var send = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(mine));
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, count * 4 * W);
    defer recv.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
    for (0..W) |p| if (p != R) {
        try lib.check(lib.api.ncclSend(send.ptr, count, .f32, @intCast(p), comm, stream.handle), "ncclSend");
        try lib.check(lib.api.ncclRecv(recv.ptr + p * count * 4, count, .f32, @intCast(p), comm, stream.handle), "ncclRecv");
    };
    try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    try stream.synchronize();
    const got = try check.download(gpu, recv);
    defer gpu.gpa.free(got);
    const want = try gpu.gpa.alloc(f32, count);
    defer gpu.gpa.free(want);
    for (0..W) |p| if (p != R) {
        for (want, 0..) |*v, i| v.* = @floatFromInt(p * 1000 + i % 977);
        try check.sameBytes("send/recv from a peer", got[p * count * 4 ..][0 .. count * 4], std.mem.sliceAsBytes(want));
    };
    try lib.check(lib.api.ncclAllReduce(send.ptr, recv.ptr, count, .f32, .sum, comm, stream.handle), "ncclAllReduce");
    try stream.synchronize();
    try recv.download(0, got[0 .. count * 4]);
    for (want, 0..) |*v, i| {
        var s: f32 = 0;
        for (0..W) |p| s += @floatFromInt(p * 1000 + i % 977);
        v.* = s;
    }
    try check.sameBytes("all-reduce", got[0 .. count * 4], std.mem.sliceAsBytes(want));
    // all-gather: every rank's pattern in rank order
    try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, count, .f32, comm, stream.handle), "ncclAllGather");
    try stream.synchronize();
    try recv.download(0, got);
    for (0..W) |p| {
        for (want, 0..) |*v, i| v.* = @floatFromInt(p * 1000 + i % 977);
        try check.sameBytes("all-gather", got[p * count * 4 ..][0 .. count * 4], std.mem.sliceAsBytes(want));
    }
    check.pass("NCCL {d}: rank {d} of {d}, send/recv with every peer ({d} MiB each), all-reduce and all-gather exact", .{ try lib.version(), rank, world, count * 4 >> 20 });
}

pub fn nccl(gpu: Gpu) !void {
    const d = gpu.d;
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const version = try lib.version();
    var comms: [1]cuda.nccl.Comm = .{null};
    const devices = [1]c_int{gpu.ctx.device};
    try lib.check(lib.api.ncclCommInitAll(&comms, 1, &devices), "ncclCommInitAll");
    defer _ = lib.api.ncclCommDestroy(comms[0]);
    const count: usize = 4096;
    const src = try gpu.gpa.alloc(f32, count);
    defer gpu.gpa.free(src);
    for (src, 0..) |*v, i| v.* = @floatFromInt(i);
    var send = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(src));
    defer send.free();
    var recv = try cuda.DeviceBuffer.alloc(d, count * 4);
    defer recv.free();
    var stream = try cuda.Stream.init(d, true);
    defer stream.deinit();
    try lib.check(lib.api.ncclAllReduce(send.ptr, recv.ptr, count, .f32, .sum, comms[0], stream.handle), "ncclAllReduce");
    try stream.synchronize();
    const got = try check.download(gpu, recv);
    defer gpu.gpa.free(got);
    try check.sameBytes("one-rank all-reduce", got, std.mem.sliceAsBytes(src));
    try recv.fill8(0, null);
    try lib.check(lib.api.ncclAllGather(send.ptr, recv.ptr, count, .f32, comms[0], stream.handle), "ncclAllGather");
    try stream.synchronize();
    try recv.download(0, got);
    try check.sameBytes("one-rank all-gather", got, std.mem.sliceAsBytes(src));
    // a send to this rank and its receive, in one group (NCCL pairs them)
    try recv.fill8(0, null);
    try lib.check(lib.api.ncclGroupStart(), "ncclGroupStart");
    try lib.check(lib.api.ncclSend(send.ptr, count, .f32, 0, comms[0], stream.handle), "ncclSend");
    try lib.check(lib.api.ncclRecv(recv.ptr, count, .f32, 0, comms[0], stream.handle), "ncclRecv");
    try lib.check(lib.api.ncclGroupEnd(), "ncclGroupEnd");
    try stream.synchronize();
    try recv.download(0, got);
    try check.sameBytes("one-rank send/recv", got, std.mem.sliceAsBytes(src));
    check.pass("NCCL {d}: one-rank communicator, all-reduce, all-gather and send/recv to itself exact", .{version});
}
