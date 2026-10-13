//! `tensorfold` on Linux with CUDA: a model run on token ids with no Python at all, plus the oracle checks.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const glm = @import("glm");
const posix = std.posix;
const core = @import("core");
const lanes = @import("lanes");
const checks = @import("cuda_checks.zig");
const kernel_checks = @import("cuda_kernel_checks.zig");
const shared_checks = @import("cuda_shared_checks.zig");
const lanes_cli = @import("cuda_lanes.zig");
const segments_cli = @import("cuda_segments.zig");
const decode = nemotron.decode;

const usage =
    \\usage: tensorfold run MODEL --tokens ID,ID,... [--max-tokens N] [--no-drafts] [--report PATH] [--kernels DIR|native] [--device N]
    \\         [--temperature T] [--top-k K] [--top-p P] [--min-p M] [--seed S]   (temperature 0: greedy)
    \\         [--context N] [--ignore-eos] [--eager] [--costs MS1,...,MS16,LEVEL]
    \\         [--carveout]   (KV caches in GB10's display memory first; also TF_CUDA_CARVEOUT=1,
    \\         sized by TF_CUDA_CARVEOUT_MIB (default 1792), card TF_DRM_CARD (default /dev/dri/card0))
    \\         [--tokens-file PATH] [--segments N]   (N whole 2048-row prompt chunks a call as staggered segments,
    \\         1-4; default TF_CUDA_SEGMENTS, else 1)
    \\       tensorfold segments MODEL IDS_FILE... [--counts 1,2,3,4] [--repeat N] [--max-tokens N] [--profile]
    \\                (each prompt prefilled at every segment count, interleaved: state hashes, prompt and decode speed)
    \\       tensorfold check-weights MODEL DIGESTS.json [--kernels DIR]
    \\       tensorfold teacher MODEL TEACHER.json [--dump DIR] [--kernels DIR]
    \\       tensorfold prefill MODEL PROMPTS.json NAME [--dump DIR] [--kernels DIR]
    \\       tensorfold rounds MODEL --tokens ID,... [--max-tokens N]   (GPU ms a serial and a window graph round)
    \\       tensorfold check-draws MODEL FIXTURES_DIR   (MTP draws against the lane fixtures)
    \\       tensorfold shared-widths MODEL PROMPTS.json --streams N [--max-tokens N] [sampling as run]
    \\                (N streams' continuations as shared windows of varying splits against serial, 2 <= N <= 8)
    \\       tensorfold check-kernels MODEL   (lane_gemv and the forked MoE against the kernels they replace, real weights)
    \\       tensorfold widths MODEL --tokens ID,... [--max-tokens N] [--eager] [sampling as run]
    \\                (every verify width 1-16 against serial decoding, a wrong draft every third window)
    \\       tensorfold lanes MODEL PROMPTS.json [--solo] [--max-tokens N] [--no-drafts] [sampling as run] [--report PATH]
    \\                (every prompt through the lane core at once, or one at a time with --solo)
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 3 and std.mem.eql(u8, args[1], "glm-serve")) return glmServe(gpa, init, args[2..]);
    if (args.len < 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var opts = Options{ .model = args[2] };
    var positional: std.ArrayList([]const u8) = .empty;
    defer positional.deinit(gpa);
    defer gpa.free(opts.tokens);
    defer if (opts.costs) |c| gpa.free(c);
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        const value = if (i + 1 < args.len) args[i + 1] else "";
        if (std.mem.eql(u8, a, "--tokens")) {
            opts.tokens = try parseIds(gpa, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--max-tokens")) {
            opts.max_tokens = try std.fmt.parseInt(u32, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--temperature")) {
            opts.sampling.temperature = try std.fmt.parseFloat(f64, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--top-k")) {
            opts.sampling.top_k = try std.fmt.parseInt(u32, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--top-p")) {
            opts.sampling.top_p = try std.fmt.parseFloat(f64, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--min-p")) {
            opts.sampling.min_p = try std.fmt.parseFloat(f64, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--seed")) {
            opts.sampling.seed = try std.fmt.parseInt(u64, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--report")) {
            opts.report = value;
            i += 1;
        } else if (std.mem.eql(u8, a, "--kernels")) {
            opts.kernels = value;
            i += 1;
        } else if (std.mem.eql(u8, a, "--device")) {
            opts.device = try std.fmt.parseInt(u32, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--dump")) {
            opts.dump = value;
            i += 1;
        } else if (std.mem.eql(u8, a, "--costs")) {
            opts.costs = try parseFloats(gpa, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--tokens-file")) {
            gpa.free(opts.tokens);
            opts.tokens = try segments_cli.readIds(gpa, init.io, value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--segments")) {
            opts.segments = try std.fmt.parseInt(usize, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--counts")) {
            opts.counts = try parseCounts(value);
            i += 1;
        } else if (std.mem.eql(u8, a, "--streams")) {
            opts.streams = try std.fmt.parseInt(usize, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--repeat")) {
            opts.repeat = try std.fmt.parseInt(usize, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--profile")) {
            opts.profile = true;
        } else if (std.mem.eql(u8, a, "--context")) {
            opts.context = try std.fmt.parseInt(u32, value, 10);
            i += 1;
        } else if (std.mem.eql(u8, a, "--no-drafts")) {
            opts.drafts = false;
        } else if (std.mem.eql(u8, a, "--solo")) {
            opts.solo = true;
        } else if (std.mem.eql(u8, a, "--ignore-eos")) {
            opts.stop_eos = false;
        } else if (std.mem.eql(u8, a, "--eager")) {
            opts.graphs = false;
        } else if (std.mem.eql(u8, a, "--carveout")) {
            opts.carveout = true;
        } else if (std.mem.startsWith(u8, a, "--")) {
            std.debug.print("unknown option {s}\n{s}", .{ a, usage });
            return 2;
        } else try positional.append(gpa, a);
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    const device = try deviceOrdinal(opts.device, init.environ_map.get("TF_CUDA_DEVICE"));
    var ctx = try cuda.Context.init(&driver, @intCast(device));
    defer ctx.deinit();
    const kernels: ?[]const u8 = switch (try cuda.aot.pick(init.arena.allocator(), init.io, opts.kernels orelse init.environ_map.get("TENSORFOLD_CUDA_KERNELS"), try ctx.capability())) {
        .native => null,
        .captured => |dir| dir,
        .missing => |dir| {
            std.debug.print("{s} holds no aot.json: give a captured set's folder, or native for our own glue kernels\n", .{dir});
            return 2;
        },
    };
    const env = init.environ_map;
    var carve: ?cuda.Carveout = null;
    if (try cuda.carveout.requested(opts.carveout, env.get("TF_CUDA_CARVEOUT"), env.get("TF_CUDA_CARVEOUT_MIB"))) |bytes| {
        const card = try init.arena.allocator().dupeSentinel(u8, env.get("TF_DRM_CARD") orelse cuda.carveout.default_card, 0);
        carve = cuda.Carveout.open(&driver, card, bytes) catch |err| {
            std.log.err("display carveout of {d} MiB from {s}: {t} (needs nvidia_drm modeset=1, the card's device node and no display in use)", .{ bytes >> 20, card, err });
            return err;
        };
        const io_flag = if (carve.?.flags & cuda.abi.host_register_iomemory != 0) "|IOMEMORY" else "";
        std.debug.print("display carveout: {d} MiB from {s}, registered DEVICEMAP{s}\n", .{ bytes >> 20, card, io_flag });
    }
    defer if (carve) |*c| c.close();
    const cmd = args[1];
    const bench = std.mem.eql(u8, cmd, "segments");
    const decoding = std.mem.eql(u8, cmd, "run") or std.mem.eql(u8, cmd, "lanes") or bench;
    const mtp = std.mem.eql(u8, cmd, "check-weights") or std.mem.eql(u8, cmd, "rounds") or std.mem.eql(u8, cmd, "check-draws") or std.mem.eql(u8, cmd, "widths") or std.mem.eql(u8, cmd, "check-kernels") or (decoding and opts.drafts);
    const widths = std.mem.eql(u8, cmd, "widths");
    const graphs = opts.graphs and (std.mem.eql(u8, cmd, "run") or std.mem.eql(u8, cmd, "rounds") or bench or widths);
    const sampling: ?lanes.Sampling = if ((std.mem.eql(u8, cmd, "run") or widths) and opts.sampling.temperature > 0) opts.sampling else null;
    const segments = opts.segments orelse if (init.environ_map.get("TF_CUDA_SEGMENTS")) |v| try std.fmt.parseInt(usize, v, 10) else 1;
    const engine = try nemotron.Engine.init(gpa, init.io, &ctx, opts.model, kernels, .{ .context = opts.context, .mtp = mtp, .graphs = graphs, .sampling = sampling, .segments = segments, .carveout = if (carve) |*c| c else null });
    defer engine.deinit();
    if (carve) |*c| {
        const free = (try ctx.memInfo()).free;
        std.debug.print("cache room: {d} MiB free on the device + {d} MiB left in the carveout; own KV planes carved {d} MiB\n", .{ free >> 20, c.freeBytes() >> 20, engine.b.carvedBytes() >> 20 });
    }
    const rest = positional.items;
    if (std.mem.eql(u8, cmd, "run")) return run(gpa, init.io, engine, opts);
    if (std.mem.eql(u8, cmd, "lanes") and rest.len == 1) {
        const s: ?lanes.Sampling = if (opts.sampling.temperature > 0) opts.sampling else null;
        return lanes_cli.run(gpa, init.io, engine, opts.model, .{ .prompts = rest[0], .max_tokens = @intCast(opts.max_tokens), .sampling = s, .drafts = opts.drafts, .solo = opts.solo, .report = opts.report });
    }
    if (bench and rest.len >= 1) {
        var drafter: ?nemotron.Drafter = if (opts.drafts) try nemotron.Drafter.init(gpa, init.io, engine, opts.model, opts.graphs, opts.costs) else null;
        defer if (drafter) |*d| d.deinit();
        const counts = opts.counts.items[0..opts.counts.len];
        return segments_cli.run(gpa, init.io, engine, if (drafter) |*d| d else null, .{ .files = rest, .counts = counts, .repeat = opts.repeat, .max_tokens = opts.max_tokens, .stop_eos = opts.stop_eos, .profile = opts.profile, .report = opts.report });
    }
    if (std.mem.eql(u8, cmd, "check-weights") and rest.len == 1) return checks.weights(gpa, init.io, engine, rest[0]);
    if (std.mem.eql(u8, cmd, "teacher") and rest.len == 1) return checks.teacher(gpa, init.io, engine, rest[0], opts.dump);
    if (std.mem.eql(u8, cmd, "prefill") and rest.len == 2) return checks.prefill(gpa, init.io, engine, rest[0], rest[1], opts.dump);
    if (std.mem.eql(u8, cmd, "rounds")) return checks.rounds(engine, opts.tokens, opts.max_tokens);
    if (widths) return checks.widths(gpa, engine, opts.tokens, opts.max_tokens);
    if (std.mem.eql(u8, cmd, "check-draws") and rest.len == 1) return checks.draws(gpa, init.io, engine, rest[0]);
    if (std.mem.eql(u8, cmd, "shared-widths") and rest.len == 1) {
        const s: ?lanes.Sampling = if (opts.sampling.temperature > 0) opts.sampling else null;
        return shared_checks.widths(gpa, init.io, engine, rest[0], opts.streams, opts.max_tokens, s);
    }
    if (std.mem.eql(u8, cmd, "check-kernels")) {
        try engine.reset();
        return kernel_checks.check(gpa, engine);
    }
    std.debug.print("{s}", .{usage});
    return 2;
}

/// glm-serve MODEL [--port N] [--context N]: the GLM CUDA engine behind a
/// minimal JSON-over-HTTP surface (token ids in, ids out; greedy). One request
/// at a time on the engine's stream - the serving slice of Phase 3.
fn glmServe(gpa: std.mem.Allocator, init: std.process.Init, rest: []const [:0]const u8) !u8 {
    if (rest.len < 1) {
        std.debug.print("usage: tensorfold glm-serve MODEL [--port N] [--context N]", .{});
        return 2;
    }
    var port: u16 = 8890;
    var context: usize = 4096;
    var i: usize = 1;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--port") and i + 1 < rest.len) {
            port = try std.fmt.parseInt(u16, rest[i + 1], 10);
            i += 1;
        } else if (std.mem.eql(u8, rest[i], "--context") and i + 1 < rest.len) {
            context = try std.fmt.parseInt(usize, rest[i + 1], 10);
            i += 1;
        }
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    // the 22+22 split: the full checkpoint fits only across both GPUs
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var eng = try glm.engine.Engine.init(gpa, init.io, &driver, rest[0], .{ .context = context, .blocks_lo = 0, .blocks_hi = 22, .stop_after_blocks = true });
    eng.quiet = true;
    defer eng.deinit();
    var ctx1 = try cuda.Context.init(&driver, 1);
    defer ctx1.deinit();
    try ctx1.makeCurrent();
    var eng_b = try glm.engine.Engine.init(gpa, init.io, &driver, rest[0], .{ .context = context, .blocks_lo = 22, .blocks_hi = 46, .cont = true });
    // engine A's mtp_emb relay buffer must land on GPU 0: allocate under A's context
    try ctx.makeCurrent();
    try eng.wantMtpEmb();
    for (rest) |a| {
        if (std.mem.eql(u8, a, "--dump-pf")) pf_dump = true;
    }
    try ctx.makeCurrent();
    try eng.wantPrefillCache();
    try ctx1.makeCurrent();
    try eng_b.wantPrefillCache();
    try ctx1.makeCurrent();
    eng_b.quiet = true;
    defer eng_b.deinit();
    // the server listener (their listener.zig pattern, raw posix here)
    const rc = posix.system.socket(posix.AF.INET, posix.SOCK.STREAM, 0);
    if (posix.errno(rc) != .SUCCESS) {
        std.debug.print("socket failed", .{});
        return 1;
    }
    const fd: posix.socket_t = @intCast(rc);
    const one: c_int = 1;
    posix.setsockopt(fd, posix.SOL.SOCKET, posix.SO.REUSEADDR, std.mem.asBytes(&one)) catch {};
    var addr: posix.sockaddr.in = .{ .family = posix.AF.INET, .port = std.mem.nativeToBig(u16, port), .addr = 0 };
    if (posix.errno(posix.system.bind(fd, @ptrCast(&addr), @sizeOf(@TypeOf(addr)))) != .SUCCESS) {
        std.debug.print("bind failed on port {d}", .{port});
        return 1;
    }
    if (posix.errno(posix.system.listen(fd, 128)) != .SUCCESS) {
        std.debug.print("listen failed", .{});
        return 1;
    }
    std.debug.print("glm-serve: {s} on 0.0.0.0:{d} (POST /generate with a JSON body)", .{ rest[0], port });
    // thread-per-connection: connections are accepted concurrently and queue on
    // the engine mutex (the GPU pair is a single serialized resource; streams
    // interleave only via the request queue for now)
    while (true) {
        const rc2 = posix.system.accept(fd, null, null);
        if (posix.errno(rc2) != .SUCCESS) continue;
        const conn_fd: posix.socket_t = @intCast(rc2);
        const th = std.Thread.spawn(.{}, handleConn, .{ gpa, eng, eng_b, ctx, ctx1, conn_fd }) catch {
            _ = posix.system.close(conn_fd);
            continue;
        };
        th.detach();
    }
    return 0;
}

var engine_mutex: std.Io.Mutex = .init;

// per-request phase timers (ns)
var t_pa: i128 = 0;
var t_h: i128 = 0;
var t_pb: i128 = 0;
var t_ra: i128 = 0;
var t_rh: i128 = 0;
var t_rb: i128 = 0;
var t_rp: i128 = 0;
inline fn tnow() i128 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

fn handleConn(gpa: std.mem.Allocator, eng: *glm.engine.Engine, eng_b: *glm.engine.Engine, ctx: cuda.Context, ctx1: cuda.Context, fd: posix.socket_t) void {
    defer _ = posix.system.close(fd);
    engine_mutex.lockUncancelable(eng.io);
    defer engine_mutex.unlock(eng.io);
    serveSplit(gpa, eng, eng_b, ctx, ctx1, fd) catch |err| {
        std.debug.print("request: {}", .{err});
    };
}

/// One request through the split: the prompt on A, handoff, tail on B; then
/// decode rows alternate A -> handoff -> B -> handoff... (the split-decode
/// test's pattern, folded into the server).
var pf_dump = false; // --dump-pf: dump state at cold-snapshot and warm-restore
var req_counter: u32 = 0;

fn dumpPfState(tag: []const u8, eng: *glm.engine.Engine, gpa: std.mem.Allocator) void {
    const pairs = .{
        .{ "x0", eng.x[0] },      .{ "x1", eng.x[1] },
        .{ "br", eng.branch },    .{ "post", eng.post },
        .{ "comb", eng.comb },    .{ "st0", eng.kda_st0 },
        .{ "st1", eng.kda_st1 },  .{ "cs", eng.kda_cs },
        .{ "mk", eng.mla_keys },  .{ "mik", eng.mla_ik },
        .{ "mig", eng.mla_ig },   .{ "mpool", eng.mla_pool },
    };
    inline for (pairs) |pr| {
        const buf = pr[1];
        if (buf.len != 0) {
            const hb = gpa.alloc(u8, buf.len) catch return;
            defer gpa.free(hb);
            buf.download(0, hb) catch return;
            var nm: [64]u8 = undefined;
            const f = std.fmt.bufPrint(&nm, "/tmp/pf_{s}_{s}.bin", .{ tag, pr[0] }) catch return;
            var dir = std.Io.Dir.cwd();
            dir.writeFile(eng.io, .{ .sub_path = f, .data = hb }) catch {};
        }
    }
}

// single-entry prefill cache: the last prompt's full state
var pf_valid: bool = false;
var pf_key: u64 = 0;
var pf_len: usize = 0;
var pf_cur_a: [64]usize = @splat(0);
var pf_cur_b: [64]usize = @splat(0);
var pf_xi_a: usize = 0;
var pf_xi_b: usize = 0;
var pf_pick0: u32 = 0;
var pf_draft0: u32 = 0;

fn hashIds(ids: []const u32) u64 {
    var h: u64 = 1469598103934665603;
    for (ids) |v| {
        h ^= v;
        h *%= 1099511628211;
    }
    return h;
}

fn serveSplit(gpa: std.mem.Allocator, eng: *glm.engine.Engine, eng_b: *glm.engine.Engine, ctx: cuda.Context, ctx1: cuda.Context, fd: posix.socket_t) !void {
    req_counter += 1;
    eng_b.mtp_tag = req_counter;
    eng_b.mtp_calls = 0;
    var buf: [1 << 16]u8 = undefined;
    var n: usize = 0;
    while (n < buf.len) {
        const got: isize = @bitCast(posix.system.read(fd, buf[n..].ptr, buf.len - n));
        if (got <= 0) break;
        n += @intCast(got);
        if (std.mem.indexOfScalar(u8, buf[0..n], '}') != null) break;
    }
    const body = blk: {
        const idx = std.mem.indexOf(u8, buf[0..n], "\r\n\r\n") orelse break :blk buf[0..n];
        break :blk buf[idx + 4 .. n];
    };
    var ids_buf: [8192]u32 = @splat(0);
    var n_ids: usize = 0;
    var max_tokens: usize = 64;
    if (std.mem.indexOf(u8, body, "\"prompt\"")) |p0| {
        var j = p0;
        while (j < body.len and body[j] != '[') j += 1;
        if (j < body.len) {
            j += 1;
            var val: u32 = 0;
            var in_num = false;
            while (j < body.len and body[j] != ']') : (j += 1) {
                const ch = body[j];
                if (ch >= '0' and ch <= '9') {
                    val = val * 10 + (ch - '0');
                    in_num = true;
                } else if (in_num) {
                    if (n_ids < ids_buf.len) {
                        ids_buf[n_ids] = val;
                        n_ids += 1;
                    }
                    val = 0;
                    in_num = false;
                }
            }
            if (in_num and n_ids < ids_buf.len) {
                ids_buf[n_ids] = val;
                n_ids += 1;
            }
        }
    }
    if (std.mem.indexOf(u8, body, "\"max_tokens\"")) |m0| {
        var j = m0;
        while (j < body.len and body[j] != ':') j += 1;
        if (j < body.len) {
            j += 1;
            var v: usize = 0;
            var any = false;
            while (j < body.len and body[j] >= '0' and body[j] <= '9') : (j += 1) {
                v = v * 10 + (body[j] - '0');
                any = true;
            }
            if (any) max_tokens = v;
        }
    }
    if (n_ids == 0) {
        _ = posix.system.write(fd, "HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n", 45);
        return;
    }
    // the prompt: A then B (or a prefill-cache hit)
    const key = hashIds(ids_buf[0..n_ids]);
    var cache_hit = false;
    if (pf_valid and pf_key == key and pf_len == n_ids) {
        try ctx.makeCurrent();
        try eng.restorePrefill();
        @memcpy(eng.kda_st_cur, pf_cur_a[0..eng.kda_st_cur.len]);
        eng.xi = pf_xi_a;
        eng.fwd_count = 1;
        try eng.stream.synchronize();
        try ctx1.makeCurrent();
        try eng_b.restorePrefill();
        @memcpy(eng_b.kda_st_cur, pf_cur_b[0..eng_b.kda_st_cur.len]);
        eng_b.xi = pf_xi_b;
        eng_b.fwd_count = 1;
        try eng_b.stream.synchronize();
        cache_hit = true;
        if (pf_dump) {
            dumpPfState("warmA", eng, gpa);
            try ctx1.makeCurrent();
            dumpPfState("warmB", eng_b, gpa);
            std.debug.print("pf: warm state dumped\n", .{});
        }
    }
    try ctx.makeCurrent();
    if (!cache_hit) try eng.resetState();
    const lx0 = eng.x[0].len;
    const lx1 = eng.x[1].len;
    const lbr = eng.branch.len;
    const lpo = eng.post.len;
    const lco = eng.comb.len;
    const lme = eng.mtp_emb.len;
    const o1 = lx0;
    const o2 = o1 + lx1;
    const o3 = o2 + lbr;
    const o4 = o3 + lpo;
    const ot = o4 + lco + lme;
    const hb = try gpa.alloc(u8, 32 * 1024 * 1024);
    defer gpa.free(hb);
    if (!cache_hit) {
        // the whole cold prompt pass + handoff: on a cache hit the restore
        // already brought BOTH engines to the post-prompt state - re-running
        // the prompt double-advances A's KDA state and resetState() zeroes
        // B's restore (the cur=0/fwd=1 signature)
        const ta0 = tnow();
        try eng.forward(ids_buf[0..n_ids], n_ids, 0);
        try eng.stream.synchronize();
        const ta1 = tnow();
        t_pa += ta1 - ta0;
        try eng.x[0].download(0, hb[0..o1]);
        try eng.x[1].download(0, hb[o1..o2]);
        try eng.branch.download(0, hb[o2..o3]);
        try eng.post.download(0, hb[o3..o4]);
        try eng.comb.download(0, hb[o4 .. o4 + lco]);
        if (lme != 0) try eng.mtp_emb.download(0, hb[o4 + lco .. ot]);
        const xi_hand = eng.xi;
        try ctx1.makeCurrent();
        try eng_b.resetState();
        try eng_b.x[0].upload(0, hb[0..o1]);
        try eng_b.x[1].upload(0, hb[o1..o2]);
        try eng_b.branch.upload(0, hb[o2..o3]);
        try eng_b.post.upload(0, hb[o3..o4]);
        try eng_b.comb.upload(0, hb[o4 .. o4 + lco]);
        if (lme != 0) try eng_b.mtp_emb.upload(0, hb[o4 + lco .. ot]);
        eng_b.xi = xi_hand;
        eng_b.fwd_count = 0;
        t_h += tnow() - ta1;
    }
    if (!cache_hit) {
        const tb0 = tnow();
        try eng_b.forward(ids_buf[0..n_ids], n_ids, 0);
        try eng_b.stream.synchronize();
        t_pb += tnow() - tb0;
        // snapshot the post-prompt state for the next identical prompt
        try eng_b.snapshotPrefill();
        @memset(pf_cur_b[0..eng_b.kda_st_cur.len], 0);
        @memcpy(pf_cur_b[0..eng_b.kda_st_cur.len], eng_b.kda_st_cur);
        pf_xi_b = eng_b.xi;
        try ctx.makeCurrent();
        try eng.snapshotPrefill();
        @memset(pf_cur_a[0..eng.kda_st_cur.len], 0);
        @memcpy(pf_cur_a[0..eng.kda_st_cur.len], eng.kda_st_cur);
        pf_xi_a = eng.xi;
        pf_key = key;
        pf_len = n_ids;
        pf_valid = true;
        if (pf_dump) {
            dumpPfState("coldA", eng, gpa);
            try ctx1.makeCurrent();
            dumpPfState("coldB", eng_b, gpa);
            try ctx.makeCurrent();
            std.debug.print("pf: cold state dumped\n", .{});
        }
        try ctx1.makeCurrent();
    }
    var pick: [128]u8 = undefined;
    var draft_ids: [512]u32 = @splat(0); // draft[i] = MTP pick made at step i (predicts token i+1)
    const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, &pick));
    if (!cache_hit) {
        try eng_b.picks.download(0, pick[0 .. 4 * n_ids]);
        if (eng_b.mtp_li != null) {
            try eng_b.mtp_pick.download(0, pick[8..12]);
            draft_ids[0] = pv[2];
        }
        pf_pick0 = pv[n_ids - 1];
        pf_draft0 = if (eng_b.mtp_li != null) pv[2] else 0;
    } else {
        draft_ids[0] = pf_draft0;
    }
    eng_b.mtp_dump_nth = if (cache_hit) 1 else 2; // the round-1 draft (cold: prompt draft is call 1)
    var pos: usize = n_ids;
    var out_ids: [512]u32 = @splat(0);

    var n_out: usize = 0;
    out_ids[0] = if (cache_hit) pf_pick0 else pv[n_ids - 1];
    n_out = 1;
    var one: [1]u32 = .{0};
    var two: [2]u32 = .{ 0, 0 };
    const have_draft = eng_b.mtp_li != null;
    var cur_draft: u32 = if (have_draft) draft_ids[0] else 0;
    var accepts: usize = 0;
    var rejects: usize = 0;
    // the handoff rides the ENGINE streams: null-stream copies are unordered
    // against the engines' non-blocking streams (the ids-race bug class)
    while (n_out < max_tokens and n_out < out_ids.len and pos < 4000) {
        if (have_draft and cur_draft != 0) {
            // verify round: [last_pick, draft] - accept banks 2 tokens
            two[0] = out_ids[n_out - 1];
            two[1] = cur_draft;
            if (pf_dump and n_out == 1) {
                eng.dump_on = true;
                eng.quiet = false;
                eng_b.dump_on = true;
                eng_b.quiet = false;
            }
            const va0 = tnow();
            try ctx.makeCurrent();
            try eng.backupCs();
            try eng.forward(two[0..2], 2, pos);
            try eng.stream.synchronize();
            t_ra += tnow() - va0;
            try eng.x[0].downloadAsync(0, hb[0..o1], eng.stream.handle);
            try eng.x[1].downloadAsync(0, hb[o1..o2], eng.stream.handle);
            try eng.branch.downloadAsync(0, hb[o2..o3], eng.stream.handle);
            try eng.post.downloadAsync(0, hb[o3..o4], eng.stream.handle);
            try eng.comb.downloadAsync(0, hb[o4 .. o4 + lco], eng.stream.handle);
            if (lme != 0) try eng.mtp_emb.downloadAsync(0, hb[o4 + lco .. ot], eng.stream.handle);
            try eng.stream.synchronize();
            const xi_v = eng.xi;
            try ctx1.makeCurrent();
            try eng_b.backupCs();
            try eng_b.x[0].uploadAsync(0, hb[0..o1], eng_b.stream.handle);
            try eng_b.x[1].uploadAsync(0, hb[o1..o2], eng_b.stream.handle);
            try eng_b.branch.uploadAsync(0, hb[o2..o3], eng_b.stream.handle);
            try eng_b.post.uploadAsync(0, hb[o3..o4], eng_b.stream.handle);
            try eng_b.comb.uploadAsync(0, hb[o4 .. o4 + lco], eng_b.stream.handle);
            if (lme != 0) try eng_b.mtp_emb.uploadAsync(0, hb[o4 + lco .. ot], eng_b.stream.handle);
            eng_b.xi = xi_v;
            if (pf_dump and n_out == 1) {
                try eng_b.stream.synchronize();
                const bb = try gpa.alloc(u8, 2 * 4 * 4096 * 4);
                defer gpa.free(bb);
                try eng_b.x[0].download(0, bb);
                var nmB: [48]u8 = undefined;
                const fB = std.fmt.bufPrint(&nmB, "/tmp/m1_{d}_bx.bin", .{req_counter}) catch unreachable;
                var dirB = std.Io.Dir.cwd();
                dirB.writeFile(eng_b.io, .{ .sub_path = fB, .data = bb }) catch {};
                const ab = try gpa.alloc(u8, 2 * 4 * 4096 * 4);
                defer gpa.free(ab);
                try eng.x[0].download(0, ab);
                var nmA: [48]u8 = undefined;
                const fA = std.fmt.bufPrint(&nmA, "/tmp/m1_{d}_ax.bin", .{req_counter}) catch unreachable;
                var dirA = std.Io.Dir.cwd();
                dirA.writeFile(eng.io, .{ .sub_path = fA, .data = ab }) catch {};
            }
            const vb0 = tnow();
            try eng_b.forward(two[0..2], 2, pos);
            try eng_b.stream.synchronize();
            t_rb += tnow() - vb0;
            try eng_b.picks.downloadAsync(0, pick[0..8], eng_b.stream.handle);
            if (eng_b.mtp_li != null) try eng_b.mtp_pick.downloadAsync(0, pick[8..12], eng_b.stream.handle);
            try eng_b.stream.synchronize();
            if (pf_dump and n_out == 1) {
                eng.dump_on = false;
                eng.quiet = true;
                eng_b.dump_on = false;
                eng_b.quiet = true;
            }
            if (pv[0] == cur_draft) {
                accepts += 1;
                if (n_out < out_ids.len) {
                    out_ids[n_out] = cur_draft;
                    n_out += 1;
                }
                if (n_out < max_tokens and n_out < out_ids.len) {
                    out_ids[n_out] = pv[1];
                    n_out += 1;
                }
                cur_draft = pv[2];
                pos += 2;
                continue;
            }
            rejects += 1;
            try ctx.makeCurrent();
            try eng.rollbackRound();
            try ctx1.makeCurrent();
            try eng_b.rollbackRound();
            cur_draft = 0; // the 1-row round regenerates a draft
        }
        one[0] = out_ids[n_out - 1];
        const oa0 = tnow();
        try ctx.makeCurrent();
        try eng.forward(one[0..1], 1, pos);
        try eng.stream.synchronize();
        t_ra += tnow() - oa0;
        try eng.x[0].downloadAsync(0, hb[0..o1], eng.stream.handle);
        try eng.x[1].downloadAsync(0, hb[o1..o2], eng.stream.handle);
        try eng.branch.downloadAsync(0, hb[o2..o3], eng.stream.handle);
        try eng.post.downloadAsync(0, hb[o3..o4], eng.stream.handle);
        try eng.comb.downloadAsync(0, hb[o4 .. o4 + lco], eng.stream.handle);
        if (lme != 0) try eng.mtp_emb.downloadAsync(0, hb[o4 + lco .. ot], eng.stream.handle);
        try eng.stream.synchronize();
        const oh0 = tnow();
        const xi2 = eng.xi;
        try ctx1.makeCurrent();
        try eng_b.x[0].uploadAsync(0, hb[0..o1], eng_b.stream.handle);
        try eng_b.x[1].uploadAsync(0, hb[o1..o2], eng_b.stream.handle);
        try eng_b.branch.uploadAsync(0, hb[o2..o3], eng_b.stream.handle);
        try eng_b.post.uploadAsync(0, hb[o3..o4], eng_b.stream.handle);
        try eng_b.comb.uploadAsync(0, hb[o4 .. o4 + lco], eng_b.stream.handle);
        if (lme != 0) try eng_b.mtp_emb.uploadAsync(0, hb[o4 + lco .. ot], eng_b.stream.handle);
        t_rh += tnow() - oh0;
        eng_b.xi = xi2;
        const ob0 = tnow();
        try eng_b.forward(one[0..1], 1, pos);
        try eng_b.stream.synchronize();
        t_rb += tnow() - ob0;
        const op0 = tnow();
        try eng_b.picks.downloadAsync(0, pick[0..8], eng_b.stream.handle);
        if (eng_b.mtp_li != null and n_out < draft_ids.len) {
            try eng_b.mtp_pick.downloadAsync(0, pick[8..12], eng_b.stream.handle);
            try eng_b.stream.synchronize();
            draft_ids[n_out - 1] = pv[2];
            cur_draft = pv[2];
        }
        try eng_b.stream.synchronize();
        t_rp += tnow() - op0;
        out_ids[n_out] = pv[0];
        n_out += 1;
        pos += 1;
    }
    std.debug.print("glm-serve: accepts {d} rejects {d} | ms: promptA {d:.1} hoff {d:.1} promptB {d:.1} | rounds: A {d:.1} hoff {d:.1} B {d:.1} pick {d:.1}\n", .{ accepts, rejects, @as(f64, @floatFromInt(t_pa)) / 1e6, @as(f64, @floatFromInt(t_h)) / 1e6, @as(f64, @floatFromInt(t_pb)) / 1e6, @as(f64, @floatFromInt(t_ra)) / 1e6, @as(f64, @floatFromInt(t_rh)) / 1e6, @as(f64, @floatFromInt(t_rb)) / 1e6, @as(f64, @floatFromInt(t_rp)) / 1e6 });
    var jbuf: [8192]u8 = undefined;
    var w: usize = 0;
    w += (std.fmt.bufPrint(jbuf[w..], "{{\"tokens\":[", .{}) catch unreachable).len;
    for (out_ids[0..n_out], 0..) |t, ti| {
        const sep: []const u8 = if (ti > 0) "," else "";
        w += (std.fmt.bufPrint(jbuf[w..], "{s}{d}", .{ sep, t }) catch unreachable).len;
    }
    w += (std.fmt.bufPrint(jbuf[w..], "],\"count\":{d},\"drafts\":[", .{ n_out }) catch unreachable).len;
    for (draft_ids[0..n_out], 0..) |t, ti| {
        const sep: []const u8 = if (ti > 0) "," else "";
        w += (std.fmt.bufPrint(jbuf[w..], "{s}{d}", .{ sep, t }) catch unreachable).len;
    }
    w += (std.fmt.bufPrint(jbuf[w..], "]}}", .{}) catch unreachable).len;
    var hdr: [128]u8 = undefined;
    const h = try std.fmt.bufPrint(&hdr, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{w});
    _ = posix.system.write(fd, h.ptr, h.len);
    _ = posix.system.write(fd, &jbuf, w);
}

fn serveFdUnused() void {}

const Options = struct {
    model: []const u8,
    tokens: []u32 = &.{},
    max_tokens: usize = 256,
    sampling: lanes.Sampling = .{ .seed = 0, .temperature = 0 },
    drafts: bool = true,
    report: ?[]const u8 = null,
    kernels: ?[]const u8 = null,
    device: ?u32 = null,
    dump: ?[]const u8 = null,
    context: ?usize = null,
    stop_eos: bool = true,
    graphs: bool = true,
    solo: bool = false,
    costs: ?[]f64 = null,
    segments: ?usize = null,
    counts: Counts = .{ .items = .{ 1, 2, 3, 4 }, .len = 4 },
    repeat: usize = 3,
    streams: usize = 4,
    profile: bool = false,
    carveout: bool = false,
};

/// Segment counts for `segments`, one to four of them.
const Counts = struct { items: [4]usize, len: usize };

fn parseCounts(text: []const u8) !Counts {
    var c: Counts = .{ .items = undefined, .len = 0 };
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |t| {
        if (c.len == c.items.len) return error.TooManyCounts;
        c.items[c.len] = try std.fmt.parseInt(usize, t, 10);
        c.len += 1;
    }
    return c;
}

/// The GPU ordinal `--device` or `TF_CUDA_DEVICE` picks before an engine loads (PR #354); nothing set means 0.
fn deviceOrdinal(flag: ?u32, env: ?[]const u8) !u32 {
    if (flag) |d| return d;
    const value = env orelse return 0;
    if (value.len == 0) return 0;
    return std.fmt.parseInt(u32, value, 10);
}

fn parseFloats(gpa: std.mem.Allocator, text: []const u8) ![]f64 {
    var out: std.ArrayList(f64) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseFloat(f64, t));
    return out.toOwnedSlice(gpa);
}

fn parseIds(gpa: std.mem.Allocator, text: []const u8) ![]u32 {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", ");
    while (it.next()) |t| try out.append(gpa, try std.fmt.parseInt(u32, t, 10));
    return out.toOwnedSlice(gpa);
}

fn run(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, o: Options) !u8 {
    if (o.tokens.len == 0) return error.NoPromptTokens;
    const room = e.max_len - o.tokens.len - nemotron.state.max_rows;
    const count = @max(1, @min(o.max_tokens, room));
    var drafter: ?nemotron.Drafter = if (o.drafts) try nemotron.Drafter.init(gpa, io, e, o.model, o.graphs, o.costs) else null;
    defer if (drafter) |*d| d.deinit();
    if (drafter) |d| {
        const v = d.rule.costs.verify;
        std.debug.print("up to 15 MTP drafts a round, each verified while it pays for its row (measured: {d:.2}/{d:.2}/{d:.2}/{d:.2}/{d:.2} ms at 1/2/4/8/16 rows, {d:.3} ms a draft)\n", .{ v[1], v[2], v[4], v[8], v[16], d.rule.costs.level });
    }
    const res = try decode.generate(gpa, io, e, if (drafter) |*d| d else null, o.tokens, count, .{ .stop_eos = o.stop_eos });
    defer gpa.free(res.tokens);
    var digest: [32]u8 = undefined;
    const text = try core.ids_json.write(gpa, res.tokens);
    defer gpa.free(text);
    std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    const steps = @max(1, res.tokens.len - 1);
    const ms = res.decode_seconds * 1e3 / @as(f64, @floatFromInt(steps));
    std.debug.print("tokens {d} sha {s} prefill {d:.4}s decode {d:.4}s {d:.3} ms/token rounds {d} accepted {d} segments {d}\n", .{ res.tokens.len, hex[0..12], res.prefill_seconds, res.decode_seconds, ms, res.rounds, res.accepted, e.segments });
    if (o.report) |path| {
        const report = .{
            .engine = "zig-cuda",
            .prompt_tokens = o.tokens,
            .tokens = res.tokens,
            .token_sha256 = hex,
            .prefill_seconds = res.prefill_seconds,
            .decode_seconds = res.decode_seconds,
            .ms_per_token = ms,
            .rounds = res.rounds,
            .accepted_drafts = res.accepted,
            .drafted = res.drafted,
            .drafts = o.drafts,
            .sampling = e.sampling,
            .graphs = o.graphs,
            .load_seconds = e.load_seconds,
            .max_len = e.max_len,
            .segments = e.segments,
            .carved_bytes = e.b.carvedBytes(),
        };
        const json = try std.json.Stringify.valueAlloc(gpa, report, .{});
        defer gpa.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }
    return 0;
}

test "the device ordinal comes from --device, then TF_CUDA_DEVICE, else 0" {
    try std.testing.expectEqual(@as(u32, 1), try deviceOrdinal(1, null));
    try std.testing.expectEqual(@as(u32, 1), try deviceOrdinal(1, "7")); // the flag wins
    try std.testing.expectEqual(@as(u32, 3), try deviceOrdinal(null, "3"));
    try std.testing.expectEqual(@as(u32, 0), try deviceOrdinal(null, null));
    try std.testing.expectEqual(@as(u32, 0), try deviceOrdinal(null, ""));
    try std.testing.expectError(error.Overflow, deviceOrdinal(null, "4294967296"));
}
