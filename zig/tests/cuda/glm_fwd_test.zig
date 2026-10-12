//! The engine's first real-weights smoke: blocks 0..3 (dense MLP + KDA attn),
//! four tokens, logits checksum + argmax ids printed.

const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const check = @import("check.zig");

const Gpu = check.Gpu;

pub fn run(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const nm: usize = if (rest.len > 1) std.fmt.parseInt(usize, rest[1], 10) catch 3 else 3;
    const split = rest.len > 2 and std.mem.eql(u8, rest[2], "split");
    const wantF16 = (rest.len > 2 and std.mem.eql(u8, rest[2], "f16")) or (rest.len > 3 and std.mem.eql(u8, rest[3], "f16"));
    if (split) return runSplit(gpu, dir, nm / 2, nm);
    const skip_mla = rest.len > 2 and std.mem.eql(u8, rest[2], "skip-mla");
    const a = gpu.gpa;
    var e = try glm.engine.Engine.init(a, gpu.io, gpu.d, dir, .{ .blocks_lo = 0, .blocks_hi = nm, .skip_mla = skip_mla });
    defer e.deinit();

    // per-block stream fingerprints (first 26 blocks)
    const sums = try a.alloc(f64, 46);
    defer a.free(sums);
    @memset(sums, 0);
    e.block_sums = sums;
    const asumsv = try a.alloc(f64, 46);
    defer a.free(asumsv);
    @memset(asumsv, 0);
    e.attn_sums = asumsv;
    if (wantF16) e.f32_streams = false;
    if (rest.len > 2 and std.mem.eql(u8, rest[2], "decode")) {
        // incremental decode: one prompt forward at pos 0, then one row at a
        // time with the carried KDA state / conv history / MLA caches
        var ids19: [24]u32 = @splat(0);
        var rows0d: usize = 4;
        {
            var cdir1 = std.Io.Dir.cwd();
            if (cdir1.openFile(gpu.io, "pf.ids", .{})) |fh1| {
                var f1 = fh1;
                defer f1.close(gpu.io);
                var buf1: [96]u8 = undefined;
                if (f1.readPositional(gpu.io, &[1][]u8{buf1[0..]}, 0)) |n1| {
                    const cnt1 = @min(n1 / 4, 24);
                    for (0..cnt1) |i| ids19[i] = std.mem.readInt(u32, buf1[i * 4 ..][0..4], .little);
                    rows0d = cnt1;
                } else |_| {}
            } else |_| {}
        }
        var pos: usize = 0;
        try e.forward(ids19[0..rows0d], rows0d, pos);
        pos += rows0d;
        var outd: []u8 = try a.alloc(u8, 4096);
        defer a.free(outd);
        var wd: usize = 0;
        var toks: [24]u32 = @splat(0);
        var ntok: usize = 0;
        {
            try e.stream.synchronize();
            const pk = try check.download(gpu, e.picks);
            defer a.free(pk);
            const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pk));
            toks[ntok] = pv[rows0d - 1];
            std.debug.print("  prompt pick {d} pv0 {d}\n", .{ pv[rows0d - 1], pv[0] });

            ntok += 1;
        }
        var cur_tok: [1]u32 = .{0};
        while (ntok < 16 and pos < 4090) : (ntok += 1) {
            cur_tok[0] = toks[ntok - 1];
            try e.forward(cur_tok[0..1], 1, pos);
            pos += 1;
            try e.stream.synchronize();
            const pk = try check.download(gpu, e.picks);
            defer a.free(pk);
            const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pk));
            toks[ntok] = pv[0];
        }
        for (toks[0..ntok]) |t| {
            const printed = std.fmt.bufPrint(outd[wd..], "{d} ", .{t}) catch unreachable;
            wd += printed.len;
        }
        std.debug.print("  decode[{d}]: {s}\n", .{ ntok, outd[0..wd] });

        return;
    }
    if (rest.len > 2 and std.mem.eql(u8, rest[2], "oneshot")) {
        // one 16-row window, full picks: the token-1 parity probe
        var ids16: [16]u32 = .{ 785, 6722, 315, 9621, 374, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        try e.forward(&ids16, 5, 0);
        try e.stream.synchronize();
        try e.stream.synchronize();
            const pk = try check.download(gpu, e.picks);
        defer a.free(pk);
        const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pk));
        var out: []u8 = try a.alloc(u8, 160);
        defer a.free(out);
        var w: usize = 0;
        for (pv[0..5]) |t| {
            const printed = std.fmt.bufPrint(out[w..], "{d} ", .{t}) catch unreachable;
            w += printed.len;
        }
        std.debug.print("  oneshot picks: {s}\n", .{out[0..w]});

        return;
    }

    {
        // state zero check: the first 64 floats of kda_st0 right after init
        try e.stream.synchronize();
        const hb = try a.alloc(u8, 256);
        defer a.free(hb);
        try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = e.kda_st0.ptr, .len = 256, .borrowed = true }).download(0, hb);
        const fv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, hb));
        var nz: usize = 0;
        for (fv) |v| if (v != 0) {
            nz += 1;
        };
        std.debug.print("  st0 head nonzero {d}/64 first {d:.4}\n", .{ nz, fv[0] });

        // full-buffer census: every float of kda_st0
        const full = try a.alloc(u8, e.kda_st0.len);
        defer a.free(full);
        try e.kda_st0.download(0, full);
        var nzf: usize = 0;
        const ff = std.mem.bytesAsSlice(f32, @as([]align(1) u8, full));
        for (ff) |x| if (x != 0) {
            nzf += 1;
        };
        std.debug.print("  st0 FULL nonzero {d}/{d}\n", .{ nzf, ff.len });
    }
    var ids_buf: [24]u32 = .{ 785, 6722, 315, 9621, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    var rows0: usize = 4;
    {
        var cdir0 = std.Io.Dir.cwd();
        if (cdir0.openFile(gpu.io, "pf.ids", .{})) |fh0| {
            var f0 = fh0;
            defer f0.close(gpu.io);
            var buf0: [96]u8 = undefined;
            if (f0.readPositional(gpu.io, &[1][]u8{buf0[0..]}, 0)) |n0| {
                const cnt0 = @min(n0 / 4, 24);
                for (0..cnt0) |i| ids_buf[i] = std.mem.readInt(u32, buf0[i * 4 ..][0..4], .little);
                rows0 = cnt0;
            } else |_| {}
        } else |_| {}
    }
    var n_gen: usize = 0;
    while (n_gen < 5) : (n_gen += 1) {
        const rows_now = rows0 + n_gen;
        if (rows_now > 23) break;
        try e.forward(ids_buf[0..rows_now], rows_now, 0);
        try e.stream.synchronize();
        try e.stream.synchronize();
            const pk = try check.download(gpu, e.picks);
        defer a.free(pk);
        const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pk));
        ids_buf[rows_now] = pv[rows_now - 1];
        if (n_gen == 0) {
            // first forward, fresh state: the parity fingerprint vs their taps
            var out2: []u8 = try a.alloc(u8, nm * 20 + 8);
            defer a.free(out2);
            var w2: usize = 0;
            for (0..@min(nm, 26)) |b| {
                const pr = std.fmt.bufPrint(out2[w2..], "{d}:{d:.4} ", .{ b, sums[b] }) catch unreachable;
                w2 += pr.len;
            }
            {
                var out3: []u8 = try a.alloc(u8, nm * 20 + 8);
                defer a.free(out3);
                var w3: usize = 0;
                for (1..@min(nm, 26)) |b| {
                    const pr = std.fmt.bufPrint(out3[w3..], "{d}:{d:.4} ", .{ b, asumsv[b - 1] }) catch unreachable;
                    w3 += pr.len;
                }
                std.debug.print("  firstpass attnsums {s}\n", .{out3[0..w3]});

            }
            std.debug.print("  firstpass blocksums {s}\n", .{out2[0..w2]});
        }
    }
    {
        var out: []u8 = try a.alloc(u8, 160);
        defer a.free(out);
        var w: usize = 0;
        for (0..rows0 + n_gen) |t| {
            if (t == rows0) {
                @memcpy(out[w .. w + 3], " | ");
                w += 3;
            }
            const printed = std.fmt.bufPrint(out[w..], "{d} ", .{ids_buf[t]}) catch unreachable;
            w += printed.len;
        }
        std.debug.print("  greedy[{d}]: {s}" ++ "\n", .{ rows0 + n_gen, out[0..w] });
    }
    {
        var w2: usize = 0;
        const out2 = try a.alloc(u8, 46 * 20);
        defer a.free(out2);
        for (0..@min(nm, 26)) |b| {
            const pr = std.fmt.bufPrint(out2[w2..], "{d}:{d:.4} ", .{ b, sums[b] }) catch unreachable;
            w2 += pr.len;
        }
        std.debug.print("  blocksums {s}" ++ "\n", .{out2[0..w2]});
    }
    const rows: usize = rows0;
    const ids = ids_buf;
    // probe: embed alone before any block
    try e.ids.upload(0, std.mem.sliceAsBytes(ids[0..rows]));
    for (0..4) |s_i| try e.ops.embed(e.stream, e.ids.ptr, e.w.embed, e.x[0].ptr + s_i * e.c.hidden * 2, rows, e.c.hidden, 4 * e.c.hidden);
    try e.stream.synchronize();
    {
        const xb = try check.download(gpu, e.x[0]);
        defer a.free(xb);
        var nans: usize = 0;
        var first: ?usize = null;
        for (0..rows * 4 * e.c.hidden) |i| {
            const h: u16 = std.mem.readInt(u16, xb[i * 2 ..][0..2], .little);
            const v = @as(f32, @bitCast(@as(u32, h) << 16));
            if (std.math.isNan(v)) {
                nans += 1;
                if (first == null) first = i;
            }
        }
        std.debug.print("  embed stage: {d} NaNs, first {s}" ++ "\n", .{ nans, if (first) |v| std.fmt.allocPrint(a, "{d}", .{v}) catch "?" else "-" });
    }
    try e.kda_st0.fill32(0, e.stream.handle);
    try e.kda_st1.fill32(0, e.stream.handle);
    try e.kda_cs.fill8(0, e.stream.handle);
@memset(e.kda_st_cur, 0);
    try e.forward(&ids, rows, 0);
    try e.stream.synchronize();
    // per-buffer NaN census after the walk
    {
        const bufs = [_]struct { name: []const u8, b: cuda.DeviceBuffer }{
            .{ .name = "x0", .b = e.x[0] }, .{ .name = "x1", .b = e.x[1] },
            .{ .name = "branch", .b = e.branch }, .{ .name = "normed", .b = e.normed },
            .{ .name = "y", .b = e.y }, .{ .name = "qkv", .b = e.qkv },
            .{ .name = "fg", .b = e.fg }, .{ .name = "mlp_act", .b = e.mlp_act },
            .{ .name = "post", .b = e.post }, .{ .name = "comb", .b = e.comb },
            .{ .name = "mla_ql", .b = e.mla_ql }, .{ .name = "mla_att", .b = e.mla_att },
            .{ .name = "mla_keys", .b = e.mla_keys }, .{ .name = "mla_vals", .b = e.mla_vals },
            .{ .name = "mla_qp", .b = e.mla_qp }, .{ .name = "mla_xp", .b = e.mla_xp },
            .{ .name = "mla_ik", .b = e.mla_ik }, .{ .name = "mla_pool", .b = e.mla_pool },
            .{ .name = "mla_iw", .b = e.mla_iw },
        };
        for (bufs) |bb| {
            const bytes = try check.download(gpu, bb.b);
            defer a.free(bytes);
            var nans: usize = 0;
            const n16 = bytes.len / 2;
            for (0..n16) |i| {
                const h: u16 = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
                const v = @as(f32, @bitCast(@as(u32, h) << 16));
                if (std.math.isNan(v)) nans += 1;
            }
            std.debug.print("  {s}: {d}/{d} NaN" ++ "\n", .{ bb.name, nans, n16 });
        }
    }
    const picks = try check.download(gpu, e.picks);
    defer a.free(picks);
    const lg = try check.download(gpu, e.logits);
    defer a.free(lg);
    var sum: f64 = 0;
    var mx: f32 = -1e30;
    var mx_at: usize = 0;
    for (0..rows * e.c.vocab) |i| {
        const h: u16 = std.mem.readInt(u16, lg[i * 2 ..][0..2], .little);
        const v = @as(f32, @bitCast(@as(u32, h) << 16));
        sum += v;
        if (v > mx) {
            mx = v;
            mx_at = i % e.c.vocab;
        }
    }
    const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, picks));
    // where does NaN begin? probe the window's stage buffers via fresh downloads
    {
        const xb = try check.download(gpu, e.x[e.xi]);
        defer a.free(xb);
        var nan_at: ?usize = null;
        for (0..rows * 4 * e.c.hidden) |i| {
            const h: u16 = std.mem.readInt(u16, xb[i * 2 ..][0..2], .little);
            const v = @as(f32, @bitCast(@as(u32, h) << 16));
            if (std.math.isNan(v)) {
                nan_at = i;
                break;
            }
        }
        std.debug.print("  streams first NaN at {s} (len {d})" ++ "\n", .{ if (nan_at) |v| std.fmt.allocPrint(a, "{d}", .{v}) catch "oom" else "none", rows * 4 * e.c.hidden });
    }
    check.pass("glm fwd smoke: logits sum {d:.1}, max {d:.2} at {d}; picks {d},{d},{d},{d}", .{ sum, mx, mx_at, pv[0], pv[1], pv[2], pv[3] });
}

fn runSplit(gpu: Gpu, dir: []const u8, split_at: usize, nm: usize) !void {
    const a = gpu.gpa;
    // engine A on this context (device 0)
    var ea = try glm.engine.Engine.init(a, gpu.io, gpu.d, dir, .{ .blocks_lo = 0, .blocks_hi = split_at, .stop_after_blocks = true });
    var rows: usize = 19;
    var ids: [24]u32 = .{ 151644, 872, 198, 4063, 374, 2207, 315, 9621, 30, 151645, 198, 151644, 77091, 198, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    // prompt file override: pf.ids in the cwd (u32 LE, count = rows from argv not supported; first 16)
    {
        var cdir = std.Io.Dir.cwd();
        if (cdir.openFile(gpu.io, "pf.ids", .{})) |fh| {
            var f = fh;
            defer f.close(gpu.io);
            var buf: [64]u8 = undefined;
            if (f.readPositional(gpu.io, &[1][]u8{buf[0..]}, 0)) |n| {
                const cnt = @min(n / 4, 24);
                for (0..cnt) |i| ids[i] = std.mem.readInt(u32, buf[i * 4 ..][0..4], .little);
                rows = cnt;
            } else |_| {}
        } else |_| {}
    }
    const sumsW = try a.alloc(f64, 46);
    @memset(sumsW, 0);
    ea.block_sums = sumsW;
    try ea.forward(&ids, rows, 0);
    try ea.stream.synchronize();
    // the handoff state through the host: x (both planes), branch, post, comb
    // download the handoff under A's context BEFORE B's context becomes current
    const Handoff = struct { x0: []u8, x1: []u8, branch: []u8, post: []u8, comb: []u8 };
    var ho: Handoff = undefined;
    ho.x0 = try check.download(gpu, ea.x[0]);
    ho.x1 = try check.download(gpu, ea.x[1]);
    ho.branch = try check.download(gpu, ea.branch);
    ho.post = try check.download(gpu, ea.post);
    ho.comb = try check.download(gpu, ea.comb);
    const xi_hand = ea.xi;
    var ctx1 = try cuda.Context.init(gpu.d, 1);
    defer ctx1.deinit();
    var eb = try glm.engine.Engine.init(a, gpu.io, gpu.d, dir, .{ .blocks_lo = split_at, .blocks_hi = @max(nm, split_at + 1), .cont = true });
    defer eb.deinit();
    const sumsWb = try a.alloc(f64, 46);
    @memset(sumsWb, 0);
    eb.block_sums = sumsWb;
    try eb.x[0].upload(0, ho.x0);
    try eb.x[1].upload(0, ho.x1);
    try eb.branch.upload(0, ho.branch);
    try eb.post.upload(0, ho.post);
    try eb.comb.upload(0, ho.comb);
    defer {
        a.free(ho.x0);
        a.free(ho.x1);
        a.free(ho.branch);
        a.free(ho.post);
        a.free(ho.comb);
    }
    eb.xi = xi_hand;
    eb.fwd_count = 0;
    try eb.forward(&ids, rows, 0);
    try eb.stream.synchronize();
    const snapA1 = try a.alloc(f64, 46);
    const snapB1 = try a.alloc(f64, 46);
    @memcpy(snapA1, sumsW);
    @memcpy(snapB1, sumsWb);
    const lg = try check.download(gpu, eb.logits);
    defer a.free(lg);
    var sum: f64 = 0;
    var mx: f32 = -1e30;
    var mx_at: usize = 0;
    for (0..rows * eb.c.vocab) |i| {
        const h: u16 = std.mem.readInt(u16, lg[i * 2 ..][0..2], .little);
        const v = @as(f32, @bitCast(@as(u32, h) << 16));
        sum += v;
        if (v > mx) {
            mx = v;
            mx_at = i % eb.c.vocab;
        }
    }
    const pk = try check.download(gpu, eb.picks);
    defer a.free(pk);
    const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pk));
    // incremental decode through the split (A -> handoff -> B per token); the
    // handoff shuttles x/branch/post/comb under A's context, then B runs one row
    var pos: usize = rows;
    var toks: [24]u32 = @splat(0);
    var ntok: usize = 0;
    var cur_tok: [1]u32 = .{0};
    toks[0] = pv[rows - 1];
    ntok = 1;
    while (ntok < 16 and pos < 4000) : (ntok += 1) {
        cur_tok[0] = toks[ntok - 1];
        try gpu.ctx.makeCurrent();
        try ea.forward(cur_tok[0..1], 1, pos);
        // handoff A -> B
        ho.x0 = try check.download(gpu, ea.x[0]);
        ho.x1 = try check.download(gpu, ea.x[1]);
        ho.branch = try check.download(gpu, ea.branch);
        ho.post = try check.download(gpu, ea.post);
        ho.comb = try check.download(gpu, ea.comb);
        try eb.x[0].upload(0, ho.x0);
        try eb.x[1].upload(0, ho.x1);
        try eb.branch.upload(0, ho.branch);
        try eb.post.upload(0, ho.post);
        try eb.comb.upload(0, ho.comb);
        eb.xi = ea.xi;
        try ctx1.makeCurrent();
        try eb.forward(cur_tok[0..1], 1, pos);
        try eb.stream.synchronize();
        const pkd = try check.download(gpu, eb.picks);
        defer a.free(pkd);
        const pvd = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pkd));
        toks[ntok] = pvd[0];
        a.free(ho.x0);
        a.free(ho.x1);
        a.free(ho.branch);
        a.free(ho.post);
        a.free(ho.comb);
        pos += 1;
    }
    {
        var outd: []u8 = try a.alloc(u8, 400);
        defer a.free(outd);
        var wd: usize = 0;
        for (toks[0..ntok]) |t| {
            const printed = std.fmt.bufPrint(outd[wd..], "{d} ", .{t}) catch unreachable;
            wd += printed.len;
        }
        std.debug.print("  split-decode[{d}]: {s}" ++ "\n", .{ ntok, outd[0..wd] });
    }
    // WINDOW 2: same prompt+decode after resetState — serving replays this per request.
    // If picks diverge, some state survives the reset.
    {
        const toks1 = toks;
        const ntok1 = ntok;
        try gpu.ctx.makeCurrent();
        try ea.resetState();
        try ctx1.makeCurrent();
        try eb.resetState();
        try gpu.ctx.makeCurrent();
        try ea.forward(&ids, rows, 0);
        try ea.stream.synchronize();
        const hx0 = try check.download(gpu, ea.x[0]);
        const hx1 = try check.download(gpu, ea.x[1]);
        const hbr = try check.download(gpu, ea.branch);
        const hpo = try check.download(gpu, ea.post);
        const hco = try check.download(gpu, ea.comb);
        try ctx1.makeCurrent();
        try eb.x[0].upload(0, hx0);
        try eb.x[1].upload(0, hx1);
        try eb.branch.upload(0, hbr);
        try eb.post.upload(0, hpo);
        try eb.comb.upload(0, hco);
        eb.xi = ea.xi;
        eb.fwd_count = 0;
        try eb.forward(&ids, rows, 0);
        try eb.stream.synchronize();
        {
            var diverged = false;
            for (0..split_at) |bi| {
                const d = sumsW[bi] - snapA1[bi];
                if (@abs(d) > 1e-3 * (1 + @abs(snapA1[bi]))) {
                    std.debug.print("  W2 A blk {d}: {d:.5} -> {d:.5} (diff {d:.5})\n", .{ bi, snapA1[bi], sumsW[bi], d });
                    diverged = true;
                }
            }
            for (split_at..nm) |bi| {
                const d = sumsWb[bi] - snapB1[bi];
                if (@abs(d) > 1e-3 * (1 + @abs(snapB1[bi]))) {
                    std.debug.print("  W2 B blk {d}: {d:.5} -> {d:.5} (diff {d:.5})\n", .{ bi, snapB1[bi], sumsWb[bi], d });
                    diverged = true;
                }
            }
            if (!diverged) std.debug.print("  W2 blocks: all fingerprints match\n", .{});
        }
        // bit-diff the per-block probe dumps: window1 pb{bi}_{tag}.bin vs window2 pb{bi}_{tag}2.bin
        {
            var cdir = std.Io.Dir.cwd();
            var ndiff: usize = 0;
            var first = true;
            const tags = [4][]const u8{ "na", "ba", "nf", "bf" };
            const fsz = rows * 4096 * 2;
            const b1 = try a.alloc(u8, fsz);
            defer a.free(b1);
            const b2 = try a.alloc(u8, fsz);
            defer a.free(b2);
            for (0..nm) |bi| {
                for (tags) |tg| {
                    var n1: [24]u8 = undefined;
                    var n2: [24]u8 = undefined;
                    const f1n = std.fmt.bufPrint(&n1, "pb{d}_{s}.bin", .{ bi, tg }) catch continue;
                    const f2n = std.fmt.bufPrint(&n2, "pb{d}_{s}2.bin", .{ bi, tg }) catch continue;
                    const fh1 = cdir.openFile(gpu.io, f1n, .{}) catch continue;
                    fh1.close(gpu.io);
                    const fh2 = cdir.openFile(gpu.io, f2n, .{}) catch continue;
                    fh2.close(gpu.io);
                    const h1 = cdir.openFile(gpu.io, f1n, .{}) catch continue;
                    defer h1.close(gpu.io);
                    const h2 = cdir.openFile(gpu.io, f2n, .{}) catch continue;
                    defer h2.close(gpu.io);
                    _ = h1.readPositional(gpu.io, &[1][]u8{b1[0..]}, 0) catch continue;
                    _ = h2.readPositional(gpu.io, &[1][]u8{b2[0..]}, 0) catch continue;
                    var off: usize = 0;
                    var nbad: usize = 0;
                    var first_off: usize = 0;
                    while (off < fsz) : (off += 2) {
                        const v1 = std.mem.readInt(u16, b1[off..][0..2], .little);
                        const v2 = std.mem.readInt(u16, b2[off..][0..2], .little);
                        if (v1 != v2) {
                            if (nbad == 0) {
                                first_off = off;
                                if (first) {
                                    std.debug.print("  FIRST DIFF blk {d} {s} at byte {d} (elem {d}): {d} vs {d}\n", .{ bi, tg, off, off / 2, v1, v2 });
                                    first = false;
                                }
                            }
                            nbad += 1;
                        }
                    }
                    if (nbad > 0) {
                        std.debug.print("  pb{d}_{s}: {d}/{d} elements differ, first at elem {d}\n", .{ bi, tg, nbad, fsz / 2, first_off / 2 });
                        ndiff += 1;
                    }
                }
            }
            if (ndiff == 0) std.debug.print("  BIT-EXACT: all probe dumps match between windows\n", .{});
        }
        const pk2 = try check.download(gpu, eb.picks);
        const pv2 = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pk2));
        var pos2: usize = rows;
        var toks2: [24]u32 = @splat(0);
        var ntok2: usize = 0;
        var cur2: [1]u32 = .{0};
        toks2[0] = pv2[rows - 1];
        ntok2 = 1;
        while (ntok2 < 16 and pos2 < 4000) : (ntok2 += 1) {
            cur2[0] = toks2[ntok2 - 1];
            try gpu.ctx.makeCurrent();
            try ea.forward(cur2[0..1], 1, pos2);
            const dx0 = try check.download(gpu, ea.x[0]);
            const dx1 = try check.download(gpu, ea.x[1]);
            const dbr = try check.download(gpu, ea.branch);
            const dpo = try check.download(gpu, ea.post);
            const dco = try check.download(gpu, ea.comb);
            try ctx1.makeCurrent();
            try eb.x[0].upload(0, dx0);
            try eb.x[1].upload(0, dx1);
            try eb.branch.upload(0, dbr);
            try eb.post.upload(0, dpo);
            try eb.comb.upload(0, dco);
            eb.xi = ea.xi;
            a.free(dx0);
            a.free(dx1);
            a.free(dbr);
            a.free(dpo);
            a.free(dco);
            try eb.forward(cur2[0..1], 1, pos2);
            try eb.stream.synchronize();
            const pkd2 = try check.download(gpu, eb.picks);
            const pvd2 = std.mem.bytesAsSlice(u32, @as([]align(1) u8, pkd2));
            toks2[ntok2] = pvd2[0];
            a.free(pkd2);
            pos2 += 1;
        }
        a.free(pk2);
        a.free(hx0);
        a.free(hx1);
        a.free(hbr);
        a.free(hpo);
        a.free(hco);
        a.free(snapA1);
        a.free(snapB1);
        a.free(sumsW);
        a.free(sumsWb);
        var same: bool = ntok2 == ntok1;
        if (same) for (0..ntok2) |i| {
            if (toks2[i] != toks1[i]) same = false;
        };
        var outd2: []u8 = try a.alloc(u8, 400);
        defer a.free(outd2);
        var wd2: usize = 0;
        for (toks2[0..ntok2]) |t| {
            const printed = std.fmt.bufPrint(outd2[wd2..], "{d} ", .{t}) catch unreachable;
            wd2 += printed.len;
        }
        std.debug.print("  split-decode2[{d}]: {s} == window1: {}\n", .{ ntok2, outd2[0..wd2], same });
    }
    {
        var out: []u8 = try a.alloc(u8, 200);
        defer a.free(out);
        var w: usize = 0;
        for (pv[0..rows]) |t| {
            const printed = std.fmt.bufPrint(out[w..], "{d} ", .{t}) catch unreachable;
            w += printed.len;
        }
        check.pass("glm fwd SPLIT 0..{d}..46: sum {d:.1} max {d:.2} at {d}", .{ split_at, sum, mx, mx_at });
        std.debug.print("  split16 picks: {s}\n", .{out[0..w]});
    }
    // top-5 of the LAST row's logits: the knife-edge probe
    {
        const vocab = eb.c.vocab;
        const last = try a.alloc(u32, vocab);
        defer a.free(last);
        var best5: [5]f32 = .{ -1e30, -1e30, -1e30, -1e30, -1e30 };
        var at5: [5]usize = .{ 0, 0, 0, 0, 0 };
        for (0..vocab) |v| {
            const h: u16 = std.mem.readInt(u16, lg[((rows - 1) * vocab + v) * 2 ..][0..2], .little);
            const val = @as(f32, @bitCast(@as(u32, h) << 16));
            var slot: usize = 5;
            var j: usize = 0;
            while (j < 5) : (j += 1) {
                if (val > best5[j]) {
                    slot = j;
                    break;
                }
            }
            if (slot < 5) {
                var k: usize = 4;
                while (k > slot) : (k -= 1) {
                    best5[k] = best5[k - 1];
                    at5[k] = at5[k - 1];
                }
                best5[slot] = val;
                at5[slot] = v;
            }
        }
        std.debug.print("  top5: {d}@{d:.2} {d}@{d:.2} {d}@{d:.2} {d}@{d:.2} {d}@{d:.2}\n", .{ at5[0], best5[0], at5[1], best5[1], at5[2], best5[2], at5[3], best5[3], at5[4], best5[4] });
    }
    {
        // deterministic-fingerprint: active rows' first 4 logits + row sums
        var ir: usize = 0;
        while (ir < rows) : (ir += 1) {
            var rs: f64 = 0;
            var jr: usize = 0;
            while (jr < eb.c.vocab) : (jr += 1) {
                const h2: u16 = std.mem.readInt(u16, lg[(ir * eb.c.vocab + jr) * 2 ..][0..2], .little);
                rs += @as(f32, @bitCast(@as(u32, h2) << 16));
            }
            const f0: u16 = std.mem.readInt(u16, lg[(ir * eb.c.vocab) * 2 ..][0..2], .little);
            std.debug.print("  split row{d}: first {d:.3} rowsum {d:.1}" ++ "\n", .{ ir, @as(f32, @bitCast(@as(u32, f0) << 16)), rs });
        }
    }
    _ = ea.deinit();
}
