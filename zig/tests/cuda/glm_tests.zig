//! GLM CUDA family: the load, then spot checks against the config's shape rules.

const std = @import("std");
const glm = @import("glm");
const check = @import("check.zig");
const Gpu = check.Gpu;

fn intArg(rest: []const [:0]const u8, i: usize) ?usize {
    if (i >= rest.len) return null;
    return std.fmt.parseInt(usize, rest[i], 10) catch null;
}

pub fn load(gpu: Gpu, dir: []const u8, rest: []const [:0]const u8) !void {
    const lo = intArg(rest, 0) orelse 0;
    const hi = intArg(rest, 1) orelse 1 << 16;
    const w = try glm.weights.load(gpu.gpa, gpu.io, gpu.d, dir, .{ .lo = lo, .hi = hi });
    defer {
        w.deinit();
        gpu.gpa.destroy(w);
    }

    const c = &w.config;
    const blocks = c.layers + 1;
    if (w.blocks_hi - w.blocks_lo != w.layers.len) return error.Range;
    var kda_n: usize = 0;
    var mla_n: usize = 0;
    for (w.layers, w.blocks_lo..) |*l, bi| {
        if (bi < c.layers) {
            switch (l.attn) {
                .kda => {
                    kda_n += 1;
                    if (c.kind(@intCast(bi)) != .kda) return error.AttnKind;
                },
                .mla => {
                    mla_n += 1;
                    if (c.kind(@intCast(bi)) != .mla) return error.AttnKind;
                },
            }
        }
        switch (l.mlp) {
            .dense => |d| {
                if (bi >= c.dense_layers) return error.MlpKind;
                if (d.gate == 0 or d.up == 0 or d.down == 0) return error.MlpMissing;
            },
            .moe => |m| {
                if (bi < c.dense_layers) return error.MlpKind;
                if (m.gate.len != c.experts or m.up.len != c.experts or m.down.len != c.experts) return error.ExpertCount;
                if (m.router == 0 or m.bias == 0) return error.RouterMissing;
                for (m.gate) |q| if (q.k != c.hidden or q.n != c.moe_inter or q.trellis == 0) return error.GateShape;
                for (m.up) |q| if (q.k != c.hidden or q.n != c.moe_inter) return error.UpShape;
                for (m.down) |q| if (q.k != c.moe_inter or q.n != c.hidden) return error.DownShape;
                for ([_][]const @import("cuda").exl3.Weight{ m.gate, m.up, m.down }) |arr| {
                    for (arr) |q| if (q.halves != 6 and q.halves != 7 and q.halves != 8) return error.Halves;
                }
                if (m.shared_gate == 0) return error.SharedMissing;
            },
        }
        if (bi < c.layers) {
            if (l.hc_attn == null or l.hc_ffn == null) return error.HcMissing;
            if (l.in_norm == 0 or l.post_norm == 0) return error.NormMissing;
        } else {
            if (l.hc_attn != null) return error.MtpHc;
            if (l.eh_proj == 0 or l.enorm == 0 or l.hnorm == 0 or l.shared_head_norm == 0) return error.MtpMissing;
        }
    }
    if (w.blocks_lo == 0 and w.embed == 0) return error.NoEmbed;
    if (w.blocks_hi == blocks and (w.head == 0 or w.final_norm == 0)) return error.NoHead;
    const usage = @import("cuda").usage(false);
    check.pass("glm-load blocks [{d},{d}) of {d}: {d} kda, {d} mla, {d} moe, {d} device MiB", .{
        w.blocks_lo, w.blocks_hi, blocks, kda_n, mla_n, w.moeBlocks(), usage.device >> 20,
    });
}
