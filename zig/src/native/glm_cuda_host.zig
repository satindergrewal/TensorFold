//! The CUDA GLM engine behind the native server's serial host: one request at
//! a time, greedy decode with the carried KDA state / conv history / MLA caches.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api.zig");
const serial = @import("serial_host.zig");
const glm = @import("glm");
const Allocator = std.mem.Allocator;

pub const CudaDriver = struct {
    gpa: Allocator,
    io: std.Io,
    eng: *glm.cuda.engine.Engine,
    pos: usize = 0,
    fed: u64 = 0,
    picked_last: u32 = 0,
    has_pick: bool = false,
    picks_buf: cuda.DeviceBuffer,

    fn selfPtr(d: *CudaDriver) *anyopaque {
        return @ptrCast(d);
    }

    pub fn driver(d: *CudaDriver) serial.Driver {
        return .{
            .ctx = d.selfPtr(),
            .info = .{
                .context_window = @intCast(d.eng.context),
                .prefill_step = 512,
                .served_name = "glm53-cuda",
            },
            .reset = reset,
            .prompt_chunk = promptChunk,
            .draw = draw,
            .advance = advance,
        };
    }

    fn reset(ptr: *anyopaque) anyerror!void {
        const d: *CudaDriver = @ptrCast(@alignCast(ptr));
        @memset(d.eng.kda_st_cur, 0);
        try d.eng.kda_st0.fill32(0, null);
        try d.eng.kda_st1.fill32(0, null);
        try d.eng.kda_cs.fill8(0, null);
        d.pos = 0;
        d.fed = 0;
        d.has_pick = false;
    }

    fn promptChunk(ptr: *anyopaque, tokens: []const u32, last: bool) anyerror!void {
        _ = last;
        const d: *CudaDriver = @ptrCast(@alignCast(ptr));
        try d.eng.forward(tokens, tokens.len, d.pos);
        d.pos += tokens.len;
        d.fed += tokens.len;
        d.has_pick = true;
        try d.readPick(tokens.len - 1);
    }

    fn draw(ptr: *anyopaque, sampling: ?api.Sampling) anyerror!u32 {
        _ = sampling; // greedy only
        const d: *CudaDriver = @ptrCast(@alignCast(ptr));
        if (!d.has_pick) return error.Failed;
        return d.picked_last;
    }

    fn advance(ptr: *anyopaque, tok: u32) anyerror!void {
        const d: *CudaDriver = @ptrCast(@alignCast(ptr));
        var one: [1]u32 = .{tok};
        try d.eng.forward(one[0..1], 1, d.pos);
        d.pos += 1;
        d.fed += 1;
        try d.readPick(0);
    }

    fn readPick(d: *CudaDriver, row: usize) !void {
        try d.eng.stream.synchronize();
        const hb = try d.gpa.alloc(u8, 4 * (row + 1));
        defer d.gpa.free(hb);
        try d.eng.picks.download(0, hb);
        const pv = std.mem.bytesAsSlice(u32, @as([]align(1) u8, hb));
        d.picked_last = pv[row];
    }
};
