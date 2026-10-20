//! Host-to-device copies for the vision path: the tower's static weights, and each image's patches.
//! Upstream v1.1.1 keeps copies in custody until the device is done with them (copy_source.zig); this
//! server blocks, so the caller's bytes may go away as soon as the call returns.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;

/// `bytes` into device address `dst`; returns once the device has them.
pub fn synchronousCopy(d: *const Driver, dst: u64, bytes: []const u8, a: std.mem.Allocator) !void {
    _ = a;
    if (bytes.len == 0) return;
    try d.check(d.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "cuMemcpyHtoD");
}

/// `bytes` into device address `dst` on `stream`; returns once the device has them.
pub fn enqueue(d: *const Driver, dst: u64, bytes: []const u8, stream: abi.Stream) !void {
    _ = stream;
    if (bytes.len == 0) return;
    try d.check(d.api.cuMemcpyHtoD_v2(dst, bytes.ptr, bytes.len), "cuMemcpyHtoD");
}
