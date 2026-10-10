//! GLM-5.3-Flash on CUDA in Zig: the Metal family's engine over our runtime —
//! routed experts as EXL3 trellis weights, everything dense in bf16.
//! The config is the Metal family's: same file, both backends.

pub const Config = @import("config.zig").Config;
pub const Kind = @import("config.zig").Kind;
pub const weights = @import("cuda_weights.zig");

test {
    _ = Config;
    _ = weights;
}
