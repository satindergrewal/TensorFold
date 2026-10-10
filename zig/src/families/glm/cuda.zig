//! GLM-5.3-Flash on CUDA in Zig: the Metal family's engine over our runtime —
//! routed experts as EXL3 trellis weights, everything dense in bf16.
//! The config is the Metal family's: same file, both backends.

pub const Config = @import("config.zig").Config;
pub const Kind = @import("config.zig").Kind;
pub const weights = @import("cuda_weights.zig");
pub const hc = @import("cuda_hc.zig");
pub const glue = @import("glm_glue.zig");
pub const kda = @import("kda.zig");
pub const dsa = @import("dsa.zig");
pub const index = @import("index.zig");
pub const engine = @import("cuda_engine.zig");

test {
    _ = Config;
    _ = weights;
    _ = hc;
    _ = glue;
    _ = kda;
    _ = dsa;
    _ = index;
    _ = engine;
}
