//! GLM-5.3-Flash (model_type glm5_next) on Metal: the Python family's decode kernels, our engine around them.
pub const config = @import("config.zig");
pub const weights = @import("weights.zig");
pub const kernels = @import("kernels.zig");
pub const state = @import("state.zig");
pub const forward = @import("forward.zig");
pub const mtp = @import("mtp.zig");
pub const prompt = @import("prompt.zig");
pub const engine = @import("engine.zig");
pub const ep = @import("ep.zig");
pub const slots = @import("slots.zig");
pub const pool = @import("pool.zig");
pub const backend = @import("backend.zig");
pub const mirror = @import("mirror.zig");
pub const timing = @import("timing.zig");
pub const vision_config = @import("vision_config.zig"); // upstream v1.1.1's checkpoint vision config
pub const image_core = @import("image_core.zig"); // the image row maths over core.picture
pub const cuda_vision = @import("cuda_vision.zig"); // the image tower (upstream v1.1.1)
pub const cuda_load_owner = @import("cuda_load_owner.zig"); // a CUDA module's owner

test {
    _ = config;
    _ = state;
    _ = ep;
    _ = @import("ep_control.zig");
    _ = @import("../../core/moe_route.zig");
    _ = @import("../../core/affine_mm.zig");
    _ = @import("../../core/hc.zig");
    _ = @import("../../core/paced_read.zig");
    _ = @import("../../core/copy_index.zig");
    _ = @import("attn_check.zig");
    _ = @import("load_plan.zig");
    _ = @import("page_cache.zig");
    _ = pool;
    _ = @import("kept.zig");
    _ = mirror;
    _ = timing;
    _ = vision_config;
    _ = image_core;
    _ = cuda_vision;
    _ = cuda_load_owner;
}
