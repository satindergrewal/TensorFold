#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

int64_t exl3_k12_configs();
void exl3_k12_prompt_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, int64_t,
                          int64_t, int64_t, int64_t, double, int64_t, const c10::optional<at::Tensor>&,
                          const c10::optional<at::Tensor>&, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), "k12: ", name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

// Xd and Y of a prompt chunk's routed pairs from the rotated rows xh [rows, D] (one a token, the layer's shared
// gate/up suh): exl3.cu ``prompt``'s outputs for shared rows and 64-member items, the same bits. blocks: int32 scratch
// of at least 4 max_items + 2 for the configurations on blocks of 80 or 128 members (3 to 6).
void prompt(const at::Tensor& xh, const at::Tensor& tg, const at::Tensor& tu, const at::Tensor& td,
            const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members, const at::Tensor& svh_g,
            const at::Tensor& svh_u, const at::Tensor& suh_d, const at::Tensor& svh_d, at::Tensor xd, at::Tensor y,
            int64_t D, int64_t N, int64_t E, int64_t max_items, double limit, int64_t slots,
            c10::optional<at::Tensor> order, c10::optional<at::Tensor> blocks, int64_t cfg) {
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&xh, &svh_g, &svh_u, &suh_d, &svh_d})
        check(*t, at::kHalf, "fp16 input");
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&tg, &tu, &td, &items, &counts, &members})
        check(*t, at::kInt, "int32 input");
    check(xd, at::kHalf, "xd");
    check(y, at::kFloat, "y");
    if (order.has_value()) check(*order, at::kInt, "order");
    if (blocks.has_value()) check(*blocks, at::kInt, "blocks");
    TORCH_CHECK(tg.dim() == 4 && tg.size(1) == D / 16 && tg.size(2) == N / 16 && tg.size(3) == 32 &&
                    tu.sizes() == tg.sizes() && td.dim() == 4 && td.size(0) == tg.size(0) && td.size(1) == N / 16 &&
                    td.size(2) == D / 16 && td.size(3) == 32,
                "k12: trellis words [E, K/16, N/16, 32] (4-bit tiles)");
    TORCH_CHECK(E <= tg.size(0) && svh_g.numel() >= E * N && svh_u.numel() >= E * N && suh_d.numel() >= E * N &&
                    svh_d.numel() >= E * D, "k12: scale shapes");
    TORCH_CHECK(xh.dim() == 2 && xh.size(1) == D, "k12: xh [rows, D]");
    // the call's pairs (its rows x slots) are what the kernels write: a prompt lane's Y and Xd are views of its own
    // rows while the plan (members) keeps the window's capacity (TF_GLM_PREFILL_LANES, TF_GLM_MOE_GLUE)
    const int64_t pairs = xh.size(0) * slots;
    TORCH_CHECK(pairs <= members.numel(), "k12: more rows than the plan holds");
    TORCH_CHECK(xd.numel() >= pairs * N && y.numel() >= pairs * D, "k12: xd / y too small for ", pairs, " pairs");
    TORCH_CHECK(items.numel() >= 3 * max_items && slots > 0, "k12: items [max_items, 3], slots");
    const at::cuda::CUDAGuard guard(xh.device());
    exl3_k12_prompt_cuda(xh, tg, tu, td, items, counts, members, svh_g, svh_u, suh_d, svh_d, xd, y, D, N, E,
                         max_items, limit, slots, order, blocks, cfg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prompt", &prompt);
    m.def("configs", &exl3_k12_configs);
}
