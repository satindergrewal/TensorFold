#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

int64_t exl3_k1_block_members(int64_t cfg, bool q);
int64_t exl3_k1_max_blocks();
int64_t exl3_k1_grid(int64_t cfg, bool q);
void exl3_k1_rot_quant_cuda(const at::Tensor&, int64_t, const at::Tensor&, at::Tensor&, at::Tensor&, int64_t, int64_t);
void exl3_k1_schedule_cuda(const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&, int64_t, int64_t,
                           int64_t, int64_t, int64_t, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&);
void exl3_k1_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, const at::Tensor&,
                  const at::Tensor&, const at::Tensor&, const c10::optional<at::Tensor>&, at::Tensor&, at::Tensor&,
                  at::Tensor&, at::Tensor&, int64_t, int64_t, int64_t, double, int64_t, int64_t, int64_t, bool,
                  const c10::optional<at::Tensor>&, const c10::optional<at::Tensor>&, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), "k1: ", name,
                ": expected a contiguous CUDA tensor of the right dtype");
}

// The tile list of a plan (items [n, 3] / counts [2], or CSR offsets [E + 1] when given): blocks of up to BN members;
// lag: the down tiles of the block at list position j follow the gate/up tiles of position j + lag.
void schedule(const at::Tensor& items, const at::Tensor& counts, c10::optional<at::Tensor> offsets, int64_t E,
              int64_t BN, int64_t NCB, int64_t NOG, int64_t lag, at::Tensor blocks, at::Tensor tiles, at::Tensor ctrl,
              at::Tensor done) {
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&items, &counts, &blocks, &tiles, &ctrl, &done})
        check(*t, at::kInt, "schedule: int32 buffer");
    if (offsets.has_value()) check(*offsets, at::kInt, "offsets");
    const at::cuda::CUDAGuard guard(items.device());
    exl3_k1_schedule_cuda(items, counts, offsets, E, BN, NCB, NOG, lag, blocks, tiles, ctrl, done);
}

// Xd and Y of a prompt chunk's routed pairs: the prompt kernels' arithmetic (exl3.cu ``prompt``, shared rows).
void run(const at::Tensor& xh, const at::Tensor& tg, const at::Tensor& tu, const at::Tensor& td,
         const at::Tensor& svh_g, const at::Tensor& svh_u, const at::Tensor& suh_d, const at::Tensor& svh_d,
         at::Tensor xd, at::Tensor y, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
         c10::optional<at::Tensor> offsets, at::Tensor blocks, at::Tensor tiles, at::Tensor ctrl, at::Tensor done,
         int64_t D, int64_t NI, int64_t slots, double limit, int64_t cfg, int64_t OB, int64_t lag, bool q,
         c10::optional<at::Tensor> xq, c10::optional<at::Tensor> xs, int64_t max_ctas) {
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&xh, &svh_g, &svh_u, &suh_d, &svh_d})
        check(*t, at::kHalf, "fp16 input");
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&tg, &tu, &td, &items, &counts, &members,
                                                                         &blocks, &tiles, &ctrl, &done})
        check(*t, at::kInt, "int32 input");
    check(xd, at::kHalf, "xd");
    check(y, at::kFloat, "y");
    if (offsets.has_value()) check(*offsets, at::kInt, "offsets");
    const int64_t E = tg.size(0);
    TORCH_CHECK(tg.dim() == 4 && tg.size(1) == D / 16 && tg.size(2) == NI / 16 && tg.size(3) == 32 &&
                    tu.sizes() == tg.sizes() && td.dim() == 4 && td.size(0) == E && td.size(1) == NI / 16 &&
                    td.size(2) == D / 16 && td.size(3) == 32,
                "k1: trellis words [E, K/16, N/16, 32] (4-bit tiles)");
    TORCH_CHECK(svh_g.numel() == E * NI && svh_u.numel() == E * NI && suh_d.numel() == E * NI &&
                    svh_d.numel() == E * D, "k1: scale shapes");
    TORCH_CHECK(xh.dim() == 2 && xh.size(1) == D, "k1: xh [rows, D]");
    // the call's pairs (its rows x slots) are what the kernels write: a prompt lane's Y and Xd are views of its own
    // rows while the plan (members) keeps the window's capacity (TF_GLM_PREFILL_LANES, TF_GLM_MOE_GLUE)
    const int64_t pairs = xh.size(0) * slots;
    TORCH_CHECK(pairs <= members.numel(), "k1: more rows than the plan holds");
    TORCH_CHECK(xd.numel() >= pairs * NI && y.numel() >= pairs * D, "k1: xd / y too small for ", pairs, " pairs");
    TORCH_CHECK(slots > 0, "k1: slots");
    if (q) {
        TORCH_CHECK(xq.has_value() && xs.has_value(), "k1q: xq and xs");
        check(*xq, at::kChar, "xq");
        check(*xs, at::kFloat, "xs");
        TORCH_CHECK(xq->dim() == 2 && xq->size(1) == D && xs->numel() >= xq->size(0), "k1q: xq [rows, D], xs [rows]");
    }
    const at::cuda::CUDAGuard guard(xh.device());
    exl3_k1_cuda(xh, tg, tu, td, svh_g, svh_u, suh_d, svh_d, xd, y, items, counts, members, offsets, blocks, tiles,
                 ctrl, done, D, NI, slots, limit, cfg, OB, lag, q, xq, xs, max_ctas);
}

// K1q's rows (quality-changing): xq [rows, D] int8 in the decode's slot order and xs [rows] fp32 scales, from the
// normed bf16 rows and the layer's shared gate/up suh.
void rot_quant(const at::Tensor& x, int64_t x_stride, const at::Tensor& suh, at::Tensor xq, at::Tensor xs,
               int64_t rows, int64_t D) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16, "k1q: x bf16 CUDA");
    check(suh, at::kHalf, "suh");
    check(xq, at::kChar, "xq");
    check(xs, at::kFloat, "xs");
    TORCH_CHECK(suh.numel() == D && xq.numel() >= rows * D && xs.numel() >= rows, "k1q: rot_quant shapes");
    const at::cuda::CUDAGuard guard(x.device());
    exl3_k1_rot_quant_cuda(x, x_stride, suh, xq, xs, rows, D);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
    m.def("schedule", &schedule);
    m.def("block_members", &exl3_k1_block_members);
    m.def("max_blocks", &exl3_k1_max_blocks);
    m.def("grid", &exl3_k1_grid);
    m.def("rot_quant", &rot_quant);
}
