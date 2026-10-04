// Bindings of moe_glue.cu (TF_GLM_MOE_GLUE): the fused route, the combine and the shared expert's gate/up + SwiGLU.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void moe_route_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                    at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, double, bool, int64_t,
                    int64_t, int64_t, int64_t);
void moe_plan_cuda(const at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&,
                   int64_t, int64_t);
int64_t moe_route_blocks(int64_t);
int64_t moe_route_bytes(int64_t, int64_t);
void moe_combine_cuda(const at::Tensor&, const c10::optional<at::Tensor>&, const at::Tensor&, at::Tensor&);
void moe_shared_gu_cuda(const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, at::Tensor&,
                        int64_t, double, int64_t);

static void need(bool ok, const char* what) { TORCH_CHECK(ok, "moe_glue: ", what); }

static bool rows16(const at::Tensor& t) {         // unit-stride rows, 16-byte aligned rows (cp.async)
    return t.dim() == 2 && t.stride(1) == 1 && (t.size(0) <= 1 || t.stride(0) % 8 == 0) &&
           reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

// x [M, D] bf16 rows, w [NE, D] bf16 (the router), bias [NE] fp32 -> logits [M, NE] fp32, pick / wts [M, 9], and the
// plan of the M * 9 pairs over ``experts`` (NE + 1: the shared expert's id NE last) in items of ``tile`` pairs:
// members [M * 9], items [>= items, 3], counts [2]; scratch rank [M * 9], hist [blocks(M) * experts], tick
// [blocks(M) + 1] (zero; the kernel leaves it zero). ``block``: glue._topk's BLOCK (next power of two above NE).
// ``plan`` false: logits, picks and weights only (the plan's buffers and scratch are not touched).
void route(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias, at::Tensor logits, at::Tensor pick,
           at::Tensor wts, at::Tensor members, at::Tensor items, at::Tensor counts, at::Tensor rank, at::Tensor hist,
           at::Tensor tick, double scale, bool norm, int64_t experts, int64_t tile, int64_t block, bool plan) {
    const int64_t M = x.size(0), D = x.size(1), NE = w.size(0);
    need(x.is_cuda() && x.scalar_type() == at::kBFloat16 && rows16(x), "x: [M, D] bf16, unit-stride 16-byte rows");
    need(w.is_cuda() && w.scalar_type() == at::kBFloat16 && w.is_contiguous() && w.size(1) == D &&
         reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "w: [NE, D] bf16, 16-byte aligned");
    need(D % 512 == 0, "D: a multiple of 512 (8 K slices of 64-input chunks)");
    need(bias.is_cuda() && bias.scalar_type() == at::kFloat && bias.is_contiguous() && bias.numel() == NE, "bias");
    need(NE % 8 == 0 && NE + 1 <= 512 && block >= NE + 1 && block <= 512, "experts: NE + 1 <= BLOCK <= 512");
    need(experts == NE + 1 && experts <= 1024, "plan experts: NE + 1");
    need(tile >= 1, "tile");
    const int64_t P = M * 9, blocks = moe_route_blocks(M);
    auto ok = [](const at::Tensor& t, at::ScalarType s, int64_t n) {
        return t.is_cuda() && t.scalar_type() == s && t.is_contiguous() && t.numel() >= n;
    };
    need(ok(logits, at::kFloat, M * NE) && logits.stride(0) == NE, "logits [M, NE] fp32");
    need(ok(pick, at::kInt, P) && ok(wts, at::kFloat, P), "pick / wts [M, 9]");
    need(ok(tick, at::kInt, blocks + 1), "tick");
    if (plan) {
        need(ok(members, at::kInt, P) && ok(counts, at::kInt, 2) && ok(items, at::kInt, 3), "plan buffers");
        need(ok(rank, at::kInt, P) && ok(hist, at::kInt, blocks * experts), "scratch");
        // items: an item per used expert plus one per tile pairs past its first, at most (experts.max_items)
        need(items.numel() / 3 >= std::min(P, experts) + P / tile, "items: experts.max_items rows");
    }
    moe_route_cuda(x, w, bias, logits, pick, wts, members, items, counts, rank, hist, tick, scale, norm, experts, tile,
                   block, plan ? 0 : 1);
}

// The plan of given picks [M, 9] int32 (values below ``experts``): experts.route's members, items and counts.
void plan(const at::Tensor& pick, at::Tensor members, at::Tensor items, at::Tensor counts, at::Tensor rank,
          at::Tensor hist, at::Tensor tick, int64_t experts, int64_t tile) {
    const int64_t M = pick.size(0), P = M * 9, blocks = moe_route_blocks(M);
    need(pick.is_cuda() && pick.scalar_type() == at::kInt && pick.is_contiguous() && pick.dim() == 2 &&
         pick.size(1) == 9, "pick [M, 9] int32");
    need(experts >= 1 && experts <= 1024 && tile >= 1, "experts / tile");
    auto ok = [](const at::Tensor& t, int64_t n) {
        return t.is_cuda() && t.scalar_type() == at::kInt && t.is_contiguous() && t.numel() >= n;
    };
    need(ok(members, P) && ok(counts, 2) && ok(items, 3 * (std::min(P, experts) + P / tile)), "plan buffers");
    need(ok(rank, P) && ok(hist, blocks * experts) && ok(tick, blocks + 1), "scratch");
    moe_plan_cuda(pick, members, items, counts, rank, hist, tick, experts, tile);
}

// y [R, 9, D] fp32, sy [R, D] fp32 or None (the last slot from y), wts [R, 9] fp32 -> out [R, D] fp32
void combine(const at::Tensor& y, const c10::optional<at::Tensor>& sy, const at::Tensor& wts, at::Tensor out) {
    const int64_t R = out.size(0), D = out.size(1);
    need(y.is_cuda() && y.scalar_type() == at::kFloat && y.is_contiguous() && y.dim() == 3 && y.size(0) == R &&
         y.size(1) == 9 && y.size(2) == D, "y [R, 9, D] fp32 contiguous");
    need(wts.is_cuda() && wts.scalar_type() == at::kFloat && wts.is_contiguous() && wts.size(0) == R && wts.size(1) == 9,
         "wts [R, 9] fp32");
    need(out.is_cuda() && out.scalar_type() == at::kFloat && out.is_contiguous() && D % 4 == 0, "out [R, D] fp32");
    need(reinterpret_cast<uintptr_t>(y.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
         "16-byte aligned rows");
    if (sy.has_value())
        need(sy->is_cuda() && sy->scalar_type() == at::kFloat && sy->is_contiguous() && sy->size(0) == R &&
             sy->size(1) == D && reinterpret_cast<uintptr_t>(sy->data_ptr()) % 16 == 0, "sy [R, D] fp32");
    moe_combine_cuda(y, sy, wts, out);
}

// x [M, K] bf16 rows, the shared expert's stacked [gate | up] 4-bit weight (shared prompt-matmul packing, groups of
// 64: weight words, scales / biases [K / 64, npad] bf16) -> act [M, ni] bf16 = SwiGLU of the bf16 gate and up
void shared_gu(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
               at::Tensor act, int64_t ni, double limit, int64_t cfg) {
    const int64_t M = x.size(0), K = x.size(1);
    need(x.is_cuda() && x.scalar_type() == at::kBFloat16 && rows16(x), "x: [M, K] bf16, unit-stride 16-byte rows");
    need(K % 64 == 0 && ni % 128 == 0, "K a multiple of 64, the intermediate width of 128");
    need(scales.dim() == 2 && scales.size(0) == K / 64 && scales.size(1) >= 2 * ni && biases.sizes() == scales.sizes() &&
         scales.is_contiguous() && biases.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
         biases.scalar_type() == at::kBFloat16, "scales / biases [K / 64, npad] bf16");
    need(w.is_cuda() && w.is_contiguous() && w.numel() * w.element_size() >= 2 * ni * K / 2 &&
         reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0, "weight words, 16-byte aligned");
    need(reinterpret_cast<uintptr_t>(scales.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(biases.data_ptr()) % 16 == 0
         && scales.size(1) % 8 == 0, "scales / biases: 16-byte aligned rows");
    need(act.is_cuda() && act.scalar_type() == at::kBFloat16 && act.is_contiguous() && act.size(0) == M &&
         act.size(1) == ni, "act [M, ni] bf16");
    moe_shared_gu_cuda(x, w, scales, biases, act, ni, limit, cfg);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("route", &route);
    m.def("plan", &plan);
    m.def("route_blocks", &moe_route_blocks);
    m.def("route_bytes", &moe_route_bytes);
    m.def("combine", &combine, py::arg("y"), py::arg("sy"), py::arg("wts"), py::arg("out"));
    m.def("shared_gu", &shared_gu);
}
