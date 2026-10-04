// Bindings of exl3s.cu: GLM-5.3-Flash's routed EXL3 experts streamed slab by slab in decode windows (the same bits
// as exl3.cu's decode kernel).
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void exl3_stream_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& Tg, const at::Tensor& Tu,
                      const at::Tensor& Td, const at::Tensor& items, const at::Tensor& counts,
                      const at::Tensor& members, const at::Tensor& svh_g, const at::Tensor& svh_u,
                      const at::Tensor& suh_d, const at::Tensor& svh_d, at::Tensor& Z, at::Tensor& xd, at::Tensor& y,
                      at::Tensor& state, int64_t D, int64_t NI, int64_t E, int64_t P, int64_t slots,
                      int64_t max_items, double limit, bool xrow, bool fine, int64_t rs, int64_t stages,
                      int64_t blocks, bool pdl, int64_t warps);

void exl3_probe_cuda(const at::Tensor& ids, const at::Tensor& Tg, const at::Tensor& Tu, const at::Tensor& Td,
                     at::Tensor& out, int64_t blocks);
void exl3_expert_prefetch_cuda(const at::Tensor& items, const at::Tensor& counts, const at::Tensor& Tg,
                               const at::Tensor& Tu, const at::Tensor& Td, int64_t max_items, int64_t budget);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t && x.is_contiguous(), "streamed experts: ", name,
                " must be a contiguous CUDA tensor of the right dtype");
}

static bool aligned16(const at::Tensor& x) { return (reinterpret_cast<uintptr_t>(x.data_ptr()) & 15) == 0; }

// X0 / X1: gate and up inputs, [rows, D] each (xrow: one rotated row a token, X1 = X0) or [pairs, D]; Tg / Tu / Td:
// the trellis words [E, D/16, NI/16, 32] and [E, NI/16, D/16, 32]; items / counts / members: the decode plan (items
// of at most 16 pairs); P: the window's pairs (rows x slots); Z: fp32 partials (coarse 8 P NI, fine 32 P NI + 4 P D);
// xd fp16 [P, NI]; y fp32 [P, D]; state: int32 zeros, 2 + (2 + D / slice) max_items of them (slice: a block's
// columns, warps * 32); warps: a block's warps, 4, 8 or 16 (1, 2 or 4 KB slices of every k row).
void stream(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& Tg, const at::Tensor& Tu,
            const at::Tensor& Td, const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
            const at::Tensor& svh_g, const at::Tensor& svh_u, const at::Tensor& suh_d, const at::Tensor& svh_d,
            at::Tensor Z, at::Tensor xd, at::Tensor y, at::Tensor state, int64_t D, int64_t NI, int64_t P,
            int64_t slots, int64_t max_items, double limit, bool xrow, bool fine, int64_t rs, int64_t stages,
            int64_t blocks, bool pdl, int64_t warps) {
    for (auto* t : {&X0, &X1, &svh_g, &svh_u, &suh_d, &svh_d}) check(*t, at::kHalf, "fp16 input");
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&Tg, &Tu, &Td, &items, &counts, &members,
                                                                        &state})
        check(*t, at::kInt, "int32 input");
    check(Z, at::kFloat, "Z");
    check(xd, at::kHalf, "xd");
    check(y, at::kFloat, "y");
    const int64_t E = Tg.size(0);
    TORCH_CHECK(warps == 4 || warps == 8 || warps == 16, "streamed experts: blocks of 4, 8 or 16 warps, not ", warps);
    const int64_t slice = warps * 2 * 16;                    // columns a block's slice: 128, 256 or 512
    TORCH_CHECK(D % slice == 0 && NI % slice == 0, "streamed experts: D and NI multiples of the slice's ", slice,
                " columns, not ", D, ", ", NI);
    TORCH_CHECK((D / 16) % 16 == 0 && (NI / 16) % 4 == 0, "streamed experts: whole chains");
    TORCH_CHECK(rs >= 1 && ((D / 16) / 16) % rs == 0 && ((NI / 16) / 4) % rs == 0,
                "streamed experts: rows a stage must divide every chain");
    TORCH_CHECK(Tg.dim() == 4 && Tg.size(1) == D / 16 && Tg.size(2) == NI / 16 && Tg.size(3) == 32 &&
                    Tu.sizes() == Tg.sizes() && Td.dim() == 4 && Td.size(0) == E && Td.size(1) == NI / 16 &&
                    Td.size(2) == D / 16 && Td.size(3) == 32,
                "streamed experts: trellis words [E, D/16, NI/16, 32] and [E, NI/16, D/16, 32]");
    TORCH_CHECK(svh_g.numel() >= E * NI && svh_u.numel() >= E * NI && suh_d.numel() >= E * NI &&
                    svh_d.numel() >= E * D, "streamed experts: scales");
    TORCH_CHECK(P >= 1 && P % slots == 0, "streamed experts: P is the window's rows x slots");
    TORCH_CHECK(X0.numel() >= (xrow ? P / slots : P) * D && X1.numel() == X0.numel(), "streamed experts: inputs");
    TORCH_CHECK(!xrow || X0.data_ptr() == X1.data_ptr(), "streamed experts: one rotated row a token takes X1 = X0");
    TORCH_CHECK(items.numel() >= 3 * max_items && members.numel() >= P && counts.numel() >= 2,
                "streamed experts: the plan");
    TORCH_CHECK(Z.numel() >= (fine ? 2 * 4 * 4 * P * NI + 4 * P * D : 2 * 4 * P * NI) && xd.numel() >= P * NI &&
                    y.numel() >= P * D, "streamed experts: outputs");
    TORCH_CHECK(state.numel() >= 2 + (2 + D / slice) * max_items, "streamed experts: state");
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&X0, &X1, &Tg, &Tu, &Td, &Z, &xd, &y})
        TORCH_CHECK(aligned16(*t), "streamed experts: 16-byte aligned tensors");
    c10::cuda::CUDAGuard guard(X0.device());
    exl3_stream_cuda(X0, X1, Tg, Tu, Td, items, counts, members, svh_g, svh_u, suh_d, svh_d, Z, xd, y, state, D, NI,
                     E, P, slots, max_items, limit, xrow, fine, rs, stages, blocks, pdl, warps);
}

// Tests and timing: every byte of the experts in ids (int32, distinct) read once with plain 16-byte loads.
void probe(const at::Tensor& ids, const at::Tensor& Tg, const at::Tensor& Tu, const at::Tensor& Td, at::Tensor out,
           int64_t blocks) {
    check(ids, at::kInt, "ids");
    for (auto* t : {&Tg, &Tu, &Td}) check(*t, at::kInt, "trellis words");
    check(out, at::kInt, "out");
    TORCH_CHECK(Tu.numel() == Tg.numel() && Td.numel() == Tg.numel() && Tg.size(0) == Td.size(0) &&
                    (Tg.numel() / Tg.size(0)) % 4 == 0, "probe: three matrices of one size an expert");
    c10::cuda::CUDAGuard guard(Tg.device());
    exl3_probe_cuda(ids, Tg, Tu, Td, out, blocks);
}

// The expert L2 prefetch (TF_GLM_L2PF_EXPERT_MB): the plan's routed experts' trellis into L2, up to budget bytes.
void prefetch(const at::Tensor& items, const at::Tensor& counts, const at::Tensor& Tg, const at::Tensor& Tu,
              const at::Tensor& Td, int64_t max_items, int64_t budget) {
    for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&items, &counts, &Tg, &Tu, &Td})
        check(*t, at::kInt, "int32 input");
    TORCH_CHECK(Tu.numel() == Tg.numel() && Td.numel() == Tg.numel() && Tg.size(0) == Td.size(0) &&
                    ((Tg.numel() / Tg.size(0)) * 4) % 16 == 0 && items.numel() >= 3 * max_items,
                "expert prefetch: three matrices of one size an expert, 16-byte multiples; the plan");
    c10::cuda::CUDAGuard guard(Tg.device());
    exl3_expert_prefetch_cuda(items, counts, Tg, Tu, Td, max_items, budget);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("stream", &stream);
    m.def("probe", &probe);
    m.def("prefetch", &prefetch);
}
