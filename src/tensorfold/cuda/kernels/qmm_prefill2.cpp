// Bindings of qmm_prefill2.cu: more launch configurations of the 4-bit prompt matmul, qmm_prefill's bits.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <vector>

void qmm_prefill2_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                       at::Tensor& out, int N, bool f32, int variant);
int qmm_prefill2_variants();
std::vector<int64_t> qmm_prefill2_tiles(int64_t variant, int64_t M, int64_t N, int64_t K, int64_t resident);

// qmm_prefill's checks (groups of 64 only), then variant ``variant`` (0 .. variants() - 1).
void qmm_prefill2(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                  at::Tensor& out, int64_t n, int64_t gs, bool f32, int64_t variant) {
    TORCH_CHECK(gs == 64, "qmm_prefill2: groups of 64");
    TORCH_CHECK(variant >= 0 && variant < qmm_prefill2_variants(), "qmm_prefill2: no such variant");
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.size(0) >= 1 &&
                x.stride(1) == 1 && x.stride(0) >= x.size(1), "x: (M, K) bf16 with contiguous rows");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && (x.size(0) == 1 || x.stride(0) % 8 == 0),
                "x rows must start on 16-byte boundaries");
    const int64_t m = x.size(0), k = x.size(1), kg = k / gs, npad = (n + 127) / 128 * 128;
    TORCH_CHECK(k % gs == 0, "K splits into whole groups");
    TORCH_CHECK(w.is_cuda() && w.is_contiguous() && w.scalar_type() == at::kInt && w.numel() == npad * k / 8,
                "packed weight does not match n and K");
    TORCH_CHECK(scales.is_contiguous() && biases.is_contiguous() && scales.scalar_type() == at::kBFloat16 &&
                biases.scalar_type() == at::kBFloat16 && scales.size(0) == kg && scales.size(1) == npad &&
                biases.sizes() == scales.sizes(), "scales and biases: (K / gs, n padded to 128) bf16");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.size(0) == m && out.size(1) == n &&
                out.scalar_type() == (f32 ? at::kFloat : at::kBFloat16), "out: (M, n)");
    c10::cuda::CUDAGuard guard(x.device());
    qmm_prefill2_cuda(x, w, scales, biases, out, static_cast<int>(n), f32, static_cast<int>(variant));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qmm_prefill2", &qmm_prefill2);
    m.def("variants", &qmm_prefill2_variants);
    m.def("tiles", &qmm_prefill2_tiles);
}
