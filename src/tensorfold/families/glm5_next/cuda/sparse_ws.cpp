// Bindings of sparse_ws.cu (TF_GLM_SPARSE_FAST=2, patch 0199): a prompt chunk's DSA sparse attention with warp roles,
// latent._sparse_onepass' bits; and of sparse_topk.cu (TF_GLM_TOPK_FAST=1): its top-512 pools, _select_rows' output.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void sparse_ws_cuda(const at::Tensor& qa, const at::Tensor& cache, const at::Tensor& tokens, const at::Tensor& counts,
                    at::Tensor& out, int64_t rs_bytes, bool fp8, double scale);
int sparse_ws_smem();
void topk_rows_cuda(const at::Tensor& scores, at::Tensor& out);
int topk_rows_smem();

// qa [R, 16, 512] bf16, cache: an FP8 cache (uint8 rows of 512 codes + 16 bytes: the fp32 scale first) or a bf16 one
// ([P, 512]), tokens [R, W] int32 (ascending, the first counts[r] valid), counts [R] int32, out [R, 16, 512] bf16
// (rows with count 0 are left alone). scale: the query scale _sparse_onepass multiplies the scores by.
void sparse_ws(const at::Tensor& qa, const at::Tensor& cache, const at::Tensor& tokens, const at::Tensor& counts,
               at::Tensor& out, double scale) {
    TORCH_CHECK(qa.is_cuda() && qa.scalar_type() == at::kBFloat16 && qa.dim() == 3 && qa.size(1) == 16 &&
                qa.size(2) == 512 && qa.is_contiguous(), "sparse_ws: qa [R, 16, 512] bf16, contiguous");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kBFloat16 && out.sizes() == qa.sizes() && out.is_contiguous(),
                "sparse_ws: out like qa");
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == at::kInt && tokens.dim() == 2 &&
                tokens.size(0) == qa.size(0) && tokens.is_contiguous(), "sparse_ws: tokens [R, W] int32, contiguous");
    TORCH_CHECK(counts.is_cuda() && counts.scalar_type() == at::kInt && counts.numel() == qa.size(0) &&
                counts.is_contiguous(), "sparse_ws: counts [R] int32");
    TORCH_CHECK(cache.is_cuda() && cache.dim() == 2 && cache.is_contiguous(), "sparse_ws: a contiguous cache");
    const bool fp8 = cache.scalar_type() == at::kByte || cache.scalar_type() == at::kFloat8_e4m3fn;
    if (fp8) {
        TORCH_CHECK(cache.size(1) == 512 + 16, "sparse_ws: FP8 cache rows of 512 codes and 16 bytes");
    } else {
        TORCH_CHECK(cache.scalar_type() == at::kBFloat16 && cache.size(1) == 512, "sparse_ws: bf16 cache rows of 512");
    }
    TORCH_CHECK(reinterpret_cast<uintptr_t>(cache.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(qa.data_ptr()) % 16 == 0
                && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0, "sparse_ws: 16-byte aligned buffers");
    if (qa.size(0) == 0) return;
    const int64_t rs = cache.stride(0) * cache.element_size();
    c10::cuda::CUDAGuard guard(qa.device());
    sparse_ws_cuda(qa, cache, tokens, counts, out, rs, fp8, scale);
}

// scores [R, NP] fp32 (contiguous rows, NP at least 512) -> out [R, 512] int64: each row's 512 best pools (order keys,
// ties to the lower pool), ascending: _select_rows' output (sparse_topk.cu).
void topk_rows(const at::Tensor& scores, at::Tensor& out) {
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type() == at::kFloat && scores.dim() == 2 && scores.is_contiguous(),
                "topk_rows: contiguous fp32 scores [R, NP]");
    TORCH_CHECK(scores.size(1) >= 512 && scores.size(1) <= (1LL << 30), "topk_rows: rows of 512 pools or more");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kLong && out.dim() == 2 && out.size(0) == scores.size(0) &&
                out.size(1) == 512 && out.is_contiguous(), "topk_rows: out [R, 512] int64");
    if (scores.size(0) == 0) return;
    c10::cuda::CUDAGuard guard(scores.device());
    topk_rows_cuda(scores, out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_ws", &sparse_ws);
    m.def("smem", &sparse_ws_smem);
    m.def("topk_rows", &topk_rows);
    m.def("topk_smem", &topk_rows_smem);
}
