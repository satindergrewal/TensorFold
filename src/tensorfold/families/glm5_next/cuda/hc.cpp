#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void hc_partial_cuda(const at::Tensor&, const at::Tensor&, at::Tensor&, int64_t);
void hc_post_partial_cuda(const at::Tensor&, const at::Tensor&, int64_t, const at::Tensor&, const at::Tensor&,
                          const at::Tensor&, at::Tensor&, int64_t, int64_t, at::Tensor&);

// PART [rows, 16, 32] fp32 = glue._hc_partial's output for x [rows, 16384] bf16 and fn [24, 16384] bf16.
void hc_partial(const at::Tensor& x, const at::Tensor& fn, at::Tensor part, int64_t rows) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == 16384,
                "hc_partial: x [rows, 16384] bf16, contiguous");
    TORCH_CHECK(fn.is_cuda() && fn.scalar_type() == at::kBFloat16 && fn.is_contiguous() && fn.size(0) == 24 &&
                fn.size(1) == 16384, "hc_partial: fn [24, 16384] bf16");
    TORCH_CHECK(part.is_cuda() && part.scalar_type() == at::kFloat && part.is_contiguous() &&
                part.numel() >= rows * 16 * 32, "hc_partial: part [rows, 16, 32] fp32");
    TORCH_CHECK(rows >= 1 && x.size(0) >= rows, "hc_partial: rows");
    c10::cuda::CUDAGuard guard(x.device());
    hc_partial_cuda(x, fn, part, rows);
}

// hc_partial of the streams glue.hc_post(x, ., g, post, comb) would write, read from the old x, which this does not
// write (decode rows; glue.hc_post_pre): g [world, rows', 4096] fp32 contiguous with rank k's rows at k rs elements,
// post [rows, 4], comb [rows, 16] fp32 (the previous site's); the new streams also go to xn [rows, 16384] bf16.
void hc_post_partial(const at::Tensor& x, const at::Tensor& g, int64_t rs, const at::Tensor& post,
                     const at::Tensor& comb, const at::Tensor& fn, at::Tensor part, int64_t rows, int64_t world,
                     at::Tensor xn) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.is_contiguous() && x.size(1) == 16384,
                "hc_post_partial: x [rows, 16384] bf16, contiguous");
    TORCH_CHECK(fn.is_cuda() && fn.scalar_type() == at::kBFloat16 && fn.is_contiguous() && fn.size(0) == 24 &&
                fn.size(1) == 16384, "hc_post_partial: fn [24, 16384] bf16");
    TORCH_CHECK(part.is_cuda() && part.scalar_type() == at::kFloat && part.is_contiguous() &&
                part.numel() >= rows * 16 * 32, "hc_post_partial: part [rows, 16, 32] fp32");
    TORCH_CHECK(g.is_cuda() && g.scalar_type() == at::kFloat && g.is_contiguous() && world >= 1 && world <= 4 &&
                rs >= rows * 4096 && g.numel() >= (world - 1) * rs + rows * 4096, "hc_post_partial: g [world, rows, 4096]");
    TORCH_CHECK(post.is_cuda() && post.scalar_type() == at::kFloat && post.is_contiguous() && post.numel() >= rows * 4,
                "hc_post_partial: post [rows, 4] fp32");
    TORCH_CHECK(comb.is_cuda() && comb.scalar_type() == at::kFloat && comb.is_contiguous() &&
                comb.numel() >= rows * 16, "hc_post_partial: comb [rows, 16] fp32");
    TORCH_CHECK(rows >= 1 && x.size(0) >= rows, "hc_post_partial: rows");
    TORCH_CHECK(xn.is_cuda() && xn.scalar_type() == at::kBFloat16 && xn.is_contiguous() && xn.dim() == 2 &&
                xn.size(1) == 16384 && xn.size(0) >= rows && xn.data_ptr() != x.data_ptr(),
                "hc_post_partial: xn [rows, 16384] bf16, contiguous, apart from x");
    c10::cuda::CUDAGuard guard(x.device());
    hc_post_partial_cuda(x, g, rs, post, comb, fn, part, rows, world, xn);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hc_partial", &hc_partial);
    m.def("hc_post_partial", &hc_post_partial);
}
