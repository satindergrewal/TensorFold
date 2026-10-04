#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

void gpu_sample_rows_cuda(const at::Tensor&, int64_t, int64_t, const at::Tensor&, at::Tensor&, int64_t);

// got: all-gathered candidate packs (fp32, contiguous; rank q's part at q * rank_stride); table [rows, 9] int64
// (gpu_sample.py builds it); out [rows, 2] int32 (token, certified).
void sample_rows(const at::Tensor& got, int64_t rank_stride, int64_t world, const at::Tensor& table, at::Tensor out,
                 int64_t rows) {
    TORCH_CHECK(got.is_cuda() && got.scalar_type() == at::kFloat && got.is_contiguous(), "sample_rows: got fp32");
    TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kLong && table.is_contiguous() && table.dim() == 2 &&
                table.size(1) == 9 && table.size(0) >= rows, "sample_rows: table [rows, 9] int64");
    TORCH_CHECK(out.is_cuda() && out.scalar_type() == at::kInt && out.is_contiguous() && out.numel() >= 2 * rows,
                "sample_rows: out [rows, 2] int32");
    TORCH_CHECK(world >= 1 && rank_stride >= 0 && (world == 1 || got.numel() >= world * rank_stride),
                "sample_rows: world / rank stride");
    c10::cuda::CUDAGuard guard(got.device());
    gpu_sample_rows_cuda(got, rank_stride, world, table, out, rows);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_rows", &sample_rows);
}
