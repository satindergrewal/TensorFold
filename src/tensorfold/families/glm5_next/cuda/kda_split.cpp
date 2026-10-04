#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

int64_t kda_split_record_bytes();
int64_t kda_split_state_smem(int64_t cols);
void kda_split_cuda(const at::Tensor&, int64_t, int64_t, const at::Tensor&, int64_t, const at::Tensor&, int64_t,
                    const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&, const at::Tensor&,
                    const at::Tensor&, double, double, int64_t, int64_t, at::Tensor&, at::Tensor&, at::Tensor&,
                    int64_t, int64_t, int64_t, int64_t);

static void check(const at::Tensor& x, at::ScalarType t, const char* name) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == t, name, ": expected a CUDA tensor of the right dtype");
}

// kda_chunk.cpp's chain (the same arguments and checks), split (kda_split.cu): ``recs`` is a uint8 CUDA buffer of at
// least window x H records, ``cols`` value columns a piece (a head takes ceil(128 / cols) pieces), ``nt`` threads a
// state block, ``window`` sub-chunks a pass, ``spb`` sub-chunks a prep block (0: one wave over the SMs).
void chain(const at::Tensor& P, int64_t p_stride, int64_t b_off, const at::Tensor& A, int64_t a_stride,
           const at::Tensor& G, int64_t g_stride, const at::Tensor& cs, const at::Tensor& cw,
           const at::Tensor& state_in, const at::Tensor& a_log, const at::Tensor& dt_bias, const at::Tensor& norm_w,
           double eps, double lower, int64_t rows, int64_t off, at::Tensor out, at::Tensor state_out,
           at::Tensor recs, int64_t cols, int64_t nt, int64_t window, int64_t spb) {
    check(P, at::kBFloat16, "P");
    check(A, at::kBFloat16, "A");
    check(G, at::kBFloat16, "G");
    check(cs, at::kBFloat16, "conv state");
    check(cw, at::kBFloat16, "conv weight");
    check(state_in, at::kFloat, "state");
    check(state_out, at::kFloat, "state out");
    check(a_log, at::kFloat, "A_log");
    check(dt_bias, at::kFloat, "dt_bias");
    check(norm_w, at::kBFloat16, "norm");
    check(out, at::kBFloat16, "out");
    check(recs, at::kByte, "records");
    TORCH_CHECK(cs.is_contiguous() && cw.is_contiguous() && state_in.is_contiguous() && state_out.is_contiguous() &&
                out.is_contiguous() && a_log.is_contiguous() && dt_bias.is_contiguous() && norm_w.is_contiguous() &&
                recs.is_contiguous(), "conv state, conv weight, states, out, the per-head weights and the records "
                "must be contiguous");
    TORCH_CHECK(P.stride(1) == 1 && A.stride(1) == 1 && G.stride(1) == 1, "rows must be contiguous along channels");
    const int64_t H = a_log.numel();
    TORCH_CHECK(dt_bias.numel() == H * 128 && norm_w.numel() == 128, "dt_bias [H*128], norm [128]");
    TORCH_CHECK(cw.size(0) == 3 * H * 128 && cw.size(1) == 4 && cs.size(0) == 3 && cs.size(1) == 3 * H * 128,
                "4-tap convolutions of 3 x H x 128 channels");
    TORCH_CHECK(state_in.numel() == H * 128 * 128 && state_out.numel() == H * 128 * 128, "states [H, 128, 128]");
    TORCH_CHECK(state_in.data_ptr() != state_out.data_ptr(), "state_out must not alias state_in");
    TORCH_CHECK(rows >= 1 && P.size(0) >= rows && A.size(0) >= rows && G.size(0) >= rows && out.size(0) >= rows,
                "rows");
    TORCH_CHECK(off >= 0 && off < 32, "off");
    TORCH_CHECK(spb >= 0, "sub-chunks a prep block: 0 (one wave) or more");
    TORCH_CHECK(window >= 1 && recs.numel() >= window * H * kda_split_record_bytes(), "records: window x H");
    TORCH_CHECK((uintptr_t)recs.data_ptr() % 16 == 0 && (uintptr_t)state_in.data_ptr() % 16 == 0 &&
                (uintptr_t)state_out.data_ptr() % 16 == 0, "records and states 16-byte aligned");
    c10::cuda::CUDAGuard guard(P.device());
    kda_split_cuda(P, p_stride, b_off, A, a_stride, G, g_stride, cs, cw, state_in, a_log, dt_bias, norm_w, eps, lower,
                   rows, off, out, state_out, recs, cols, nt, window, spb);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chain", &chain);
    m.def("record_bytes", &kda_split_record_bytes);
    m.def("state_smem", &kda_split_state_smem);
}
