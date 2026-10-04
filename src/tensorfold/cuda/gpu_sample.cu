// The keyed top-k draw (tensorfold.engine.exact_sampling.choose_rows) and the greedy pick on the GPU, certified.
//
// Everything that decides a token is either exact on the GPU or certified by a margin:
//   exact: the merge of every rank's candidates by (value descending, id ascending), the top-k cut, the scaling
//     value / max(T, 1e-6) (IEEE double division), the splitmix64 uniform of (seed, position, id) (integer mixing, an
//     exact product by 2^-53 and one rounded add), the min_p threshold (one double add) and its comparisons;
//   certified: the two decisions that go through log / exp, which the host computes with numpy's float64 functions
//     and the GPU with CUDA's (each within a few ulps of the exact value, not always the same ulp): the argmax of
//     scaled - log(-log(u)) is taken only when the best score leads the next by more than 2 x DS x max(1, |scores|)
//     (DS = 2^-40, far above the difference of two few-ulp logs), and the top_p cut only when no cumulative
//     probability lies within DP = 2^-34 of top_p. A row that misses either margin (or holds a NaN) is flagged
//     uncertain and the host draws it with the exact host rule; a flagged row's token is never used.
// So a certified token equals the host rule's token bit for bit, and every other token comes from the host rule.
//
// Layout: candidates of rank q for row r are got[q * rank_stride + off_r + j] (values, fp32) and
// got[q * rank_stride + off_r + n_r + j] (token ids, int32 bits), j < n_r: the all-gathered pack the samplers build.
// table [rows, COLS] int64: off, n, mode (0 greedy, 1 keyed top-k), top_k (0: all), seed, position, then the bits of
// the doubles max(T, 1e-6), top_p, ln(min_p) (-inf: off). out [rows, 2] int32: token, certified (1) or not (0).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <math_constants.h>

namespace {

constexpr int COLS = 9;
constexpr int MAXC = 1024;          // candidates a row (world x n)
constexpr int THREADS = 256;
constexpr double DS = 0x1.0p-40;    // score margin (relative)
constexpr double DP = 0x1.0p-34;    // top_p margin (absolute)

__device__ __forceinline__ unsigned long long mix(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// exact_sampling.uniform: (seed, position, id) -> a double in (0, 1], the same bits
__device__ __forceinline__ double keyed_uniform(unsigned long long seed, long long position, long long id) {
    unsigned long long x = mix(seed + 0x9E3779B97F4A7C15ull);
    x = mix(x ^ ((unsigned long long)position * 0xD1B54A32D192ED03ull));
    x = mix(x ^ (unsigned long long)id);
    return __dadd_rn(__dmul_rn((double)(x >> 11), 0x1.0p-53), 0x1.0p-54);
}

__global__ void __launch_bounds__(THREADS) sample_rows_kernel(const float* __restrict__ got, long long rank_stride,
                                                             int world, const long long* __restrict__ table,
                                                             int* __restrict__ out) {
    const int r = blockIdx.x;
    const long long* t = table + (size_t)r * COLS;
    const long long off = t[0];
    const int n = (int)t[1], mode = (int)t[2], top_k = (int)t[3];
    const unsigned long long seed = (unsigned long long)t[4];
    const long long position = t[5];
    const double temp = __longlong_as_double(t[6]), top_p = __longlong_as_double(t[7]);
    const double min_log = __longlong_as_double(t[8]);
    const int W = world * n;
    __shared__ float sv[MAXC];
    __shared__ int sid[MAXC];
    __shared__ float ov[MAXC];
    __shared__ int oid[MAXC];
    __shared__ double sc[MAXC];
    __shared__ double score[MAXC];
    __shared__ double ex[MAXC];                                // exp(scaled - top), for the top_p cut
    __shared__ int bad;
    if (threadIdx.x == 0) bad = (W < 1 || W > MAXC);
    __syncthreads();
    if (bad) {
        if (threadIdx.x == 0) { out[2 * r] = -1; out[2 * r + 1] = 0; }
        return;
    }
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        const int q = i / n, j = i % n;
        const float* base = got + (size_t)q * rank_stride + off;
        sv[i] = base[j];
        sid[i] = __float_as_int(base[n + j]);
    }
    __syncthreads();
    // np.lexsort((ids, -values)): value descending (IEEE order, -0 == +0), then id ascending; ids are distinct
    int nan_here = 0;
    for (int i = threadIdx.x; i < W; i += blockDim.x) {
        const float v = sv[i];
        const int id = sid[i];
        nan_here |= v != v;                                    // NaN: the host rule decides
        int rank = 0;
        for (int j = 0; j < W; ++j) {
            const float u = sv[j];
            rank += (u > v) || (u == v && sid[j] < id);
        }
        if (rank < W) { ov[rank] = v; oid[rank] = id; }
    }
    if (__syncthreads_or(nan_here) && threadIdx.x == 0) bad = 1;
    __syncthreads();
    if (mode == 0) {                                           // greedy: the first in that order
        if (threadIdx.x == 0) { out[2 * r] = oid[0]; out[2 * r + 1] = bad ? 0 : 1; }
        return;
    }
    const int k = max(1, min(top_k > 0 ? top_k : W, W));
    const double top = __ddiv_rn((double)ov[0], temp);         // scaled.max(): the first in the order
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        const double s = __ddiv_rn((double)ov[i], temp);
        sc[i] = s;
        score[i] = __dsub_rn(s, log(-log(keyed_uniform(seed, position, (long long)oid[i]))));
        ex[i] = exp(__dsub_rn(s, top));
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    int certain = !bad;
    int keep = k;
    if (top_p > 0.0 && top_p < 1.0) {                          // (cumsum(probs) < top_p).sum() + 1
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += ex[i];
        double cum = 0.0;
        int below = 0;
        for (int i = 0; i < k; ++i) {
            cum += ex[i] / sum;
            if (fabs(cum - top_p) <= DP) certain = 0;
            below += cum < top_p;
        }
        keep = below + 1;
    }
    const double thr = __dadd_rn(sc[0], min_log);             // min_p off: -inf, nothing below it
    int best = -1;
    double s1 = -CUDART_INF, s2 = -CUDART_INF;
    for (int i = 0; i < k && i < keep; ++i) {
        if (sc[i] < thr) continue;
        const double s = score[i];
        if (s != s) certain = 0;                               // a NaN score (-inf - -inf): numpy's argmax takes it
        if (best < 0 || s > s1) { s2 = s1; s1 = s; best = i; }
        else if (s > s2) s2 = s;
    }
    if (best < 0) { best = 0; certain = 0; }
    if (!isfinite(s1)) certain = 0;
    if (s2 > -CUDART_INF) {
        const double scale = fmax(1.0, fmax(fabs(s1), fabs(s2)));
        if (!(s1 - s2 > 2.0 * DS * scale)) certain = 0;
    }
    out[2 * r] = oid[best];
    out[2 * r + 1] = certain;
}

}  // namespace

void gpu_sample_rows_cuda(const at::Tensor& got, int64_t rank_stride, int64_t world, const at::Tensor& table,
                          at::Tensor& out, int64_t rows) {
    if (rows <= 0) return;
    sample_rows_kernel<<<(unsigned)rows, THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        got.data_ptr<float>(), (long long)rank_stride, (int)world,
        reinterpret_cast<const long long*>(table.data_ptr<int64_t>()), out.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
