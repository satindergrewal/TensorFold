// GLM's hyper-connection boundaries: the 4-stream residual expand, the 24 mix
// features, and the split that makes the next boundary's post/comb, the block
// input's collapse and its RMSNorm. The spec is kernels/metal/core/hc_boundary;
// D is the arch's hidden (4096), S its stream count (4) — constexpr like metal.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_glm_hc {

constexpr int S = 4;
constexpr int D = 4096;
constexpr int F = S * D;
constexpr int MIX = 24;       // 2*S pre/post + S*S comb features
constexpr int SLICES = 16;    // F / 1024
constexpr int PARTS = MIX + 1; // the mixes and the square sum

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

// ─── expand: fold the branch into the streams by the last split's post/comb
// (expand), or pass the streams through (first boundary); either way the 24 mix
// dots and the square sum, as PART[row, slice, 25] partials.
// Grid (SLICES, rows), 256 threads; thread t takes f = 1024*j + 4*t .. +3.

extern "C" __global__ void hc_expand_mix(
    const __nv_bfloat16* __restrict__ xold,  // [rows, F]
    const __nv_bfloat16* __restrict__ branch,// [rows, D]
    const float* __restrict__ post,          // [rows, S]
    const float* __restrict__ comb,          // [rows, S, S]
    const __nv_bfloat16* __restrict__ fn,    // [MIX, F]
    __nv_bfloat16* __restrict__ xnew,        // [rows, F], untouched when expand == 0
    float* __restrict__ part,                // [rows, SLICES, PARTS]
    int rows, int expand
) {
    const int j = blockIdx.x;
    const int r = blockIdx.y;
    const int t = threadIdx.x;

    const size_t rb = (size_t)r * F;
    const float* c = comb + (size_t)r * S * S;
    float pm[MIX];
    for (int m = 0; m < MIX; ++m) pm[m] = 0.0f;
    float ss = 0.0f;

    for (int tn = 0; tn < 4; ++tn) {
        const int f = 1024 * j + 4 * t + tn;
        const int s = f / D, d = f - s * D;
        float v = __bfloat162float(xold[rb + f]);
        if (expand) {
            float mm = c[0 * S + s] * __bfloat162float(xold[rb + 0 * D + d]);
            mm = fmaf(c[1 * S + s], __bfloat162float(xold[rb + 1 * D + d]), mm);
            mm = fmaf(c[2 * S + s], __bfloat162float(xold[rb + 2 * D + d]), mm);
            mm = fmaf(c[3 * S + s], __bfloat162float(xold[rb + 3 * D + d]), mm);
            v = bf(post[(size_t)r * S + s] * __bfloat162float(branch[(size_t)r * D + d]) + mm);
            xnew[rb + f] = __float2bfloat16(v);
        }
        ss = fmaf(v, v, ss);
        for (int m = 0; m < MIX; ++m)
            pm[m] = fmaf(__bfloat162float(fn[(size_t)m * F + f]), v, pm[m]);
    }
    // deterministic fixed-order reduce: warp shuffle down, then warps in index order
    {
        const int lane = t & 31, warp = t >> 5;
        #pragma unroll
        for (int m = 0; m < MIX; ++m)
            for (int off = 16; off > 0; off >>= 1) pm[m] += __shfl_down_sync(0xffffffff, pm[m], off);
        for (int off = 16; off > 0; off >>= 1) ss += __shfl_down_sync(0xffffffff, ss, off);
        __shared__ float wred[8][PARTS];
        if (lane == 0) {
            #pragma unroll
            for (int m = 0; m < MIX; ++m) wred[warp][m] = pm[m];
            wred[warp][MIX] = ss;
        }
        __syncthreads();
        if (t < PARTS) {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) v += wred[w][t];
            part[((size_t)r * SLICES + j) * PARTS + t] = v;
        }
    }
}

// ─── split: the slices' sums, the mixes' norm, pre/post z's, the comb's
// Sinkhorn, the streams' collapse by pre and its RMSNorm.
// Grid (rows), 1024 threads; thread t of the collapse part takes d = 4t .. +3.

extern "C" __global__ void hc_split(
    const __nv_bfloat16* __restrict__ x,     // [rows, F] the expanded streams
    const float* __restrict__ part,          // [rows, SLICES, PARTS]
    const float* __restrict__ scale,         // [3] pre, post, comb
    const float* __restrict__ base,          // [MIX]
    const __nv_bfloat16* __restrict__ norm_w,// [D]
    const float* __restrict__ eps,
    __nv_bfloat16* __restrict__ normed,      // [rows, D]
    float* __restrict__ post_out,            // [rows, S]
    float* __restrict__ comb_out,            // [rows, S, S]
    int rows
) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    __shared__ float tot[PARTS];
    __shared__ float mix[MIX]; // shared: the Sinkhorn thread reads every feature
    __shared__ float pre_s[S];
    __shared__ float inv_s;

    if (t < PARTS) {
        float a = part[((size_t)r * SLICES + 0) * PARTS + t];
        for (int j = 1; j < SLICES; ++j) a += part[((size_t)r * SLICES + j) * PARTS + t];
        tot[t] = a;
    }
    __syncthreads();
    const float inv_n = rsqrtf(tot[MIX] / (float)F + eps[0]);
    if (t < MIX) mix[t] = tot[t] * inv_n;
    __syncthreads();

    if (t < S) {
        const float z = mix[t] * scale[0] + base[t];
        pre_s[t] = 1.0f / (1.0f + __expf(-z)) + eps[0];
        const float zp = mix[S + t] * scale[1] + base[S + t];
        post_out[(size_t)r * S + t] = 2.0f / (1.0f + __expf(-zp));
    }
    // the comb's Sinkhorn, one thread: 4x4, an eps on every norm
    if (t == 0) {
        float mm[4][4];
        for (int i = 0; i < S; ++i)
            for (int jj = 0; jj < S; ++jj)
                mm[i][jj] = mix[2 * S + i * S + jj] * scale[2] + base[2 * S + i * S + jj];
        float row[4];
        for (int i = 0; i < S; ++i) {
            row[i] = -INFINITY;
            for (int jj = 0; jj < S; ++jj) row[i] = fmaxf(row[i], mm[i][jj]);
        }
        float rr[4][4];
        for (int i = 0; i < S; ++i) {
            float e[4], sum = 0.0f;
            for (int jj = 0; jj < S; ++jj) {
                e[jj] = __expf(mm[i][jj] - row[i]);
                sum += e[jj];
            }
            for (int jj = 0; jj < S; ++jj) rr[i][jj] = e[jj] / (sum + eps[0]) + eps[0];
        }
        for (int it = 0; it < 20; ++it) {
            for (int i = 0; i < S; ++i) {
                float sum = 0.0f;
                for (int jj = 0; jj < S; ++jj) sum += rr[i][jj];
                for (int jj = 0; jj < S; ++jj) rr[i][jj] /= sum + eps[0];
            }
            for (int jj = 0; jj < S; ++jj) {
                float sum = 0.0f;
                for (int i = 0; i < S; ++i) sum += rr[i][jj];
                for (int i = 0; i < S; ++i) rr[i][jj] /= sum + eps[0];
            }
        }
        for (int i = 0; i < S; ++i)
            for (int jj = 0; jj < S; ++jj)
                comb_out[(size_t)r * S * S + i * S + jj] = rr[i][jj];
    }
    __syncthreads();

    // the collapse by pre and its RMS norm; thread t takes d = 4t .. +3
    float xc[4], ss = 0.0f;
    const size_t rb = (size_t)r * F;
    for (int i = 0; i < 4; ++i) {
        const int d = 4 * t + i;
        float v = pre_s[0] * __bfloat162float(x[rb + d]);
        v = fmaf(pre_s[1], __bfloat162float(x[rb + D + d]), v);
        v = fmaf(pre_s[2], __bfloat162float(x[rb + 2 * D + d]), v);
        v = fmaf(pre_s[3], __bfloat162float(x[rb + 3 * D + d]), v);
        xc[i] = bf(v);
        ss = fmaf(xc[i], xc[i], ss);
    }
    __shared__ float red[32];
    const int lane = t & 31, warp = t >> 5;
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffff, ss, o);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (t == 0) {
        float a = 0.0f;
        for (int w = 0; w < 32; ++w) a += red[w];
        inv_s = rsqrtf(a / (float)D + eps[0]);
    }
    __syncthreads();
    for (int i = 0; i < 4; ++i) {
        const int d = 4 * t + i;
        normed[(size_t)r * D + d] = __float2bfloat16(__bfloat162float(norm_w[d]) * bf(xc[i] * inv_s));
    }
}

} // namespace tf_glm_hc
