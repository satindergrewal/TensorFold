// GLM's dense ops: embed gather, plain RMSNorm, the swiglu-limit activation
// (asymmetric: the gate clamps above only), the MoE combine, the streams' mean,
// the MTP add, greedy argmax, and the sigmoid+bias route. The semantics are the
// metal family's (glm_glue.metal, moogateup/moe_combine/moe_route), bf16
// roundings at the same points.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_glm_ops {

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float sig_precise(float x) {
    const float e = expf(fabsf(x));
    const float y = 1.0f / (1.0f + e);
    return x < 0 ? y : 1.0f - y;
}

// out[r, d] = E[ids[r], d]; grid over rows*D / 256.
extern "C" __global__ void glm_embed(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ e,
                                     __nv_bfloat16* __restrict__ out, int rows, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * d;
    if (i >= total) return;
    const int r = i / d, c = i - r * d;
    out[i] = __ldg(e + (size_t)__ldg(ids + r) * d + c);
}

// y[r, c] = w[c] * bf16(x[r, c] * inv_rms), fp32 square sum; grid rows, block 512 x 8.
extern "C" __global__ void glm_rms(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                   __nv_bfloat16* __restrict__ y, int rows, int width, int x_stride, int y_stride,
                                   float eps) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const int base = t * 8; // eight consecutive, 4096 at 512 threads
    float sq = 0.0f;
    for (int j = 0; j < 8; ++j) {
        const int c = base + j;
        if (c >= width) break;
        const float v = __bfloat162float(x[(size_t)r * x_stride + c]);
        sq = fmaf(v, v, sq);
    }
    __shared__ float red[32];
    const int lane = t & 31, warp = t >> 5;
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_down_sync(0xffffffff, sq, off);
    if (lane == 0) red[warp] = sq;
    __syncthreads();
    __shared__ float inv;
    if (t == 0) {
        float a = 0.0f;
        for (int w_ = 0; w_ < (int)(blockDim.x + 31) / 32; ++w_) a += red[w_];
        inv = rsqrtf(a / (float)width + eps);
    }
    __syncthreads();
    for (int j = 0; j < 8; ++j) {
        const int c = base + j;
        if (c >= width) break;
        y[(size_t)r * y_stride + c] = __float2bfloat16(__bfloat162float(w[c]) * bf(__bfloat162float(x[(size_t)r * x_stride + c]) * inv));
    }
}

// act[r, s, j] from gate/up [rows, slots, N]: the gate clamps above only.
extern "C" __global__ void glm_act(const __nv_bfloat16* __restrict__ g, const __nv_bfloat16* __restrict__ u,
                                   __nv_bfloat16* __restrict__ act, int rows, int slots, int n, float lim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * slots * n;
    if (i >= total) return;
    // glue._swiglu: fp32 clamps (unrounded), silu by the division form rounded
    // once, then the up product rounded once
    const float gv = __bfloat162float(g[i]);
    const float uv = __bfloat162float(u[i]);
    const float gt = fminf(gv, lim);
    const float up = fminf(fmaxf(uv, -lim), lim);
    act[i] = __float2bfloat16(bf(gt / (1.0f + expf(-gt))) * up);
}

// out[r, d] = bf16(sum_k wts[r, k] * ye[r, k, d] fused) + ys[r, d].
extern "C" __global__ void glm_combine(const __nv_bfloat16* __restrict__ ys, const __nv_bfloat16* __restrict__ ye,
                                      const float* __restrict__ wts, __nv_bfloat16* __restrict__ out, int topk, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= d) return;
    const __nv_bfloat16* y = ye + (size_t)r * topk * d + i;
    float acc = wts[r * topk] * __bfloat162float(y[0]);
    for (int k = 1; k < topk; ++k) acc = fmaf(wts[r * topk + k], __bfloat162float(y[(size_t)k * d]), acc);
    out[(size_t)r * d + i] = __float2bfloat16(bf(acc) + __bfloat162float(ys[(size_t)r * d + i]));
}

extern "C" __global__ void glm_embed_f32(const int* __restrict__ ids, const __nv_bfloat16* __restrict__ e,
                                         float* __restrict__ out, int rows, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rows * d;
    if (i >= total) return;
    const int r = i / d, c = i - r * d;
    out[i] = __bfloat162float(__ldg(e + (size_t)__ldg(ids + r) * d + c));
}

// y[r, d] = bf16(sum of the 4 f32 streams in order, * 0.25).
extern "C" __global__ void glm_stream_mean_f32(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= d) return;
    const float* row = x + (size_t)r * 4 * d + i;
    float v = row[0];
    for (int s = 1; s < 4; ++s) v += row[(size_t)s * d];
    y[(size_t)r * d + i] = __float2bfloat16(v * 0.25f);
}

// y[r, d] = bf16(sum of the 4 streams in order, * 0.25).
extern "C" __global__ void glm_stream_mean(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= d) return;
    const __nv_bfloat16* row = x + (size_t)r * 4 * d + i;
    float v = __bfloat162float(row[0]);
    for (int s = 1; s < 4; ++s) v += __bfloat162float(row[(size_t)s * d]);
    y[(size_t)r * d + i] = __float2bfloat16(v * 0.25f);
}

extern "C" __global__ void glm_add(const __nv_bfloat16* __restrict__ a, const __nv_bfloat16* __restrict__ b,
                                   __nv_bfloat16* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

// out[row]: the larger logit, the lower index on a tie; block 1024 a row.
extern "C" __global__ void glm_argmax(const __nv_bfloat16* __restrict__ logits, int* __restrict__ out, int vocab) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const __nv_bfloat16* x = logits + (size_t)r * vocab;
    float best = -INFINITY;
    int at = 0x7fffffff;
    for (int i = t; i < vocab; i += blockDim.x) {
        const float v = __bfloat162float(__ldg(x + i));
        if (v > best) {
            best = v;
            at = i;
        }
    }
    __shared__ float vb[32];
    __shared__ int va[32];
    const int lane = t & 31, warp = t >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffff, best, off);
        const int oa = __shfl_xor_sync(0xffffffff, at, off);
        if (ob > best || (ob == best && oa < at)) {
            best = ob;
            at = oa;
        }
    }
    if (lane == 0) {
        vb[warp] = best;
        va[warp] = at;
    }
    __syncthreads();
    if (warp != 0) return;
    best = vb[lane];
    at = va[lane];
    for (int off = 16; off > 0; off >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffff, best, off);
        const int oa = __shfl_xor_sync(0xffffffff, at, off);
        if (ob > best || (ob == best && oa < at)) {
            best = ob;
            at = oa;
        }
    }
    if (lane == 0) out[r] = at;
}

// The route: sigmoid + bias, top-k ties to the lower id, weights the sigmoid
// scaled by the normalized share; one block of 512 a row, e < experts holds.
// `pick` [rows, topk] i32, `wts` [rows, topk] f32.
extern "C" __global__ void glm_route(const float* __restrict__ logits, const float* __restrict__ bias,
                                     int* __restrict__ pick, float* __restrict__ wts, int experts, int topk,
                                     float scale) {
    const int r = blockIdx.x;
    const int e = threadIdx.x;
    __shared__ float c[512];
    __shared__ float sc[512];
    __shared__ float wr[32]; // the raw weight of round k
    __shared__ int win;
    const bool has = e < experts;
    if (has) {
        const float s = sig_precise(logits[(size_t)r * experts + e]);
        sc[e] = s;
        c[e] = s + bias[e];
    } else {
        c[e] = -INFINITY;
        sc[e] = 0.0f;
    }
    __syncthreads();
    int win_s = 0;
    for (int k = 0; k < topk; ++k) {
        float best = c[e]; // fresh: the last round's winner is now -INF
        int at = e;
        for (int off = 16; off > 0; off >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffff, best, off);
            const int oa = __shfl_xor_sync(0xffffffff, at, off);
            if (ob > best || (ob == best && oa < at)) {
                best = ob;
                at = oa;
            }
        }
        __shared__ float rb[16];
        __shared__ int ra[16];
        const int lane = e & 31, warp = e >> 5;
        if (lane == 0) {
            rb[warp] = best;
            ra[warp] = at;
        }
        __syncthreads();
        if (warp == 0) {
            // all 32 lanes join the shuffle; only the low 16 read the stage array
            best = lane < 16 ? rb[lane] : -INFINITY;
            at = lane < 16 ? ra[lane] : 0x7fffffff;
            for (int off = 8; off > 0; off >>= 1) { // 16 warps
                const float ob = __shfl_xor_sync(0xffffffff, best, off);
                const int oa = __shfl_xor_sync(0xffffffff, at, off);
                if (ob > best || (ob == best && oa < at)) {
                    best = ob;
                    at = oa;
                }
            }
            if (lane == 0) {
                win = at;
                wr[k] = sc[at];
            }
        }
        __syncthreads();
        win_s = win;
        if (e == win_s) c[e] = -INFINITY;
        if (e < topk) pick[(size_t)r * topk + k] = win_s;
    }
    __shared__ float total;
    if (e == 0) {
        float t = wr[0];
        for (int k = 1; k < topk; ++k) t = t + wr[k];
        total = t;
    }
    __syncthreads();
    if (e < topk) wts[(size_t)r * topk + e] = (wr[e] / total) * scale;
}

// dst[r, c] = src[r * src_stride + c]; rows x cols bf16.
extern "C" __global__ void glm_copy2d(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                      int rows, int cols, int src_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    const int r = blockIdx.y;
    dst[(size_t)r * cols + i] = src[(size_t)r * src_stride + i];
}

// dst[i] = src[i * src_stride]; n bf16 elements.
extern "C" __global__ void glm_gather(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                      int n, int src_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[(size_t)i * src_stride];
}

// LayerNorm with weight and bias (the indexer's k_norm): rows x width, eps.
// grid rows, block 256 x 8.
extern "C" __global__ void glm_layer_norm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                          const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ y,
                                          int width, int x_stride, int y_stride, float eps) {
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    __shared__ float red[2][8];
    float mean_acc = 0.0f, var_acc = 0.0f;
    const int base = t * 8;
    for (int j = 0; j < 8; ++j) {
        const int c = base + j;
        if (c >= width) break;
        const float v = __bfloat162float(x[(size_t)r * x_stride + c]);
        mean_acc += v;
        var_acc = fmaf(v, v, var_acc);
    }
    // block reduce both (each thread holds partials; a tree over shared)
    __shared__ float means[32], vars[32];
    {
        const int lane = t & 31, warp = t >> 5;
        for (int off = 16; off > 0; off >>= 1) {
            mean_acc += __shfl_down_sync(0xffffffff, mean_acc, off);
            var_acc += __shfl_down_sync(0xffffffff, var_acc, off);
        }
        if (lane == 0) {
            means[warp] = mean_acc;
            vars[warp] = var_acc;
        }
    }
    __syncthreads();
    if (t == 0) {
        float m = 0, v2 = 0;
        const int warps = (blockDim.x + 31) / 32;
        for (int i = 0; i < warps; ++i) {
            m += means[i];
            v2 += vars[i];
        }
        red[0][0] = m / (float)width;
        red[1][0] = rsqrtf(v2 / (float)width - (m / (float)width) * (m / (float)width) + eps);
    }
    __syncthreads();
    const float mean = red[0][0], inv = red[1][0];
    for (int j = 0; j < 8; ++j) {
        const int c = base + j;
        if (c >= width) break;
        const float v = __bfloat162float(x[(size_t)r * x_stride + c]);
        y[(size_t)r * y_stride + c] = __float2bfloat16(bf((v - mean) * inv) * __bfloat162float(w[c]) + __bfloat162float(b[c]));
    }
    (void)red;
}

// The pooled indexer keys: pool[blk, d] = sum_j softmax(ig + ape)[j] * ik[j] over
// the kpool of 4; precise exp, each op rounded as the metal's glm_pool.
// grid (128, blocks).
extern "C" __global__ void glm_pool(const __nv_bfloat16* __restrict__ ik, const __nv_bfloat16* __restrict__ ig,
                                    const __nv_bfloat16* __restrict__ ape, __nv_bfloat16* __restrict__ pool,
                                    int first_blk, int n_blocks) {
    const int d = blockIdx.x; // the 128 dims
    const int blk = first_blk + blockIdx.y;
    if (blockIdx.y >= n_blocks) return;
    float logit[4], k[4];
    for (int j = 0; j < 4; ++j) {
        const size_t at = (size_t)(blk * 4 + j) * 128 + d;
        logit[j] = __bfloat162float(ig[at]) + __bfloat162float(ape[j * 128 + d]);
        k[j] = __bfloat162float(ik[at]);
    }
    float top = logit[0];
    for (int j = 1; j < 4; ++j) top = fmaxf(top, logit[j]);
    float e[4], total = 0.0f;
    for (int j = 0; j < 4; ++j) {
        e[j] = expf(logit[j] - top);
        total = total + e[j];
    }
    float out = (e[0] / total) * k[0];
    for (int j = 1; j < 4; ++j) out = out + (e[j] / total) * k[j];
    pool[(size_t)blk * 128 + d] = __float2bfloat16(out);
}

// dst[i] = bf16(src[i] * s) over n elements.
extern "C" __global__ void glm_scale(const __nv_bfloat16* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                     int n, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(__bfloat162float(src[i]) * s);
}

} // namespace tf_glm_ops
