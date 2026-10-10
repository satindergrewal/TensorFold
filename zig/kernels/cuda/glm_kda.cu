// GLM's KDA (gated delta net) step: the conv1d over the q/k/v projections and
// the delta rule on the [D, D] fp32 state, the metal kda_rows.metal's math with
// bf16 dense projections on cublasLt (this checkpoint has no 4-bit KDA tails).
// H=64 heads of D=128; one block per head, a thread per state row; the block
// reduces are the three L2 norms. The conv's sigmoid lives in kda_conv; the
// f_b/g_b outputs and the gate join here.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_glm_kda {

constexpr int H = 64;
constexpr int D = 128;
constexpr int W = H * D; // 8192
constexpr int TAPS = 4;

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float sig_precise(float x) { return 1.0f / (1.0f + expf(-x)); } // the python engine's exact form

__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float a = 0.0f;
    const int warps = (blockDim.x + 31) / 32;
    if (threadIdx.x < 32) {
        a = threadIdx.x < warps ? red[threadIdx.x] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffff, a, off);
        if (threadIdx.x == 0) red[0] = a;
    }
    __syncthreads();
    const float out = red[0];
    __syncthreads();
    return out;
}

// qkv[r, i] = bf(conv(src window) ) then sigmoided: sl = bf(x * sig(x)).
// grid (3W/256, rows); cs [TAPS-1, 3W] the history, cw [TAPS, 3W].
extern "C" __global__ void kda_conv(const __nv_bfloat16* __restrict__ p, int ps,
                                    const __nv_bfloat16* __restrict__ cs, const __nv_bfloat16* __restrict__ cw,
                                    __nv_bfloat16* __restrict__ qkv) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (i >= 3 * W) return;
    float acc = 0.0f;
    for (int j = 0; j < TAPS; ++j) {
        const int e = r + j;
        const float xv = e < TAPS - 1 ? __bfloat162float(cs[(size_t)e * 3 * W + i])
                                      : __bfloat162float(p[(size_t)(e - (TAPS - 1)) * ps + i]);
        acc = fmaf(xv, __bfloat162float(cw[(size_t)j * 3 * W + i]), acc);
    }
    qkv[(size_t)r * 3 * W + i] = __float2bfloat16_rn(acc / (1.0f + expf(-acc))); // silu exactly as chain_kernel
}

// The recurrent step over the whole window; grid H, block D (a state row each).
// qkv [rows, 3W] post-conv sigmoided; fg [rows, 2W]: the f_b then g_b outputs;
// bv [rows, H]; a [H]; dtb [H, D]; st/st_out [H, D, D] fp32; onw [W] bf16.
// y [rows, W]. The conv history's rotate is a host memcpy (TAPS-1 rows).
extern "C" __global__ void kda_step(const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ fg,
                                    const __nv_bfloat16* __restrict__ bv, const float* __restrict a,
                                    const float* __restrict dtb, const float* __restrict st_in,
                                    const __nv_bfloat16* __restrict onw, float lb, float eps, int rows,
                                    float* __restrict st_out, __nv_bfloat16* __restrict y) {
    const int h = blockIdx.x;
    const int t = threadIdx.x; // the state row dv
    __shared__ float sq[D], skD[D], kvec[D], qvec[D], graw[D], red[8];
    float st[D];
    const float* sh = st_in + (size_t)h * D * D;
    for (int i = 0; i < D; ++i) st[i] = sh[(size_t)t * D + i];
    const float a_h = a[h];
    const float qscale = rsqrtf((float)D);

    for (int r = 0; r < rows; ++r) {
        const size_t qb = (size_t)r * 3 * W + (size_t)h * D;
        // q, k, v: the conv'd sigmoid values
        const float vq = __bfloat162float(qkv[qb + t]);
        const float vk = __bfloat162float(qkv[qb + W + t]);
        const float vv = __bfloat162float(qkv[qb + 2 * W + t]);
        const float sa = __bfloat162float(fg[(size_t)r * 2 * W + (size_t)h * D + t]);
        graw[t] = sa; // the f_b out (already bf16), dk-indexed in the state, dv in the out
        const float beta = sig_precise(__bfloat162float(bv[(size_t)r * H + h]));

        // the L2 norms over the raw values; then everyone's normalized vector
        sq[t] = vq;
        skD[t] = vk;
        __syncthreads();
        if (t == 0) {
            float pq2 = 0.0f, pk2 = 0.0f;
            for (int i = 0; i < D; ++i) {
                pq2 = fmaf(sq[i], sq[i], pq2);
                pk2 = fmaf(skD[i], skD[i], pk2);
            }
            red[0] = rsqrtf(pq2 + 1e-6f);
            red[1] = rsqrtf(pk2 + 1e-6f);
        }
        __syncthreads();
        const float rq = red[0], rk = red[1];
        qvec[t] = (vq * rq) * qscale; // fp32: the python engine keeps the normalized vectors unrounded
        kvec[t] = vk * rk;
        __syncthreads();

        // this dv's state row: the dk gate, the key dot, the delta, the out
        float kv = 0.0f, oo = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float g = expf(lb * sig_precise(a_h * (graw[i] + dtb[(size_t)h * D + i])));
            st[i] *= g;
            kv = fmaf(st[i], kvec[i], kv);
        }
        const float delta = (vv - kv) * beta;
        for (int i = 0; i < D; ++i) {
            st[i] = fmaf(kvec[i], delta, st[i]);
            oo = fmaf(st[i], qvec[i], oo);
        }
        __shared__ float syv[D];
        syv[t] = bf(oo);
        __syncthreads();
        // every thread sums the whole array: identical, no reduce needed
        float po = 0.0f;
        for (int i = 0; i < D; ++i) po = fmaf(syv[i], syv[i], po);
        const float pon = 1.0f / sqrtf(po / (float)D + eps);
        const float gate = sig_precise(__bfloat162float(fg[(size_t)r * 2 * W + W + (size_t)h * D + t])); // g_b's, dv-indexed
        float x = syv[t] * pon;
        x = __bfloat162float(onw[t]) * x; // shared [D] across heads
        x *= gate;
        y[(size_t)r * W + (size_t)h * D + t] = __float2bfloat16(x);
        __syncthreads();
    }
    float* so = st_out + (size_t)h * D * D;
    for (int i = 0; i < D; ++i) so[(size_t)t * D + i] = st[i];
}

} // namespace tf_glm_kda
