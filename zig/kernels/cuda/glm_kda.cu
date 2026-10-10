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
__device__ __forceinline__ float sig_precise(float x) {
    const float e = expf(fabsf(x));
    const float y = 1.0f / (1.0f + e);
    return x < 0 ? y : 1.0f - y;
}

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
    const float xb = bf(acc);
    qkv[(size_t)r * 3 * W + i] = __float2bfloat16(xb * bf(xb * sig_precise(xb)));
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
    __shared__ float sq[D], sk[D], red[32];
    float st[D];
    const float* sh = st_in + (size_t)h * D * D;
    for (int i = 0; i < D; ++i) st[i] = sh[(size_t)t * D + i];
    const float a_h = a[h];
    const float qscale = rsqrtf((float)D);

    for (int r = 0; r < rows; ++r) {
        const size_t qb = (size_t)r * 3 * W + (size_t)h * D;
        // q, k, v: the conv'd sigmoid values
        float vq = __bfloat162float(qkv[qb + t]);
        float vk = __bfloat162float(qkv[qb + W + t]);
        float vv = __bfloat162float(qkv[qb + 2 * W + t]);
        // the gate from f_b's out and dtb; the L2 norms need all threads first
        const float sa = __bfloat162float(fg[(size_t)r * W + (size_t)h * D + t]);
        const float av = bf(sa) + dtb[(size_t)h * D + t];
        const float sg = expf(lb * sig_precise(a_h * av));
        const float beta = sig_precise(__bfloat162float(bv[(size_t)r * H + h]));

        sq[t] = vq; // the values; the norm squares them
        sk[t] = vk;
        __syncthreads();
        // the L2 norms: one thread sums the shared arrays (D=128, cheap and order-fixed)
        if (t == 0) {
            float pq2 = 0.0f, pk2 = 0.0f;
            for (int i = 0; i < D; ++i) {
                pq2 = fmaf(sq[i], sq[i], pq2);
                pk2 = fmaf(sk[i], sk[i], pk2);
            }
            red[0] = rsqrtf(pq2 + 1e-6f);
            red[1] = rsqrtf(pk2 + 1e-6f);
        }
        __syncthreads();
        const float rq = red[0], rk = red[1];
        const float q = bf((vq * rq) * qscale);
        const float k = bf(vk * rk);

        // the state row update, the out: this thread's dv
        float kv = 0.0f, oo = 0.0f;
        for (int i = 0; i < D; ++i) {
            st[i] *= sg;
            kv = fmaf(st[i], k, kv);
        }
        const float delta = (vv - kv) * beta;
        for (int i = 0; i < D; ++i) {
            st[i] = fmaf(k, delta, st[i]);
            oo = fmaf(st[i], q, oo);
        }
        // the out norm needs all sy: two-pass via shared
        __shared__ float syv[D];
        syv[t] = bf(oo);
        __syncthreads();
        float po = 0.0f;
        for (int i = 0; i < D; ++i) po = fmaf(syv[i], syv[i], po);
        // block reduce po
        float pon;
        {
            const int lane = t & 31, warp = t >> 5;
            for (int off = 16; off > 0; off >>= 1) po += __shfl_down_sync(0xffffffff, po, off);
            if (lane == 0) red[2 + warp] = po;
            __syncthreads();
            if (t == 0) {
                float a2 = 0.0f;
                for (int w_ = 0; w_ < 4; ++w_) a2 += red[2 + w_];
                red[2] = rsqrtf(a2 / (float)D + eps);
            }
            __syncthreads();
            pon = red[2];
        }
        const float gate = sig_precise(bf(__bfloat162float(fg[(size_t)r * W + W + (size_t)h * D + t])));
        float x = syv[t] * pon;
        x = __bfloat162float(onw[(size_t)h * D + t]) * x;
        x *= gate;
        y[(size_t)r * W + (size_t)h * D + t] = __float2bfloat16(x);
        __syncthreads();
    }
    float* so = st_out + (size_t)h * D * D;
    for (int i = 0; i < D; ++i) so[(size_t)t * D + i] = st[i];
}

} // namespace tf_glm_kda
