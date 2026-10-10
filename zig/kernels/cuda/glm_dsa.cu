// GLM's indexed sparse attention (DSA): the absorbed-MLA flash form over
// top-k selected latent rows. q [rows*H, 512] (q_b absorbed, unscaled), kv
// [key_len, 512] the shared latent (K and V are the same rows), idx [rows, K]
// int32 selections (-1 or out-of-range pads skip), meta[0] the key length,
// scale fp32. One block per (row, head): 32 warps stream selections s, s+32..,
// online softmax per warp, then the block merge through per-warp (m, l) and a
// [32][512] partial plane - the metal's rotation, stated plainly.
// Grid (1, rows*H)? x covers nothing: blockIdx.y = row*H+head... blocks 1024.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace tf_glm_dsa {

constexpr int HDIM = 512;      // the latent/qk width
constexpr int VPC = 16;        // dims a lane owns (HDIM / 32)
constexpr int WARPS = 32;
constexpr int TOPK = 2051;     // the index row width (i_topk + kpool - 1)

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
    return v;
}

__device__ __forceinline__ float block_max(float v) {
    __shared__ float red[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x < 32) {
        v = red[lane];
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        if (lane == 0) red[0] = v;
    }
    __syncthreads();
    const float o = red[0];
    __syncthreads();
    return o;
}

extern "C" __global__ void dsa_attention(
    const __nv_bfloat16* __restrict__ q,     // [rows*H, HDIM]
    const __nv_bfloat16* __restrict__ kv,    // [key_len, HDIM]
    const int* __restrict__ idx,             // [rows, TOPK]
    const float* __restrict__ scale,
    const int* __restrict__ meta,            // [0] key_len
    __nv_bfloat16* __restrict__ out,         // [rows*H, HDIM]
    int heads
) {
    const int gid = blockIdx.y;       // row * heads + head
    const int row = gid / heads;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    __shared__ float sm[WARPS];
    __shared__ float sl[WARPS];

    // this lane's 16 q dims, pre-scaled
    float qr[VPC];
    const __nv_bfloat16* qp = q + (size_t)gid * HDIM + lane * VPC;
    #pragma unroll
    for (int j = 0; j < VPC; ++j) qr[j] = scale[0] * __bfloat162float(qp[j]);

    float m = -3.402823466e38f, l = 0.0f, o[VPC]; // finite: all-invalid warps keep expf(0)
    #pragma unroll
    for (int j = 0; j < VPC; ++j) o[j] = 0.0f;

    const int key_len = meta[0];
    const int* my = idx + (size_t)row * TOPK;
    for (int s = warp; s < TOPK; s += WARPS) {
        const int pos = my[s];
        const bool valid = pos >= 0 && pos < key_len;
        float score = -INFINITY;
        if (valid) {
            const __nv_bfloat16* kp = kv + (size_t)pos * HDIM + lane * VPC;
            float dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < VPC; ++j) dot = fmaf(qr[j], __bfloat162float(kp[j]), dot);
            score = warp_sum(dot);
        }
        const float m_new = fmaxf(m, score);
        const float factor = __expf(m - m_new);
        const float w = valid ? __expf(score - m_new) : 0.0f;
        m = m_new;
        l = l * factor + w;
        if (valid) {
            const __nv_bfloat16* vp = kv + (size_t)pos * HDIM + lane * VPC;
            #pragma unroll
            for (int j = 0; j < VPC; ++j) o[j] = o[j] * factor + w * __bfloat162float(vp[j]);
        } else {
            #pragma unroll
            for (int j = 0; j < VPC; ++j) o[j] *= factor;
        }
    }
    sl[warp] = l;
    sm[warp] = m;
    __syncthreads();

    // the block's M, then a FIXED-ORDER merge: warp w writes its plane, one
    // warp sums the planes in warp order (no atomics: bit-reproducible)
    float M = -3.402823466e38f;
    for (int d2 = threadIdx.x; d2 < WARPS; d2 += blockDim.x) M = fmaxf(M, sm[d2]);
    M = block_max(M);
    const float fac = __expf(sm[warp] - M);
    __syncthreads();
    {
        extern __shared__ float plane[]; // [WARPS][HDIM] dynamic
        float* mine = plane + (size_t)warp * HDIM;
        for (int j = 0; j < VPC; ++j) mine[lane * VPC + j] = o[j] * fac;
        if (lane == 0) sl[warp] = l * fac; // reuse as the plane's l sum
        __syncthreads();
        if (warp == 0) {
            float ls = 0.0f;
            #pragma unroll 8
            for (int w2 = 0; w2 < WARPS; ++w2) ls += sl[w2];
            const float inv = ls == 0.0f ? 0.0f : 1.0f / ls;
            for (int d2 = 0; d2 < HDIM; ++d2) {
                float v = 0.0f;
                #pragma unroll 8
                for (int w2 = 0; w2 < WARPS; ++w2) v += plane[(size_t)w2 * HDIM + d2];
                out[(size_t)gid * HDIM + d2] = __float2bfloat16(v * inv);
            }
        }
    }
}

} // namespace tf_glm_dsa
