// GLM-5.3-Flash's vision tower helpers: 2-D rope apply, naive per-head
// attention (fp32 softmax), GEMM bias epilogue, GELU and layer-norm (the
// merger's mid-norm carries a bias, unlike the tower's rms norms).
// The rows/heads geometry and the reference math: session notes / the
// extracted lane module (glm.py) - vision_ref_feat.bin is the parity target.

#include <cuda_bf16.h>
#include <math.h>

namespace tf_glm_vision {

__device__ __forceinline__ float bf(float v) { return __bfloat162float(__float2bfloat16(v)); }

// y = x + bias[col]: the tower's linears all carry biases and the shared
// cublasLt plan has no epilogue. grid(rows), block(256).
// debug: o[0..3] = (m, denom, score0*scale, q0) for row 0 head 0
extern "C" __global__ void vv_attn_dbg(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                        const __nv_bfloat16* __restrict__ v, float* __restrict__ dbg_out,
                                        int S, int heads, int hd, float scale) {
    if (blockIdx.x != 0) return;
    const int t = threadIdx.x;
    if (t >= hd) return;
    extern __shared__ float sh[];
    const int nwarps = (blockDim.x + 31) / 32;
    float* wd = sh;
    const int lane = t & 31, warp = t >> 5;
    const float qv = __bfloat162float(q[t]);
    float m = -INFINITY; float s0 = 0.0f; float s0b = -INFINITY; int am = -1;
    for (int s2 = 0; s2 < S; ++s2) {
        float d = qv * __bfloat162float(k[((size_t)s2 * heads) * hd + t]);
        for (int off = 16; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffffu, d, off);
        if (lane == 0) wd[warp] = d;
        __syncthreads();
        float tot = 0.0f;
        for (int w = 0; w < nwarps; ++w) tot += wd[w];
        __syncthreads();
        m = fmaxf(m, tot * scale);
        if (s2 == 0) s0 = tot * scale;
        if (tot * scale > s0b) { s0b = tot * scale; am = s2; }
        if (s2 == 134) { if (lane == 0) dbg_out[6] = wd[0]; if (lane == 0) dbg_out[7] = wd[1]; }
    }
    float denom = 0.0f;
    for (int s2 = 0; s2 < S; ++s2) {
        float d = qv * __bfloat162float(k[((size_t)s2 * heads) * hd + t]);
        for (int off = 16; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffffu, d, off);
        if (lane == 0) wd[warp] = d;
        __syncthreads();
        float tot = 0.0f;
        for (int w = 0; w < nwarps; ++w) tot += wd[w];
        __syncthreads();
        denom += __expf(tot * scale - m);
    }
    if (t == 0) { dbg_out[0] = m; dbg_out[1] = denom; dbg_out[2] = s0; dbg_out[3] = qv; dbg_out[4] = s0b; dbg_out[5] = (float)am; }
}

extern "C" __global__ void vv_bias(__nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ bias,
                                   int rows, int n, int stride) {
    const int r = blockIdx.x;
    for (int c = threadIdx.x; c < n; c += blockDim.x)
        y[(size_t)r * stride + c] = __float2bfloat16(__bfloat162float(y[(size_t)r * stride + c]) + __bfloat162float(bias[c]));
}

// rope: x[r, h, d] rotated with cos/sin[r, d] (precomputed host-side).
// grid(r * heads), block(head_dim).
extern "C" __global__ void vv_rope(__nv_bfloat16* __restrict__ x, const float* __restrict__ cos,
                                   const float* __restrict__ sin, int rows, int heads, int hd, int stride) {
    const int r = blockIdx.x / heads, h = blockIdx.x % heads;
    const int half = hd / 2, t = threadIdx.x;
    if (t >= hd) return;
    const float v = __bfloat162float(x[(size_t)r * stride + h * hd + t]);
    const float pair = (t < half) ? __bfloat162float(x[(size_t)r * stride + h * hd + t + half]) : __bfloat162float(x[(size_t)r * stride + h * hd + t - half]);
    const float rot = (t < half) ? -pair : pair;
    x[(size_t)r * stride + h * hd + t] = __float2bfloat16(bf(v * cos[r * hd + t] + rot * sin[r * hd + t]));
}

// naive attention: out[r, h, :] = softmax(q.k/8).v; q, k, v, out [S, H, D]
// contiguous. One block per (row, head); K/V rows streamed, scores in fp32
// registers via two passes (max+exp sum, then weighted V).
// naive attention: out[r, h, :] = softmax(q.k*scale).v; q, k, v, out [S, H, D]
// contiguous, one block per (row, head). The dot crosses warps via shared.
extern "C" __global__ void vv_attn(const __nv_bfloat16* __restrict__ qkv, int qoff, int koff, int voff,
                                   int rowstride, __nv_bfloat16* __restrict__ o,
                                   int S, int heads, int hd, float scale) {
    const int r = blockIdx.x / heads, h = blockIdx.x % heads;
    const int t = threadIdx.x;
    if (t >= hd) return;
    extern __shared__ float sh[]; // [warps] dots + [hd] acc
    const int nwarps = (blockDim.x + 31) / 32;
    float* wd = sh;
    float* acc = sh + nwarps;
    const int lane = t & 31, warp = t >> 5;
    const float qv = __bfloat162float(qkv[(size_t)r * rowstride + qoff + h * hd + t]);
    acc[t] = 0.0f;
    // pass 1: the row max
    float m = -INFINITY;
    for (int s2 = 0; s2 < S; ++s2) {
        float d = qv * __bfloat162float(qkv[(size_t)s2 * rowstride + koff + h * hd + t]);
        for (int off = 16; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffffu, d, off);
        if (lane == 0) wd[warp] = d;
        __syncthreads();
        float tot = 0.0f;
        for (int w = 0; w < nwarps; ++w) tot += wd[w];
        __syncthreads();
        m = fmaxf(m, tot * scale);
    }
    // pass 2: exp-sum and the weighted V accumulation
    float denom = 0.0f;
    for (int s2 = 0; s2 < S; ++s2) {
        float d = qv * __bfloat162float(qkv[(size_t)s2 * rowstride + koff + h * hd + t]);
        for (int off = 16; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffffu, d, off);
        if (lane == 0) wd[warp] = d;
        __syncthreads();
        float tot = 0.0f;
        for (int w = 0; w < nwarps; ++w) tot += wd[w];
        __syncthreads();
        const float p = __expf(tot * scale - m);
        denom += p;
        acc[t] += p * __bfloat162float(qkv[(size_t)s2 * rowstride + voff + h * hd + t]);
    }
    o[((size_t)r * heads + h) * hd + t] = __float2bfloat16(acc[t] / denom);
}

// gelu (the merger's mid activation), exact form tanh-free: 0.5x(1+erf(x/sqrt2))
extern "C" __global__ void vv_gelu(__nv_bfloat16* __restrict__ x, long total) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)));
}

// layernorm rows with weight+bias (the merger's post_projection_norm).
extern "C" __global__ void vv_layernorm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                        const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ y,
                                        int rows, int n, int x_stride, int y_stride, float eps) {
    const int r = blockIdx.x, t = threadIdx.x;
    float s = 0.0f, s2 = 0.0f;
    for (int c = t; c < n; c += blockDim.x) {
        const float v = __bfloat162float(x[(size_t)r * x_stride + c]);
        s += v; s2 = fmaf(v, v, s2);
    }
    __shared__ float red[32];
    const int lane = t & 31, warp = t >> 5;
    for (int off = 16; off > 0; off >>= 1) { s += __shfl_down_sync(0xffffffffu, s, off); s2 += __shfl_down_sync(0xffffffffu, s2, off); }
    if (lane == 0) { red[warp] = s; red[16 + warp] = s2; }
    __syncthreads();
    __shared__ float mu, iv;
    if (t == 0) {
        float as = 0.0f, as2 = 0.0f;
        const int warps = (blockDim.x + 31) / 32;
        for (int w_ = 0; w_ < warps; ++w_) { as += red[w_]; as2 += red[16 + w_]; }
        mu = as / n;
        iv = rsqrtf(as2 / n - mu * mu + eps);
    }
    __syncthreads();
    for (int c = t; c < n; c += blockDim.x)
        y[(size_t)r * y_stride + c] = __float2bfloat16(bf((__bfloat162float(x[(size_t)r * x_stride + c]) - mu) * iv * __bfloat162float(w[c]) + __bfloat162float(b[c])));
}


// rms (per-head rows of width hd) + 2-D rope, fused, IN PLACE on the q and k
// slices of the qkv buffer [S, 3H, hd]: qoff/koff in elements. The v slice is
// untouched. grid(S * heads), block(hd); one block normalizes+rotates one
// head-row exactly as the reference (rms over hd, then the half rotation).
extern "C" __global__ void vv_qk_norm_rope(__nv_bfloat16* __restrict__ qkv, int qoff, int koff,
                                           const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
                                           const float* __restrict__ cos, const float* __restrict__ sin,
                                           int rows, int heads, int hd, int stride, float eps) {
    const int r = blockIdx.x / heads, h = blockIdx.x % heads;
    const int t = threadIdx.x;
    if (t >= hd) return;
    const size_t qb = (size_t)r * stride + qoff + h * hd;
    const size_t kb = (size_t)r * stride + koff + h * hd;
    // per-row sums (one block = one row of hd)
    extern __shared__ float sh[];
    float sq = 0.0f, sk = 0.0f;
    {
        const float v = __bfloat162float(qkv[qb + t]);
        sq = v * v;
    }
    {
        const float v = __bfloat162float(qkv[kb + t]);
        sk = v * v;
    }
    __shared__ float redq[32], redk[32];
    const int lane = t & 31, warp = t >> 5;
    for (int off = 16; off > 0; off >>= 1) { sq += __shfl_down_sync(0xffffffffu, sq, off); sk += __shfl_down_sync(0xffffffffu, sk, off); }
    if (lane == 0) { redq[warp] = sq; redk[warp] = sk; }
    __syncthreads();
    __shared__ float iq, ik;
    if (t == 0) {
        float aq = 0.0f, ak = 0.0f;
        const int warps = (blockDim.x + 31) / 32;
        for (int w = 0; w < warps; ++w) { aq += redq[w]; ak += redk[w]; }
        iq = rsqrtf(aq / (float)hd + eps);
        ik = rsqrtf(ak / (float)hd + eps);
    }
    __syncthreads();
    const float qv = __bfloat162float(qkv[qb + t]) * iq * __bfloat162float(qw[t]);
    const float kv = __bfloat162float(qkv[kb + t]) * ik * __bfloat162float(kw[t]);
    const int half = hd / 2;
    const float qpair = (t < half) ? __bfloat162float(qkv[qb + t + half]) * iq * __bfloat162float(qw[t + half]) : __bfloat162float(qkv[qb + t - half]) * iq * __bfloat162float(qw[t - half]);
    const float kpair = (t < half) ? __bfloat162float(qkv[kb + t + half]) * ik * __bfloat162float(kw[t + half]) : __bfloat162float(qkv[kb + t - half]) * ik * __bfloat162float(kw[t - half]);
    const float qrot = (t < half) ? -qpair : qpair;
    const float krot = (t < half) ? -kpair : kpair;
    qkv[qb + t] = __float2bfloat16(qv * cos[r * hd + t] + qrot * sin[r * hd + t]);
    qkv[kb + t] = __float2bfloat16(kv * cos[r * hd + t] + krot * sin[r * hd + t]);
}

// dst += src elementwise (bf16, total elements). grid/256 blocks.
extern "C" __global__ void vv_add(__nv_bfloat16* __restrict__ dst, const __nv_bfloat16* __restrict__ src, long total) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    dst[i] = __float2bfloat16(__bfloat162float(dst[i]) + __bfloat162float(src[i]));
}

// the 2x2 merge gather: x [S, D] patch rows -> y [S/4, 4D] merge-block rows,
// y[g, j*D + c] = x[(2gy+dy)*gw + (2gx+dx), c] with j = dy*2+dx. The
// downsample conv2d is then an exact matmul over y. grid(S/4), block(D).
extern "C" __global__ void vv_gather_merge(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ y,
                                           int gh, int gw, int D) {
    // rows are MERGE-BLOCK-MAJOR: group g is the CONSECUTIVE rows 4g..4g+3
    const int g = blockIdx.x;
    const int c = threadIdx.x;
    if (c >= D) return;
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        y[(size_t)g * (4 * D) + (size_t)j * D + c] = x[((size_t)4 * g + j) * D + c];
}

}  // namespace tf_glm_vision
