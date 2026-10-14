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
extern "C" __global__ void vv_attn(const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
                                   const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
                                   int S, int heads, int hd, float scale) {
    const int r = blockIdx.x / heads, h = blockIdx.x % heads;
    const int t = threadIdx.x; // hd threads
    if (t >= hd) return;
    const float qv = __bfloat162float(q[((size_t)r * heads + h) * hd + t]);
    // pass 1: scores -> max and sumexp
    extern __shared__ float red[]; // [blockDim] reductions + [hd] accumulator
    float* acc = red + blockDim.x;
    for (int i = 0; i < hd; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    for (int s = 0; s < S; ++s) {
        const float kv = __bfloat162float(k[((size_t)s * heads + h) * hd + t]);
        const float sc = qv * kv * scale;
        // warp-reduce the dot across hd lanes (hd <= 1024 assumed one block)
        float d = sc;
        for (int off = blockDim.x / 2; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffffu, d, off);
        if (t == 0) red[0] = d;
        __syncthreads();
        const float sc_all = red[0];
        __syncthreads();
        m = fmaxf(m, sc_all);
    }
    // pass 2 with the true max
    float denom = 0.0f;
    for (int s = 0; s < S; ++s) {
        const float kv = __bfloat162float(k[((size_t)s * heads + h) * hd + t]);
        float d = qv * kv * scale;
        for (int off = blockDim.x / 2; off > 0; off >>= 1) d += __shfl_down_sync(0xffffffffu, d, off);
        if (t == 0) red[0] = d;
        __syncthreads();
        const float p = __expf(red[0] - m);
        __syncthreads();
        denom += p;
        const float vv = __bfloat162float(v[((size_t)s * heads + h) * hd + t]);
        acc[t] += p * vv; // each lane's dim accumulates its own v component
    }
    if (t < hd) o[((size_t)r * heads + h) * hd + t] = __float2bfloat16(bf(acc[t] / denom));
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

}  // namespace tf_glm_vision
