// GLM-5.3-Flash's hyper-connection mixing dots for decode rows (hc_pre's first half) with more blocks than
// glue._hc_partial, reproducing its bits: that Triton kernel (4 warps, a [32, 128] bf16 tile in the blocked layout
// sizePerThread [1, 8], threadsPerWarp [2, 16]) sums, for each of 8 steps of 128 inputs, a row's 8 products per
// thread in input order (products of bf16 values are exact in fp32, so fused or not), then across the 16 lanes of
// the row with butterflies 8, 4, 2, 1, and adds the steps to a zero accumulator in order; the sum of squares keeps
// one input per thread (4 warps x 32 lanes), adds the 8 steps' squares to zero in order, then butterflies 16 .. 1
// within each warp and (w0 + w2) + (w1 + w3) across the warps. PART [rows, 16 blocks, 32]: the 24 dots, then the sum.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

namespace {

constexpr int WIDE = 16384, NB = 16, KB = WIDE / NB, SUB = 128, STEPS = KB / SUB, ROWS = 24;

__device__ __forceinline__ void unpack8(const uint4 v, float (&f)[8]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        f[2 * i] = __uint_as_float(w[i] << 16);
        f[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
    }
}

// Block (K block, part, row group): warps 0..7 the dots of mixing rows 8 part .. 8 part + 7 (their weights read once
// for up to RPB rows), warp 8 of part 0 the squares.
constexpr int RPB = 4;
constexpr int D = WIDE / 4;              // a stream's width (4 streams, hc_mult 4)

template <bool SH>
__device__ __forceinline__ uint4 ld16(const __nv_bfloat16* p) {
    return SH ? *reinterpret_cast<const uint4*>(p) : __ldg(reinterpret_cast<const uint4*>(p));
}

// POST (the previous site's hc_post fused in front, decode rows): the block first writes its K block of the NEW
// streams (stream s = b / 4, columns (b % 4) KB ..) for its rows into shared memory from the old X, that site's
// gathered partials G [WORLD, rows, D] (rank k at k RS) and its POST / COMB, with the operations Triton 3.7 emits for
// glue._hc_post on sm_120 (its PTX, every element of a row, read symbolically for 1 to 4 ranks): branch =
// bf16(((g0 + g1) + g2) + g3); m = x1 c1, then fma(x0, c0, m), except where Triton packs the pair the other way round
// (3 ranks: column % 8 == 6 - 2 s; 4 ranks: stream 0's even columns): m = x0 c0, then fma(x1, c1, m); then
// m = fma(x2, c2, m); m = fma(x3, c3, m); v = fma(branch, ps, m); bf16(v). The blocks of part 0 also store their K
// block of the new streams to XN (each element once); X is not written here (other blocks still read the old
// streams): glue._hc_finish_copy copies XN to X row by row. fuse.py checks the bits on the GPU before use.
template <int WORLD>
__device__ __forceinline__ bool post_swapped(int st, int col) {
    return (WORLD == 3 && (col & 7) == 6 - 2 * st) || (WORLD == 4 && st == 0 && (col & 1) == 0);
}

template <bool POST, int WORLD>
__global__ void __launch_bounds__(288) hc_partial_kernel(const __nv_bfloat16* __restrict__ X,
                                                        const __nv_bfloat16* __restrict__ FN, float* __restrict__ part,
                                                        int rows, const float* __restrict__ G, long long RS,
                                                        const float* __restrict__ POSTW,
                                                        const float* __restrict__ COMB,
                                                        __nv_bfloat16* __restrict__ XN) {
    const int b = blockIdx.x, p = blockIdx.y, r0 = blockIdx.z * RPB;
    const int r1 = min(r0 + RPB, rows);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    __shared__ __align__(16) __nv_bfloat16 xn[POST ? RPB : 1][POST ? KB : 8];
    if constexpr (POST) {
        const int st = b / (D / KB), c0 = (b % (D / KB)) * KB;
        for (int i = threadIdx.x; i < (r1 - r0) * KB; i += blockDim.x) {
            const int rr = i / KB, c = i % KB, r = r0 + rr, col = c0 + c;
            const float* gr = G + (size_t)r * D + col;
            float acc = gr[0];
#pragma unroll
            for (int k = 1; k < WORLD; ++k) acc = __fadd_rn(acc, gr[(size_t)k * RS]);
            const float branch = __bfloat162float(__float2bfloat16_rn(acc));
            const __nv_bfloat16* xr = X + (size_t)r * WIDE + col;
            const float x0 = __bfloat162float(xr[0]), x1 = __bfloat162float(xr[D]);
            const float x2 = __bfloat162float(xr[2 * D]), x3 = __bfloat162float(xr[3 * D]);
            const float* cm = COMB + (size_t)r * 16 + st;                  // column st of comb: cm[4 j] = comb[j][st]
            float mixed;
            if (post_swapped<WORLD>(st, col)) {
                mixed = __fmul_rn(x0, cm[0]);
                mixed = __fmaf_rn(x1, cm[4], mixed);
            } else {
                mixed = __fmul_rn(x1, cm[4]);
                mixed = __fmaf_rn(x0, cm[0], mixed);
            }
            mixed = __fmaf_rn(x2, cm[8], mixed);
            mixed = __fmaf_rn(x3, cm[12], mixed);
            const __nv_bfloat16 v = __float2bfloat16_rn(__fmaf_rn(branch, POSTW[(size_t)r * 4 + st], mixed));
            xn[rr][c] = v;
            if (p == 0) XN[(size_t)r * WIDE + st * D + col] = v;
        }
        __syncthreads();
    }
    // a row's KB inputs of this K block: the new streams in shared memory (POST) or X
    auto xrow = [&](int r) -> const __nv_bfloat16* {
        if constexpr (POST) return &xn[r - r0][0];
        else return X + (size_t)r * WIDE + b * KB;
    };
    if (warp < 8) {
        const int m = p * 8 + warp;
        const int h = lane >> 4, l = lane & 15;
        const __nv_bfloat16* wr = FN + (size_t)m * WIDE + b * KB;
        float wf[4][8];
#pragma unroll
        for (int i = 0; i < 4; ++i)                    // steps 2 i + h, inputs 8 l .. 8 l + 7 of the step
            unpack8(__ldg(reinterpret_cast<const uint4*>(wr + (2 * i + h) * SUB + 8 * l)), wf[i]);
        for (int r = r0; r < r1; ++r) {
            const __nv_bfloat16* xr = xrow(r);
            uint4 xv[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) xv[i] = ld16<POST>(xr + (2 * i + h) * SUB + 8 * l);
            float s[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                float xf[8];
                unpack8(xv[i], xf);
                float a = __fmul_rn(wf[i][0], xf[0]);
#pragma unroll
                for (int j = 1; j < 8; ++j) a = __fmaf_rn(wf[i][j], xf[j], a);
#pragma unroll
                for (int o = 8; o; o >>= 1) a = __fadd_rn(a, __shfl_xor_sync(0xffffffffu, a, o));
                s[i] = a;
            }
            float acc = 0.f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc = __fadd_rn(acc, s[i]);                                      // step 2 i (half 0)
                acc = __fadd_rn(acc, __shfl_sync(0xffffffffu, s[i], 16));       // step 2 i + 1 (half 1)
            }
            if (lane == 0) part[((size_t)r * NB + b) * 32 + m] = acc;
        }
    } else if (p == 0) {
        for (int r = r0; r < r1; ++r) {
            const __nv_bfloat16* xr = xrow(r);
            float w[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {              // the Triton kernel's warp j: input 32 j + lane of each step
                float ss = 0.f;
#pragma unroll
                for (int t = 0; t < STEPS; ++t) {
                    const float x = __bfloat162float(xr[t * SUB + 32 * j + lane]);
                    ss = __fmaf_rn(x, x, ss);
                }
#pragma unroll
                for (int o = 16; o; o >>= 1) ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, o));
                w[j] = ss;
            }
            if (lane == 0) part[((size_t)r * NB + b) * 32 + ROWS] = __fadd_rn(__fadd_rn(w[0], w[2]), __fadd_rn(w[1], w[3]));
        }
    }
}

}  // namespace

void hc_partial_cuda(const at::Tensor& x, const at::Tensor& fn, at::Tensor& part, int64_t rows) {
    hc_partial_kernel<false, 1><<<dim3(NB, 3, (unsigned)((rows + RPB - 1) / RPB)), 288, 0,
                                  at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(fn.data_ptr()),
        part.data_ptr<float>(), (int)rows, nullptr, 0, nullptr, nullptr, nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// hc_partial of the rows hc_post(x, ., g, post, comb) would write, from the old x (not written here); those rows
// also go to xn.
void hc_post_partial_cuda(const at::Tensor& x, const at::Tensor& g, int64_t rs, const at::Tensor& post,
                          const at::Tensor& comb, const at::Tensor& fn, at::Tensor& part, int64_t rows, int64_t world,
                          at::Tensor& xn) {
    const dim3 grid(NB, 3, (unsigned)((rows + RPB - 1) / RPB));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    auto fp = reinterpret_cast<const __nv_bfloat16*>(fn.data_ptr());
#define GO(W_)                                                                                                     \
    hc_partial_kernel<true, W_><<<grid, 288, 0, stream>>>(xp, fp, part.data_ptr<float>(), (int)rows,               \
                                                          g.data_ptr<float>(), (long long)rs, post.data_ptr<float>(), \
                                                          comb.data_ptr<float>(),                                     \
                                                          reinterpret_cast<__nv_bfloat16*>(xn.data_ptr()))
    if (world == 1) GO(1);
    else if (world == 2) GO(2);
    else if (world == 3) GO(3);
    else if (world == 4) GO(4);
    else TORCH_CHECK(false, "hc_post_partial: 1 to 4 ranks");
#undef GO
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
