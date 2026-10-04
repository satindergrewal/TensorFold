// More launch configurations of the 4-bit prompt matmul (qmm_prefill.cu), every output with qmm_prefill's bits.
//
// qmm_prefill computes each output as one chain of m16n8k16 bf16 tensor-core steps over K in order (a 64-input group
// at a time, its four k16 steps in order), its A fragments by ldmatrix from row-major inputs, its B fragments the
// weights w = bf16(fma(q, s, b)) decoded from the packed nibbles by pair() and one fma.rn.bf16x2, from a zero
// accumulator; the epilogue rounds once (bf16) or not at all (fp32). Which CTA, warp or stage computes an output, and
// how many outputs a CTA holds, never enter that chain. The kernels here keep that per-output code verbatim (the same
// loads into the same shared-memory layouts, fragments, pair(), fma2() and mma() calls in the same order) and change
// only what does not touch it:
//   - the CTA tile and warp tile (16 to 256 rows, 64 to 256 columns; 4 or 8 warps; the 16- and 32-row tiles give a
//     prompt piece of a few hundred rows enough CTAs: patch 0197),
//   - the pipeline: stages in flight and 64-input groups a stage (one barrier a stage),
//   - a tail split: the columns that would make a last, mostly idle wave of CTAs are covered by a smaller tile in the
//     same launch (the grid's last CTAs), so the GPU's SMs finish together; an output is still computed by exactly one
//     CTA, with its own chain,
//   - fp32 outputs stored as pairs (the same values).
// tests/k5/test_q4p2_cpu.py compares the per-group code with qmm_prefill's as text and checks that every launch covers
// every output once; the engine compares outputs byte for byte on the GPU before it uses a variant (qmm_prefill2.py).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "qmm_frag.cuh"

namespace {

using namespace qmm_frag;

constexpr int GS = 64;                     // inputs a group (the scales' and biases' group)
constexpr int ROW = GS * 2;                // bytes of one input row's group
constexpr int CHUNKS = ROW / 16;
constexpr int TILE_BYTES = 64 * GS / 2;    // one stored 64-column tile's group block of packed weights

template <int BM_, int BN_, int WM_, int WN_>
struct Tile {
    static constexpr int BM = BM_, BN = BN_, WM = WM_, WN = WN_;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;         // m16 tiles a warp
    static constexpr int NT = BN / WN / 8;          // n8 tiles a warp
    static constexpr int X = BM * ROW;              // a group's inputs,
    static constexpr int W = BN * GS / 2;           // weights,
    static constexpr int S = BN * 2;                // scales and biases (bf16)
    static constexpr int GROUP = X + W + 2 * S;     // a group's bytes in a stage
    static_assert(BM % (WM * 16) == 0 && BN % (WN * 8) == 0 && BN % 64 == 0, "tile shape");
};

__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// One CTA tile at (m0, n0): qmm_prefill's prefill_kernel body with GPS groups a stage.
template <class T, int THREADS, int STAGES, int GPS, bool F32>
__device__ __forceinline__ void run_tile(const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w,
                                         const __nv_bfloat16* __restrict__ scales,
                                         const __nv_bfloat16* __restrict__ biases, void* __restrict__ out, int M,
                                         int N, int K, int npad, int ldx, int m0, int n0, bool pairs,
                                         unsigned char* buf) {
    static_assert(T::THREADS == THREADS, "a launch's tiles share its warps");
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / T::WN, wn = warp % T::WN;
    const int KG = K / GS;
    const int per = (KG + GPS - 1) / GPS;            // stages of GPS groups (the last may hold fewer)

    auto stage = [&](int s) { return buf + s * (GPS * T::GROUP); };
    // groups st * GPS .. of stage st into buffer s: qmm_prefill's load() for each, one after the other
    auto load = [&](int s, int st) {
#pragma unroll
        for (int u = 0; u < GPS; ++u) {
            const int g = st * GPS + u;
            if (g >= KG) break;
            unsigned char* p = stage(s) + u * T::GROUP;
            for (int c = tid; c < T::BM * CHUNKS; c += THREADS) {
                const int r = c / CHUNKS, ch = c % CHUNKS;
                const int row = min(m0 + r, M - 1);
                cp16z(p + r * ROW + swz<CHUNKS>(r, ch) * 16, x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                      m0 + r < M);
            }
            unsigned char* pw = p + T::X;
            for (int c = tid; c < T::W / 16; c += THREADS) {
                const int t = c / (TILE_BYTES / 16), off = c % (TILE_BYTES / 16);
                const size_t tile = static_cast<size_t>(n0 / 64 + t) * KG + g;
                if (n0 + t * 64 < npad)                   // a wide block's last tiles may pass the padded columns
                    cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + off * 16);
            }
            unsigned char* ps = pw + T::W;
            for (int c = tid; c < 2 * (T::S / 16); c += THREADS) {
                const int which = c / (T::S / 16), off = c % (T::S / 16);
                const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad + n0 + off * 8;
                if (n0 + off * 8 < npad) cp16(ps + which * T::S + off * 16, src);
            }
        }
    };

    float acc[T::MT][T::NT][4];
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < per) load(s, s);
        commit();
    }
    for (int it = 0; it < per; ++it) {
        wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < per) load(next % STAGES, next);
        commit();
#pragma unroll
        for (int u = 0; u < GPS; ++u) {
            if (it * GPS + u >= KG) break;
            // one group: qmm_prefill's compute, verbatim
            const unsigned char* p = stage(it % STAGES) + u * T::GROUP;
            const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
            const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
            uint32_t words[T::NT][GS / 32], sv[T::NT], bv[T::NT];
#pragma unroll
            for (int j = 0; j < T::NT; ++j) {
#pragma unroll
                for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[((wn * T::NT + j) * 32 + lane) * (GS / 32) + v];
                const int col = wn * (T::BN / T::WN) + j * 8 + (lane >> 2);         // this lane's B-fragment column
                sv[j] = ps[col] * 0x10001u;                                         // (s, s) and (b, b) as bf16 pairs
                bv[j] = ps[T::BN + col] * 0x10001u;
            }
#pragma unroll
            for (int kt = 0; kt < GS / 16; ++kt) {
                uint32_t a[T::MT][4];
#pragma unroll
                for (int i = 0; i < T::MT; ++i) {
                    const int r = wm * (T::BM / T::WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                    const int ch = kt * 2 + (lane >> 4);
                    ldmatrix4(a[i], p + r * ROW + swz<CHUNKS>(r, ch) * 16);
                }
#pragma unroll
                for (int j = 0; j < T::NT; ++j) {
                    const uint32_t b0 = fma2(pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]);
                    const uint32_t b1 = fma2(pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]);
#pragma unroll
                    for (int i = 0; i < T::MT; ++i) mma(acc[i][j], a[i], b0, b1);
                }
            }
        }
    }
    wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < T::MT; ++i)
#pragma unroll
        for (int j = 0; j < T::NT; ++j) {
            const int col = n0 + wn * (T::BN / T::WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (T::BM / T::WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                const float v0 = acc[i][j][2 * h], v1 = acc[i][j][2 * h + 1];
                if (F32) {
                    float* dst = reinterpret_cast<float*>(out) + static_cast<size_t>(row) * N + col;
                    if (pairs && col + 1 < N) {
                        *reinterpret_cast<float2*>(dst) = make_float2(v0, v1);        // the same two values
                    } else {
                        if (col < N) dst[0] = v0;
                        if (col + 1 < N) dst[1] = v1;
                    }
                } else {
                    __nv_bfloat16* dst = reinterpret_cast<__nv_bfloat16*>(out) + static_cast<size_t>(row) * N + col;
                    if (col + 1 < N && (N & 1) == 0)
                        *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                    else {
                        if (col < N) dst[0] = __float2bfloat16_rn(v0);
                        if (col + 1 < N) dst[1] = __float2bfloat16_rn(v1);
                    }
                }
            }
        }
}

// Block b's (first row, first column) over an M x N region of BM x BN tiles: qmm_frag's tile_of, host and device (the
// host side lists a launch's tiles for the coverage test).
__host__ __device__ __forceinline__ int2 tile_at(int b, int M, int N, int BM, int BN, int group) {
    const int rows_t = (M + BM - 1) / BM, cols_t = (N + BN - 1) / BN, band = group * cols_t;
    const int first = b / band * group, in_band = b % band;
    const int height = group < rows_t - first ? group : rows_t - first;
    return make_int2((first + in_band % height) * BM, in_band / height * BN);
}

// The launch: CTAs 0 .. nbig - 1 take Big tiles over columns [0, cbig), the rest Small tiles over [cbig, N).
template <class Big, class Small, int STAGES, int GPS, bool F32>
__global__ void __launch_bounds__(Big::THREADS) prefill2_kernel(
        const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, void* __restrict__ out, int M, int N, int K, int npad, int ldx,
        int nbig, int cbig, int gbig, int gsmall, int pairs) {
    extern __shared__ __align__(128) unsigned char buf[];
    const int b = blockIdx.x;
    if (b < nbig) {
        const int2 at = tile_at(b, M, cbig, Big::BM, Big::BN, gbig);
        run_tile<Big, Big::THREADS, STAGES, GPS, F32>(x, w, scales, biases, out, M, N, K, npad, ldx, at.x, at.y,
                                                      pairs != 0, buf);
    } else {
        const int2 at = tile_at(b - nbig, M, N - cbig, Small::BM, Small::BN, gsmall);
        run_tile<Small, Big::THREADS, STAGES, GPS, F32>(x, w, scales, biases, out, M, N, K, npad, ldx, at.x,
                                                        cbig + at.y, pairs != 0, buf);
    }
}

int l2_group(int rows_t, int BM, int K) {
    // a group's inputs stay near 12 MB of L2 while its blocks sweep the column tiles (qmm_prefill's rule)
    return std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K * 2))));
}

struct Plan {
    int nbig, cbig, nsmall, gbig, gsmall;
};

// The tail split (speed only): Big tiles over the most columns that fill whole waves of resident CTAs, Small tiles
// over the rest; with split false (or no whole wave, or nothing left over) the whole width at one tile size.
Plan plan(int bBM, int bBN, int sBM, int sBN, int M, int N, int K, int resident, bool split) {
    const int rows_b = (M + bBM - 1) / bBM, cols_b = (N + bBN - 1) / bBN;
    Plan p{rows_b * cols_b, N, 0, l2_group(rows_b, bBM, K), 1};
    if (!split || resident < 1) return p;
    const long long total = static_cast<long long>(rows_b) * cols_b;
    const long long waves = total / resident;
    const int cols = waves > 0 ? static_cast<int>(std::min<long long>(cols_b, waves * resident / rows_b)) : 0;
    if (cols >= cols_b) return p;                     // whole waves already
    const int cbig = cols * bBN;
    const int rows_s = (M + sBM - 1) / sBM, cols_s = (N - cbig + sBN - 1) / sBN;
    p.nbig = rows_b * cols;
    p.cbig = cbig;
    p.nsmall = rows_s * cols_s;
    p.gsmall = l2_group(rows_s, sBM, K);
    return p;
}

template <class Big, class Small, int STAGES, int GPS, bool F32>
void launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
            at::Tensor& out, int N, bool split) {
    constexpr int SMEM = STAGES * GPS * (Big::GROUP > Small::GROUP ? Big::GROUP : Small::GROUP);
    const int M = x.size(0), K = x.size(1);
    auto kernel = prefill2_kernel<Big, Small, STAGES, GPS, F32>;
    static int resident = -1;                          // CTAs an SM holds (this kernel, this device: one per process)
    if (resident < 0) {
        int per_sm = 0, sms = 0, dev = 0, optin = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        TORCH_CHECK(SMEM <= optin, "qmm_prefill2: this variant needs ", SMEM, " bytes of shared memory a block, the GPU "
                    "allows ", optin);
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, Big::THREADS, SMEM));
        resident = std::max(0, per_sm) * sms;
    }
    const Plan p = plan(Big::BM, Big::BN, Small::BM, Small::BN, M, N, K, resident, split);
    const bool pairs = reinterpret_cast<uintptr_t>(out.data_ptr()) % 8 == 0 && (N & 1) == 0;
    kernel<<<p.nbig + p.nsmall, Big::THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(w.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr()), out.data_ptr(), M, N, K,
        static_cast<int>(scales.size(1)), M == 1 ? K : static_cast<int>(x.stride(0)), p.nbig, p.cbig, p.gbig,
        p.gsmall, pairs ? 1 : 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class Big, class Small, int STAGES, int GPS>
void launch_any(const at::Tensor& x, const at::Tensor& w, const at::Tensor& s, const at::Tensor& b, at::Tensor& out,
                int N, bool f32, bool split) {
    if (f32) launch<Big, Small, STAGES, GPS, true>(x, w, s, b, out, N, split);
    else launch<Big, Small, STAGES, GPS, false>(x, w, s, b, out, N, split);
}

using T128 = Tile<128, 128, 2, 2>;     // qmm_prefill's tiles 5, 6, 9: 64 x 64 a warp, four warps
using T64 = Tile<64, 64, 2, 2>;        // 32 x 32 a warp
using T64x128 = Tile<64, 128, 2, 2>;   // 32 x 64 a warp
using T128x64 = Tile<128, 64, 2, 2>;   // 64 x 32 a warp
using T64x256 = Tile<64, 256, 1, 4>;   // 64 x 64 a warp, each warp its own 64 columns (no shared weight decode)
using W128x256 = Tile<128, 256, 2, 4>; // eight warps, 64 x 64 each
using W64x128 = Tile<64, 128, 2, 4>;   // eight warps, 32 x 32 each
using W256x128 = Tile<256, 128, 4, 2>; // eight warps, 64 x 64 each
using W128x64 = Tile<128, 64, 4, 2>;   // eight warps, 32 x 32 each
// patch 0197: tiles for calls of a few hundred rows (a prompt chunk's 256-row pieces, the index split's 512-row blocks),
// where 128-row tiles leave most SMs idle (a 1,024-column projection of 256 rows: 8 CTAs of 128 x 256 on 188 SMs)
using S16x64 = Tile<16, 64, 1, 4>;     // 16 x 16 a warp
using S16x128 = Tile<16, 128, 1, 4>;   // 16 x 32 a warp
using S32x64 = Tile<32, 64, 1, 4>;     // 32 x 16 a warp
using S32x128 = Tile<32, 128, 1, 4>;   // 32 x 32 a warp
using S32x256 = Tile<32, 256, 1, 4>;   // 32 x 64 a warp, each warp its own 64 columns
using S64x64 = Tile<64, 64, 1, 4>;     // 64 x 16 a warp (qmm_prefill's tile 3)

} // namespace

// The variants, one list for the launches and the table below (a launch table names variant v as tile 32 + v):
// id, Big tile, Small tile (the tail's; the Big one when there is no split), tail split, stages, groups a stage.
#define Q4P2_VARIANTS(X)                                                                                              \
    X(0, T128, T128, 0, 2, 1)         /* tile 9's launch, fp32 outputs stored as pairs */                            \
    X(1, T128, T64, 1, 2, 1)          /* + tail split, 64 x 64 tail tiles */                                         \
    X(2, T128, T128x64, 1, 2, 1)      /* + tail split, 128 x 64 tail tiles */                                        \
    X(3, T128, T128, 0, 2, 2)         /* two groups a stage (one barrier for 128 inputs) */                          \
    X(4, T128, T64, 1, 2, 2)                                                                                          \
    X(5, T128, T128, 0, 3, 1)         /* three stages */                                                             \
    X(6, T128, T64, 1, 3, 1)                                                                                          \
    X(7, T64x128, T64, 1, 3, 1)                                                                                       \
    X(8, T64, T64, 0, 4, 1)                                                                                           \
    X(9, T128x64, T64, 1, 3, 1)                                                                                       \
    X(10, T64x256, T64x256, 0, 2, 1)  /* 4 warps side by side: each decodes only its own weights */                  \
    X(11, T64x256, T64x256, 0, 3, 1)                                                                                  \
    X(12, W128x256, W64x128, 1, 2, 1) /* eight warps */                                                              \
    X(13, W128x256, W64x128, 1, 3, 1)                                                                                 \
    X(14, W256x128, W128x64, 1, 2, 1)                                                                                 \
    X(15, T64x256, T64, 1, 2, 1)      /* 4 warps side by side + 64 x 64 tail split */                                \
    X(16, S32x64, S32x64, 0, 4, 1)    /* patch 0197: 16- and 32-row tiles for calls of a few hundred rows */         \
    X(17, S32x128, S32x128, 0, 4, 1)                                                                                  \
    X(18, S16x64, S16x64, 0, 4, 1)                                                                                    \
    X(19, S16x128, S16x128, 0, 4, 1)                                                                                  \
    X(20, S32x256, S32x256, 0, 3, 1)                                                                                  \
    X(21, S32x64, S32x64, 0, 6, 1)    /* six stages: a long K on small tiles */                                      \
    X(22, S64x64, S32x64, 1, 4, 1)    /* 64 x 64 + 32 x 64 tail split */                                             \
    X(23, T64x128, S32x64, 1, 3, 1)   /* 64 x 128 + 32 x 64 tail split */

struct VariantInfo {
    int bBM, bBN, sBM, sBN, split, stages, gps, threads;
};
#define Q4P2_INFO(id, B, S, split, stages, gps) {B::BM, B::BN, S::BM, S::BN, split, stages, gps, B::THREADS},
constexpr VariantInfo VARIANTS[] = {Q4P2_VARIANTS(Q4P2_INFO)};
#undef Q4P2_INFO
constexpr int NVARIANTS = sizeof(VARIANTS) / sizeof(VARIANTS[0]);
int qmm_prefill2_variants() { return NVARIANTS; }

// The tiles a launch of ``variant`` covers for an (M, N, K) matmul with ``resident`` CTAs a GPU: [nbig, cbig, nsmall,
// then (first row, first column, rows, columns) a CTA in launch order] (the coverage test reads it; no GPU needed).
std::vector<int64_t> qmm_prefill2_tiles(int64_t variant, int64_t M, int64_t N, int64_t K, int64_t resident) {
    TORCH_CHECK(variant >= 0 && variant < NVARIANTS, "qmm_prefill2: no such variant");
    const VariantInfo& v = VARIANTS[variant];
    const Plan p = plan(v.bBM, v.bBN, v.sBM, v.sBN, M, N, K, static_cast<int>(resident), v.split != 0);
    std::vector<int64_t> out{p.nbig, p.cbig, p.nsmall};
    for (int b = 0; b < p.nbig + p.nsmall; ++b) {
        int2 at;
        int bm, bn;
        if (b < p.nbig) {
            at = tile_at(b, M, p.cbig, v.bBM, v.bBN, p.gbig);
            bm = v.bBM, bn = v.bBN;
        } else {
            at = tile_at(b - p.nbig, M, N - p.cbig, v.sBM, v.sBN, p.gsmall);
            at.y += p.cbig;
            bm = v.sBM, bn = v.sBN;
        }
        out.insert(out.end(), {at.x, at.y, bm, bn});
    }
    return out;
}

void qmm_prefill2_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                       at::Tensor& out, int N, bool f32, int variant) {
    switch (variant) {
#define Q4P2_CASE(id, B, S, split, stages, gps)                                                                       \
    case id: launch_any<B, S, stages, gps>(x, w, scales, biases, out, N, f32, split != 0); break;
        Q4P2_VARIANTS(Q4P2_CASE)
#undef Q4P2_CASE
        default: TORCH_CHECK(false, "qmm_prefill2: no such variant");
    }
}
