// TF_GLM_SPARSE_FAST=2 (patch 0199): a prompt chunk's DSA sparse attention (latent._sparse_onepass, 16 heads, 32-key
// tiles, one latent slice, the 512-wide latent) restructured with warp roles, every output with _sparse_onepass' bits.
//
// _sparse_onepass runs a row on eight warps that take turns: all of them gather a tile, compute its 16 x 32 scores
// (a product only four warps' worth wide: warps 4-7 recompute warps 0-3's scores), its softmax and its PV product,
// then the next tile. Here a CTA's eight warps split the work of a row:
//   - four score warps hold the row's query in registers (its A fragments for all 32 k16 steps) and each computes one
//     8-key block of every tile's scores (an m16n8k16 chain over the 512-wide latent), applies the fp8 row scale,
//     the query scale and the mask past the row's count, and hands the 16 x 32 tile to the PV warps (shared memory);
//   - four PV warps each own 128 output columns: the tile's softmax (every PV warp computes the whole tile's, from
//     the scores), the rescale of their output columns and the PV product over those columns, and they gather the
//     tiles ahead (each its 128-column slice of the 32 key rows; fp8 codes widened to the bf16 values _sparse_onepass
//     widens them to, by integer ops and one exact multiply: e4m3x4_bf16) into one of two tile buffers.
// So the scores of tile p + 1 are computed while the PV warps run tile p, nothing is computed twice, and the query is
// loaded from memory once a row. mbarriers order the two roles (tile full, scores full, scores consumed).
//
// Bits: every operation of _sparse_onepass' per-tile code is done on the same values in the same order, with the
// same PTX instructions (read from its sm_120 PTX, Triton 3.7):
//   - a score is one m16n8k16 bf16 chain over k16 steps 0 .. 31 in order from a zero accumulator, operands in the
//     natural fragment order (Triton's k width 2), then fp8: x * row scale, then x * scale, then -inf past the count;
//   - the tile's row maximum: max.f32 of each thread's column pair (2t, 2t + 1), then within the 8-column block
//     (max(max(a0, a2), max(a1, a3))), then across the four blocks (max(max(B0, B2), max(B1, B3)));
//   - next maximum, the rescale factor (sub, mul by log2 e, ex2.approx, the -inf and inactive selects), the
//     probabilities (sub, mul by log2 e, ex2.approx, the inactive and mask selects), fp8: probability * row scale;
//     bf16 by cvt.rn.bf16x2.f32;
//   - the output's rescale (mul.f32 in place) and the PV chain per output: the rescaled output as the accumulator,
//     keys 0-15 then keys 16-31;
//   - the row sum: the column pair's add, then within the block ((a0 + a2) + (a1 + a3)), then across the blocks
//     ((B0 + B2) + (B1 + B3)); l = fma(l, alpha, sum);
//   - the output: div.full.f32 by l, cvt.rn.bf16x2.f32.
// Float operations are written as inline PTX with explicit .rn (no contraction can merge them); the selects and
// maxima keep the reference's operand order. Rows past the count are zeros, as the reference's masked loads give.
//
// Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
// the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int H = 16;                       // heads a row (a rank's DSA heads at TP4)
constexpr int LW = 512;                     // latent width
constexpr int KT = 32;                      // keys a tile
constexpr int QK_WARPS = 4;                 // score warps: one 8-key block of a tile each
constexpr int PV_WARPS = 4;                 // PV warps: 128 output columns each
constexpr int THREADS = (QK_WARPS + PV_WARPS) * 32;
constexpr int ROWB = LW * 2;                // bytes of a bf16 tile row
constexpr int TILEB = KT * ROWB;            // bytes of a bf16 tile (32 KB)
constexpr int SROW = KT + 8;                // floats a score row in shared memory (conflict-free pairs)
constexpr int KSLOTS = 4;                   // fp8 row scales: a ring of tiles (PV warps can be a tile apart)
constexpr uint32_t LOG2E = 0x3FB8AA3Bu;     // the constant Triton multiplies by before ex2.approx

struct __align__(128) Smem {
    unsigned char tile[2][TILEB];           // two tiles of bf16 key rows; 16-byte chunk c of row r at c ^ (r & 7)
    float s[2][H * SROW];                   // two tiles of scores
    float ks[KSLOTS][KT];                   // fp8 row scales of the tiles in flight
    unsigned long long full[2];             // a tile buffer holds its tile (the PV warps' 128 threads arrive)
    unsigned long long sfull[2];            // a score buffer holds its tile's scores (the score warps' 128 threads)
    unsigned long long sempty[2];           // a score buffer was read (the PV warps' 128 threads)
};

// -- PTX wrappers -------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t saddr(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.shared::cta.b64 st, [%0];\n}" ::"r"(bar) : "memory");
}

__device__ __forceinline__ bool mbar_try(uint32_t bar, uint32_t parity) {
    uint32_t ok;
    asm volatile("{\n .reg .pred P;\n mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n selp.u32 %0, 1, 0, P;\n}"
                 : "=r"(ok) : "r"(bar), "r"(parity) : "memory");
    return ok != 0;
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
    while (!mbar_try(bar, parity)) {
    }
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr) : "memory");
}

__device__ __forceinline__ void mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
                 "{%0, %1, %2, %3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float fmul(float a, float b) {
    float d;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float fsub(float a, float b) {
    float d;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float fadd(float a, float b) {
    float d;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float ffma(float a, float b, float c) {
    float d;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
    return d;
}

__device__ __forceinline__ float fmax_(float a, float b) {        // max.f32, operands in this order
    float d;
    asm("max.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ float ex2(float a) {
    float d;
    asm("ex2.approx.f32 %0, %1;" : "=f"(d) : "f"(a));
    return d;
}

__device__ __forceinline__ float fdiv(float a, float b) {
    float d;
    asm("div.full.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
    return d;
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {     // cvt.rn.bf16x2.f32 d, hi, lo
    uint32_t d;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
    return d;
}

__device__ __forceinline__ float shfl_xor(float v, int m) {
    float d;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 31, -1;" : "=f"(d) : "f"(v), "r"(m));
    return d;
}

// Four e4m3 codes (byte k of w) -> two bf16x2 words ((c0, c1), (c2, c3)), with the bits _sparse_onepass' conversion
// gives (cvt.rn.f16x2.e4m3x2 then cvt.bf16.f16: both exact) for every code a cache holds (patch 0291).
// The reference's conversion instructions run on the GPU's conversion pipe (F2FP / F2F, about 16 results a clock an
// SM), which a tile's 16,384 codes hold for ~1,000+ cycles; here integer ops place each code's bits in a bf16 whose
// value is the code's times 2^-120 (sign to bit 15, exponent field e, mantissa bits m << 4: an e4m3 subnormal lands on
// a bf16 subnormal), and one packed bf16 multiply by 2^120 scales it back: a power of two, no rounding, the code's
// exact value. NaN codes (0x7F, 0xFF) differ, and a cache never holds them: kv8 writes codes within +-256 (0x78).
__device__ __forceinline__ void e4m3x4_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
    const uint32_t L = (w << 4) & 0xF0F0F0F0u;                         // byte k: c_k's e0 m2 m1 m0 0 0 0 0
    const uint32_t H = ((w >> 4) & 0x07070707u) | (w & 0x80808080u);   // byte k: c_k's s 0 0 0 0 e3 e2 e1
    const uint32_t p0 = __byte_perm(L, H, 0x5140);                     // (c0, c1) as bf16x2 bits, x 2^-120
    const uint32_t p1 = __byte_perm(L, H, 0x7362);                     // (c2, c3)
    // x 2^120 (bf16 0x7B80); the -0 addend keeps a product's zero sign
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(lo) : "r"(p0), "r"(0x7B807B80u), "r"(0x80008000u));
    asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(hi) : "r"(p1), "r"(0x7B807B80u), "r"(0x80008000u));
}

__device__ __forceinline__ uint4 ldg16(const void* p) {
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ void sts16(uint32_t addr, uint4 v) {
    asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" ::"r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ uint32_t chunk_addr(uint32_t tile, int row, int chunk) {     // the swizzled 16 bytes
    return tile + row * ROWB + ((chunk ^ (row & 7)) << 4);
}

// -- the score warps --------------------------------------------------------------------------------------------
template <bool FP8>
__device__ __forceinline__ void score_warp(Smem& sm, const __nv_bfloat16* __restrict__ q_row, int n, int T, int w,
                                           int lane, float scale) {
    const int g = lane >> 2, t = lane & 3;
    // the row's 16 x 512 query as A fragments of all 32 k16 steps: a0 = q[g][16k + 2t, +1], a1 = q[g + 8][..],
    // a2 = q[g][16k + 8 + 2t, +1], a3 = q[g + 8][16k + 8 + 2t, +1]
    const uint32_t* q32 = reinterpret_cast<const uint32_t*>(q_row);
    uint32_t qf[32][4];
#pragma unroll
    for (int k = 0; k < 32; ++k) {
        qf[k][0] = __ldg(q32 + g * (LW / 2) + k * 8 + t);
        qf[k][1] = __ldg(q32 + (g + 8) * (LW / 2) + k * 8 + t);
        qf[k][2] = __ldg(q32 + g * (LW / 2) + k * 8 + 4 + t);
        qf[k][3] = __ldg(q32 + (g + 8) * (LW / 2) + k * 8 + 4 + t);
    }
    const uint32_t tile0 = saddr(&sm.tile[0][0]);
    const int row = 8 * w + (lane & 7);              // the key row this lane addresses for ldmatrix
    const int mat = lane >> 3;                       // the 8 x 8 matrix it addresses
    for (int p = 0; p < T; ++p) {
        const int b = p & 1;
        mbar_wait(saddr(&sm.full[b]), (p >> 1) & 1);
        const uint32_t tb = tile0 + b * TILEB;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int kp = 0; kp < 16; ++kp) {           // k16 steps 2 kp, 2 kp + 1: 16-byte chunks 4 kp .. 4 kp + 3
            uint32_t bf[4];
            ldsm_x4(bf, chunk_addr(tb, row, 4 * kp + mat));
            mma(acc, qf[2 * kp], bf[0], bf[1]);
            mma(acc, qf[2 * kp + 1], bf[2], bf[3]);
        }
        const int key = p * KT + 8 * w + 2 * t;      // acc: (g, key), (g, key + 1), (g + 8, key), (g + 8, key + 1)
        float v0 = acc[0], v1 = acc[1], v2 = acc[2], v3 = acc[3];
        if (FP8) {
            const float k0 = sm.ks[p & (KSLOTS - 1)][8 * w + 2 * t];
            const float k1 = sm.ks[p & (KSLOTS - 1)][8 * w + 2 * t + 1];
            v0 = fmul(v0, k0);
            v1 = fmul(v1, k1);
            v2 = fmul(v2, k0);
            v3 = fmul(v3, k1);
        }
        v0 = fmul(v0, scale);
        v1 = fmul(v1, scale);
        v2 = fmul(v2, scale);
        v3 = fmul(v3, scale);
        const float ninf = __int_as_float(0xFF800000);
        if (!(key < n)) v0 = ninf, v2 = ninf;
        if (!(key + 1 < n)) v1 = ninf, v3 = ninf;
        mbar_wait(saddr(&sm.sempty[b]), ((p >> 1) & 1) ^ 1);
        float* s = sm.s[b];
        *reinterpret_cast<float2*>(&s[g * SROW + 8 * w + 2 * t]) = make_float2(v0, v1);
        *reinterpret_cast<float2*>(&s[(g + 8) * SROW + 8 * w + 2 * t]) = make_float2(v2, v3);
        mbar_arrive(saddr(&sm.sfull[b]));
    }
}

// -- the PV warps' gather: their 128-column slice of a tile's 32 key rows ------------------------------------------
// fp8: 8 chunks of 16 codes a lane (rows (lane >> 3) + 4 i, chunk lane & 7 of the slice's 8) and, lanes 0-7, the scale
// of row 8 j + lane; bf16: 16 chunks of 8 values a lane (rows (lane >> 4) + 2 i, chunk lane & 15 of 16). Rows past
// the count are zeros and their scales 1 (the reference's masked loads). The ids of a tile are loaded one step before
// its rows (slice_ids, then slice_load), so neither waits on the other's latency inside the loop.
template <bool FP8>
struct Slice {
    static constexpr int N = FP8 ? 8 : 16;          // 16-byte loads a lane
    static constexpr int STEP = FP8 ? 4 : 2;        // rows between a lane's loads
    uint4 v[N];
    int id[N];
    int sid;                                        // fp8, lanes 0-7: the id of row 8 j + lane
    float sc;                                       // ... and its scale
};

template <bool FP8>
__device__ __forceinline__ void slice_ids(Slice<FP8>& s, const int32_t* __restrict__ tok_row, int p, int n, int j,
                                          int lane) {
    const int r0 = FP8 ? (lane >> 3) : (lane >> 4);
#pragma unroll
    for (int i = 0; i < Slice<FP8>::N; ++i) {
        const int key = p * KT + r0 + Slice<FP8>::STEP * i;
        s.id[i] = key < n ? __ldg(tok_row + key) : -1;
    }
    if (FP8) {
        const int key = p * KT + 8 * j + (lane & 7);
        s.sid = (lane < 8 && key < n) ? __ldg(tok_row + key) : -1;
    }
}

template <bool FP8>
__device__ __forceinline__ void slice_load(Slice<FP8>& s, const unsigned char* __restrict__ cache, int RS, int j,
                                           int lane) {
    const int c = FP8 ? (lane & 7) : (lane & 15);
#pragma unroll
    for (int i = 0; i < Slice<FP8>::N; ++i) {
        if (s.id[i] >= 0) {
            const unsigned char* src = cache + static_cast<size_t>(s.id[i]) * RS + (FP8 ? 128 * j : 256 * j) + 16 * c;
            s.v[i] = ldg16(src);
        } else {
            s.v[i] = make_uint4(0u, 0u, 0u, 0u);
        }
    }
    if (FP8)
        s.sc = s.sid >= 0 ? __ldg(reinterpret_cast<const float*>(cache + static_cast<size_t>(s.sid) * RS + LW)) : 1.0f;
}

template <bool FP8>
__device__ __forceinline__ void slice_store(const Slice<FP8>& s, Smem& sm, uint32_t tb, int p, int j, int lane) {
#pragma unroll
    for (int i = 0; i < Slice<FP8>::N; ++i) {
        if (FP8) {
            const int row = (lane >> 3) + 4 * i, c = lane & 7;          // 16 codes -> two 16-byte bf16 chunks
            uint4 lo, hi;
            e4m3x4_bf16(s.v[i].x, lo.x, lo.y);
            e4m3x4_bf16(s.v[i].y, lo.z, lo.w);
            e4m3x4_bf16(s.v[i].z, hi.x, hi.y);
            e4m3x4_bf16(s.v[i].w, hi.z, hi.w);
            sts16(chunk_addr(tb, row, 16 * j + 2 * c), lo);
            sts16(chunk_addr(tb, row, 16 * j + 2 * c + 1), hi);
        } else {
            const int row = (lane >> 4) + 2 * i, c = lane & 15;
            sts16(chunk_addr(tb, row, 16 * j + c), s.v[i]);
        }
    }
    if (FP8 && lane < 8) sm.ks[p & (KSLOTS - 1)][8 * j + lane] = s.sc;
}

// -- the PV warps ------------------------------------------------------------------------------------------------
// The block maximum of a quad's column pairs: max(max(a0, a2), max(a1, a3)) in lane t = 0, as the reference has it.
// The other lanes may hold it with the operands of a max swapped, which can only change the sign of a zero maximum;
// no value downstream depends on that sign (it enters only s - m and m - m', whose exp is the same for +0 and -0,
// and the -inf test), so every lane keeps its own.
__device__ __forceinline__ float quad_max(float a) {
    const float x = fmax_(a, shfl_xor(a, 2));
    return fmax_(x, shfl_xor(x, 1));
}

// The block sum: (a0 + a2) + (a1 + a3) (addition is commutative: every lane of the quad holds the same bits).
__device__ __forceinline__ float quad_sum(float a) {
    const float x = fadd(a, shfl_xor(a, 2));
    return fadd(x, shfl_xor(x, 1));
}

template <bool FP8>
__device__ __forceinline__ void pv_warp(Smem& sm, const unsigned char* __restrict__ cache,
                                        const int32_t* __restrict__ tok_row, __nv_bfloat16* __restrict__ out_row,
                                        int n, int T, int j, int lane, int RS) {
    const int g = lane >> 2, t = lane & 3;
    const float ninf = __int_as_float(0xFF800000);
    const float log2e = __int_as_float(LOG2E);
    const uint32_t tile0 = saddr(&sm.tile[0][0]);
    // prologue: tiles 0 and 1 into their buffers, tile 2's slice in flight
    Slice<FP8> sl;
    for (int p = 0; p < 2 && p < T; ++p) {
        slice_ids<FP8>(sl, tok_row, p, n, j, lane);
        slice_load<FP8>(sl, cache, RS, j, lane);
        slice_store<FP8>(sl, sm, tile0 + p * TILEB, p, j, lane);
        mbar_arrive(saddr(&sm.full[p]));
    }
    if (2 < T) {
        slice_ids<FP8>(sl, tok_row, 2, n, j, lane);
        slice_load<FP8>(sl, cache, RS, j, lane);
    }
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = ninf, m1 = ninf, l0 = 0.0f, l1 = 0.0f;     // rows g and g + 8
    const int mat = lane >> 3;
    for (int p = 0; p < T; ++p) {
        const int b = p & 1;
        if (p + 3 < T) slice_ids<FP8>(sl, tok_row, p + 3, n, j, lane);      // tile p + 3's ids (the rows: below)
        mbar_wait(saddr(&sm.sfull[b]), (p >> 1) & 1);
        float sv[4][2], su[4][2];                    // scores of rows g and g + 8, columns 8 w + 2 t, + 1
        const float* s = sm.s[b];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const float2 x = *reinterpret_cast<const float2*>(&s[g * SROW + 8 * w + 2 * t]);
            const float2 y = *reinterpret_cast<const float2*>(&s[(g + 8) * SROW + 8 * w + 2 * t]);
            sv[w][0] = x.x, sv[w][1] = x.y, su[w][0] = y.x, su[w][1] = y.y;
        }
        mbar_arrive(saddr(&sm.sempty[b]));
        // the tile's row maxima, in the reference's order
        float bv[4], bu[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            bv[w] = quad_max(fmax_(sv[w][0], sv[w][1]));
            bu[w] = quad_max(fmax_(su[w][0], su[w][1]));
        }
        const float tm0 = fmax_(fmax_(bv[0], bv[2]), fmax_(bv[1], bv[3]));
        const float tm1 = fmax_(fmax_(bu[0], bu[2]), fmax_(bu[1], bu[3]));
        const bool act0 = tm0 != ninf, act1 = tm1 != ninf;
        const float mx0 = fmax_(m0, tm0), mx1 = fmax_(m1, tm1);
        const float nm0 = act0 ? mx0 : m0, nm1 = act1 ? mx1 : m1;
        float al0 = ex2(fmul(fsub(m0, nm0), log2e)), al1 = ex2(fmul(fsub(m1, nm1), log2e));
        al0 = m0 == ninf ? 0.0f : al0;
        al1 = m1 == ninf ? 0.0f : al1;
        al0 = act0 ? al0 : 1.0f;
        al1 = act1 ? al1 : 1.0f;
        // probabilities
        float pv[4][2], pu[4][2];
#pragma unroll
        for (int w = 0; w < 4; ++w)
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const bool ok = p * KT + 8 * w + 2 * t + c < n;
                float x = ex2(fmul(fsub(sv[w][c], nm0), log2e));
                float y = ex2(fmul(fsub(su[w][c], nm1), log2e));
                x = act0 ? x : 0.0f;
                y = act1 ? y : 0.0f;
                pv[w][c] = ok ? x : 0.0f;
                pu[w][c] = ok ? y : 0.0f;
            }
        // the output's rescale
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            o[i][0] = fmul(o[i][0], al0);
            o[i][1] = fmul(o[i][1], al0);
            o[i][2] = fmul(o[i][2], al1);
            o[i][3] = fmul(o[i][3], al1);
        }
        // the PV product's A fragments (fp8: probability * row scale), k16 steps 0 (keys 0-15) and 1 (16-31)
        uint32_t A[2][4];
#pragma unroll
        for (int st = 0; st < 2; ++st) {
            float x[2][2], y[2][2];
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int w = 2 * st + h;
                x[h][0] = pv[w][0], x[h][1] = pv[w][1], y[h][0] = pu[w][0], y[h][1] = pu[w][1];
                if (FP8) {
                    const float k0 = sm.ks[p & (KSLOTS - 1)][8 * w + 2 * t];
                    const float k1 = sm.ks[p & (KSLOTS - 1)][8 * w + 2 * t + 1];
                    x[h][0] = fmul(x[h][0], k0);
                    x[h][1] = fmul(x[h][1], k1);
                    y[h][0] = fmul(y[h][0], k0);
                    y[h][1] = fmul(y[h][1], k1);
                }
            }
            A[st][0] = pack_bf16(x[0][0], x[0][1]);
            A[st][1] = pack_bf16(y[0][0], y[0][1]);
            A[st][2] = pack_bf16(x[1][0], x[1][1]);
            A[st][3] = pack_bf16(y[1][0], y[1][1]);
        }
        // the row sums: column pairs, the blocks, then across the blocks; l = fma(l, alpha, sum)
        float qv[4], qu[4];
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            qv[w] = quad_sum(fadd(pv[w][0], pv[w][1]));
            qu[w] = quad_sum(fadd(pu[w][0], pu[w][1]));
        }
        l0 = ffma(l0, al0, fadd(fadd(qv[0], qv[2]), fadd(qv[1], qv[3])));
        l1 = ffma(l1, al1, fadd(fadd(qu[0], qu[2]), fadd(qu[1], qu[3])));
        m0 = nm0;
        m1 = nm1;
        // PV over this warp's 128 columns: 8 pairs of n8 blocks, keys 0-15 then 16-31 for each
        const uint32_t tb = tile0 + b * TILEB;
#pragma unroll
        for (int bp = 0; bp < 8; ++bp)
#pragma unroll
            for (int st = 0; st < 2; ++st) {
                uint32_t bf[4];
                const int krow = 16 * st + 8 * (mat & 1) + (lane & 7);
                ldsm_x4_t(bf, chunk_addr(tb, krow, 16 * j + 2 * bp + (mat >> 1)));
                mma(o[2 * bp], A[st], bf[0], bf[1]);
                mma(o[2 * bp + 1], A[st], bf[2], bf[3]);
            }
        __syncwarp();
        // refill: tile p + 2 into this buffer (this warp's slice: only this warp reads it until the buffer is full),
        // then tile p + 3's rows into registers (consumed at the next step's refill)
        if (p + 2 < T) {
            slice_store<FP8>(sl, sm, tb, p + 2, j, lane);
            mbar_arrive(saddr(&sm.full[b]));
        }
        if (p + 3 < T) slice_load<FP8>(sl, cache, RS, j, lane);
    }
    // the output: o / l, bf16
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int col = 128 * j + 8 * i + 2 * t;
        *reinterpret_cast<uint32_t*>(out_row + g * LW + col) = pack_bf16(fdiv(o[i][0], l0), fdiv(o[i][1], l0));
        *reinterpret_cast<uint32_t*>(out_row + (g + 8) * LW + col) = pack_bf16(fdiv(o[i][2], l1), fdiv(o[i][3], l1));
    }
}

template <bool FP8>
__global__ void __launch_bounds__(THREADS, 1) sparse_ws_kernel(const __nv_bfloat16* __restrict__ qa,
                                                                const unsigned char* __restrict__ cache,
                                                                const int32_t* __restrict__ tokens,
                                                                const int32_t* __restrict__ counts,
                                                                __nv_bfloat16* __restrict__ out, int W, int RS,
                                                                float scale) {
    extern __shared__ __align__(128) unsigned char smem_raw[];
    Smem& sm = *reinterpret_cast<Smem*>(smem_raw);
    const int r = blockIdx.x;
    const int n = counts[r];
    if (n <= 0) return;                              // rows with no selected token are left alone
    const int T = (n + KT - 1) / KT;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (threadIdx.x == 0) {
        for (int b = 0; b < 2; ++b) {
            mbar_init(saddr(&sm.full[b]), PV_WARPS * 32);
            mbar_init(saddr(&sm.sfull[b]), QK_WARPS * 32);
            mbar_init(saddr(&sm.sempty[b]), PV_WARPS * 32);
        }
    }
    __syncthreads();
    const size_t base = static_cast<size_t>(r) * H * LW;
    if (warp < QK_WARPS)
        score_warp<FP8>(sm, qa + base, n, T, warp, lane, scale);
    else
        pv_warp<FP8>(sm, cache, tokens + static_cast<size_t>(r) * W, out + base, n, T, warp - QK_WARPS, lane, RS);
}

template <bool FP8>
void launch(const at::Tensor& qa, const at::Tensor& cache, const at::Tensor& tokens, const at::Tensor& counts,
            at::Tensor& out, int rs_bytes, double scale) {
    auto kernel = sparse_ws_kernel<FP8>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeof(Smem)));
        configured = true;
    }
    const int R = static_cast<int>(qa.size(0));
    kernel<<<R, THREADS, sizeof(Smem), at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(qa.data_ptr()), reinterpret_cast<const unsigned char*>(cache.data_ptr()),
        tokens.data_ptr<int32_t>(), counts.data_ptr<int32_t>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int>(tokens.size(1)), rs_bytes, static_cast<float>(scale));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

int sparse_ws_smem() { return static_cast<int>(sizeof(Smem)); }

void sparse_ws_cuda(const at::Tensor& qa, const at::Tensor& cache, const at::Tensor& tokens, const at::Tensor& counts,
                    at::Tensor& out, int64_t rs_bytes, bool fp8, double scale) {
    if (fp8)
        launch<true>(qa, cache, tokens, counts, out, static_cast<int>(rs_bytes), scale);
    else
        launch<false>(qa, cache, tokens, counts, out, static_cast<int>(rs_bytes), scale);
}
