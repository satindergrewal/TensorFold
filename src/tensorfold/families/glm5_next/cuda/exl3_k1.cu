// K1: prompt chunks' EXL3 routed experts (GLM-5.3-Flash) with the prompt kernels' bits (exl3.cu), decoded once a
// block of up to 128 members and run as one persistent launch for gate/up and down.
//
// The same arithmetic as exl3.cu's prompt kernels (prompt_gateup_kernel / prompt_down_kernel), so every output keeps
// its bits: each output element is one fp32 chain of mma.m16n8k16.f32.f16.f16.f32 over the ascending k tiles from +0,
// with A = the decoded W^T tile (decode_tile_p and its fragment order, a0 = b0[0], a1 = b1[0], a2 = b0[1],
// a3 = b1[1]) and B = the member's rows of Xh (gate/up) or Xd (down) through ldmatrix; the gate/up and down
// epilogues are exl3.cu's formulas in its order (fwht128, the bf16 roundings, expf, the fp16 rounding of Xd, the
// fp32 Y). A row's outputs depend on its own values only, whatever else the chunk holds.
//
// What changes is data movement and scheduling:
// - a tile is (one expert's block of up to BN members, 16 m tiles): a trellis tile is decoded once for up to 128
//   members (the prompt kernels: once a pass of 64), straight into the A fragment of the one warp that owns it;
// - the inner loop is specialised on the block's 16-member groups (no branch a group and k tile), and the mma
//   instruction is not volatile, so loads, decode and mma interleave freely;
// - one persistent grid takes gate/up and down tiles from one list built on the GPU from the plan (no host sync):
//   blocks ranked by their expert's members, interleaved from both ends (the heaviest, the lightest, the next
//   heaviest, ...), an expert's blocks together; a block's down tiles come `lag` list positions after its gate/up
//   tiles and wait for them (a counter a block, release / acquire), so the projections overlap and neither ends in
//   its own tail. A down tile runs OB output blocks of 256 columns in one cp.async pipeline, which keeps loading
//   through each block's epilogue (the epilogue has its own shared memory).
//
// Shapes: D (model width) % 256 == 0, NI (the rank's expert width) % 128 == 0; trellis words int32
// [E, K/16, N/16, 32] (4-bit tiles); experts that share one gate/up suh (one rotated row a token, Xh [rows, D]).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <type_traits>

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// ---- exl3.cu's arithmetic, verbatim ---------------------------------------------------------------------------------

// Two values of the "mcg" codebook from two 16-bit states, as a half2 (first state in .x): exl3.cu's mcg2, with
// (x & 0x8FFF8FFF) ^ 0x3B603B60 as one lop3 whose mask comes in a register (M, a kernel argument: the compiler
// otherwise rematerialises the constant before every lop3). The same function of x, so the same bits.
__device__ __forceinline__ uint32_t lop_mask_xor(uint32_t x, uint32_t M) {
    uint32_t y;
    asm("lop3.b32 %0, %1, %2, 0x3B603B60, 0x6a;\n" : "=r"(y) : "r"(x), "r"(M));   // (x & M) ^ 0x3B603B60
    return y;
}
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1, uint32_t M) {
    const uint32_t x0 = lop_mask_xor(s0 * 0xCBAC1FEDu, M);
    const uint32_t x1 = lop_mask_xor(s1 * 0xCBAC1FEDu, M);
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// The prompt kernels' lane decode (exl3.cu decode_tile_p): this lane's eight values of a 4-bit tile as the B
// fragments of the tile's two n8 halves.
__device__ __forceinline__ void decode_tile_p(uint32_t w, int lane, uint32_t M, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    b0[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu, M);       // (s >> 8) & 0xffff, (s >> 4) & 0xffff
    b0[1] = mcg2(s & 0xffffu, w >> 16, M);
    b1[0] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421), M);   // (w >> 12) & 0xffff, (w >> 8) & 0xffff
    b1[1] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu, M);               // (w >> 4) & 0xffff, w & 0xffff
}

// d += a @ b (fp16 inputs, fp32 accumulators). Not volatile: a pure function of its registers, so the compiler may
// move it among the loads and the decode; its operands and its order on each accumulator are what the code says.
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
        "{%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Fast Walsh-Hadamard transform of 128 values held 4 per lane (lane L: values 4L..4L+3), exl3.cu's butterfly order.
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

// ---- K1q: 8-bit gate/up (TF_GLM_EXPERT_PROMPT_KERNEL=k1q*: quality-changing, deterministic, batch-invariant) ------
// The codebook's values lie in (-4, 4) (each fp16 half of the mcg hash has exponent field 12..15), so one fixed scale
// serves every weight: q = round(31.75 v), |q| <= 127. In fp16, fma(v, 31.75, 1536) is exact before its one rounding
// and lands in [1024, 2048), where the ulp is 1: the result is 1536 + q, whose low byte is q in two's complement.
// A row's int8 values (one fp32 scale a token) are stored with each 16-value group in the order the lane decode leaves
// the weights (logical k 2t, 2t+1, 2t+8, 2t+9 in slots 4t..4t+3), so two decoded k tiles are an m16n8k32 A fragment
// as they stand and ldmatrix gives the matching B fragments; int32 sums do not depend on the order of k.
constexpr float Q8_WEIGHT = 31.75f;

// Four codebook values (two half2: lo.x, lo.y, hi.x, hi.y) as four int8 round(31.75 v), in that order.
__device__ __forceinline__ uint32_t q8x4(uint32_t lo2, uint32_t hi2) {
    const half2 S = __halves2half2(__ushort_as_half(0x4FF0), __ushort_as_half(0x4FF0));   // 31.75
    const half2 B = __halves2half2(__ushort_as_half(0x6600), __ushort_as_half(0x6600));   // 1536
    const half2 a = __hfma2(*reinterpret_cast<const half2*>(&lo2), S, B);
    const half2 b = __hfma2(*reinterpret_cast<const half2*>(&hi2), S, B);
    return __byte_perm(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b), 0x6420);
}

// d += a @ b, int8 inputs, int32 sums (exact).
__device__ __forceinline__ void imma16832(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---- data movement --------------------------------------------------------------------------------------------------

__device__ __forceinline__ void cp_async16(unsigned smem, const void* gmem, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem), "l"(gmem), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

// A block's finished gate/up tiles: published with release semantics after every thread's Xd stores and a fence,
// read with acquire semantics before a down tile loads that Xd (through L2: cp.async.cg).
__device__ __forceinline__ void count_release_add(int* c, int v) {
    asm volatile("red.release.gpu.global.add.s32 [%0], %1;\n" ::"l"(c), "r"(v) : "memory");
}
// The wait is bounded: 30 s without the count means a broken list (a down tile before its own gate/up tiles), and
// the kernel traps (a CUDA error the engine raises) rather than hangs.
__device__ __forceinline__ void count_wait(const int* c, int target) {
    unsigned long long t0;
    asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t0));
    for (;;) {
        int v;
        asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(c) : "memory");
        if (v >= target) break;
        __nanosleep(200);
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;\n" : "=l"(t));
        if (t - t0 > 30000000000ull) __trap();
    }
}

// ---- configuration --------------------------------------------------------------------------------------------------

// NW warps a block (each owns MT = 16 / NW of the tile's 16 m tiles, for all of the block's members), up to NGM groups
// of 16 members (BN = 16 NGM), DW stages of trellis words and DX stages of member rows (32 k values a stage),
// epilogue passes of EPM members, CTAS blocks an SM. The first TW threads load the words (WPT 16-byte pieces each),
// the others the member rows (XPT pieces each): each side has its own cp.async group sequence, so the words run DW - 1
// stages ahead (DRAM) and the rows DX - 1 (L2), each waited for at its own depth.
template <int NW_, int NGM_, int DW_, int DX_, int EPM_, int CTAS_, int WPT_>
struct Cfg {
    static constexpr int NW = NW_, MT = 16 / NW_, NGM = NGM_, BN = 16 * NGM_, DW = DW_, DX = DX_, EPM = EPM_;
    static constexpr int CTAS = CTAS_, T = NW_ * 32;
    static constexpr int BK = 32, PKT = BK / 16, PXS = BK + 8;          // stage row stride +16 B: ldmatrix rows apart
    static constexpr int W_PIECES = 16 * PKT * 8;                       // 16-byte pieces of a stage's words
    static constexpr int WPT = WPT_, TW = W_PIECES / WPT_, TX = T - TW;
    static constexpr int XPT = BN * (BK / 8) / TX;                      // pieces of a stage's member rows a thread
    static constexpr int X_BYTES = BN * PXS * 2;                        // a stage's member rows
    static constexpr int PXSQ = BK + 16;                                // K1q: an int8 row's stride (bytes)
    static constexpr int XPTQ = BN * (BK / 16) / TX;                    // K1q: int8 row pieces a thread
    static_assert(BN * PXSQ <= X_BYTES && XPTQ * TX == BN * (BK / 16) && (XPTQ == 1 || XPTQ == 2), "K1q rows");
    static constexpr int W_BYTES = 16 * PKT * 32 * 4;                   // a stage's trellis words (16 m tiles)
    static constexpr int GU_ES = 128 + 4, DN_ES = 256 + 4;              // epilogue row strides (floats)
    static constexpr int EP_GU = 2 * EPM * GU_ES * 4, EP_DN = EPM * DN_ES * 4;
    static constexpr int EP_BYTES = EP_GU > EP_DN ? EP_GU : EP_DN;
    static constexpr int SMEM = DW * W_BYTES + DX * X_BYTES + EP_BYTES;
    static_assert(TW % 32 == 0 && TX % 32 == 0 && TW + TX == T, "whole warps on each side");
    static_assert(XPT * TX == BN * (BK / 8) && (XPT == 1 || XPT == 2 || XPT == 4), "row pieces split evenly");
    static_assert(WPT == 1 || WPT == 2, "word pieces a thread");
    static_assert(16 % NW == 0 && NW * MT == 16, "16 m tiles over the warps");
    static_assert(EPM % 8 == 0 && BN % EPM == 0, "epilogue passes of whole n8 tiles");
    static_assert(DW >= 2 && DX >= 2, "two stages at least");
    // sm_12x: 100 KB of shared memory an SM, 1 KB of it reserved a block, 99 KB at most a block (the static arrays,
    // up to 3 KB, included): CTAS blocks must fit side by side
    static_assert(CTAS * (SMEM + 3 * 1024 + 1024) <= 100 * 1024 && SMEM + 3 * 1024 <= 99 * 1024, "shared memory");
};

using CfgWide = Cfg<16, 8, 8, 3, 32, 1, 1>;   // "k1-w16": 16 warps, 1 m tile each, up to 128 members; 1 block an SM
using CfgPair = Cfg<8, 4, 3, 3, 16, 2, 2>;    // "k1-64": 8 warps, 2 m tiles each, up to 64 members; 2 blocks an SM
                                              // (3 word stages: two blocks' shared memory within an SM's 100 KB)
using CfgDeep = Cfg<8, 8, 8, 3, 32, 1, 2>;    // "k1": 8 warps, 2 m tiles each, up to 128 members (up to 255
                                              // registers a thread: no rematerialised constants); 1 block an SM

struct Params {
    const half* xh;              // [rows, D]: one rotated input row a token (gate/up)
    const uint32_t* tg;          // gate trellis [E, D/16, NI/16, 32]
    const uint32_t* tu;          // up
    const uint32_t* td;          // down [E, NI/16, D/16, 32]
    const half* svh_g;           // [E, NI]
    const half* svh_u;
    const half* suh_d;           // [E, NI]
    const half* svh_d;           // [E, D]
    half* xd;                    // [P, NI]: gate/up writes, down reads
    float* y;                    // [P, D]
    const int* members;          // [P]: the plan's pairs grouped by expert
    const int* blocks;           // [blocks, 3]: (expert, first member, members)
    const int* tiles;            // [tiles]: block << 8 | down << 7 | column block (gate/up) or output group (down)
    int* ctrl;                   // [0] next tile, [1] tiles, [2] blocks, [3] blocks the plan needs
    int* done;                   // [blocks]: finished gate/up tiles
    int D, NI, slots, OB;
    float limit;
    uint32_t mask;               // 0x8FFF8FFF (the mcg codebook's mask), in a register rather than an immediate
    const int8_t* xq;            // K1q: [rows, D] int8 rows (16-value groups in the decode's order)
    const float* xs;             // K1q: [rows] their scales
};

// ---- one tile -------------------------------------------------------------------------------------------------------

// This warp's MT m tiles x all of the block's members for one stage (PKT k tiles), exl3.cu's prompt mainloop body.
template <class C, int NG>
__device__ __forceinline__ void stage_mma(const unsigned char* xst, const unsigned char* wst,
                                          float (&acc)[C::MT][2 * NG][4], bool up_last, int warp, int lane,
                                          uint32_t M) {
    const half* xs = reinterpret_cast<const half*>(xst);
    const uint32_t* ws = reinterpret_cast<const uint32_t*>(wst);
    const int lrow = (lane & 7) + ((lane >> 4) & 1) * 8, lcol = ((lane >> 3) & 1) * 8;
#pragma unroll
    for (int kk = 0; kk < C::PKT; ++kk) {
        uint32_t a[C::MT][4];
#pragma unroll
        for (int i = 0; i < C::MT; ++i) {
            uint32_t b0[2], b1[2];
            decode_tile_p(ws[((warp * C::MT + i) * C::PKT + kk) * 32 + lane], lane, M, b0, b1);
            a[i][0] = b0[0]; a[i][1] = b1[0]; a[i][2] = b0[1]; a[i][3] = b1[1];
        }
#pragma unroll
        for (int j = 0; j < NG; ++j) {
            uint32_t r[4];
            ldsm_x4(r, xs + (j * 16 + lrow) * C::PXS + kk * 16 + lcol);
            const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
            for (int i = 0; i < C::MT; ++i) {
                mma16816(acc[i][2 * j], a[i], bl);
                if (j + 1 < NG || up_last) mma16816(acc[i][2 * j + 1], a[i], bh);   // members 8..15 of the group
            }
        }
    }
}

// K1q: this warp's MT m tiles x the block's members for one stage of 32 k values: two decoded k tiles an m tile become
// one int8 A fragment (q8x4), then a 16-member group at a time the int8 rows' ldmatrix fragments are B, one m16n8k32
// an n8 tile. The last group's upper n8 tile runs whether or not it has members (their rows are zero-filled and their
// sums never stored): no branch in the loop.
template <class C, int NG>
__device__ __forceinline__ void stage_mma_q(const unsigned char* xst, const unsigned char* wst,
                                            int (&acc)[C::MT][2 * NG][4], int warp, int lane, uint32_t M) {
    const uint32_t* ws = reinterpret_cast<const uint32_t*>(wst);
    const int lrow = (lane & 7) + ((lane >> 4) & 1) * 8, lcol = ((lane >> 3) & 1) * 16;
    uint32_t a[C::MT][4];
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            uint32_t b0[2], b1[2];
            decode_tile_p(ws[((warp * C::MT + i) * C::PKT + kk) * 32 + lane], lane, M, b0, b1);
            a[i][2 * kk] = q8x4(b0[0], b0[1]);          // row g:     k 2t, 2t+1, 2t+8, 2t+9 of k tile kk
            a[i][2 * kk + 1] = q8x4(b1[0], b1[1]);      // row g + 8: the same k
        }
#pragma unroll
    for (int j = 0; j < NG; ++j) {
        uint32_t r[4];
        ldsm_x4(r, xst + (j * 16 + lrow) * C::PXSQ + lcol);
        const uint32_t bl[2] = {r[0], r[1]}, bh[2] = {r[2], r[3]};
#pragma unroll
        for (int i = 0; i < C::MT; ++i) {
            imma16832(acc[i][2 * j], a[i], bl);
            imma16832(acc[i][2 * j + 1], a[i], bh);
        }
    }
}

// Gate/up epilogue of column block cb (exl3.cu's prompt_gateup_kernel epilogue, EPM members a pass): Xd rows.
template <class C, int NG, bool Q8, class Acc>
__device__ __forceinline__ void epilogue_gu(const Params& p, const Acc (&acc)[C::MT][2 * NG][4], int e, int cb,
                                            int cnt, const int* rows_sh, const float* dq_sh, float* E) {
    constexpr int MT = C::MT, NW = C::NW, EPM = C::EPM, ES = C::GU_ES, U = EPM / 8;
    constexpr int PASSES = (2 * NG + U - 1) / U;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int q0 = warp * MT, mat = q0 >> 3, col0 = (q0 & 7) * 16;   // a warp's m tiles are all gate or all up
    const int N = p.NI, n = cb * 128 + 4 * lane;
    float* Em = E + mat * EPM * ES;
#pragma unroll
    for (int ps = 0; ps < PASSES; ++ps) {
        if (ps * EPM >= cnt) break;
        __syncthreads();                                   // E's previous readers are done
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int jn = ps * U + u;
                if (jn < 2 * NG) {
                    const int tok = u * 8 + 2 * t, col = col0 + i * 16 + g;
                    if constexpr (Q8) {                  // K1q: int32 sums times the member's scale / 31.75
                        const float d0 = dq_sh[ps * EPM + tok], d1 = dq_sh[ps * EPM + tok + 1];
                        Em[tok * ES + col] = __int2float_rn(acc[i][jn][0]) * d0;
                        Em[(tok + 1) * ES + col] = __int2float_rn(acc[i][jn][1]) * d1;
                        Em[tok * ES + col + 8] = __int2float_rn(acc[i][jn][2]) * d0;
                        Em[(tok + 1) * ES + col + 8] = __int2float_rn(acc[i][jn][3]) * d1;
                    } else {
                        Em[tok * ES + col] = acc[i][jn][0];
                        Em[(tok + 1) * ES + col] = acc[i][jn][1];
                        Em[tok * ES + col + 8] = acc[i][jn][2];
                        Em[(tok + 1) * ES + col + 8] = acc[i][jn][3];
                    }
                }
            }
        __syncthreads();
        for (int m = warp; m < EPM && ps * EPM + m < cnt; m += NW) {
            const int pr = rows_sh[ps * EPM + m];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = E[m * ES + 4 * lane + j];
                uv[j] = E[(EPM + m) * ES + 4 * lane + j];
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(p.svh_g[(size_t)e * N + n + j])), p.limit);
                float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(p.svh_u[(size_t)e * N + n + j])),
                                       -p.limit),
                                 p.limit);
                float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                v[j] = act * __half2float(p.suh_d[(size_t)e * N + n + j]);
            }
            fwht128(v, lane);
            const half2 o01 = __halves2half2(__float2half_rn(v[0] * HAD_SCALE), __float2half_rn(v[1] * HAD_SCALE));
            const half2 o23 = __halves2half2(__float2half_rn(v[2] * HAD_SCALE), __float2half_rn(v[3] * HAD_SCALE));
            uint2 packed;
            packed.x = *reinterpret_cast<const uint32_t*>(&o01);
            packed.y = *reinterpret_cast<const uint32_t*>(&o23);
            *reinterpret_cast<uint2*>(p.xd + (size_t)pr * N + n) = packed;
        }
    }
}

// Down epilogue of output block ob (256 columns; exl3.cu's prompt_down_kernel epilogue, EPM members a pass): Y rows,
// stored streaming (they are read once, by the combine, after the call).
template <class C, int NG>
__device__ __forceinline__ void epilogue_dn(const Params& p, const float (&acc)[C::MT][2 * NG][4], int e, int ob,
                                            int cnt, const int* rows_sh, float* E) {
    constexpr int MT = C::MT, NW = C::NW, EPM = C::EPM, ES = C::DN_ES, U = EPM / 8;
    constexpr int PASSES = (2 * NG + U - 1) / U;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int col0 = warp * MT * 16;
    const int D = p.D;
#pragma unroll
    for (int ps = 0; ps < PASSES; ++ps) {
        if (ps * EPM >= cnt) break;
        __syncthreads();
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int jn = ps * U + u;
                if (jn < 2 * NG) {
                    const int tok = u * 8 + 2 * t, col = col0 + i * 16 + g;
                    E[tok * ES + col] = acc[i][jn][0];
                    E[(tok + 1) * ES + col] = acc[i][jn][1];
                    E[tok * ES + col + 8] = acc[i][jn][2];
                    E[(tok + 1) * ES + col + 8] = acc[i][jn][3];
                }
            }
        __syncthreads();
        for (int m = warp; m < EPM && ps * EPM + m < cnt; m += NW) {
            const int pr = rows_sh[ps * EPM + m];
#pragma unroll
            for (int blk = 0; blk < 2; ++blk) {
                const int n = ob * 256 + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = E[m * ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                const uint2 sv = *reinterpret_cast<const uint2*>(p.svh_d + (size_t)e * D + n);
                const half2 s01 = *reinterpret_cast<const half2*>(&sv.x), s23 = *reinterpret_cast<const half2*>(&sv.y);
                __stcs(reinterpret_cast<float4*>(p.y + (size_t)pr * D + n),
                       make_float4(v[0] * HAD_SCALE * __low2float(s01), v[1] * HAD_SCALE * __high2float(s01),
                                   v[2] * HAD_SCALE * __low2float(s23), v[3] * HAD_SCALE * __high2float(s23)));
            }
        }
    }
}

// A tile of one block of cnt members (NG = ceil(cnt / 16) groups): gate/up column block `sub` (one segment, K = D),
// or down output group `sub` (OB segments of 256 output columns, K = NI each), in one pipeline whose loads run on
// through each segment's epilogue. Q8 (K1q, gate/up only): int8 rows and weights, int32 sums.
template <class C, int NG, bool Q8>
__device__ __forceinline__ void run_tile(const Params& p, int dn, int e, int sub, int cnt, const int* rows_sh,
                                         const int* xrow_sh, const float* dq_sh, unsigned char* smem) {
    constexpr int DW = C::DW, DX = C::DX, PKT = C::PKT, BK = C::BK, MT = C::MT;
    using Acc = typename std::conditional<Q8, int, float>::type;
    if constexpr (Q8) dn = 0;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int K = dn ? p.NI : p.D;
    const int CPS = K / BK;                              // stages a segment
    const int nseg = dn ? p.OB : 1;
    const int total = nseg * CPS;
    const int NTILES = (dn ? p.D : p.NI) >> 4, KT = K >> 4;
    unsigned char* wring = smem;
    unsigned char* xring = smem + DW * C::W_BYTES;
    float* E = reinterpret_cast<float*>(smem + DW * C::W_BYTES + DX * C::X_BYTES);
    const unsigned wring_s = (unsigned)__cvta_generic_to_shared(wring);
    const unsigned xring_s = (unsigned)__cvta_generic_to_shared(xring);

    // this thread's pieces: words (tid < TW: WPT adjacent 16-byte pieces of tile q, k tile kk) or member rows
    // (adjacent 16-byte pieces of member xi); src offsets in elements from a base, dst offsets in bytes into a stage
    const bool wthr = tid < C::TW;
    const int widx = tid * C::WPT;
    const int wq = (widx >> 4) & 15, wkk = (widx >> 3) & (PKT - 1), wpart = widx & 7;   // (in range on every thread)
    constexpr int XPT = Q8 ? C::XPTQ : C::XPT;
    const int xidx = (tid - C::TW) * XPT;
    const int xi = (Q8 ? xidx >> 1 : xidx >> 2) & (C::BN - 1), xpart = Q8 ? xidx & 1 : xidx & 3;
    const int xr = wthr ? -1 : xrow_sh[xi];
    const bool xok = xr >= 0;
    const unsigned xo = (unsigned)(xok ? xr : 0) * (unsigned)K + (unsigned)(xpart * (Q8 ? 16 : 8));
    const unsigned dst = wthr ? (unsigned)((wq * PKT + wkk) * 32 + wpart * 4) * 4
                              : (Q8 ? (unsigned)(xi * C::PXSQ + xpart * 16) : (unsigned)(xi * C::PXS + xpart * 8) * 2);
    const unsigned wstep = (unsigned)(PKT * NTILES * 32);
    auto wseg = [&](int s) -> const uint32_t* {
        const uint32_t* T0;
        int nt;
        if (dn) {
            T0 = p.td;
            nt = (sub * p.OB + s) * 16 + wq;
        } else {
            T0 = (wq >> 3) ? p.tu : p.tg;
            nt = sub * 8 + (wq & 7);
        }
        return T0 + (((size_t)e * KT + wkk) * NTILES + nt) * 32 + wpart * 4;
    };
    const uint32_t* wb = wseg(0);
    int lc = 0, ls = 0;                                  // the next load's stage within its segment, and segment
    auto load = [&](int slot) {                          // this thread's pieces of the next stage
        if (wthr) {
            const uint32_t* src = wb + (size_t)((unsigned)lc * wstep);
            const unsigned d = wring_s + (unsigned)(slot * C::W_BYTES) + dst;
#pragma unroll
            for (int k = 0; k < C::WPT; ++k) cp_async16(d + 16 * k, src + 4 * k, true);
        } else {
            const unsigned d = xring_s + (unsigned)(slot * C::X_BYTES) + dst;
            if constexpr (Q8) {
                const int8_t* src = p.xq + (xo + (unsigned)(lc * BK));          // BK int8 values = BK bytes a stage
#pragma unroll
                for (int k = 0; k < XPT; ++k) cp_async16(d + 16 * k, src + 16 * k, xok);
            } else {
                const half* src = (dn ? p.xd : p.xh) + (xo + (unsigned)(lc * BK));
#pragma unroll
                for (int k = 0; k < XPT; ++k) cp_async16(d + 16 * k, src + 8 * k, xok);
            }
        }
        if (++lc == CPS) {
            lc = 0;
            if (++ls < nseg && wthr) wb = wseg(ls);
        }
    };
    const int depth = wthr ? DW : DX;                    // stages this thread loads ahead

    for (int s = 0; s < depth - 1; ++s) {
        if (s < total) load(s);
        cp_async_commit();
    }
    Acc acc[MT][2 * NG][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < 2 * NG; ++j)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][j][c] = Acc(0);
    const bool up_last = cnt > (NG - 1) * 16 + 8;
    int wsl = 0, xsl = 0, c = 0, seg = 0;                // stage gc's slots, its stage within the segment, segment
    for (int gc = 0; gc < total; ++gc) {
        if (wthr)
            cp_async_wait<DW - 2>();
        else
            cp_async_wait<DX - 2>();
        __syncthreads();                                 // stage gc landed for all; stage gc - 1's readers are done
        if (gc + depth - 1 < total) {
            const int sl = wthr ? wsl : xsl;
            load(sl == 0 ? depth - 1 : sl - 1);          // stage gc + depth - 1 into the slot stage gc - 1 left
        }
        cp_async_commit();
        if constexpr (Q8)
            stage_mma_q<C, NG>(xring + xsl * C::X_BYTES, wring + wsl * C::W_BYTES, acc, warp, lane, p.mask);
        else
            stage_mma<C, NG>(xring + xsl * C::X_BYTES, wring + wsl * C::W_BYTES, acc, up_last, warp, lane, p.mask);
        if (++wsl == DW) wsl = 0;
        if (++xsl == DX) xsl = 0;
        if constexpr (!Q8) {
            if (++c == CPS) {                            // a segment's last stage: its epilogue (loads run on)
                c = 0;
                if (dn)
                    epilogue_dn<C, NG>(p, acc, e, sub * p.OB + seg, cnt, rows_sh, E);
                else
                    epilogue_gu<C, NG, false>(p, acc, e, sub, cnt, rows_sh, dq_sh, E);
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int j = 0; j < 2 * NG; ++j)
#pragma unroll
                        for (int q = 0; q < 4; ++q) acc[i][j][q] = Acc(0);
                ++seg;
            }
        }
    }
    // K1q's gate/up tile is one segment: its epilogue after the loop (inside it, the int32 sums' reads at the
    // segment's end make the compiler keep the mma's results out of place and copy them back every stage)
    if constexpr (Q8) epilogue_gu<C, NG, true>(p, acc, e, sub, cnt, rows_sh, dq_sh, E);
    cp_async_wait<0>();
    __syncthreads();                                     // the rings, E and the member rows are free
}

// K1q's gate/up blocks hold as many members as K1's (its epilogue runs after the loop, so the int32 sums stay in
// place and nothing spills; K1_Q_MAXNG caps them at compile time if a compiler ever needs it).
#ifndef K1_Q_MAXNG
#define K1_Q_MAXNG 8
#endif
template <class C>
constexpr int q_maxng() { return C::NGM < K1_Q_MAXNG ? C::NGM : K1_Q_MAXNG; }
template <class C, bool Q8, int NG = 1>
__device__ __forceinline__ void run_groups(int ng, const Params& p, int dn, int e, int sub, int cnt, const int* rows_sh,
                                           const int* xrow_sh, const float* dq_sh, unsigned char* smem) {
    if constexpr (NG < (Q8 ? q_maxng<C>() : C::NGM)) {
        if (ng > NG) {
            run_groups<C, Q8, NG + 1>(ng, p, dn, e, sub, cnt, rows_sh, xrow_sh, dq_sh, smem);
            return;
        }
    }
    run_tile<C, NG, Q8>(p, dn, e, sub, cnt, rows_sh, xrow_sh, dq_sh, smem);
}

// The persistent grid: tiles in list order through an atomic counter (the next one claimed a tile ahead). Q (K1q):
// gate/up tiles take the int8 path; down tiles are K1's in either case.
template <class C, bool Q>
__global__ void __launch_bounds__(C::T, C::CTAS) k1_kernel(const Params p) {
    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ int rows_sh[C::BN];      // the members' pair indices (-1 past the block's members)
    __shared__ int xrow_sh[C::BN];      // the X row each member reads: its token (gate/up) or pair (down)
    __shared__ float dq_sh[Q ? C::BN : 1];   // K1q: a member's row scale / 31.75
    __shared__ int tile_sh[2];
    const int tid = threadIdx.x;
    const int ntiles = p.ctrl[1];
    const int ncb = p.NI >> 7;          // gate/up tiles a block
    if (tid == 0) tile_sh[0] = atomicAdd(p.ctrl, 1);
    for (int it = 0;; ++it) {
        __syncthreads();
        const int t = tile_sh[it & 1];
        if (t >= ntiles) break;
        if (tid == 0) tile_sh[(it + 1) & 1] = atomicAdd(p.ctrl, 1);
        const int code = p.tiles[t];
        const int b = code >> 8, dn = (code >> 7) & 1, sub = code & 127;
        const int e = p.blocks[3 * b], first = p.blocks[3 * b + 1], cnt = p.blocks[3 * b + 2];
        if (dn && tid == 0) count_wait(p.done + b, ncb);
        if (tid < C::BN) {
            const int q = tid < cnt ? p.members[first + tid] : -1;
            rows_sh[tid] = q;
            xrow_sh[tid] = q < 0 ? -1 : (dn ? q : q / p.slots);
            if constexpr (Q) dq_sh[tid] = (q < 0 || dn) ? 0.f : p.xs[q / p.slots] / Q8_WEIGHT;
        }
        __syncthreads();
        if (Q && !dn)
            run_groups<C, true>((cnt + 15) >> 4, p, dn, e, sub, cnt, rows_sh, xrow_sh, dq_sh, smem);
        else
            run_groups<C, false>((cnt + 15) >> 4, p, dn, e, sub, cnt, rows_sh, xrow_sh, dq_sh, smem);
        if (!dn) {
            __threadfence();                             // this thread's Xd stores, before the count
            __syncthreads();
            if (tid == 0) count_release_add(p.done + b, 1);
        }
    }
}

// K1q's rows: one block of 8 warps a row. v = fwht128((x * suh) a 128 block) / sqrt(128) in fp32 (rot_rows'
// arithmetic before its fp16 rounding); the row's scale s = max|v| / 127; q = rint(v * (127 / max|v|)) as int8,
// each 16-value group in the decode's slot order (logical k 2t, 2t+1, 2t+8, 2t+9 in slots 4t..4t+3). A row's values
// depend on that row alone.
__global__ void __launch_bounds__(256) k1q_rot_quant_kernel(const __nv_bfloat16* __restrict__ x, int x_stride,
                                                            const half* __restrict__ suh, int8_t* __restrict__ xq,
                                                            float* __restrict__ xs, int D) {
    const int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nblk = D >> 7;                             // at most 64 (D <= 8192)
    float v[8][4];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int blk = warp + 8 * i;
        if (blk < nblk) {
            const __nv_bfloat16* xr = x + (size_t)row * x_stride + blk * 128 + 4 * lane;
            const half* sr = suh + blk * 128 + 4 * lane;
#pragma unroll
            for (int j = 0; j < 4; ++j) v[i][j] = __bfloat162float(xr[j]) * __half2float(sr[j]);
            fwht128(v[i], lane);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                v[i][j] = v[i][j] * HAD_SCALE;
                amax = fmaxf(amax, fabsf(v[i][j]));
            }
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    __shared__ float wmax[8];
    if (lane == 0) wmax[warp] = amax;
    __syncthreads();
    float m = wmax[0];
#pragma unroll
    for (int w = 1; w < 8; ++w) m = fmaxf(m, wmax[w]);
    const float inv = m > 0.f ? 127.f / m : 0.f;
    if (threadIdx.x == 0) xs[row] = m / 127.f;
    const int lq = lane & 3;
    const int s0 = ((lq & 1) << 3) | ((lq >> 1) << 1);   // logical 4 lq + j -> slots s0, s0 + 1, s0 + 4, s0 + 5
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int blk = warp + 8 * i;
        if (blk < nblk) {
            int q[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) q[j] = max(-127, min(127, __float2int_rn(v[i][j] * inv)));
            unsigned char* base = reinterpret_cast<unsigned char*>(xq) + (size_t)row * D + blk * 128 + (lane >> 2) * 16;
            *reinterpret_cast<unsigned short*>(base + s0) = (unsigned short)((q[0] & 0xff) | ((q[1] & 0xff) << 8));
            *reinterpret_cast<unsigned short*>(base + s0 + 4) = (unsigned short)((q[2] & 0xff) | ((q[3] & 0xff) << 8));
        }
    }
}

// ---- the tile list --------------------------------------------------------------------------------------------------

constexpr int SCHED_THREADS = 1024;
constexpr int MAX_BLOCKS = 4096;

// Block-wide exclusive scan of one int a thread; *total = the sum (all threads).
__device__ __forceinline__ int block_scan(int v, int* wsum, int* total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += u;
    }
    __syncthreads();                                     // wsum's previous readers are done
    if (lane == 31) wsum[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = wsum[lane];
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += u;
        }
        wsum[lane] = w;
    }
    __syncthreads();
    *total = wsum[31];
    return (warp ? wsum[warp - 1] : 0) + x - v;
}

// Where the tiles of the block at list position `pos` (of n) go: its NCB gate/up tiles at gate_up_start, its NOG down
// tiles at down_start. Down tiles of position j follow the gate/up tiles of position j + lag (the tail when that is
// past the end), so a down tile is claimed about lag blocks after its own gate/up tiles.
__host__ __device__ __forceinline__ int gate_up_start(int pos, int lag, int NCB, int NOG) {
    return pos * NCB + (pos > lag ? pos - lag : 0) * NOG;
}
__host__ __device__ __forceinline__ int down_start(int pos, int n, int lag, int NCB, int NOG) {
    return pos + lag < n ? (pos + lag + 1) * NCB + pos * NOG : n * NCB + pos * NOG;
}

// One block: the plan's experts (items in expert order, the shared expert's id E last and left out; or CSR offsets
// [E + 1]) as blocks of up to BN members, and the tile list. Blocks are ranked by their expert's first block's n8 tiles
// (descending; an expert's blocks together, in order), then interleaved from both ends (the heaviest, the lightest,
// the next heaviest, ...) so DRAM-bound and tensor-bound tiles run side by side and the list ends with middling ones.
__global__ void __launch_bounds__(SCHED_THREADS) k1_schedule_kernel(
    const int* __restrict__ items, const int* __restrict__ counts, int cap_items, const int* __restrict__ offsets,
    int E, int BN, int NCB, int NOG, int lag, int cap_blocks, int* __restrict__ blocks, int* __restrict__ tiles,
    int* __restrict__ ctrl, int* __restrict__ done) {
    __shared__ int key_sh[MAX_BLOCKS];
    __shared__ int wsum[32];
    const int tid = threadIdx.x;
    const int n_cand = offsets ? E : min(counts[0], cap_items);
    const int per = (n_cand + SCHED_THREADS - 1) / SCHED_THREADS;
    const int lo = min(n_cand, tid * per), hi = min(n_cand, lo + per);
    auto run = [&](int i, int& e, int& first, int& c) -> bool {   // candidate i starts a routed expert's members
        if (offsets) {
            e = i;
            first = offsets[i];
            c = offsets[i + 1] - first;
            return c > 0;
        }
        e = items[3 * i];
        if (e < 0 || e >= E || (i > 0 && items[3 * (i - 1)] == e)) return false;
        first = items[3 * i + 1];
        c = 0;
        for (int j = i; j < n_cand && items[3 * j] == e; ++j) c += items[3 * j + 2];
        return c > 0;
    };
    int nb = 0;
    for (int i = lo; i < hi; ++i) {
        int e, f, c;
        if (run(i, e, f, c)) nb += (c + BN - 1) / BN;
    }
    int want;
    int base = block_scan(nb, wsum, &want);
    // the host sizes cap_blocks with the bound every plan meets (min(pairs, experts) + pairs / BN, exl3_k1.py
    // max_blocks); ctrl[3] keeps the count a plan needed, for tests
    const int n = min(want, min(cap_blocks, MAX_BLOCKS));
    for (int i = lo; i < hi; ++i) {
        int e, f, c;
        if (!run(i, e, f, c)) continue;
        const int key = (min(c, BN) + 7) >> 3;
        for (int k = 0; k * BN < c && base < n; ++k, ++base) {
            blocks[3 * base] = e;
            blocks[3 * base + 1] = f + k * BN;
            blocks[3 * base + 2] = min(BN, c - k * BN);
            key_sh[base] = key;
        }
    }
    __syncthreads();
    // rank = blocks of larger keys + blocks of the same key before it (a scan over the blocks a key value)
    const int bper = (n + SCHED_THREADS - 1) / SCHED_THREADS;
    const int blo = min(n, tid * bper), bhi = min(n, blo + bper);
    const int half_n = (n + 1) >> 1;
    int above = 0;
    for (int k = (BN + 7) >> 3; k >= 1; --k) {
        int mine = 0;
        for (int b = blo; b < bhi; ++b) mine += key_sh[b] == k;
        int tot;
        int r = above + block_scan(mine, wsum, &tot);
        for (int b = blo; b < bhi; ++b) {
            if (key_sh[b] != k) continue;
            const int pos = r < half_n ? 2 * r : 2 * (n - 1 - r) + 1;
            const int g0 = gate_up_start(pos, lag, NCB, NOG), d0 = down_start(pos, n, lag, NCB, NOG);
            for (int cb = 0; cb < NCB; ++cb) tiles[g0 + cb] = (b << 8) | cb;
            for (int og = 0; og < NOG; ++og) tiles[d0 + og] = (b << 8) | 128 | og;
            ++r;
        }
        above += tot;
    }
    for (int b = tid; b < n; b += SCHED_THREADS) done[b] = 0;
    if (tid == 0) {
        ctrl[0] = 0;
        ctrl[1] = n * (NCB + NOG);
        ctrl[2] = n;
        ctrl[3] = want;                                  // > ctrl[2]: blocks past the capacity (tests check it)
    }
}

template <class C, bool Q>
int grid_of() {
    static int grid = [] {
        cudaFuncSetAttribute(k1_kernel<C, Q>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM);
        int per_sm = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k1_kernel<C, Q>, C::T, C::SMEM);
        per_sm = per_sm < 1 ? 1 : (per_sm > C::CTAS ? C::CTAS : per_sm);
        return per_sm * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    }();
    return grid;
}

}  // namespace

// Configurations: 0 = CfgDeep ("k1"), 1 = CfgPair ("k1-64"), 2 = CfgWide ("k1-w16"); q: K1q's int8 gate/up.
int64_t exl3_k1_block_members(int64_t cfg, bool q) {
    if (q) return 16 * (cfg == 1 ? q_maxng<CfgPair>() : cfg == 2 ? q_maxng<CfgWide>() : q_maxng<CfgDeep>());
    return cfg == 1 ? CfgPair::BN : cfg == 2 ? CfgWide::BN : CfgDeep::BN;
}

int64_t exl3_k1_max_blocks() { return MAX_BLOCKS; }

// The persistent grid of a configuration on this device (its blocks a call).
int64_t exl3_k1_grid(int64_t cfg, bool q) {
    if (q) return cfg == 1 ? grid_of<CfgPair, true>() : cfg == 2 ? grid_of<CfgWide, true>() : grid_of<CfgDeep, true>();
    return cfg == 1 ? grid_of<CfgPair, false>() : cfg == 2 ? grid_of<CfgWide, false>() : grid_of<CfgDeep, false>();
}

// The tile list alone (tests read it): blocks [cap_blocks, 3], tiles [cap_blocks * (NCB + NOG)], ctrl [4],
// done [cap_blocks]. lag: list positions between a block's gate/up tiles and its down tiles.
void exl3_k1_schedule_cuda(const at::Tensor& items, const at::Tensor& counts, const c10::optional<at::Tensor>& offsets,
                           int64_t E, int64_t BN, int64_t NCB, int64_t NOG, int64_t lag, at::Tensor& blocks,
                           at::Tensor& tiles, at::Tensor& ctrl, at::Tensor& done) {
    const int64_t cap_blocks = done.numel();
    TORCH_CHECK(cap_blocks <= MAX_BLOCKS, "k1: at most ", MAX_BLOCKS, " member blocks");
    TORCH_CHECK(blocks.numel() >= 3 * cap_blocks && tiles.numel() >= cap_blocks * (NCB + NOG) && ctrl.numel() >= 4,
                "k1: schedule buffers too small");
    TORCH_CHECK(NCB >= 1 && NCB <= 128 && NOG >= 1 && NOG <= 128 && BN >= 8 && BN <= 128, "k1: schedule shape");
    const int* off = nullptr;
    if (offsets.has_value()) {
        TORCH_CHECK(offsets->numel() >= E + 1, "k1: offsets [E + 1]");
        off = offsets->data_ptr<int>();
    }
    TORCH_CHECK(lag >= 0, "k1: lag >= 0");
    k1_schedule_kernel<<<1, SCHED_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        items.data_ptr<int>(), counts.data_ptr<int>(), (int)(items.numel() / 3), off, (int)E, (int)BN, (int)NCB,
        (int)NOG, (int)(lag > MAX_BLOCKS ? MAX_BLOCKS : lag), (int)cap_blocks, blocks.data_ptr<int>(),
        tiles.data_ptr<int>(), ctrl.data_ptr<int>(), done.data_ptr<int>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// K1q's rows: xq [rows, D] int8 (decode slot order), xs [rows] fp32 scales, from bf16 x and the layer's shared suh.
void exl3_k1_rot_quant_cuda(const at::Tensor& x, int64_t x_stride, const at::Tensor& suh, at::Tensor& xq,
                            at::Tensor& xs, int64_t rows, int64_t D) {
    TORCH_CHECK(D % 128 == 0 && D <= 8192, "k1q: D a multiple of 128, at most 8,192");
    if (rows <= 0) return;
    k1q_rot_quant_kernel<<<(unsigned)rows, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), (int)x_stride,
        reinterpret_cast<const half*>(suh.data_ptr()), xq.data_ptr<int8_t>(), xs.data_ptr<float>(), (int)D);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Xd [P, NI] (fp16) and Y [P, D] (fp32) of every routed pair of the plan; the shared expert's pairs are left alone.
// q: K1q (gate/up from xq / xs in int8; xh unused).
void exl3_k1_cuda(const at::Tensor& xh, const at::Tensor& tg, const at::Tensor& tu, const at::Tensor& td,
                  const at::Tensor& svh_g, const at::Tensor& svh_u, const at::Tensor& suh_d, const at::Tensor& svh_d,
                  at::Tensor& xd, at::Tensor& y, const at::Tensor& items, const at::Tensor& counts,
                  const at::Tensor& members, const c10::optional<at::Tensor>& offsets, at::Tensor& blocks,
                  at::Tensor& tiles, at::Tensor& ctrl, at::Tensor& done, int64_t D, int64_t NI, int64_t slots,
                  double limit, int64_t cfg, int64_t OB, int64_t lag, bool q, const c10::optional<at::Tensor>& xq,
                  const c10::optional<at::Tensor>& xs, int64_t max_ctas) {
    TORCH_CHECK(D % 256 == 0 && NI % 128 == 0 && NI >= 128, "k1: D % 256 and NI % 128");
    TORCH_CHECK(OB >= 1 && (D / 256) % OB == 0, "k1: OB divides D / 256");
    TORCH_CHECK(cfg >= 0 && cfg <= 2, "k1: configuration 0 (k1), 1 (k1-64) or 2 (k1-w16)");
    TORCH_CHECK(!q || (xq.has_value() && xs.has_value()), "k1q: xq and xs");
    const int64_t E = tg.size(0);
    const int64_t BN = exl3_k1_block_members(cfg, q);
    const int64_t NCB = NI / 128, NOG = (D / 256) / OB;
    int64_t grid = exl3_k1_grid(cfg, q);
    if (max_ctas > 0 && max_ctas < grid) grid = max_ctas;   // leave SMs to another stream (scheduling only)
    if (lag < 0) lag = (3 * grid + NCB - 1) / NCB;
    exl3_k1_schedule_cuda(items, counts, offsets, E, BN, NCB, NOG, lag, blocks, tiles, ctrl, done);
    Params prm;
    prm.xh = reinterpret_cast<const half*>(xh.data_ptr());
    prm.tg = reinterpret_cast<const uint32_t*>(tg.data_ptr());
    prm.tu = reinterpret_cast<const uint32_t*>(tu.data_ptr());
    prm.td = reinterpret_cast<const uint32_t*>(td.data_ptr());
    prm.svh_g = reinterpret_cast<const half*>(svh_g.data_ptr());
    prm.svh_u = reinterpret_cast<const half*>(svh_u.data_ptr());
    prm.suh_d = reinterpret_cast<const half*>(suh_d.data_ptr());
    prm.svh_d = reinterpret_cast<const half*>(svh_d.data_ptr());
    prm.xd = reinterpret_cast<half*>(xd.data_ptr());
    prm.y = y.data_ptr<float>();
    prm.members = members.data_ptr<int>();
    prm.blocks = blocks.data_ptr<int>();
    prm.tiles = tiles.data_ptr<int>();
    prm.ctrl = ctrl.data_ptr<int>();
    prm.done = done.data_ptr<int>();
    prm.D = (int)D;
    prm.NI = (int)NI;
    prm.slots = (int)slots;
    prm.OB = (int)OB;
    prm.limit = (float)limit;
    prm.mask = 0x8FFF8FFFu;
    prm.xq = q ? xq->data_ptr<int8_t>() : nullptr;
    prm.xs = q ? xs->data_ptr<float>() : nullptr;
    auto stream = at::cuda::getCurrentCUDAStream();
#define K1_GO(CFG, Q_) k1_kernel<CFG, Q_><<<(unsigned)grid, CFG::T, CFG::SMEM, stream>>>(prm)
    if (q) {
        if (cfg == 1) K1_GO(CfgPair, true); else if (cfg == 2) K1_GO(CfgWide, true); else K1_GO(CfgDeep, true);
    } else {
        if (cfg == 1) K1_GO(CfgPair, false); else if (cfg == 2) K1_GO(CfgWide, false); else K1_GO(CfgDeep, false);
    }
#undef K1_GO
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
