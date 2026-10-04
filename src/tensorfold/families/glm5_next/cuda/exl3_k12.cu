// GLM-5.3-Flash's routed EXL3 experts in prompt chunks: the K12 kernels (TF_GLM_EXPERT_PROMPT_KERNEL=k12...).
//
// The bits of exl3.cu's prompt kernels (prompt_gateup_kernel / prompt_down_kernel): the same plan items (an item = up
// to 64 members of one expert), the same blocks (16 m tiles: 128 gate + 128 up columns, or 256 down columns), every
// output one fp32 chain of mma.m16n8k16 over the ascending k tiles from zero, and the same epilogue formulas in the
// same order. What changes is how the main loop feeds the tensor pipe (that kernel keeps it busy 53-62% of cycles:
// a stage's code is cut into nine basic blocks by per-block branches, the decode and the mma of a warp run in turns,
// and a third of the integer work re-creates the codebook mask):
//  - the item's member count picks a main loop compiled for its number of 8-member column blocks (NBA), so a stage
//    is one basic block with no branch, no padding mma and no predicate;
//  - a warp keeps two A fragment sets: the next k tile's trellis decode is issued among the current k tile's mma,
//    and across the stage barrier (the next stage's words are waited for one stage early);
//  - a lane reads its neighbour's word from shared memory instead of a shuffle, and the codebook mask is an argument
//    (a register), not a constant the compiler rebuilds;
//  - the epilogue reads its per-column scales once a block.
// Variant ``ws`` (warp-specialised): two producer warps decode every tile of a k tile once into shared memory, eight
// consumer warps only load fragments and issue mma; the same chains, so the same bits.
//
// Trellis decode after exl3.cu (format after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp).

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

namespace cg = cooperative_groups;

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)
constexpr uint32_t MCG_MASK = 0x8FFF8FFFu;          // the codebook mask (a kernel argument: a register)

// (x & m) ^ 0x3B603B60 in one LOP3 with m in a register.
__device__ __forceinline__ uint32_t lop_mask_xor(uint32_t x, uint32_t m) {
    uint32_t y;
    asm("lop3.b32 %0, %1, %2, 0x3B603B60, 0x6a;\n" : "=r"(y) : "r"(x), "r"(m));
    return y;
}

// exl3.cu's mcg2: two codebook values from two 16-bit states, as a half2 (first state in .x).
__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1, uint32_t m) {
    const uint32_t x0 = lop_mask_xor(s0 * 0xCBAC1FEDu, m);
    const uint32_t x1 = lop_mask_xor(s1 * 0xCBAC1FEDu, m);
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// A tile's mma A fragment for this lane from its word w and the word of lane - 1 (p): exl3.cu decode_tile_p's values
// (b0, b1) in its A order, a0 = b0[0], a1 = b1[0], a2 = b0[1], a3 = b1[1].
__device__ __forceinline__ void decode_a(uint32_t w, uint32_t p, uint32_t m, uint32_t (&a)[4]) {
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    a[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu, m);   // (s >> 8) & 0xffff, (s >> 4) & 0xffff
    a[1] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421), m);
    a[2] = mcg2(s & 0xffffu, w >> 16, m);
    a[3] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu, m);
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, int bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], uint32_t s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], uint32_t s) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
__device__ __forceinline__ uint32_t lds32(uint32_t s) {
    uint32_t v;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(v) : "r"(s));
    return v;
}
__device__ __forceinline__ uint4 lds128(uint32_t s) {
    uint4 v;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(s));
    return v;
}
__device__ __forceinline__ void sts128(uint32_t s, uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};\n" ::"r"(s), "r"(a), "r"(b), "r"(c), "r"(d));
}
// Named barriers of a "ws" block (id 0 is __syncthreads').
template <int ID, int THREADS>
__device__ __forceinline__ void bar_sync() {
    asm volatile("bar.sync %0, %1;\n" ::"n"(ID), "n"(THREADS) : "memory");
}
template <int ID, int THREADS>
__device__ __forceinline__ void bar_arrive() {
    asm volatile("bar.arrive %0, %1;\n" ::"n"(ID), "n"(THREADS) : "memory");
}

// exl3.cu's fwht128: the same butterflies in the same order, the same bits. A cross-lane butterfly, v + o in the
// lower lane and o - v in the upper one, is one FFMA, fmaf(s, v, o) with s = +1 or -1: s v is exact, so the FFMA
// rounds o + v or o - v once, as the add does (exl3.cu's code compiles to an FSEL of -v and an FADD).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c; v[1] = b + d; v[2] = a - c; v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const float s = (lane & m) ? -1.f : 1.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = fmaf(s, v[j], o);
        }
    }
}
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

struct Params {
    const half* x;              // gate/up: the rotated rows [rows, D] (one a token); down: Xd [pairs, NI]
    const uint32_t* t0;         // trellis words [E, K/16, N/16, 32]: gate (gate/up) or down
    const uint32_t* t1;         // up (gate/up)
    const int* items;           // the plan: items [n, 3] (expert, first member, count), counts [2], members
    const int* counts;
    const int* members;
    const int* order;           // the launch order (nullptr: plan order)
    const half* svh_g;
    const half* svh_u;
    const half* suh_d;
    const half* svh_d;
    half* xd;                   // [pairs, NI]
    float* y;                   // [pairs, D]
    int K, N, E, slots;         // K, N: this kernel's matmul (gate/up: D, NI; down: NI, D)
    float limit;
    uint32_t mask;              // MCG_MASK
};

// A block's shape. MATS: 2 for gate/up (the first TILES / 2 m tiles are the gate's), 1 for down or for one matrix of
// a gate/up pair of blocks (k12_gateup_c2_kernel). A warp holds MT m tiles for all PM members; a stage holds PKT k
// tiles (PBK = 16 PKT values) of the members' rows and of the TILES tiles' words; PS stages. WS: the
// warp-specialised variant (NW consumer warps plus PW producer warps). Blocks: 16 m tiles x 64 members (exl3.cu's
// items), 16 m tiles x 80 members, or 8 m tiles (128 columns, one Hadamard block) x 128 members (blocks of an
// expert's members cut from its first: k12_blocks_kernel).
template <int MATS_, int MT_, int NW_, int PM_, int PKT_, int PS_, bool WS_ = false, int PW_ = 0>
struct Cfg {
    static constexpr int MATS = MATS_, MT = MT_, NW = NW_, PM = PM_, PKT = PKT_, PS = PS_, PW = PW_;
    static constexpr bool WS = WS_;
    static constexpr int TILES = MT * NW, TPM = TILES / MATS, NB = PM / 8, PBK = PKT * 16, PXS = PBK + 8;
    static constexpr int TC = NW * 32, T = TC + PW * 32;      // consumer threads, all threads
    static constexpr int X_BYTES = PM * PXS * 2, W_BYTES = TILES * PKT * 32 * 4;
    static constexpr int STAGE_BYTES = X_BYTES + W_BYTES;
    // ws: decoded A fragments, ASLOTS k tiles in flight, a k tile = TILES tiles x 32 lanes x 16 bytes
    static constexpr int ASLOTS = 2, A_BYTES = WS ? ASLOTS * TILES * 512 : 0;
    static constexpr int RD_MAX = WS ? 4 : NB;                 // ws: items of up to RD_MAX blocks run register-direct
    static constexpr int MAIN_BYTES = PS * STAGE_BYTES + A_BYTES;
    static constexpr int ES = TPM * 16 + 4;                   // the epilogue's fp32 row stride (exl3.cu's 132 / 260)
    static constexpr int EP_BYTES = MATS * (PM / 2) * ES * 4;  // half the members' sums, every matrix
    static constexpr int SMEM = MAIN_BYTES > EP_BYTES ? MAIN_BYTES : EP_BYTES;
    static constexpr int XPIECES = PM * PBK / 8, WPIECES = TILES * PKT * 8;
    static constexpr int XPT = (XPIECES + TC - 1) / TC;       // member-row pieces a (consumer) thread loads a stage
    static constexpr int CTAS = 2;                             // blocks an SM (the register cap of launch bounds)
    static_assert(PS >= 3 && PS <= 4, "a stage of lookahead needs three or four stages");
    static_assert(PKT % 2 == 0, "two A fragment sets: an even number of k tiles a stage");
    static_assert((TILES == 16 && PM == 64) || (TILES == 16 && PM == 80 && !WS) ||
                      (TILES == 8 && PM == 128 && MATS == 1 && !WS),
                  "blocks of 16 m tiles x 64 or 80 members, or 8 m tiles x 128 members");
    static_assert(CTAS * (SMEM + 1024 + 2 * PM * 4) <= 100 * 1024, "shared memory for two blocks an SM");
    static_assert(!WS || (PW > 0 && TILES % PW == 0), "producer warps share the tiles evenly");
};

// m tile q of block cb: its matrix and its n tile (exl3.cu prompt_tile).
template <class C>
__device__ __forceinline__ void tile_of(int q, int cb, int& mat, int& nt) {
    mat = q / C::TPM;
    nt = cb * C::TPM + q % C::TPM;
}

// This thread's cp.async pieces of a stage: member-row pieces (consumer threads) and word pieces (WT word loaders).
template <class C, int NBA, int WT>
struct Loads {
    static constexpr int WPT = (C::WPIECES + WT - 1) / WT;
    const half* xs[C::XPT];
    uint32_t xd[C::XPT];
    int xb[C::XPT];
    bool xon[C::XPT];
    const uint32_t* ws[WPT];
    uint32_t wd[WPT];
    bool won[WPT];
    size_t wstep;

    // xt / wt: this thread's index among the row loaders / the word loaders (-1: none; its pointers then stay inside
    // the tensors, unused)
    __device__ __forceinline__ void init(const Params& p, const int* xrows, int e, int cb, int xt, int wt,
                                         uint32_t sbase) {
        const int K = p.K, KT = K >> 4, NT = p.N >> 4;
#pragma unroll
        for (int j = 0; j < C::XPT; ++j) {
            const int idx = (xt < 0 ? 0 : xt) + j * C::TC, i = idx / (C::PBK / 8), part = idx % (C::PBK / 8);
            const bool in = xt >= 0 && idx < C::XPIECES;
            const int r = in ? xrows[i] : -1;
            xs[j] = p.x + (size_t)(r < 0 ? 0 : r) * K + part * 8;
            xb[j] = r >= 0 ? 16 : 0;                             // rows past the count: zeros (exl3.cu's)
            xon[j] = in && i < NBA * 8;                         // rows no mma reads are not loaded
            xd[j] = sbase + (uint32_t)(i * C::PXS + part * 8) * 2;
        }
#pragma unroll
        for (int j = 0; j < WPT; ++j) {
            const int idx = (wt < 0 ? 0 : wt) + j * WT, q = idx / (C::PKT * 8), kk = (idx / 8) % C::PKT, part = idx % 8;
            won[j] = wt >= 0 && idx < C::WPIECES;
            int mat, nt;
            tile_of<C>(won[j] ? q : 0, cb, mat, nt);
            ws[j] = (mat ? p.t1 : p.t0) + (((size_t)e * KT + kk) * NT + nt) * 32 + part * 4;
            wd[j] = sbase + C::X_BYTES + (uint32_t)((q * C::PKT + kk) * 32 + part * 4) * 4;
        }
        wstep = (size_t)C::PKT * NT * 32;
    }
    __device__ __forceinline__ void issue_x(int c, int slot) const {
        const uint32_t off = (uint32_t)slot * C::STAGE_BYTES;
#pragma unroll
        for (int j = 0; j < C::XPT; ++j)
            if (xon[j]) cp_async16(xd[j] + off, xs[j] + (size_t)c * C::PBK, xb[j]);
    }
    __device__ __forceinline__ void issue_w(int c, int slot) const {
        const uint32_t off = (uint32_t)slot * C::STAGE_BYTES;
#pragma unroll
        for (int j = 0; j < WPT; ++j)
            if (won[j]) cp_async16(wd[j] + off, ws[j] + (size_t)c * wstep, 16);
    }
    __device__ __forceinline__ void issue(int c, int slot) const {
        issue_x(c, slot);
        issue_w(c, slot);
    }
};

// Register-direct main loop: warp w holds m tiles w MT .. w MT + MT - 1 for the item's NBA 8-member column blocks.
// acc[i][b] = the fp32 chain of m tile i, members 8b .. 8b + 7, over k tiles 0, 1, ..., K/16 - 1 (exl3.cu's). Run by
// the TC consumer threads (all of a register-direct block; in a warp-specialised block, its small items, while the
// producer warps wait at the epilogue's barrier): stage turns on barrier 0, or 6 among the consumers.
template <class C, int NBA>
__device__ __forceinline__ void mainloop_rd(const Params& p, const int* xrows, int e, int cb, unsigned char* smem,
                                            float (&acc)[C::MT][NBA][4]) {
    constexpr int MT = C::MT, PKT = C::PKT, PS = C::PS, PXS = C::PXS, NP = (NBA + 1) / 2;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int chunks = p.K / C::PBK;
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t m = p.mask;
    auto stage_sync = [] {
        if constexpr (C::WS) bar_sync<6, C::TC>(); else __syncthreads();
    };

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int b = 0; b < NBA; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][b][c] = 0.f;

    Loads<C, NBA, C::TC> ld;
    ld.init(p, xrows, e, cb, tid, tid, sbase);
#pragma unroll
    for (int s = 0; s < PS - 1; ++s) {
        if (s < chunks) ld.issue(s, s);
        cp_async_commit();
    }
    // per-lane offsets: ldmatrix rows (members) and columns (k), this warp's words and the lane - 1 word
    const uint32_t xlane = (uint32_t)((((lane & 7) + ((lane >> 4) & 1) * 8) * PXS + ((lane >> 3) & 1) * 8) * 2);
    const uint32_t wl0 = (uint32_t)(C::X_BYTES + (warp * MT * PKT * 32 + lane) * 4);
    const uint32_t wl1 = (uint32_t)(C::X_BYTES + (warp * MT * PKT * 32 + ((lane + 31) & 31)) * 4);

    cp_async_wait<PS - 2>();
    stage_sync();                                      // stage 0
    uint32_t A[2][MT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
        decode_a(lds32(sbase + wl0 + i * PKT * 128), lds32(sbase + wl1 + i * PKT * 128), m, A[0][i]);

    uint32_t sc = sbase, sn = sbase + C::STAGE_BYTES;  // this stage's and the next stage's buffers
    int fill = PS - 1;                                 // the buffer the next load goes to
    for (int c = 0; c < chunks; ++c) {
        cp_async_wait<PS - 3>();                       // stage c + 1 landed: its first words decode in this one
        stage_sync();                                  // and every warp is past stage c - 1
        if (c + PS - 1 < chunks) ld.issue(c + PS - 1, fill);
        cp_async_commit();
        fill = fill + 1 == PS ? 0 : fill + 1;
        const uint32_t xs = sc + xlane;
#pragma unroll
        for (int kk = 0; kk < PKT; ++kk) {
            // the next k tile's A set (free since the previous k tile's mma) decodes while this one's mma issue
            const int cur = kk & 1, nxt = cur ^ 1;
            const uint32_t src = kk + 1 < PKT ? sc + (uint32_t)((kk + 1) * 128) : sn;   // the next k tile's words
#pragma unroll
            for (int i = 0; i < MT; ++i)
                decode_a(lds32(src + wl0 + i * PKT * 128), lds32(src + wl1 + i * PKT * 128), m, A[nxt][i]);
#pragma unroll
            for (int jp = 0; jp < NP; ++jp) {
                const uint32_t xa = xs + (uint32_t)((jp * 16 * PXS + kk * 16) * 2);
                if (2 * jp + 1 < NBA) {
                    uint32_t r[4];
                    ldsm_x4(r, xa);
#pragma unroll
                    for (int i = 0; i < MT; ++i) {
                        mma16816(acc[i][2 * jp], A[cur][i], r[0], r[1]);
                        mma16816(acc[i][2 * jp + 1], A[cur][i], r[2], r[3]);
                    }
                } else {
                    uint32_t r[2];
                    ldsm_x2(r, xa);
#pragma unroll
                    for (int i = 0; i < MT; ++i) mma16816(acc[i][2 * jp], A[cur][i], r[0], r[1]);
                }
            }
        }
        sc = sn;
        sn = sn + C::STAGE_BYTES == sbase + PS * C::STAGE_BYTES ? sbase : sn + C::STAGE_BYTES;
    }
}

// Warp-specialised main loop. Producer warps (tid >= TC) load the words and decode each k tile's TILES tiles once
// into A slots (lane-major 16-byte fragments); consumer warps load the member rows and run the mma, reading their
// fragments with one 16-byte load a tile. Each side turns its own stages (barrier 5: producers, 6: consumers); slot
// s of the A ring: barrier 1 + s full (producers arrive, consumers wait), 3 + s free (consumers arrive, producers
// wait). A k tile's fragments are the decode_a values the register-direct loop feeds the same mma, so the same bits.
template <class C, int NBA>
__device__ __forceinline__ void mainloop_ws(const Params& p, const int* xrows, int e, int cb, unsigned char* smem,
                                            float (&acc)[C::MT][NBA][4]) {
    constexpr int MT = C::MT, PKT = C::PKT, PS = C::PS, PXS = C::PXS, NP = (NBA + 1) / 2;
    constexpr int TPW = C::TILES / C::PW;              // tiles a producer warp decodes a k tile
    constexpr int SLOT = C::TILES * 512, T = C::T, TP = C::PW * 32, TC = C::TC;
    static_assert(C::ASLOTS == 2, "two A slots: k tile k uses slot k & 1 (an even PKT keeps it kk & 1)");
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const bool producer = tid >= TC;
    const int chunks = p.K / C::PBK, ksteps = chunks * PKT;
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);
    const uint32_t abase = sbase + PS * C::STAGE_BYTES;
    const uint32_t m = p.mask;

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int b = 0; b < NBA; ++b)
#pragma unroll
            for (int c = 0; c < 4; ++c) acc[i][b][c] = 0.f;

    Loads<C, NBA, TP> ld;
    ld.init(p, xrows, e, cb, producer ? -1 : tid, producer ? tid - TC : -1, sbase);
    if (producer) {
#pragma unroll
        for (int s = 0; s < PS - 1; ++s) {
            if (s < chunks) ld.issue_w(s, s);
            cp_async_commit();
        }
        const int pw = warp - C::NW;
        const uint32_t wl0 = (uint32_t)(C::X_BYTES + (pw * TPW * PKT * 32 + lane) * 4);
        const uint32_t wl1 = (uint32_t)(C::X_BYTES + (pw * TPW * PKT * 32 + ((lane + 31) & 31)) * 4);
        const uint32_t al = abase + (uint32_t)((pw * TPW * 32 + lane) * 16);
        int sb = 0;
        for (int c = 0; c < chunks; ++c) {
            cp_async_wait<PS - 2>();                   // stage c's words (this thread's pieces)
            bar_sync<5, TP>();                         // every producer's pieces; every producer past stage c - 1
            if (c + PS - 1 < chunks) ld.issue_w(c + PS - 1, sb == 0 ? PS - 1 : sb - 1);
            cp_async_commit();
            const uint32_t wc = sbase + (uint32_t)sb * C::STAGE_BYTES;
#pragma unroll
            for (int kk = 0; kk < PKT; ++kk) {
                const int s = kk & 1;
                uint32_t w[TPW], q[TPW];
#pragma unroll
                for (int i = 0; i < TPW; ++i) {
                    w[i] = lds32(wc + wl0 + (uint32_t)((i * PKT + kk) * 128));
                    q[i] = lds32(wc + wl1 + (uint32_t)((i * PKT + kk) * 128));
                }
                if (c * PKT + kk >= 2) {               // slot s held k tile k - 2: wait until it is read
                    if (s) bar_sync<4, T>(); else bar_sync<3, T>();
                }
#pragma unroll
                for (int i = 0; i < TPW; ++i) {
                    uint32_t a[4];
                    decode_a(w[i], q[i], m, a);
                    sts128(al + (uint32_t)(s * SLOT + i * 512), a[0], a[1], a[2], a[3]);
                }
                if (s) bar_arrive<2, T>(); else bar_arrive<1, T>();
            }
            sb = sb + 1 == PS ? 0 : sb + 1;
        }
    } else {
#pragma unroll
        for (int s = 0; s < PS - 1; ++s) {
            if (s < chunks) ld.issue_x(s, s);
            cp_async_commit();
        }
        const uint32_t xlane = (uint32_t)((((lane & 7) + ((lane >> 4) & 1) * 8) * PXS + ((lane >> 3) & 1) * 8) * 2);
        const uint32_t al = abase + (uint32_t)((warp * MT * 32 + lane) * 16);
        int sb = 0;
        for (int c = 0; c < chunks; ++c) {
            cp_async_wait<PS - 2>();                   // stage c's member rows
            bar_sync<6, TC>();
            if (c + PS - 1 < chunks) ld.issue_x(c + PS - 1, sb == 0 ? PS - 1 : sb - 1);
            cp_async_commit();
            const uint32_t xs = sbase + (uint32_t)sb * C::STAGE_BYTES + xlane;
#pragma unroll
            for (int kk = 0; kk < PKT; ++kk) {
                const int s = kk & 1;
                if (s) bar_sync<2, T>(); else bar_sync<1, T>();   // slot s holds k tile c PKT + kk
                uint32_t a[MT][4];
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    const uint4 v = lds128(al + (uint32_t)(s * SLOT + i * 512));
                    a[i][0] = v.x; a[i][1] = v.y; a[i][2] = v.z; a[i][3] = v.w;
                }
                if (c * PKT + kk + 2 < ksteps) {       // fragments in registers: slot s is free for k tile + 2
                    if (s) bar_arrive<4, T>(); else bar_arrive<3, T>();
                }
#pragma unroll
                for (int jp = 0; jp < NP; ++jp) {
                    const uint32_t xa = xs + (uint32_t)((jp * 16 * PXS + kk * 16) * 2);
                    if (2 * jp + 1 < NBA) {
                        uint32_t r[4];
                        ldsm_x4(r, xa);
#pragma unroll
                        for (int i = 0; i < MT; ++i) {
                            mma16816(acc[i][2 * jp], a[i], r[0], r[1]);
                            mma16816(acc[i][2 * jp + 1], a[i], r[2], r[3]);
                        }
                    } else {
                        uint32_t r[2];
                        ldsm_x2(r, xa);
#pragma unroll
                        for (int i = 0; i < MT; ++i) mma16816(acc[i][2 * jp], a[i], r[0], r[1]);
                    }
                }
            }
            sb = sb + 1 == PS ? 0 : sb + 1;
        }
    }
}

// The item's main loop, then every thread of the block past it (the stages become the epilogue's rows). A
// warp-specialised block runs items of up to RD_MAX column blocks register-direct (decode-bound: all eight warps
// decode), its producer warps idle until the epilogue.
template <class C, int NBA>
__device__ __forceinline__ void mainloop(const Params& p, const int* xrows, int e, int cb, unsigned char* smem,
                                         float (&acc)[C::MT][NBA][4]) {
    if constexpr (C::WS && NBA > C::RD_MAX)
        mainloop_ws<C, NBA>(p, xrows, e, cb, smem, acc);
    else if ((int)threadIdx.x < C::TC)
        mainloop_rd<C, NBA>(p, xrows, e, cb, smem, acc);
    cp_async_wait<0>();
    __syncthreads();
}

// acc (this warp's m tiles, the 8-member blocks of half hf) -> E[member - PM / 2 hf][col], exl3.cu prompt_spill's
// places; col0: this warp's first column within E's columns.
template <class C, int NBA>
__device__ __forceinline__ void spill(float* E, int ES, int col0, const float (&acc)[C::MT][NBA][4], int hf) {
    constexpr int HB = C::NB / 2;
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < C::MT; ++i)
#pragma unroll
        for (int q = 0; q < HB; ++q) {
            const int nb = hf * HB + q;
            if (nb < NBA) {
                const int tok = q * 8 + 2 * t, col = col0 + i * 16 + g;
                E[(size_t)tok * ES + col] = acc[i][nb][0];
                E[(size_t)(tok + 1) * ES + col] = acc[i][nb][1];
                E[(size_t)tok * ES + col + 8] = acc[i][nb][2];
                E[(size_t)(tok + 1) * ES + col + 8] = acc[i][nb][3];
            }
        }
}

// Gate/up of one item (exl3.cu prompt_gateup_kernel's epilogue): Xd[pair][cb 128 ..] = fp16 of the down input
// rotation of GLM's limited SwiGLU of the rotated, scaled gate and up sums. Consumer warps spill, every warp of the
// block works rows.
template <class C, int NBA>
__device__ __forceinline__ void gateup_item(const Params& p, int e, int cb, int cnt, const int* pairs,
                                            const int* xrows, unsigned char* smem) {
    constexpr int PH = C::PM / 2, NWALL = C::T / 32;
    static_assert(C::MATS == 2, "a gate/up block holds both matrices (a one-matrix block: gateup_c2_item)");
    float acc[C::MT][NBA][4];
    mainloop<C, NBA>(p, xrows, e, cb, smem, acc);
    float* Eg = reinterpret_cast<float*>(smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool consumer = (int)threadIdx.x < C::TC;
    int mat, nt;
    tile_of<C>(consumer ? warp * C::MT : 0, cb, mat, nt);
    const int col0 = (nt - cb * C::TPM) * 16;
    const int N = p.N, n = cb * 128 + 4 * lane;
    float sg[4], su[4], sd[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sg[j] = __half2float(p.svh_g[(size_t)e * N + n + j]);
        su[j] = __half2float(p.svh_u[(size_t)e * N + n + j]);
        sd[j] = __half2float(p.suh_d[(size_t)e * N + n + j]);
    }
    const float limit = p.limit;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        if (hf * PH >= cnt) break;
        if (hf) __syncthreads();                       // the first half's rows are read
        if (consumer) spill<C, NBA>(Eg + (size_t)mat * PH * C::ES, C::ES, col0, acc, hf);
        __syncthreads();
        for (int i = warp; i < PH && hf * PH + i < cnt; i += NWALL) {
            const int pr = pairs[hf * PH + i];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * C::ES + 4 * lane + j];
                uv[j] = Eg[(size_t)(PH + i) * C::ES + 4 * lane + j];
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = fminf(bf16r(gv[j] * HAD_SCALE * sg[j]), limit);
                float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * su[j]), -limit), limit);
                float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                v[j] = act * sd[j];
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

// Down of one item (exl3.cu prompt_down_kernel's epilogue): Y[pair][the block's columns] fp32 = the rotated sums x
// svh, a 128-column Hadamard block at a time (two a 256-column block, one a 128-column block).
template <class C, int NBA>
__device__ __forceinline__ void down_item(const Params& p, int e, int cb, int cnt, const int* pairs,
                                          unsigned char* smem) {
    constexpr int PH = C::PM / 2, NWALL = C::T / 32, NBLK = C::TPM * 16 / 128, COLS = C::TPM * 16;
    float acc[C::MT][NBA][4];
    mainloop<C, NBA>(p, pairs, e, cb, smem, acc);
    float* Ed = reinterpret_cast<float*>(smem);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool consumer = (int)threadIdx.x < C::TC;
    const int D = p.N;
    float sv[NBLK][4];
#pragma unroll
    for (int blk = 0; blk < NBLK; ++blk) {
        const uint2 s = *reinterpret_cast<const uint2*>(p.svh_d + (size_t)e * D + cb * COLS + blk * 128 + 4 * lane);
        const half2 s01 = *reinterpret_cast<const half2*>(&s.x), s23 = *reinterpret_cast<const half2*>(&s.y);
        sv[blk][0] = __low2float(s01);
        sv[blk][1] = __high2float(s01);
        sv[blk][2] = __low2float(s23);
        sv[blk][3] = __high2float(s23);
    }
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        if (hf * PH >= cnt) break;
        if (hf) __syncthreads();
        if (consumer) spill<C, NBA>(Ed, C::ES, warp * C::MT * 16, acc, hf);
        __syncthreads();
        for (int i = warp; i < PH && hf * PH + i < cnt; i += NWALL) {
            const int pr = pairs[hf * PH + i];
#pragma unroll
            for (int blk = 0; blk < NBLK; ++blk) {
                const int n = cb * COLS + blk * 128 + 4 * lane;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = Ed[(size_t)i * C::ES + blk * 128 + 4 * lane + j];
                fwht128(v, lane);
                *reinterpret_cast<float4*>(p.y + (size_t)pr * D + n) =
                    make_float4(v[0] * HAD_SCALE * sv[blk][0], v[1] * HAD_SCALE * sv[blk][1],
                                v[2] * HAD_SCALE * sv[blk][2], v[3] * HAD_SCALE * sv[blk][3]);
            }
        }
    }
}

// Gate/up of a 128-member block as a pair of blocks in a cluster (k12_gateup_c2_kernel): rank 0 holds the gate's
// 128 columns of the block's Hadamard block, rank 1 the up's, each for all 128 members (one decoded tile feeds 16
// mma). The epilogue, a half (64 members) at a time: each rank spills its sums, then works half the half's rows,
// reading the gate row from rank 0's shared memory and the up row from rank 1's (distributed shared memory), with
// exl3.cu's formulas: the same values in, the same arithmetic, the same bits.
template <class C, int NBA>
__device__ __forceinline__ void gateup_c2_item(const Params& p, int e, int cb, int cnt, int rank, const int* pairs,
                                               const int* xrows, unsigned char* smem) {
    constexpr int PH = C::PM / 2, ES = C::ES, RH = PH / 2;
    float acc[C::MT][NBA][4];
    mainloop<C, NBA>(p, xrows, e, cb, smem, acc);
    cg::cluster_group cl = cg::this_cluster();
    float* Eown = reinterpret_cast<float*>(smem);
    const float* Eg = cl.map_shared_rank(Eown, 0);
    const float* Eu = cl.map_shared_rank(Eown, 1);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int N = p.N, n = cb * 128 + 4 * lane;
    float sg[4], su[4], sd[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        sg[j] = __half2float(p.svh_g[(size_t)e * N + n + j]);
        su[j] = __half2float(p.svh_u[(size_t)e * N + n + j]);
        sd[j] = __half2float(p.suh_d[(size_t)e * N + n + j]);
    }
    const float limit = p.limit;
#pragma unroll
    for (int hf = 0; hf < 2; ++hf) {
        if (hf * PH >= cnt) break;                     // both ranks: the same count, the same cluster barriers
        if (hf) cl.sync();                             // the partner has read this block's first half
        spill<C, NBA>(Eown, ES, warp * C::MT * 16, acc, hf);
        cl.sync();                                     // both ranks' sums of this half are in place
        for (int i = rank * RH + warp; i < (rank + 1) * RH && hf * PH + i < cnt; i += C::NW) {
            const int pr = pairs[hf * PH + i];
            float gv[4], uv[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                gv[j] = Eg[(size_t)i * ES + 4 * lane + j];
                uv[j] = Eu[(size_t)i * ES + 4 * lane + j];
            }
            fwht128(gv, lane);
            fwht128(uv, lane);
            float v[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float gg = fminf(bf16r(gv[j] * HAD_SCALE * sg[j]), limit);
                float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * su[j]), -limit), limit);
                float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
                v[j] = act * sd[j];
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
    cl.sync();                                         // the partner has read this block's last half: exit
}

// The item of this block (exl3.cu's grid: x = column block, y = launch slot); false: nothing to do.
template <class C>
__device__ __forceinline__ bool item_of(const Params& p, int& e, int& first, int& cnt) {
    if ((int)blockIdx.y >= p.counts[0]) return false;
    const int item = p.order ? p.order[blockIdx.y] : (int)blockIdx.y;
    if (item < 0) return false;                        // the order leaves the shared expert's items out
    e = p.items[3 * item];
    first = p.items[3 * item + 1];
    cnt = p.items[3 * item + 2];
    return e < p.E;                                    // the shared expert
}

// The item's main loop and epilogue compiled for its number of 8-member column blocks (1 .. NB).
#define K12_CASES(CALL)                                                                                           \
    if constexpr (C::NB == 16) {                                                                                  \
        switch ((cnt + 7) >> 3) {                                                                                 \
            case 1: CALL(1); break;                                                                               \
            case 2: CALL(2); break;                                                                               \
            case 3: CALL(3); break;                                                                               \
            case 4: CALL(4); break;                                                                               \
            case 5: CALL(5); break;                                                                               \
            case 6: CALL(6); break;                                                                               \
            case 7: CALL(7); break;                                                                               \
            case 8: CALL(8); break;                                                                               \
            case 9: CALL(9); break;                                                                               \
            case 10: CALL(10); break;                                                                             \
            case 11: CALL(11); break;                                                                             \
            case 12: CALL(12); break;                                                                             \
            case 13: CALL(13); break;                                                                             \
            case 14: CALL(14); break;                                                                             \
            case 15: CALL(15); break;                                                                             \
            default: CALL(16); break;                                                                             \
        }                                                                                                         \
    } else if constexpr (C::NB == 10) {                                                                           \
        switch ((cnt + 7) >> 3) {                                                                                 \
            case 1: CALL(1); break;                                                                               \
            case 2: CALL(2); break;                                                                               \
            case 3: CALL(3); break;                                                                               \
            case 4: CALL(4); break;                                                                               \
            case 5: CALL(5); break;                                                                               \
            case 6: CALL(6); break;                                                                               \
            case 7: CALL(7); break;                                                                               \
            case 8: CALL(8); break;                                                                               \
            case 9: CALL(9); break;                                                                               \
            default: CALL(10); break;                                                                             \
        }                                                                                                         \
    } else {                                                                                                      \
        static_assert(C::NB == 8, "8, 10 or 16 column blocks");                                                   \
        switch ((cnt + 7) >> 3) {                                                                                 \
            case 1: CALL(1); break;                                                                               \
            case 2: CALL(2); break;                                                                               \
            case 3: CALL(3); break;                                                                               \
            case 4: CALL(4); break;                                                                               \
            case 5: CALL(5); break;                                                                               \
            case 6: CALL(6); break;                                                                               \
            case 7: CALL(7); break;                                                                               \
            default: CALL(8); break;                                                                              \
        }                                                                                                         \
    }

template <class C>
__global__ void __launch_bounds__(C::T, C::CTAS) k12_gateup_kernel(const Params p) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int pairs_sh[C::PM], xrows_sh[C::PM];
    int e, first, cnt;
    if (!item_of<C>(p, e, first, cnt)) return;
    if ((int)threadIdx.x < C::PM) {
        const int q = (int)threadIdx.x < cnt ? p.members[first + threadIdx.x] : -1;
        pairs_sh[threadIdx.x] = q;
        xrows_sh[threadIdx.x] = q < 0 ? -1 : q / p.slots;
    }
    __syncthreads();
    const int cb = blockIdx.x;
#define K12_GU(NBA_) gateup_item<C, NBA_>(p, e, cb, cnt, pairs_sh, xrows_sh, smem)
    K12_CASES(K12_GU)
#undef K12_GU
}

template <class C>
__global__ void __launch_bounds__(C::T, C::CTAS) k12_down_kernel(const Params p) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int pairs_sh[C::PM];
    int e, first, cnt;
    if (!item_of<C>(p, e, first, cnt)) return;
    if ((int)threadIdx.x < C::PM) pairs_sh[threadIdx.x] = (int)threadIdx.x < cnt ? p.members[first + threadIdx.x] : -1;
    __syncthreads();
    const int cb = blockIdx.x;
#define K12_DN(NBA_) down_item<C, NBA_>(p, e, cb, cnt, pairs_sh, smem)
    K12_CASES(K12_DN)
#undef K12_DN
}

// Gate/up of 128-member blocks: grid (2 x N / 128, blocks), a cluster of the two blocks of one column block (x = 2 cb
// + rank: rank 0 the gate's 8 m tiles, rank 1 the up's). Both blocks of a cluster read the same list slot, so they
// return together or run together.
template <class C>
__global__ void __cluster_dims__(2, 1, 1) __launch_bounds__(C::T, C::CTAS) k12_gateup_c2_kernel(const Params p) {
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int pairs_sh[C::PM], xrows_sh[C::PM];
    int e, first, cnt;
    if (!item_of<C>(p, e, first, cnt)) return;
    if ((int)threadIdx.x < C::PM) {
        const int q = (int)threadIdx.x < cnt ? p.members[first + threadIdx.x] : -1;
        pairs_sh[threadIdx.x] = q;
        xrows_sh[threadIdx.x] = q < 0 ? -1 : q / p.slots;
    }
    __syncthreads();
    const int rank = (int)(blockIdx.x & 1), cb = (int)(blockIdx.x >> 1);
    Params q = p;
    q.t0 = rank ? p.t1 : p.t0;                         // this block's matrix
#define K12_GU2(NBA_) gateup_c2_item<C, NBA_>(q, e, cb, cnt, rank, pairs_sh, xrows_sh, smem)
    K12_CASES(K12_GU2)
#undef K12_GU2
}

// Blocks of up to BM members (80 or 128) from a 64-member plan, in plan order: each expert's member range (its items
// are consecutive, its members contiguous) cut from its first member, blocks[b] = (expert, first member, count),
// bcount[0] = the routed blocks (the shared expert's items, the plan's last, left out); the same blocks as a plan of
// BM-member items. One block of 1024 threads.
__global__ void __launch_bounds__(1024) k12_blocks_kernel(const int* __restrict__ items,
                                                          const int* __restrict__ counts, int* __restrict__ blocks,
                                                          int* __restrict__ bcount, int cap, int E, int BM) {
    __shared__ int wsum[32];
    __shared__ int routed;
    const int all = min(counts[0], cap);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) routed = 0;
    __syncthreads();
    for (int i = tid; i < all; i += 1024)              // items are in expert order: the routed ones come first
        if (items[3 * i] < E && (i + 1 == all || items[3 * (i + 1)] >= E)) routed = i + 1;
    __syncthreads();
    const int n = routed;
    const int per = (n + 1023) / 1024;
    const int lo = min(n, tid * per), hi = min(n, lo + per);
    // an expert's run of items starts at i: its members, and its blocks
    auto members_from = [&](int i) {
        const int e = items[3 * i];
        int c = 0;
        for (int j = i; j < n && items[3 * j] == e; ++j) c += items[3 * j + 2];
        return c;
    };
    auto starts_run = [&](int i) { return i == 0 || items[3 * (i - 1)] != items[3 * i]; };
    int nb = 0;
    for (int i = lo; i < hi; ++i)
        if (starts_run(i)) nb += (members_from(i) + BM - 1) / BM;
    int c = nb;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, c, o);
        if (lane >= o) c += u;
    }
    if (lane == 31) wsum[warp] = c;
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
    int b = (warp ? wsum[warp - 1] : 0) + c - nb;      // blocks before lo
    for (int i = lo; i < hi; ++i) {
        if (!starts_run(i)) continue;
        const int e = items[3 * i], first = items[3 * i + 1], m = members_from(i);
        for (int j = 0; j * BM < m; ++j, ++b) {
            blocks[3 * b] = e;
            blocks[3 * b + 1] = first + j * BM;
            blocks[3 * b + 2] = min(BM, m - j * BM);
        }
    }
    if (tid == 0) bcount[0] = wsum[31];
}

// exl3.cu prompt_order_kernel (the same launch order: single-item and multi-item experts' items merged evenly).
__global__ void __launch_bounds__(1024) k12_order_kernel(const int* __restrict__ items,
                                                         const int* __restrict__ counts, int* __restrict__ order,
                                                         int cap, int E) {
    __shared__ int wsum[32];
    __shared__ int routed;
    const int all = min(counts[0], cap);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0) routed = 0;
    __syncthreads();
    for (int i = tid; i < all; i += 1024)
        if (items[3 * i] < E && (i + 1 == all || items[3 * (i + 1)] >= E)) routed = i + 1;
    __syncthreads();
    const int n = routed;
    const int per = (n + 1023) / 1024;
    const int lo = min(n, tid * per), hi = min(n, lo + per);
    auto heavy = [&](int i) {
        const int e = items[3 * i];
        return (i > 0 && items[3 * (i - 1)] == e) || (i + 1 < n && items[3 * (i + 1)] == e);
    };
    int h = 0;
    for (int i = lo; i < hi; ++i) h += heavy(i);
    int v = h;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int u = __shfl_up_sync(0xffffffffu, v, o);
        if (lane >= o) v += u;
    }
    if (lane == 31) wsum[warp] = v;
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
    const int nh = wsum[31], nl = n - nh;
    int rh = (warp ? wsum[warp - 1] : 0) + v - h;
    for (int i = lo; i < hi; ++i) {
        if (heavy(i)) {
            const int r = rh++;
            const long long X = (2LL * r + 1) * nl;
            const int cl = X > nh ? (int)min((long long)nl, (X - nh + 2LL * nh - 1) / (2LL * nh)) : 0;
            order[r + cl] = i;
        } else {
            const int l = i - rh;
            const long long Y = (2LL * l + 1) * nh;
            const int ch = Y >= nl ? (int)min((long long)nh, (Y - nl) / (2LL * nl) + 1) : 0;
            order[l + ch] = i;
        }
    }
    for (int i = n + tid; i < all; i += 1024) order[i] = -1;
}

// The configurations (exl3_k12.py CONFIGS), a (gate/up, down) pair of kernels each:
//   0 k12      register-direct, four stages               1 k12-s3  register-direct, three stages
//   2 k12-ws   warp-specialised (8 consumer + 2 producer warps), three stages
//   3 k12-m128 128-member blocks: gate/up as cluster pairs (8 m tiles of one matrix x 128 members), down 128 x 128
//   4 k12-g128 the 128-member gate/up, k12's down          5 k12-d128 k12's gate/up, the 128-member down
//   6 k12-p80  k12's blocks (16 m tiles, 2 a warp) on blocks of up to 80 members: one block for an expert of up to
//              80 rows (no 1-16-member remainder item), a decoded tile feeds up to 10 mma
using GuRd4 = Cfg<2, 2, 8, 64, 2, 4>;
using DnRd4 = Cfg<1, 2, 8, 64, 2, 4>;
using GuRd3 = Cfg<2, 2, 8, 64, 2, 3>;
using DnRd3 = Cfg<1, 2, 8, 64, 2, 3>;
using GuWs3 = Cfg<2, 2, 8, 64, 2, 3, true, 2>;
using DnWs3 = Cfg<1, 2, 8, 64, 2, 3, true, 2>;
using Gu128 = Cfg<1, 1, 8, 128, 2, 3>;
using Dn128 = Cfg<1, 1, 8, 128, 2, 3>;
using GuP80 = Cfg<2, 2, 8, 80, 2, 4>;
using DnP80 = Cfg<1, 2, 8, 80, 2, 4>;

template <class G>
void launch_gu(const Params& p, int64_t N, int64_t slots_y, cudaStream_t stream) {
    static bool once = [] {
        cudaFuncSetAttribute(k12_gateup_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
        return true;
    }();
    (void)once;
    k12_gateup_kernel<G><<<dim3((unsigned)(N / 128), (unsigned)slots_y), G::T, G::SMEM, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class G>
void launch_gu_c2(const Params& p, int64_t N, int64_t slots_y, cudaStream_t stream) {
    static bool once = [] {
        cudaFuncSetAttribute(k12_gateup_c2_kernel<G>, cudaFuncAttributeMaxDynamicSharedMemorySize, G::SMEM);
        return true;
    }();
    (void)once;
    k12_gateup_c2_kernel<G><<<dim3((unsigned)(2 * N / 128), (unsigned)slots_y), G::T, G::SMEM, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <class Dn>
void launch_dn(const Params& p, int64_t D, int64_t slots_y, cudaStream_t stream) {
    static bool once = [] {
        cudaFuncSetAttribute(k12_down_kernel<Dn>, cudaFuncAttributeMaxDynamicSharedMemorySize, Dn::SMEM);
        return true;
    }();
    (void)once;
    k12_down_kernel<Dn><<<dim3((unsigned)(D / (Dn::TPM * 16)), (unsigned)slots_y), Dn::T, Dn::SMEM, stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

int64_t exl3_k12_configs() { return 7; }

// Xd [pairs, N] and Y [pairs, D] of a prompt plan's routed pairs, exl3.cu exl3_prompt_cuda's outputs for shared rows
// (xh = the rotated rows, one a token) and 64-member items. blk_buf (configurations 3-6): int32, at least
// 4 max_items + 2: the blocks (128 members, or 80 for 6), their count and their launch order.
void exl3_k12_prompt_cuda(const at::Tensor& xh, const at::Tensor& tg, const at::Tensor& tu, const at::Tensor& td,
                          const at::Tensor& items, const at::Tensor& counts, const at::Tensor& members,
                          const at::Tensor& svh_g, const at::Tensor& svh_u, const at::Tensor& suh_d,
                          const at::Tensor& svh_d, at::Tensor& xd, at::Tensor& y, int64_t D, int64_t N, int64_t E,
                          int64_t max_items, double limit, int64_t slots, const c10::optional<at::Tensor>& order_buf,
                          const c10::optional<at::Tensor>& blk_buf, int64_t cfg) {
    TORCH_CHECK(D % 256 == 0 && N % 128 == 0 && D % 32 == 0 && N % 32 == 0 && N >= 64,
                "k12: D a multiple of 256, N of 128");
    TORCH_CHECK(cfg >= 0 && cfg < exl3_k12_configs(), "k12: configuration 0 to 6, not ", cfg);
    auto stream = at::cuda::getCurrentCUDAStream();
    const bool gu_blk = cfg == 3 || cfg == 4 || cfg == 6, dn_blk = cfg == 3 || cfg == 5 || cfg == 6;  // on blocks
    const int BM = cfg == 6 ? 80 : 128;
    auto h = [](const at::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); };
    auto w = [](const at::Tensor& t) { return reinterpret_cast<const uint32_t*>(t.data_ptr()); };
    Params gu{h(xh), w(tg), w(tu), items.data_ptr<int>(), counts.data_ptr<int>(), members.data_ptr<int>(), nullptr,
              h(svh_g), h(svh_u), h(suh_d), h(svh_d), reinterpret_cast<half*>(xd.data_ptr()), y.data_ptr<float>(),
              (int)D, (int)N, (int)E, (int)slots, (float)limit, MCG_MASK};
    Params blk = gu;                                   // the same, reading the 128-member blocks
    if (!gu_blk || !dn_blk) {                          // a kernel on the plan's items: their launch order
        if (order_buf.has_value()) {
            TORCH_CHECK(order_buf->numel() >= max_items, "k12: order buffer too small");
            int* o = order_buf->data_ptr<int>();
            k12_order_kernel<<<1, 1024, 0, stream>>>(items.data_ptr<int>(), counts.data_ptr<int>(), o,
                                                     (int)max_items, (int)E);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            gu.order = o;
        }
    }
    if (gu_blk || dn_blk) {
        TORCH_CHECK(blk_buf.has_value() && blk_buf->scalar_type() == at::kInt && blk_buf->is_contiguous() &&
                        blk_buf->numel() >= 4 * max_items + 2, "k12: blocks buffer of 4 max_items + 2 int32");
        int* b = blk_buf->data_ptr<int>();
        int* bcount = b + 3 * max_items;
        k12_blocks_kernel<<<1, 1024, 0, stream>>>(items.data_ptr<int>(), counts.data_ptr<int>(), b, bcount,
                                                  (int)max_items, (int)E, BM);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        blk.items = b;
        blk.counts = bcount;
        blk.order = nullptr;
        if (order_buf.has_value()) {                   // the same launch order rule over the blocks
            int* bo = bcount + 2;
            k12_order_kernel<<<1, 1024, 0, stream>>>(b, bcount, bo, (int)max_items, (int)E);
            C10_CUDA_KERNEL_LAUNCH_CHECK();
            blk.order = bo;
        }
    }
    auto down = [&](Params p) {
        p.x = reinterpret_cast<const half*>(xd.data_ptr());
        p.t0 = p.t1 = w(td);
        p.K = (int)N;
        p.N = (int)D;
        return p;
    };
    switch (cfg) {
        case 0: launch_gu<GuRd4>(gu, N, max_items, stream); launch_dn<DnRd4>(down(gu), D, max_items, stream); break;
        case 1: launch_gu<GuRd3>(gu, N, max_items, stream); launch_dn<DnRd3>(down(gu), D, max_items, stream); break;
        case 2: launch_gu<GuWs3>(gu, N, max_items, stream); launch_dn<DnWs3>(down(gu), D, max_items, stream); break;
        case 3: launch_gu_c2<Gu128>(blk, N, max_items, stream); launch_dn<Dn128>(down(blk), D, max_items, stream); break;
        case 4: launch_gu_c2<Gu128>(blk, N, max_items, stream); launch_dn<DnRd4>(down(gu), D, max_items, stream); break;
        case 5: launch_gu<GuRd4>(gu, N, max_items, stream); launch_dn<Dn128>(down(blk), D, max_items, stream); break;
        default: launch_gu<GuP80>(blk, N, max_items, stream); launch_dn<DnP80>(down(blk), D, max_items, stream); break;
    }
}
