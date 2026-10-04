// GLM-5.3-Flash's routed EXL3 experts for decode windows of many rows (TF_GLM_EXL3_STREAM): every routed expert's
// trellis streamed once a window in long contiguous runs, the same bits as the decode kernel (exl3.cu, dec_kernel).
//
// Why: dec_kernel gives each block an (item, 128-column block, K split): its warps read 1 KB pieces of every k row
// (4 KB apart for gate/up, 32 KB apart for down), so a window's experts reach DRAM as thousands of short strided
// streams. Here a block takes a SLAB: a column slice (1, 2 or 4 KB: TF_GLM_EXL3_STREAM_SLICE) of a run of k rows of
// one expert matrix (gate/up: the 64 rows of one K split; down: every k row), loaded with 16-byte cp.async a stage of
// rows at a time, several stages in flight, into shared memory; the block's warps decode their own tiles from there.
// Narrow slices run several blocks a multiprocessor, so one block's gaps between slabs overlap the others' streaming.
//
// One persistent launch a layer: blocks take work from a device queue in order: first every item's gate/up slabs
// (2 matrices x 4 K splits x NI / slice column blocks), then every item's down slabs (D / slice blocks). The last
// gate/up slab of an item to finish runs the item's gate/up epilogue (exl3.cu's gateup_epilogue_kernel arithmetic) and
// raises the item's ready flag; a down slab waits for its item's flag (every gate/up slab is taken before any down slab
// and a taken slab belongs to a running block, so the wait always ends). The last block to leave resets the queue and
// the flags, so CUDA graphs replay the launch.
//
// Bits: each output is dec_kernel's (and grouped_kernel's) float chain: a warp's mma chain runs k tiles
// [split * KT / SK + c * KT / (SK * 4), + KT / (SK * 4)) from zero for chain c = 0..3 (dec_kernel's warp c), the
// chains are added in order (s = c0; s += c1; s += c2; s += c3: dec_kernel's red[0] + red[1] + ...), gate/up write
// that per split to Z as dec_kernel (fuse 0) does and the epilogue adds the splits from zero in split order; down
// takes 0 + s into its fused epilogue (dec_kernel fuse 1). The A fragments hold the same fp16 values (rows past an
// item's pairs are zeros, as load_pair gives), the B fragments come from the same decode, the same mma instruction
// multiplies them, and mma keeps rows independent. Only data movement and which block computes what change.
//
// Format and decode: exl3.cu (after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp); kernels of the GLM-5.3-Flash
// recipe (MiaAI-Lab, Apache-2.0) on TensorFold (Ash Hart and the TensorFold contributors, Apache-2.0).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace {

constexpr float HAD_SCALE = 0.08838834764831845f;   // 1 / sqrt(128)

// ---- exl3.cu's arithmetic, verbatim (the same decode, mma, transforms and roundings) ----------------------------

__device__ __forceinline__ uint32_t mcg2(uint32_t s0, uint32_t s1) {
    uint32_t x0 = s0 * 0xCBAC1FEDu;
    uint32_t x1 = s1 * 0xCBAC1FEDu;
    x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
    uint32_t lo = __byte_perm(x0, x1, 0x5410);
    uint32_t hi = __byte_perm(x0, x1, 0x7632);
    half2 r = __hadd2(*reinterpret_cast<half2*>(&lo), *reinterpret_cast<half2*>(&hi));
    return *reinterpret_cast<uint32_t*>(&r);
}

// exl3.cu's decode_tile_p: decode_tile's states with fewer integer instructions (the same states, so the same values)
__device__ __forceinline__ void decode_tile(uint32_t w, int lane, uint32_t (&b0)[2], uint32_t (&b1)[2]) {
    const uint32_t p = __shfl_sync(0xffffffffu, w, (lane + 31) & 31);
    const uint32_t s = __funnelshift_r(w, p, 20);
    const uint32_t t = w >> 4;
    b0[0] = mcg2(__byte_perm(s, 0u, 0x4421), (s >> 4) & 0xffffu);          // (s >> 8) & 0xffff, (s >> 4) & 0xffff
    b0[1] = mcg2(s & 0xffffu, w >> 16);
    b1[0] = mcg2(__byte_perm(t, 0u, 0x4421), __byte_perm(w, 0u, 0x4421));   // (w >> 12) & 0xffff, (w >> 8) & 0xffff
    b1[1] = mcg2(__byte_perm(t, 0u, 0x4410), w & 0xffffu);                  // (w >> 4) & 0xffff, w & 0xffff
}

__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

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

// ---- data movement --------------------------------------------------------------------------------------------------

// Copies: plain and zero-fill 16-byte cp.async and ldmatrix x4, the forms the engine's other kernels already run
// (qmm_frag.cuh, experts.cuh, kda_chunk.cu, qmm.cu). No L2 cache-hint copies: with a hint (createpolicy +
// cp.async.L2::cache_hint) ptxas folds this loop's shared base into the hint's descriptor registers, a form only the
// hinted kernels had, and their first launch on the RTX PRO 6000 (sm_120) stopped with "illegal instruction" (Xid 13
// Illegal Instruction Parameter). tests/K3/triage_exl3_stream.py runs that form alone, in a process of its own.
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
__device__ __forceinline__ int ld_acquire(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release(int* p, int v) {
    asm volatile("st.release.gpu.global.b32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}

// ---- the launch -----------------------------------------------------------------------------------------------------

constexpr int NTW = 2;                    // n tiles a warp
constexpr int SK = 4;                     // gate/up K splits (exl3_mm.GATEUP_CFG), down 1 (DOWN_CFG)
constexpr int CHAINS = 4;                 // chains a split: dec_kernel's warps along K

// A block's geometry (TF_GLM_EXL3_STREAM_SLICE): W warps of NTW n tiles take a column slice of BT = 2 W tiles, BT * 128
// bytes of every k row: W 16 (4 KB slices, one 512-thread block a multiprocessor), 8 (2 KB, two blocks) or 4 (1 KB,
// four blocks): with several blocks a multiprocessor, one block's gaps between slabs (the claim, the item's rows, the
// pipeline's first stage, the partials and counts) overlap the other blocks' streaming. Columns are independent: every
// output keeps its chain whatever slice holds its column.
template <int W>
struct Geo {
    static constexpr int THREADS = W * 32;
    static constexpr int BT = W * NTW;            // n tiles a block
    static constexpr int ROWB = BT * 128;         // bytes of a k row a block reads
    static constexpr int PR = ROWB / 16;          // 16-byte pieces of it
    static constexpr int ES = BT * 16 + 8;        // the down epilogue's row stride in floats
    static constexpr int MINB = 16 / W;           // blocks a multiprocessor the launch bounds ask for (128 registers)
    static_assert(W == 4 || W == 8 || W == 16, "slices of 1, 2 or 4 KB");
};

struct Args {
    const half* X0;                       // gate input: [rows, D] (xrow) or [pairs, D]
    const half* X1;                       // up input (xrow: X0)
    const uint32_t* Tg;                   // trellis words [E, D/16, NI/16, 32]
    const uint32_t* Tu;
    const uint32_t* Td;                   // [E, NI/16, D/16, 32]
    const int* items;                     // [max_items, 3]: expert, first member, count
    const int* counts;                    // [2]: items, distinct experts
    const int* members;                   // pair ids grouped by expert
    const half* svh_g;                    // [E, NI]
    const half* svh_u;
    const half* suh_d;                    // [E, NI]
    const half* svh_d;                    // [E, D]
    float* Z;                             // partials (a window's pairs P): coarse [2][SK][P][NI]; fine
                                          // [2][SK][4][P][NI] then [4][P][D]
    half* xd;                             // [pairs, NI]: gate/up output, down input
    float* y;                             // [pairs, D]
    int* state;                           // queue head, blocks out, gate/up counts [cap], ready flags [cap], down
                                          // counts [cap][D / (BT * 16)] (fine): zeros before and after every launch
    int D, NI, E, P, slots, max_items;
    float limit;
    int xrow, fine;
};

template <int RS, int W>
struct Stage {
    static constexpr int APITCH = RS * 32 + 16;              // bytes a row of A (RS k tiles + 16: no bank conflicts)
    static constexpr int BYTES = RS * Geo<W>::ROWB + 16 * APITCH;
};

template <int RS, int STAGES, int W>
constexpr int smem_bytes() {
    constexpr int pipe = STAGES * Stage<RS, W>::BYTES;
    constexpr int epi = 16 * Geo<W>::ES * 4;
    return pipe > epi ? pipe : epi;
}

// One slab: rows k0 .. k0 + nrows - 1 of an expert matrix (row k at wrow + k * pitch words, the block's ROWB bytes of
// it), A = the item's member rows (arow[i], null past the item's pairs) at k values 16 k .. 16 k + 15; chains of CL
// rows added in order into `sum` (this warp's NTW tiles, rows g / g + 8 of the 16-row mma tile). Each thread moves
// PER 16-byte pieces of every stage's k rows (RS * PR pieces over the block's threads) and threads below ACH one
// piece of A; a stage lands, every warp decodes its tiles of its RS rows, and the slot is refilled STAGES - 1 ahead.
template <int RS, int STAGES, int W>
__device__ __forceinline__ void run_slab(unsigned char* smem, const uint32_t* wrow, size_t pitch, int k0, int nrows,
                                         int CL, const half* const* arow, const half* adummy,
                                         float (&sum)[NTW][2][4]) {
    using G = Geo<W>;
    using S = Stage<RS, W>;
    constexpr int PER = RS * G::PR / G::THREADS;             // weight pieces a thread a stage
    static_assert(RS * G::PR % G::THREADS == 0, "a stage's weight pieces split evenly over the threads");
    constexpr int ACH = 16 * RS * 2;                         // A: 16 rows x RS k tiles x 2 pieces of 16 bytes
    static_assert(ACH <= G::THREADS, "a thread a piece of A");
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int nst = nrows / RS;
    // this thread's weight pieces: piece q = tid + j * THREADS of a stage, row q / PR, bytes 16 (q % PR) of it
    const char* wsrc = reinterpret_cast<const char*>(wrow + (size_t)k0 * pitch);
    const size_t wpitch = pitch * 4;                         // bytes from one k row to the next
    const unsigned sbase = (unsigned)__cvta_generic_to_shared(smem);
    const int ai = tid / (RS * 2), apart = tid % (RS * 2);
    const half* aptr = tid < ACH ? arow[ai] : nullptr;
    const char* asrc = reinterpret_cast<const char*>(aptr ? aptr + (size_t)k0 * 16 + apart * 8 : adummy);
    const unsigned adst = sbase + RS * G::ROWB + ai * S::APITCH + apart * 16;
    const int abytes = aptr != nullptr ? 16 : 0;
    auto load = [&](int slot, int st) {
        const unsigned off = slot * S::BYTES;
        const char* src = wsrc + (size_t)(st * RS) * wpitch;
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int q = tid + j * G::THREADS;
            const unsigned d = sbase + off + q * 16;
            const char* g = src + (size_t)(q / G::PR) * wpitch + (q % G::PR) * 16;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(g));
        }
        if (tid < ACH)
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(adst + off),
                         "l"(abytes ? asrc + st * RS * 32 : asrc), "r"(abytes));
    };
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < nst) load(st, st);
        cp_async_commit();
    }
    const int arow_l = (lane & 7) + ((lane >> 3) & 1) * 8;   // ldmatrix: this lane's A row and k half
    const unsigned a_lane = RS * G::ROWB + arow_l * S::APITCH + (lane >> 4) * 16;
    const unsigned w_lane = ((warp * NTW) * 32 + lane) * 4;
    int st = 0;
    for (int c = 0; c * CL < nrows; ++c) {                   // chain c: rows c * CL .. c * CL + CL - 1, from zero
        float acc[NTW][2][4];
#pragma unroll
        for (int i = 0; i < NTW; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][h][e] = 0.f;
        for (int left = CL; left > 0; left -= RS, ++st) {    // RS divides CL: chains end on stage ends
            cp_async_wait<STAGES - 2>();
            __syncthreads();                                 // stage st landed; stage st - 1 is free again
            if (st + STAGES - 1 < nst) load((st + STAGES - 1) % STAGES, st + STAGES - 1);
            cp_async_commit();
            const unsigned char* base = smem + (st % STAGES) * S::BYTES;
#pragma unroll
            for (int rr = 0; rr < RS; ++rr) {
                uint32_t a[4];
                ldsm_x4(a, base + a_lane + rr * 32);
                uint32_t w[NTW];
#pragma unroll
                for (int i = 0; i < NTW; ++i)
                    w[i] = *reinterpret_cast<const uint32_t*>(base + rr * G::ROWB + w_lane + i * 128);
#pragma unroll
                for (int i = 0; i < NTW; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile(w[i], lane, b0, b1);
                    mma16816(acc[i][0], a, b0);
                    mma16816(acc[i][1], a, b1);
                }
            }
        }
        // dec_kernel adds its warps' chains in warp order: s = c0, then s += c1, s += c2, s += c3
#pragma unroll
        for (int i = 0; i < NTW; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int e = 0; e < 4; ++e) sum[i][h][e] = c == 0 ? acc[i][h][e] : sum[i][h][e] + acc[i][h][e];
    }
    cp_async_wait<0>();
    __syncthreads();                                         // every warp is done with the stages
}

// The gate/up epilogue of an item (its last gate/up slab runs it): gateup_epilogue_kernel's arithmetic on each pair
// and 128 columns. Each split's sum S is the coarse slab's register sum, or (fine) its four chains added in the same
// order from the partials; g = 0 + S0 + S1 + S2 + S3 in split order. Partials are read through L2 (other blocks wrote
// them in this launch).
template <int W>
__device__ __forceinline__ void gateup_epilogue(const Args& a, int e, const int* rows_sh, int cnt) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int NI = a.NI, P = a.P;
    const int nb = NI / 128;
    for (int task = warp; task < cnt * nb; task += W) {
        const int p = rows_sh[task / nb];
        const int n = (task % nb) * 128 + 4 * lane;
        float gv[4], uv[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float sg = 0.f, su = 0.f;
            for (int s = 0; s < SK; ++s) {
                float vg, vu;
                if (a.fine) {
                    const float* zg = a.Z + ((size_t)((0 * SK + s) * CHAINS) * P + p) * NI + n + j;
                    const float* zu = a.Z + ((size_t)((1 * SK + s) * CHAINS) * P + p) * NI + n + j;
                    const size_t cs = (size_t)P * NI;                // one chain's partials to the next
                    vg = __ldcg(zg);
                    vu = __ldcg(zu);
#pragma unroll
                    for (int c = 1; c < CHAINS; ++c) {
                        vg += __ldcg(zg + c * cs);
                        vu += __ldcg(zu + c * cs);
                    }
                } else {
                    vg = __ldcg(a.Z + ((size_t)(0 * SK + s) * P + p) * NI + n + j);
                    vu = __ldcg(a.Z + ((size_t)(1 * SK + s) * P + p) * NI + n + j);
                }
                sg += vg;
                su += vu;
            }
            gv[j] = sg;
            uv[j] = su;
        }
        fwht128(gv, lane);
        fwht128(uv, lane);
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float gg = fminf(bf16r(gv[j] * HAD_SCALE * __half2float(a.svh_g[(size_t)e * NI + n + j])), a.limit);
            float uu = fminf(fmaxf(bf16r(uv[j] * HAD_SCALE * __half2float(a.svh_u[(size_t)e * NI + n + j])), -a.limit),
                             a.limit);
            float act = bf16r(bf16r(gg / (1.f + expf(-gg))) * uu);
            v[j] = act * __half2float(a.suh_d[(size_t)e * NI + n + j]);
        }
        fwht128(v, lane);
        half* o = a.xd + (size_t)p * NI + n;
#pragma unroll
        for (int j = 0; j < 4; ++j) o[j] = __float2half_rn(v[j] * HAD_SCALE);
    }
}

// dec_kernel's fused down epilogue for one 128-column block of one pair: v = 0 + s, rotated, times svh -> Y.
__device__ __forceinline__ void down_out(const Args& a, int e, int p, int n, const float (&s)[4]) {
    const int lane = threadIdx.x & 31;
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float z = 0.f;
        z += s[j];
        v[j] = z;
    }
    fwht128(v, lane);
    float* o = a.y + (size_t)p * a.D + n;
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = v[j] * HAD_SCALE * __half2float(a.svh_d[(size_t)e * a.D + n + j]);
}

template <int RS, int STAGES, bool PDL, int W>
__global__ void __launch_bounds__(Geo<W>::THREADS, Geo<W>::MINB) stream_kernel(const Args a) {
    using G = Geo<W>;
    extern __shared__ __align__(16) unsigned char smem[];
    __shared__ int rows_sh[16];
    __shared__ const half* arow_sh[16];
    __shared__ int work_sh, flag_sh;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    // a programmatic dependent launch (TF_GLM_EXL3_STREAM_PDL; only its own instantiation has the instruction): the
    // plan and inputs of earlier launches before the first read
    if constexpr (PDL) asm volatile("griddepcontrol.wait;\n" ::: "memory");
    const int D = a.D, NI = a.NI, P = a.P;
    const int cbg = NI / (G::BT * 16), cbd = D / (G::BT * 16);   // column blocks of gate/up and of down
    const int ch = a.fine ? CHAINS : 1;                      // slabs a chain range splits into
    const int GU = 2 * SK * ch * cbg, DN = cbd * ch;         // slabs an item, each phase
    const int n_items = min(a.counts[0], a.max_items);
    const int phase1 = n_items * GU, total = phase1 + n_items * DN;
    int* head = a.state;
    int* outc = a.state + 1;
    int* cnt = a.state + 2;
    int* ready = cnt + a.max_items;
    int* cntd = ready + a.max_items;
    float* Zd = a.Z + (size_t)2 * SK * CHAINS * P * NI;      // fine: the down chains' partials

    if (tid == 0) work_sh = atomicAdd(head, 1);
    __syncthreads();
    for (;;) {
        const int w = work_sh;
        if (w >= total) break;
        const bool gu = w < phase1;
        const int item = gu ? w / GU : (w - phase1) / DN;
        const int sub = gu ? w % GU : (w - phase1) % DN;
        const int e = a.items[3 * item], first = a.items[3 * item + 1], cnt_i = a.items[3 * item + 2];
        if (e < a.E) {
            // gate/up: sub = ((mat * SK + split) * ch + chain) * cbg + cb; down: sub = cb * ch + chain
            const int cb = gu ? sub % cbg : sub / ch;
            const int chain = gu ? (sub / cbg) % ch : sub % ch;
            const int ms = gu ? sub / (cbg * ch) : 0;          // mat * SK + split
            const int mat = ms / SK, split = ms % SK;
            if (!gu && tid == 0)                             // the item's gate/up epilogue has run
                while (ld_acquire(ready + item) == 0) __nanosleep(128);
            if (tid < 16) {
                const int p = tid < cnt_i ? a.members[first + tid] : -1;
                rows_sh[tid] = p;
                const half* r = nullptr;
                if (p >= 0) r = gu ? (a.xrow ? a.X0 + (size_t)(p / a.slots) * D : (mat ? a.X1 : a.X0) + (size_t)p * D)
                                   : a.xd + (size_t)p * NI;
                arow_sh[tid] = r;
            }
            __syncthreads();
            float sum[NTW][2][4] = {};
            if (gu) {
                const int KT = D >> 4, NT = NI >> 4, rows = KT / SK, CL = rows / CHAINS;
                const uint32_t* T = mat ? a.Tu : a.Tg;
                const uint32_t* wrow = T + (((size_t)e * KT) * NT + cb * G::BT) * 32;
                // coarse: the split's 64 rows, its four chains added in registers; fine: one chain's rows
                if (a.fine) run_slab<RS, STAGES, W>(smem, wrow, (size_t)NT * 32, split * rows + chain * CL, CL, CL,
                                                 arow_sh, a.X0, sum);
                else run_slab<RS, STAGES, W>(smem, wrow, (size_t)NT * 32, split * rows, rows, CL, arow_sh, a.X0, sum);
                // partials: coarse Z[mat][split][pair][n] as dec_kernel (fuse 0) writes them; fine per chain
                float* Zs = a.Z + (size_t)(a.fine ? ms * CHAINS + chain : ms) * P * NI;
#pragma unroll
                for (int i = 0; i < NTW; ++i)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int col = cb * G::BT * 16 + (warp * NTW + i) * 16 + h * 8 + 2 * t;
                        const int p0 = rows_sh[g], p1 = rows_sh[g + 8];
                        if (p0 >= 0) *reinterpret_cast<float2*>(Zs + (size_t)p0 * NI + col) =
                                         make_float2(sum[i][h][0], sum[i][h][1]);
                        if (p1 >= 0) *reinterpret_cast<float2*>(Zs + (size_t)p1 * NI + col) =
                                         make_float2(sum[i][h][2], sum[i][h][3]);
                    }
                __threadfence();                             // this block's partials before its count
                __syncthreads();
                if (tid == 0) flag_sh = atomicAdd(cnt + item, 1) == GU - 1;
                __syncthreads();
                if (flag_sh) {                               // the item's last gate/up slab: its epilogue
                    __threadfence();                         // every slab's partials are visible from here
                    gateup_epilogue<W>(a, e, rows_sh, cnt_i);
                    __threadfence();
                    __syncthreads();
                    if (tid == 0) {
                        cnt[item] = 0;                       // the next launch counts from zero
                        st_release(ready + item, 1);
                    }
                }
            } else {
                const int KT = NI >> 4, NT = D >> 4, CL = KT / CHAINS;
                const uint32_t* wrow = a.Td + (((size_t)e * KT) * NT + cb * G::BT) * 32;
                constexpr int HB = G::BT * 16 / 128;         // 128-column blocks a slab: 1, 2 or 4
                if (a.fine) {
                    run_slab<RS, STAGES, W>(smem, wrow, (size_t)NT * 32, chain * CL, CL, CL, arow_sh, a.xd, sum);
                    float* Zc = Zd + (size_t)chain * P * D;
#pragma unroll
                    for (int i = 0; i < NTW; ++i)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int col = cb * G::BT * 16 + (warp * NTW + i) * 16 + h * 8 + 2 * t;
                            const int p0 = rows_sh[g], p1 = rows_sh[g + 8];
                            if (p0 >= 0) *reinterpret_cast<float2*>(Zc + (size_t)p0 * D + col) =
                                             make_float2(sum[i][h][0], sum[i][h][1]);
                            if (p1 >= 0) *reinterpret_cast<float2*>(Zc + (size_t)p1 * D + col) =
                                             make_float2(sum[i][h][2], sum[i][h][3]);
                        }
                    __threadfence();
                    __syncthreads();
                    int* c = cntd + (size_t)item * cbd + cb;
                    if (tid == 0) flag_sh = atomicAdd(c, 1) == CHAINS - 1;
                    __syncthreads();
                    if (flag_sh) {                           // the last chain of these 512 columns: s = c0 + c1 + ...
                        __threadfence();
                        for (int task = warp; task < cnt_i * HB; task += W) {
                            const int p = rows_sh[task / HB];
                            const int n = cb * G::BT * 16 + (task % HB) * 128 + 4 * lane;
                            float s[4];
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const float* z = Zd + (size_t)p * D + n + j;
                                float v = __ldcg(z);
#pragma unroll
                                for (int k = 1; k < CHAINS; ++k) v += __ldcg(z + (size_t)k * P * D);
                                s[j] = v;
                            }
                            down_out(a, e, p, n, s);
                        }
                        if (tid == 0) *c = 0;                // the next launch counts from zero
                    }
                } else {
                    run_slab<RS, STAGES, W>(smem, wrow, (size_t)NT * 32, 0, KT, CL, arow_sh, a.xd, sum);
                    float* E = reinterpret_cast<float*>(smem);   // the stages are free (run_slab's last barrier)
#pragma unroll
                    for (int i = 0; i < NTW; ++i)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const int col = (warp * NTW + i) * 16 + h * 8 + 2 * t;
                            *reinterpret_cast<float2*>(E + g * G::ES + col) = make_float2(sum[i][h][0], sum[i][h][1]);
                            *reinterpret_cast<float2*>(E + (g + 8) * G::ES + col) =
                                make_float2(sum[i][h][2], sum[i][h][3]);
                        }
                    __syncthreads();
                    for (int task = warp; task < cnt_i * HB; task += W) {
                        const int i = task / HB, c = (task % HB) * 128 + 4 * lane;
                        const float4 s4 = *reinterpret_cast<const float4*>(E + i * G::ES + c);
                        const float s[4] = {s4.x, s4.y, s4.z, s4.w};
                        down_out(a, e, rows_sh[i], cb * G::BT * 16 + c, s);
                    }
                }
            }
        }
        __syncthreads();                                     // rows_sh, arow_sh, flag_sh and E are free again
        if (tid == 0) work_sh = atomicAdd(head, 1);
        __syncthreads();
    }
    // leaving: the last block out resets the queue, its own count and the flags for the next launch
    if (tid == 0) flag_sh = atomicAdd(outc, 1) == (int)gridDim.x - 1;
    __syncthreads();
    if (flag_sh) {
        __threadfence();
        for (int i = tid; i < a.max_items; i += G::THREADS) ready[i] = 0;
        if (tid == 0) {
            *head = 0;
            *outc = 0;
        }
    }
}


// ---- a read probe (tests and timing only): every byte of the listed experts' three matrices once, 16-byte loads ----
// in a grid-stride loop, four in flight a thread; the bandwidth the card gives these bytes read plainly (no decode).
__global__ void __launch_bounds__(256) probe_kernel(const uint4* __restrict__ Tg, const uint4* __restrict__ Tu,
                                                    const uint4* __restrict__ Td, const int* __restrict__ ids, int n,
                                                    long long per, uint32_t* __restrict__ out) {
    const long long total = (long long)n * 3 * per;          // 16-byte pieces: per a matrix of an expert
    uint32_t x = 0;
    const long long stride = (long long)gridDim.x * blockDim.x;
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    for (; c + 3 * stride < total; c += 4 * stride) {
        uint4 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const long long q = c + u * stride;
            const long long seg = q / per, off = q % per;
            const int e = ids[seg / 3], m = (int)(seg % 3);
            const uint4* base = m == 0 ? Tg : m == 1 ? Tu : Td;
            v[u] = __ldg(base + (size_t)e * per + off);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) x ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
    }
    for (; c < total; c += stride) {
        const long long seg = c / per, off = c % per;
        const int e = ids[seg / 3], m = (int)(seg % 3);
        const uint4* base = m == 0 ? Tg : m == 1 ? Tu : Td;
        const uint4 v = __ldg(base + (size_t)e * per + off);
        x ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    out[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = x;
}

// ---- the expert L2 prefetch (TF_GLM_L2PF_EXPERT_MB): right after routing, on the L2 prefetcher's side stream ------
// The window's routed experts' trellis, in the order the streamed kernel reads it (every item's gate and up, then every
// item's down; an expert of several items once), brought into L2 with cp.async.bulk.prefetch.L2 in pieces of PIECE
// bytes, up to `budget` bytes, while the main stream runs the shared expert and the input rotation. It only reads.
constexpr long long PF_PIECE = 32 << 10;

__global__ void __launch_bounds__(128) expert_prefetch_kernel(const int* __restrict__ items,
                                                              const int* __restrict__ counts, const char* Tg,
                                                              const char* Tu, const char* Td, int E, int max_items,
                                                              long long mat_bytes, long long budget) {
    const int n_items = min(counts[0], max_items);
    const long long per = (mat_bytes + PF_PIECE - 1) / PF_PIECE;          // pieces a matrix
    const long long gu_pieces = (long long)n_items * 2 * per;
    const long long total = min(gu_pieces + (long long)n_items * per, (budget + PF_PIECE - 1) / PF_PIECE);
    for (long long q = (long long)blockIdx.x * blockDim.x + threadIdx.x; q < total;
         q += (long long)gridDim.x * blockDim.x) {
        const bool gu = q < gu_pieces;
        const long long r = gu ? q : q - gu_pieces;
        const long long m = r / per;                                       // gu: item * 2 + mat; down: item
        const long long off = (r % per) * PF_PIECE;
        const int item = (int)(gu ? m / 2 : m);
        const int e = items[3 * item];
        if (e >= E || (item > 0 && items[3 * (item - 1)] == e)) continue;  // the shared expert; an expert's 2nd item
        const char* base = gu ? (m % 2 ? Tu : Tg) : Td;
        const char* a = base + (size_t)e * mat_bytes + off;
        const unsigned n = (unsigned)min(PF_PIECE, mat_bytes - off);
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;\n" ::"l"(a), "r"(n) : "memory");
    }
}

template <int RS, int STAGES, bool PDL, int W>
void launch(const Args& args, int blocks, cudaStream_t stream) {
    using G = Geo<W>;
    constexpr int SMEM = smem_bytes<RS, STAGES, W>();
    auto fn = stream_kernel<RS, STAGES, PDL, W>;
    static int per_sm = [&] {
        cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
        int n = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, fn, G::THREADS, SMEM);
        return n > 0 ? n : 1;
    }();
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int cbg = args.NI / (G::BT * 16), cbd = args.D / (G::BT * 16), ch = args.fine ? CHAINS : 1;
    const long long work = (long long)args.max_items * (2 * SK * cbg + cbd) * ch;
    long long grid = blocks > 0 ? blocks : (long long)per_sm * sms;
    grid = std::min<long long>(std::max<long long>(grid, 1), std::max<long long>(work, 1));
    if constexpr (PDL) {
        // programmatic dependent launch: this grid's blocks may be scheduled before the previous launch has finished
        // and wait for it (griddepcontrol.wait) before their first read. The kernel never triggers its dependents
        // early: a persistent grid's later blocks must be able to start.
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3((unsigned)grid);
        cfg.blockDim = dim3(G::THREADS);
        cfg.dynamicSmemBytes = SMEM;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, fn, args));
    } else {
        fn<<<(unsigned)grid, G::THREADS, SMEM, stream>>>(args);
    }
}

}  // namespace

// Routed experts of a decode window, one launch: Xd [pairs, NI] and Y [pairs, D] (fp32) for every routed pair of the
// plan (P: the window's pairs, rows x slots); Z holds the partials (coarse: each split's; fine: each chain's, gate/up
// and down); state zeroed once (the kernel leaves it zeroed). fine: slabs of one chain (small windows: 4x the slabs);
// rs / stages the pipeline (rows a stage, stages); blocks the grid (0: as many as fit at once); pdl a programmatic
// dependent launch. All data movement and placement only: the same bits.
void exl3_stream_cuda(const at::Tensor& X0, const at::Tensor& X1, const at::Tensor& Tg, const at::Tensor& Tu,
                      const at::Tensor& Td, const at::Tensor& items, const at::Tensor& counts,
                      const at::Tensor& members, const at::Tensor& svh_g, const at::Tensor& svh_u,
                      const at::Tensor& suh_d, const at::Tensor& svh_d, at::Tensor& Z, at::Tensor& xd, at::Tensor& y,
                      at::Tensor& state, int64_t D, int64_t NI, int64_t E, int64_t P, int64_t slots,
                      int64_t max_items, double limit, bool xrow, bool fine, int64_t rs, int64_t stages,
                      int64_t blocks, bool pdl, int64_t warps) {
    Args a;
    a.X0 = reinterpret_cast<const half*>(X0.data_ptr());
    a.X1 = reinterpret_cast<const half*>(X1.data_ptr());
    a.Tg = reinterpret_cast<const uint32_t*>(Tg.data_ptr());
    a.Tu = reinterpret_cast<const uint32_t*>(Tu.data_ptr());
    a.Td = reinterpret_cast<const uint32_t*>(Td.data_ptr());
    a.items = items.data_ptr<int>();
    a.counts = counts.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.svh_g = reinterpret_cast<const half*>(svh_g.data_ptr());
    a.svh_u = reinterpret_cast<const half*>(svh_u.data_ptr());
    a.suh_d = reinterpret_cast<const half*>(suh_d.data_ptr());
    a.svh_d = reinterpret_cast<const half*>(svh_d.data_ptr());
    a.Z = Z.data_ptr<float>();
    a.xd = reinterpret_cast<half*>(xd.data_ptr());
    a.y = y.data_ptr<float>();
    a.state = state.data_ptr<int>();
    a.D = (int)D;
    a.NI = (int)NI;
    a.E = (int)E;
    a.P = (int)P;
    a.slots = (int)slots;
    a.max_items = (int)max_items;
    a.limit = (float)limit;
    a.xrow = xrow ? 1 : 0;
    a.fine = fine ? 1 : 0;
    auto stream = at::cuda::getCurrentCUDAStream();
    const int b = (int)blocks;
#define GO_W(RS_, ST_, W_)                                                                                         \
    do {                                                                                                           \
        if (pdl) launch<RS_, ST_, true, W_>(a, b, stream); else launch<RS_, ST_, false, W_>(a, b, stream);         \
    } while (0)
#define GO(RS_, ST_)                                                                                               \
    do {                                                                                                           \
        if (warps == 4) GO_W(RS_, ST_, 4); else if (warps == 8) GO_W(RS_, ST_, 8); else GO_W(RS_, ST_, 16);        \
    } while (0)
    TORCH_CHECK(warps == 4 || warps == 8 || warps == 16, "streamed experts: blocks of 4, 8 or 16 warps, not ", warps);
    if (rs == 2 && stages == 6) GO(2, 6);
    else if (rs == 2 && stages == 4) GO(2, 4);
    else if (rs == 4 && stages == 3) GO(4, 3);
    else if (rs == 4 && stages == 4) GO(4, 4);
    else TORCH_CHECK(false, "streamed experts: pipelines 4x3, 4x4, 2x4 or 2x6 (rows a stage x stages), not ",
                     rs, "x", stages);
#undef GO
#undef GO_W
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The read probe: ids [n] int32 (distinct experts), Tg / Tu / Td stacked [E, ...] of equal per-expert size; out
// uint32 [blocks * 256] (blocks: 0 = four a multiprocessor).
void exl3_probe_cuda(const at::Tensor& ids, const at::Tensor& Tg, const at::Tensor& Tu, const at::Tensor& Td,
                     at::Tensor& out, int64_t blocks) {
    const long long per = (long long)(Tg.numel() / Tg.size(0)) * 4 / 16;
    const int sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int grid = blocks > 0 ? (int)blocks : 4 * sms;
    TORCH_CHECK(out.numel() >= (int64_t)grid * 256, "probe: out holds a word a thread");
    probe_kernel<<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint4*>(Tg.data_ptr()), reinterpret_cast<const uint4*>(Tu.data_ptr()),
        reinterpret_cast<const uint4*>(Td.data_ptr()), ids.data_ptr<int>(), (int)ids.numel(), per,
        reinterpret_cast<uint32_t*>(out.data_ptr()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The expert L2 prefetch: a window's routed experts (plan items / counts), up to budget bytes; 0 bytes: nothing.
void exl3_expert_prefetch_cuda(const at::Tensor& items, const at::Tensor& counts, const at::Tensor& Tg,
                               const at::Tensor& Tu, const at::Tensor& Td, int64_t max_items, int64_t budget) {
    if (budget <= 0) return;
    const long long mat_bytes = (long long)(Tg.numel() / Tg.size(0)) * 4;
    const long long pieces = (budget + PF_PIECE - 1) / PF_PIECE;
    const int grid = (int)std::min<long long>(std::max<long long>((pieces + 127) / 128, 1), 64);
    expert_prefetch_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        items.data_ptr<int>(), counts.data_ptr<int>(), reinterpret_cast<const char*>(Tg.data_ptr()),
        reinterpret_cast<const char*>(Tu.data_ptr()), reinterpret_cast<const char*>(Td.data_ptr()),
        (int)Tg.size(0), (int)max_items, mat_bytes, (long long)budget);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
