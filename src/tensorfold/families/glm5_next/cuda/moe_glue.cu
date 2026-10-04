// The MoE layer's glue (TF_GLM_MOE_GLUE, moe_glue.py) in fewer kernels with the bits of the kernels it replaces.
//
// route:   router matmul + slice sums + sigmoid / bias top-k + normalization + the plan (stable counting sort of the
//          (row, slot) pairs by expert, items of T pairs, counts) in ONE launch: a cluster of 8 CTAs a (row block,
//          expert block), one CTA a K slice; the slices' partials are added in slice order through distributed shared
//          memory; the cluster that completes a row block (a ticket) selects its rows' experts and ranks their pairs;
//          the cluster that completes the last row block (a second ticket) finishes the plan.
//          Bits: every logit is glue._router_part's chain (32 mma.sync m16n8k16 steps of its K slice in ascending k,
//          from a zero accumulator) and glue._router_sum's slice-order adds; the selection, weights and plan are
//          glue._topk's and experts.cu's (same instructions: sub.f32, mul.f32 by log2e, ex2.approx.f32, add.f32,
//          div.full.f32; max value, lowest index among equals; pairs of an expert in pair order).
// combine: a rank's fp32 MoE share, one thread 4 columns: glue._combine(_sy)'s fma.rn.f32 chain in slot order.
// shared:  a prompt chunk's shared-expert gate/up matmul with SwiGLU in its epilogue: the prompt matmul's chain
//          (qmm_prefill.cu: weights bf16(fma(q, s, b)), one fp32 chain over K in k16 steps), its bf16 rounding, then
//          glue._swiglu's operations; the gate and up columns of an output sit in one thread's fragments.
//
// Licensed under the Apache License, Version 2.0. Builds on TensorFold (its Triton glue kernels, the grouped-expert
// plan and the 4-bit prompt matmul, whose arithmetic these kernels reproduce) and on the GLM-5.3-Flash recipe and
// patches 0001-0056 by MiaAI-Lab.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <algorithm>

namespace cg = cooperative_groups;

namespace {

// ------------------------------------------------------------------------------------------------- PTX helpers
__device__ __forceinline__ uint32_t sptr(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// 16 bytes into shared memory, or zeros without reading when ``read`` is false
__device__ __forceinline__ void cp16z(void* dst, const void* src, bool read) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sptr(dst)), "l"(src), "r"(read ? 16 : 0));
}

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(sptr(dst)), "l"(src));
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(sptr(p)));
}

__device__ __forceinline__ void ldsm2(uint32_t (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(sptr(p)));
}

// the instruction Triton's tl.dot of bf16 tiles emits on sm_120 (and qmm_frag.cuh's mma)
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// fp32 operations as the Triton glue kernels compile them on sm_120 (tests/k4/ptx_contract.py reads their PTX): the
// uncontracted IEEE ops (.rn: never fused with a neighbour), the approximate exp2 and division Triton emits for
// tl.exp and an fp32 '/', and min / max without NaN propagation.
__device__ __forceinline__ float add_rn(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float sub_rn(float a, float b) {
    float r;
    asm("sub.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float mul_rn(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float ex2_approx(float a) {
    float r;
    asm("ex2.approx.f32 %0, %1;\n" : "=f"(r) : "f"(a));
    return r;
}

__device__ __forceinline__ float div_full(float a, float b) {
    float r;
    asm("div.full.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float min_f32(float a, float b) {
    float r;
    asm("min.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float max_f32(float a, float b) {
    float r;
    asm("max.f32 %0, %1, %2;\n" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float fma_rn(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;\n" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}

// Triton's tl.exp(x) on sm_120: ex2.approx.f32(x * 0f3FB8AA3B)
__device__ __forceinline__ float log2e() { return __int_as_float(0x3FB8AA3B); }

// glue._topk's score: 1.0 / (1.0 + tl.exp(-lg)) = div.full(1, ex2(sub(0, lg) * log2e) + 1)
__device__ __forceinline__ float sigmoid_tf(float lg) {
    return div_full(1.0f, add_rn(ex2_approx(mul_rn(sub_rn(0.0f, lg), log2e())), 1.0f));
}

// glue._swiglu on bf16 gate / up values (as fp32): bf16(u * bf16(g / (1 + exp(-g)))) with g = min(g, L) and
// u = clip(u, -L, L), in the order its PTX holds them
__device__ __forceinline__ __nv_bfloat16 swiglu_tf(float g, float u, float limit) {
    g = min_f32(g, limit);
    u = min_f32(max_f32(u, sub_rn(0.0f, limit)), limit);
    const float silu = div_full(g, add_rn(ex2_approx(mul_rn(sub_rn(0.0f, g), log2e())), 1.0f));
    const float sb = __bfloat162float(__float2bfloat16_rn(silu));
    return __float2bfloat16_rn(mul_rn(u, sb));
}

// ======================================================================================================= route
constexpr int KS = 8;             // K slices (glue.ROUTER_KS): a logit is 8 slice chains added in slice order
constexpr int BK = 64;            // inputs a pipeline chunk: one 128-byte row of bf16 a tile row
constexpr int TOPK = 8;           // GLM-5.3-Flash's experts a token (the host refuses others)
constexpr int SLOTS = TOPK + 1;   // the shared expert's slot last
constexpr int QB = 16;            // glue._topk's BLOCK (next power of two above NE) / 32 at most: NE + 1 <= 512

struct RouteArgs {
    const __nv_bfloat16* x;
    const __nv_bfloat16* w;
    const float* bias;
    float* logits;
    int* pick;
    float* wts;
    int* members;
    int* items;
    int* counts;
    int* rank;
    int* hist;
    int* tick;
    float scale;
    int ldx, M, D, NE, E, T, norm, NRB, NEB, block;     // block: glue._topk's BLOCK
    int mode;                                            // 0: route and plan; 1: route only; 2: plan only (from pick)
};

template <int BM, int BE, int WM, int WN, int STAGES>
struct RouteTile {
    static constexpr int WARPS = WM * WN, THREADS = WARPS * 32;
    static constexpr int MT = BM / WM / 16, NT = BE / WN / 8;
    static constexpr int XB = BM * BK * 2, WB = BE * BK * 2, STAGE = XB + WB;
    static constexpr int ACC = MT * NT * 4;                 // a thread's accumulators
    static constexpr int PART = ACC * THREADS * 4;          // a slice's partial tile, bytes
    static constexpr int RPC = BM / KS;                     // rows of a row block each CTA of its finisher selects
    static_assert(MT >= 1 && NT >= 1 && BM % (WM * 16) == 0 && BE % (WN * 8) == 0, "tile");
    static_assert(RPC >= 1 && BM % KS == 0, "rows a CTA finishes");
    static_assert((ACC * THREADS) % KS == 0, "slices share the reduction");
};

__device__ __forceinline__ bool better(float v, int i, float bv, int bi) {
    // glue._topk's rule: the largest value (max.f32 ignores NaN: a NaN never wins), then the lowest index among equals
    return v > bv || (v == bv && i < bi);
}

// One row's experts and weights (glue._topk on the row's logits), by one warp; the picks also into ``spick``. Lane l
// holds candidates l, l + 32, ... below BLOCK (glue._topk's masked ones too: score sigmoid(0), choice -inf); a pick is
// the largest choice, the lowest index among equals (NaN never: max.f32 drops it, == never holds); a picked candidate
// becomes -inf and stays a candidate, as there. No array is indexed by a run-time value (registers only).
__device__ void topk_row(const RouteArgs& a, int r, int lane, int* spick) {
    const float neg = __int_as_float(0xff800000);
    const float sig0 = sigmoid_tf(0.0f);              // a masked lane's score (its logit loads as 0.0 in glue._topk)
    float sc[QB], ch[QB];
#pragma unroll
    for (int q = 0; q < QB; ++q) {
        const int e = lane + 32 * q;
        if (e < a.NE) {
            const float lg = __ldcg(a.logits + static_cast<size_t>(r) * a.NE + e);
            sc[q] = sigmoid_tf(lg);
            ch[q] = add_rn(sc[q], __ldg(a.bias + e));
        } else {
            sc[q] = sig0;
            ch[q] = neg;                               // masked (NE .. BLOCK - 1), and nothing past BLOCK below
        }
    }
    float total = 0.0f, myw = 1.0f;                    // lane k < TOPK: pick k's weight; lane TOPK: the shared slot's
    int myp = a.NE;                                    // lane k < TOPK: pick k; lane TOPK: the shared expert's id
#pragma unroll
    for (int k = 0; k < TOPK; ++k) {
        float bv = neg, bs = 0.0f;
        int bi = 1 << 30;
#pragma unroll
        for (int q = 0; q < QB; ++q) {
            const int e = lane + 32 * q;
            if (e < a.block && better(ch[q], e, bv, bi)) {
                bv = ch[q];
                bi = e;
                bs = sc[q];
            }
        }
#pragma unroll
        for (int o = 16; o; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            const float os = __shfl_xor_sync(0xffffffffu, bs, o);
            if (better(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
                bs = os;
            }
        }
        // every lane now holds the pick (bi) and its score (bs: glue._topk's sk, the sum of one score and zeros)
#pragma unroll
        for (int q = 0; q < QB; ++q) ch[q] = (lane + 32 * q == bi) ? neg : ch[q];
        if (lane == k) {
            myp = bi;
            myw = bs;
        }
        total = add_rn(total, bs);
    }
    if (lane < TOPK) {
        if (a.norm) myw = div_full(myw, add_rn(total, __int_as_float(0x1E3CE508)));   // / (total + 1e-20)
        myw = mul_rn(a.scale, myw);
    }
    if (lane < SLOTS) {
        a.pick[static_cast<size_t>(r) * SLOTS + lane] = myp;
        a.wts[static_cast<size_t>(r) * SLOTS + lane] = myw;
        spick[lane] = myp;
    }
}

// Exclusive scans over e < E of tot[e] (into off) and of its items ceil(tot / T) (into ioff); returns
// (items, experts used) in scr[2 WARPS], scr[2 WARPS + 1]. One CTA; ``scr`` holds 2 * WARPS ints + 2, zeroed.
template <int THREADS>
__device__ void scan_experts(const int* tot, int* off, int* ioff, int E, int T, int* scr) {
    constexpr int WARPS = THREADS / 32;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int per = (E + THREADS - 1) / THREADS;
    const int lo = min(E, tid * per), hi = min(E, lo + per);
    int a = 0, b = 0, u = 0;
    for (int e = lo; e < hi; ++e) {
        a += tot[e];
        b += (tot[e] + T - 1) / T;
        u += tot[e] > 0;
    }
    int ia = a, ib = b;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int va = __shfl_up_sync(0xffffffffu, ia, o), vb = __shfl_up_sync(0xffffffffu, ib, o);
        if (lane >= o) {
            ia += va;
            ib += vb;
        }
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) u += __shfl_xor_sync(0xffffffffu, u, o);
    if (lane == 31) {
        scr[warp] = ia;
        scr[WARPS + warp] = ib;
    }
    if (lane == 0) atomicAdd(scr + 2 * WARPS + 1, u);
    __syncthreads();
    int wa = 0, wb = 0;
    for (int k = 0; k < warp; ++k) {
        wa += scr[k];
        wb += scr[WARPS + k];
    }
    int ra = wa + ia - a, rbb = wb + ib - b;
    for (int e = lo; e < hi; ++e) {
        off[e] = ra;
        ioff[e] = rbb;
        ra += tot[e];
        rbb += (tot[e] + T - 1) / T;
    }
    if (tid == THREADS - 1) scr[2 * WARPS] = wb + ib;
    __syncthreads();
}

template <int BM, int BE, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32) route_kernel(const RouteArgs a) {
    using T = RouteTile<BM, BE, WM, WN, STAGES>;
    constexpr int THREADS = T::THREADS, WARPS = T::WARPS, NT = T::NT, MT = T::MT, RPC = T::RPC;
    extern __shared__ __align__(128) unsigned char smem[];
    cg::cluster_group cluster = cg::this_cluster();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, wm = warp / WN, wn = warp % WN;
    const int s = static_cast<int>(cluster.block_rank());          // this CTA's K slice
    const int clus = blockIdx.x / KS;
    const int eb = clus % a.NEB, rb = clus / a.NEB;
    const int m0 = rb * BM, e0 = eb * BE;
    const int KSL = a.D / KS, kbase = s * KSL, NCH = KSL / BK;

    // region B (control) after region A (the pipeline, then the slice partial, then the plan's prefix table): the
    // host computes the same sizes (route_launch)
    const int region_a = max(max(STAGES * T::STAGE, T::PART), a.NRB * a.E * 4);
    int* ctl = reinterpret_cast<int*>(smem + ((region_a + 127) & ~127));
    int* sflag = ctl;                      // [0]: the row block is done here, [1]: the plan is finished here
    int* cnt = ctl + 4;                    // [E]
    int* base = cnt + a.E;                 // [E]
    int* spick = base + a.E;               // [RPC * SLOTS]
    int* slr = spick + RPC * SLOTS;        // [RPC * SLOTS]
    int* tot = slr + RPC * SLOTS;          // [E]
    int* off = tot + a.E;                  // [E]
    int* ioff = off + a.E;                 // [E]
    int* scr = ioff + a.E;                 // [2 * WARPS + 2]

    if (a.mode != 2) {
    // ---- 1. this slice's partial of the tile: glue._router_part's chain (k16 steps in ascending k from zero)
    auto load = [&](int st, int chunk) {
        unsigned char* px = smem + st * T::STAGE;
        unsigned char* pw = px + T::XB;
        const int kk = kbase + chunk * BK;
        for (int c = tid; c < BM * 8; c += THREADS) {
            const int r = c >> 3, q = c & 7, row = m0 + r;
            const bool ok = row < a.M;
            cp16z(px + r * 128 + ((q ^ (r & 7)) << 4),
                  a.x + static_cast<size_t>(ok ? row : 0) * a.ldx + kk + q * 8, ok);
        }
        for (int c = tid; c < BE * 8; c += THREADS) {
            const int r = c >> 3, q = c & 7, e = e0 + r;
            const bool ok = e < a.NE;
            cp16z(pw + r * 128 + ((q ^ (r & 7)) << 4), a.w + static_cast<size_t>(ok ? e : 0) * a.D + kk + q * 8, ok);
        }
    };
    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < NCH) load(st, st);
        cp_commit();
    }
    for (int chunk = 0; chunk < NCH; ++chunk) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int nx = chunk + STAGES - 1;
        if (nx < NCH) load(nx % STAGES, nx);
        cp_commit();
        const unsigned char* px = smem + (chunk % STAGES) * T::STAGE;
        const unsigned char* pw = px + T::XB;
#pragma unroll
        for (int kt = 0; kt < BK / 16; ++kt) {
            uint32_t af[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int q = kt * 2 + (lane >> 4);
                ldsm4(af[i], px + r * 128 + ((q ^ (r & 7)) << 4));
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                uint32_t bf[2];
                const int r = wn * (BE / WN) + j * 8 + (lane & 7);
                const int q = kt * 2 + ((lane >> 3) & 1);
                ldsm2(bf, pw + r * 128 + ((q ^ (r & 7)) << 4));
#pragma unroll
                for (int i = 0; i < MT; ++i) mma16816(acc[i][j], af[i], bf[0], bf[1]);
            }
        }
    }
    cp_wait<0>();
    __syncthreads();

    // ---- 2. the partial into shared memory, every slice's at once
    float* part = reinterpret_cast<float*>(smem);
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) part[((i * NT + j) * 4 + e) * THREADS + tid] = acc[i][j][e];
    cluster.sync();

    // ---- 3. logits: the 8 slice partials added in slice order (glue._router_sum), this CTA's eighth of the tile
    {
        constexpr int F = T::ACC * THREADS, PER = F / KS;
        const float* peer[KS];
#pragma unroll
        for (int j = 0; j < KS; ++j) peer[j] = cluster.map_shared_rank(part, j);
        for (int f = s * PER + tid; f < (s + 1) * PER; f += THREADS) {
            float l = peer[0][f];
#pragma unroll
            for (int j = 1; j < KS; ++j) l = add_rn(l, peer[j][f]);
            const int t = f % THREADS, v = f / THREADS;
            const int tl = t & 31, tw = t >> 5, twm = tw / WN, twn = tw % WN;
            const int i = v / (NT * 4), j = (v >> 2) % NT, e = v & 3;
            const int row = m0 + twm * (BM / WM) + i * 16 + (tl >> 2) + (e >> 1) * 8;
            const int col = e0 + twn * (BE / WN) + j * 8 + (tl & 3) * 2 + (e & 1);
            if (row < a.M && col < a.NE) a.logits[static_cast<size_t>(row) * a.NE + col] = l;
        }
    }
    __threadfence();
    cluster.sync();                         // every CTA's logits written and fenced; the partials no longer read

    // ---- 4. the row block's ticket: the cluster that completes it selects its rows' experts
    if (s == 0 && tid == 0) {
        __threadfence();
        const int old = atomicAdd(a.tick + rb, 1);
        const int last = old == a.NEB - 1;
        if (last) a.tick[rb] = 0;           // every expert block of this row block has arrived: ready for the next call
        for (int j = 0; j < KS; ++j) *cluster.map_shared_rank(sflag, j) = last;
    }
    cluster.sync();
    if (!sflag[0]) return;
    __threadfence();
    }

    // ---- 5. selection (glue._topk), RPC rows a CTA, a warp a row; or (plan only) the rows' picks as given
    const int r0 = m0 + s * RPC;
    const int nrows = max(0, min(RPC, a.M - r0));
    if (a.mode != 2) {
        for (int lr = warp; lr < nrows; lr += WARPS) topk_row(a, r0 + lr, lane, spick + lr * SLOTS);
        if (a.mode == 1) return;
    } else {
        for (int p = tid; p < nrows * SLOTS; p += THREADS) spick[p] = a.pick[static_cast<size_t>(r0) * SLOTS + p];
    }
    for (int e = tid; e < a.E; e += THREADS) cnt[e] = 0;
    __syncthreads();

    // ---- 6. the plan's ranks: pairs of an expert in pair order (experts.cu); within the CTA, then the row block
    const int np = nrows * SLOTS;
    if (warp == 0) {
        for (int c0 = 0; c0 < np; c0 += 32) {
            const int p = c0 + lane;
            const bool ok = p < np;
            const int e = ok ? spick[p] : -1 - lane;
            const unsigned same = __match_any_sync(0xffffffffu, e);
            const int below = __popc(same & ((1u << lane) - 1u));
            const int b0 = ok ? cnt[e] : 0;
            __syncwarp();
            if (ok && below == 0) cnt[e] = b0 + __popc(same);
            __syncwarp();
            if (ok) slr[p] = b0 + below;
        }
    }
    cluster.sync();
    for (int e = tid; e < a.E; e += THREADS) {
        int before = 0, all = 0;
#pragma unroll
        for (int j = 0; j < KS; ++j) {
            const int v = *cluster.map_shared_rank(cnt + e, j);
            before += j < s ? v : 0;
            all += v;
        }
        base[e] = before;
        if (s == 0) a.hist[static_cast<size_t>(rb) * a.E + e] = all;
    }
    __syncthreads();
    for (int p = tid; p < np; p += THREADS) a.rank[static_cast<size_t>(r0) * SLOTS + p] = base[spick[p]] + slr[p];
    __threadfence();
    cluster.sync();                         // the counts read by every peer; hist and ranks fenced

    // ---- 7. the chunk's ticket: the cluster that completes the last row block finishes the plan (one row block,
    // a decode window: this cluster, no ticket)
    if (a.NRB > 1) {
        if (s == 0 && tid == 0) {
            __threadfence();
            const int old = atomicAdd(a.tick + a.NRB, 1);
            const int last = old == a.NRB - 1;
            if (last) a.tick[a.NRB] = 0;
            for (int j = 0; j < KS; ++j) *cluster.map_shared_rank(sflag + 1, j) = last;
        }
        cluster.sync();
        if (!sflag[1]) return;
        __threadfence();
    }

    int* pre = reinterpret_cast<int*>(smem);          // [NRB][E]: an expert's pairs in earlier row blocks
    for (int e = tid; e < a.E; e += THREADS) {
        int run = 0;
        for (int b = 0; b < a.NRB; ++b) {
            const int h = __ldcg(a.hist + static_cast<size_t>(b) * a.E + e);
            pre[b * a.E + e] = run;
            run += h;
        }
        tot[e] = run;
    }
    if (tid < 2 * WARPS + 2) scr[tid] = 0;
    __syncthreads();
    scan_experts<THREADS>(tot, off, ioff, a.E, a.T, scr);
    if (s == 0) {                                     // experts.cu's place_items: items of <= T pairs, expert order
        for (int e = tid; e < a.E; e += THREADS) {
            const int c = tot[e], tiles = (c + a.T - 1) / a.T;
            for (int j = 0; j < tiles; ++j) {
                int* it = a.items + 3 * (ioff[e] + j);
                it[0] = e;
                it[1] = off[e] + a.T * j;
                it[2] = min(a.T, c - a.T * j);
            }
        }
        if (tid == 0) {
            a.counts[0] = scr[2 * WARPS];
            a.counts[1] = scr[2 * WARPS + 1];
        }
    }
    const int P = a.M * SLOTS, per = (P + KS - 1) / KS;
    const int plo = s * per, phi = min(P, plo + per);
    for (int p = plo + tid; p < phi; p += THREADS) {
        const int b = (p / SLOTS) / BM;
        const int e = __ldcg(a.pick + p);
        a.members[off[e] + pre[b * a.E + e] + __ldcg(a.rank + p)] = p;
    }
}

// The route kernel's dynamic shared memory for ``nrb`` row blocks and ``E`` plan experts (route_kernel's regions A
// and B: the pipeline / slice partial / plan prefix table, then the control arrays).
template <int BM, int BE, int WM, int WN, int STAGES>
int route_smem(int nrb, int E) {
    using T = RouteTile<BM, BE, WM, WN, STAGES>;
    const int region_a = std::max(std::max(STAGES * T::STAGE, T::PART), nrb * E * 4);
    const int region_b = (4 + 5 * E + 2 * T::RPC * SLOTS + 2 * T::WARPS + 2) * 4;
    return ((region_a + 127) & ~127) + region_b;
}

template <int BM, int BE, int WM, int WN, int STAGES>
void route_launch(RouteArgs a, cudaStream_t stream) {
    using T = RouteTile<BM, BE, WM, WN, STAGES>;
    auto kernel = route_kernel<BM, BE, WM, WN, STAGES>;
    a.NRB = (a.M + BM - 1) / BM;
    a.NEB = a.mode == 2 ? 1 : (a.NE + BE - 1) / BE;      // plan only: a cluster a row block
    const int bytes = route_smem<BM, BE, WM, WN, STAGES>(a.NRB, a.E);
    int dev = 0;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    int most = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&most, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    TORCH_CHECK(bytes <= most, "moe_glue route: ", bytes, " bytes of shared memory, the GPU takes ", most,
                " (fewer rows a chunk)");
    static int configured = 0;
    if (bytes > configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, most));
        configured = most;
    }
    cudaLaunchConfig_t config = {};
    config.gridDim = dim3(KS * a.NEB * a.NRB, 1, 1);
    config.blockDim = dim3(T::THREADS, 1, 1);
    config.dynamicSmemBytes = bytes;
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = KS;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    config.attrs = attr;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, a));
}

// ===================================================================================================== combine
// out[r, d] = the fma chain over the slots in order (y[r, k, d] * w[r, k], from 0), the last slot's row from sy[r, d]
// (SY) or y[r, SLOTS - 1, d]: glue._combine / _combine_sy's fma.rn.f32 chain. A thread 4 columns of one row; every
// row's slots are loaded at once (streaming: each is read once), then added in slot order.
template <bool SY>
__global__ void __launch_bounds__(256) combine_kernel(const float4* __restrict__ y, const float4* __restrict__ sy,
                                                      const float* __restrict__ wts, float4* __restrict__ out, int R,
                                                      int D4) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(R) * D4) return;
    const int r = static_cast<int>(idx / D4), c = static_cast<int>(idx % D4);
    const float* w = wts + static_cast<size_t>(r) * SLOTS;
    const float4* yr = y + static_cast<size_t>(r) * SLOTS * D4 + c;
    float4 v[SLOTS];
#pragma unroll
    for (int k = 0; k < SLOTS; ++k)
        v[k] = (SY && k == SLOTS - 1) ? __ldcs(sy + static_cast<size_t>(r) * D4 + c) : __ldcs(yr + static_cast<size_t>(k) * D4);
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int k = 0; k < SLOTS; ++k) {
        const float wk = __ldg(w + k);
        acc.x = fma_rn(v[k].x, wk, acc.x);
        acc.y = fma_rn(v[k].y, wk, acc.y);
        acc.z = fma_rn(v[k].z, wk, acc.z);
        acc.w = fma_rn(v[k].w, wk, acc.w);
    }
    out[static_cast<size_t>(r) * D4 + c] = acc;
}

// ============================================================================================ shared gate / up
// qmm_prefill.cu's prompt matmul on the shared expert's stacked [gate | up] 4-bit weight (n = 2 NI), a CTA BM rows by
// BNA output columns of each half: its gate columns n0 .. n0 + BNA and up columns NI + n0 .. NI + n0 + BNA in one
// tile (virtual columns 0 .. 2 BNA), a warp BNA / WN columns of each, so one thread holds an element's gate and up.
// Every gate / up value is the prompt matmul's (the same weights' bf16(fma(q, s, b)), the same mma chain over K) and is
// rounded to bf16 as it stores them; then SwiGLU (glue._swiglu's operations) into act [M, NI] bf16.
__device__ __forceinline__ uint32_t fma2(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// qmm_frag.cuh's pair(): nibbles at bits [s, s + 4) and [16 + s, 20 + s) as a bf16 pair
__device__ __forceinline__ uint32_t nib_pair(uint32_t w, int s) {
    const uint32_t t = ((w >> s) & 0x000F000Fu) | 0x43004300u;
    uint32_t r;
#if __CUDA_ARCH__ >= 900
    asm("sub.rn.bf16x2 %0, %1, %2;\n" : "=r"(r) : "r"(t), "r"(0x43004300u));
#else
    asm("fma.rn.bf16x2 %0, %1, %2, %3;\n" : "=r"(r) : "r"(t), "r"(0x3F803F80u), "r"(0xC300C300u));
#endif
    return r;
}

// qmm_frag.cuh's tile_of: block b's (first row, first column), row tiles fastest in bands of ``group``
__device__ __forceinline__ int2 tile_of(int b, int M, int N, int BM, int BN, int group) {
    const int rows_t = (M + BM - 1) / BM, cols_t = (N + BN - 1) / BN, band = group * cols_t;
    const int first = b / band * group, in_band = b % band, height = min(group, rows_t - first);
    return make_int2((first + in_band % height) * BM, in_band / height * BN);
}

template <int BM, int BNA, int WM, int WN, int STAGES>
struct GuTile {
    static constexpr int GS = 64;                        // the dense 4-bit groups (TF_GLM_DENSE=q4)
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int MT = BM / WM / 16;
    static constexpr int NTH = BNA / WN / 8;             // n8 tiles a warp, of each half
    static constexpr int ROW = GS * 2, CHUNKS = ROW / 16;
    static constexpr int X = BM * ROW;                   // stage bytes: inputs,
    static constexpr int W = 2 * BNA * GS / 2;           // weights (gate tiles, then up tiles),
    static constexpr int S = 2 * BNA * 2;                // scales (then biases), bf16
    static constexpr int STAGE = X + W + 2 * S;
    static constexpr int SMEM = STAGES * STAGE;
    static_assert(BNA % 64 == 0 && NTH >= 1 && MT >= 1, "tile");
};

template <int BM, int BNA, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32) shared_gu_kernel(
        const __nv_bfloat16* __restrict__ x, const uint32_t* __restrict__ w, const __nv_bfloat16* __restrict__ scales,
        const __nv_bfloat16* __restrict__ biases, __nv_bfloat16* __restrict__ act, int M, int NI, int K, int npad,
        int ldx, int group, float limit) {
    using T = GuTile<BM, BNA, WM, WN, STAGES>;
    constexpr int GS = T::GS, MT = T::MT, NTH = T::NTH, NT = 2 * NTH;
    extern __shared__ __align__(128) unsigned char buf[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / WN, wn = warp % WN;
    const int KG = K / GS;
    const int2 at = tile_of(blockIdx.x, M, NI, BM, BNA, group);
    const int m0 = at.x, n0 = at.y;                      // act columns n0 .. n0 + BNA
    constexpr int TILE_BYTES = 64 * GS / 2;              // one stored 64-column tile's group block
    constexpr int HALF_TILES = BNA / 64;

    auto stage = [&](int s) { return buf + s * T::STAGE; };
    auto load = [&](int s, int g) {
        unsigned char* p = stage(s);
        for (int c = tid; c < BM * T::CHUNKS; c += T::THREADS) {
            const int r = c / T::CHUNKS, ch = c % T::CHUNKS;
            const int row = min(m0 + r, M - 1);
            cp16z(p + r * T::ROW + ((ch ^ (r % T::CHUNKS)) << 4), x + static_cast<size_t>(row) * ldx + g * GS + ch * 8,
                  m0 + r < M);
        }
        unsigned char* pw = p + T::X;
        for (int c = tid; c < T::W / 16; c += T::THREADS) {
            const int t = c / (TILE_BYTES / 16), o = c % (TILE_BYTES / 16);
            const int col = (t < HALF_TILES ? n0 : NI + n0) + (t % HALF_TILES) * 64;      // gate tiles, then up
            const size_t tile = static_cast<size_t>(col / 64) * KG + g;
            cp16(pw + c * 16, reinterpret_cast<const unsigned char*>(w) + tile * TILE_BYTES + o * 16);
        }
        unsigned char* ps = pw + T::W;
        for (int c = tid; c < 2 * (T::S / 16); c += T::THREADS) {      // scales: gate then up; then biases alike
            const int which = c / (T::S / 16), o = c % (T::S / 16);
            const int half = (o * 8) / BNA, within = (o * 8) % BNA;
            const __nv_bfloat16* src = (which ? biases : scales) + static_cast<size_t>(g) * npad +
                                       (half ? NI + n0 : n0) + within;
            cp16(ps + which * T::S + o * 16, src);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < KG) load(s, s);
        cp_commit();
    }
    for (int it = 0; it < KG; ++it) {
        cp_wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < KG) load(next % STAGES, next);
        cp_commit();
        const unsigned char* p = stage(it % STAGES);
        const uint32_t* pw = reinterpret_cast<const uint32_t*>(p + T::X);
        const uint16_t* ps = reinterpret_cast<const uint16_t*>(p + T::X + T::W);
        uint32_t words[NT][GS / 32], sv[NT], bv[NT];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            // virtual n8 tile: the warp's j-th gate tile, then (j >= NTH) its up tile for the same columns
            const int jv = (j < NTH ? 0 : BNA / 8) + wn * NTH + (j % NTH);
#pragma unroll
            for (int v = 0; v < GS / 32; ++v) words[j][v] = pw[(jv * 32 + lane) * (GS / 32) + v];
            const int col = jv * 8 + (lane >> 2);
            sv[j] = ps[col] * 0x10001u;
            bv[j] = ps[2 * BNA + col] * 0x10001u;
        }
#pragma unroll
        for (int kt = 0; kt < GS / 16; ++kt) {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * (BM / WM) + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
                const int ch = kt * 2 + (lane >> 4);
                ldsm4(a[i], p + r * T::ROW + ((ch ^ (r % T::CHUNKS)) << 4));
            }
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                const uint32_t b0 = fma2(nib_pair(words[j][kt / 2], (kt & 1) * 8), sv[j], bv[j]);
                const uint32_t b1 = fma2(nib_pair(words[j][kt / 2], (kt & 1) * 8 + 4), sv[j], bv[j]);
#pragma unroll
                for (int i = 0; i < MT; ++i) mma16816(acc[i][j], a[i], b0, b1);
            }
        }
    }
    cp_wait<0>();
    __syncthreads();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NTH; ++j) {
            const int col = n0 + wn * (BNA / WN) + j * 8 + (lane & 3) * 2;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int row = m0 + wm * (BM / WM) + i * 16 + (lane >> 2) + h * 8;
                if (row >= M) continue;
                // the prompt matmul stores gate / up as bf16 (round to nearest); SwiGLU reads them back
                const float g0 = __bfloat162float(__float2bfloat16_rn(acc[i][j][2 * h]));
                const float g1 = __bfloat162float(__float2bfloat16_rn(acc[i][j][2 * h + 1]));
                const float u0 = __bfloat162float(__float2bfloat16_rn(acc[i][j + NTH][2 * h]));
                const float u1 = __bfloat162float(__float2bfloat16_rn(acc[i][j + NTH][2 * h + 1]));
                __nv_bfloat162 o;
                o.x = swiglu_tf(g0, u0, limit);
                o.y = swiglu_tf(g1, u1, limit);
                *reinterpret_cast<__nv_bfloat162*>(act + static_cast<size_t>(row) * NI + col) = o;
            }
        }
}

template <int BM, int BNA, int WM, int WN, int STAGES>
void shared_gu_launch(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                      at::Tensor& act, int NI, float limit) {
    using T = GuTile<BM, BNA, WM, WN, STAGES>;
    const int M = x.size(0), K = x.size(1);
    auto kernel = shared_gu_kernel<BM, BNA, WM, WN, STAGES>;
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, T::SMEM));
        configured = true;
    }
    const int rows_t = (M + BM - 1) / BM;
    // a group's inputs stay near 12 MB of L2 while its blocks sweep the column tiles (as the prompt matmul)
    const int group = std::max(1, std::min(rows_t, static_cast<int>((12LL << 20) / (static_cast<long long>(BM) * K * 2))));
    kernel<<<rows_t * (NI / BNA), T::THREADS, T::SMEM, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<const uint32_t*>(w.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(act.data_ptr()), M, NI, K, static_cast<int>(scales.size(1)),
        M == 1 ? K : static_cast<int>(x.stride(0)), group, limit);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

// ============================================================================================= host entries
void moe_route_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& bias, at::Tensor& logits,
                    at::Tensor& pick, at::Tensor& wts, at::Tensor& members, at::Tensor& items, at::Tensor& counts,
                    at::Tensor& rank, at::Tensor& hist, at::Tensor& tick, double scale, bool norm, int64_t experts,
                    int64_t tile, int64_t block, int64_t mode) {
    const c10::cuda::CUDAGuard guard(x.device());
    RouteArgs a{};
    a.x = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    a.w = reinterpret_cast<const __nv_bfloat16*>(w.data_ptr());
    a.bias = bias.data_ptr<float>();
    a.logits = logits.data_ptr<float>();
    a.pick = pick.data_ptr<int>();
    a.wts = wts.data_ptr<float>();
    a.members = members.data_ptr<int>();
    a.items = items.data_ptr<int>();
    a.counts = counts.data_ptr<int>();
    a.rank = rank.data_ptr<int>();
    a.hist = hist.data_ptr<int>();
    a.tick = tick.data_ptr<int>();
    a.scale = static_cast<float>(scale);
    a.ldx = static_cast<int>(x.stride(0));
    a.M = static_cast<int>(x.size(0));
    a.D = static_cast<int>(x.size(1));
    a.NE = static_cast<int>(w.size(0));
    a.E = static_cast<int>(experts);
    a.T = static_cast<int>(tile);
    a.norm = norm ? 1 : 0;
    a.block = static_cast<int>(block);
    a.mode = static_cast<int>(mode);
    if (a.M == 0) return;
    const auto stream = at::cuda::getCurrentCUDAStream();
    // decode windows: every K chunk's load in flight at once (latency-bound); prompt chunks: two stages
    if (a.M <= 16) route_launch<16, 32, 1, 4, 8>(a, stream);
    else if (a.M <= 32) route_launch<32, 32, 2, 2, 8>(a, stream);
    else if (a.M <= 64) route_launch<64, 32, 4, 1, 6>(a, stream);
    else route_launch<128, 96, 4, 2, 2>(a, stream);
}

// The plan alone from given picks [M, 9] (route mode 2): the same kernel's ranking and finishing, a cluster a row block.
void moe_plan_cuda(const at::Tensor& pick, at::Tensor& members, at::Tensor& items, at::Tensor& counts, at::Tensor& rank,
                   at::Tensor& hist, at::Tensor& tick, int64_t experts, int64_t tile) {
    const c10::cuda::CUDAGuard guard(pick.device());
    RouteArgs a{};
    a.pick = pick.data_ptr<int>();
    a.members = members.data_ptr<int>();
    a.items = items.data_ptr<int>();
    a.counts = counts.data_ptr<int>();
    a.rank = rank.data_ptr<int>();
    a.hist = hist.data_ptr<int>();
    a.tick = tick.data_ptr<int>();
    a.M = static_cast<int>(pick.size(0));
    a.D = 512;
    a.NE = static_cast<int>(experts) - 1;
    a.E = static_cast<int>(experts);
    a.T = static_cast<int>(tile);
    a.mode = 2;
    if (a.M == 0) return;
    const auto stream = at::cuda::getCurrentCUDAStream();
    // decode windows: every K chunk's load in flight at once (latency-bound); prompt chunks: two stages
    if (a.M <= 16) route_launch<16, 32, 1, 4, 8>(a, stream);
    else if (a.M <= 32) route_launch<32, 32, 2, 2, 8>(a, stream);
    else if (a.M <= 64) route_launch<64, 32, 4, 1, 6>(a, stream);
    else route_launch<128, 96, 4, 2, 2>(a, stream);
}

int64_t moe_route_blocks(int64_t rows) {          // row blocks a call of ``rows`` rows takes (scratch sizes)
    return rows <= 64 ? 1 : (rows + 127) / 128;
}

// The dynamic shared memory a route / plan call of ``rows`` rows and ``experts`` plan experts takes (the configs
// moe_route_cuda picks), for the caller to compare with the device's opt-in maximum before using the kernel.
int64_t moe_route_bytes(int64_t rows, int64_t experts) {
    const int E = static_cast<int>(experts), nrb = static_cast<int>(moe_route_blocks(rows));
    if (rows <= 16) return route_smem<16, 32, 1, 4, 8>(nrb, E);
    if (rows <= 32) return route_smem<32, 32, 2, 2, 8>(nrb, E);
    if (rows <= 64) return route_smem<64, 32, 4, 1, 6>(nrb, E);
    return route_smem<128, 96, 4, 2, 2>(nrb, E);
}

void moe_combine_cuda(const at::Tensor& y, const c10::optional<at::Tensor>& sy, const at::Tensor& wts, at::Tensor& out) {
    const c10::cuda::CUDAGuard guard(y.device());
    const int R = static_cast<int>(out.size(0)), D = static_cast<int>(out.size(1));
    TORCH_CHECK(wts.size(1) == SLOTS && D % 4 == 0, "moe_glue combine: ", SLOTS, " slots and a width of 4s");
    if (R == 0) return;
    const long long n = static_cast<long long>(R) * (D / 4);
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    const auto stream = at::cuda::getCurrentCUDAStream();
    const float4* yp = reinterpret_cast<const float4*>(y.data_ptr<float>());
    float4* op = reinterpret_cast<float4*>(out.data_ptr<float>());
    if (sy.has_value())
        combine_kernel<true><<<blocks, threads, 0, stream>>>(yp, reinterpret_cast<const float4*>(sy->data_ptr<float>()),
                                                             wts.data_ptr<float>(), op, R, D / 4);
    else
        combine_kernel<false><<<blocks, threads, 0, stream>>>(yp, nullptr, wts.data_ptr<float>(), op, R, D / 4);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void moe_shared_gu_cuda(const at::Tensor& x, const at::Tensor& w, const at::Tensor& scales, const at::Tensor& biases,
                        at::Tensor& act, int64_t ni, double limit, int64_t cfg) {
    const c10::cuda::CUDAGuard guard(x.device());
    if (x.size(0) == 0) return;
    const int NI = static_cast<int>(ni);
    const float L = static_cast<float>(limit);
    switch (cfg) {
        case 1: shared_gu_launch<64, 64, 1, 4, 3>(x, w, scales, biases, act, NI, L); break;
        case 2: shared_gu_launch<128, 64, 4, 2, 3>(x, w, scales, biases, act, NI, L); break;
        case 3: shared_gu_launch<64, 128, 2, 4, 3>(x, w, scales, biases, act, NI, L); break;
        default: shared_gu_launch<128, 64, 2, 4, 3>(x, w, scales, biases, act, NI, L); break;
    }
}
