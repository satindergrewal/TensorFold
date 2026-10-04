// TF_GLM_TOPK_FAST=1 (patch 0199): a prompt chunk's top-512 pools a row (sparse.top_pools on its _select_rows path)
// in two reads of the row's scores instead of five, the same pools in the same order.
//
// _select_rows finds the 512th best score's order key by a radix select of 8 bits a pass (four passes over the row),
// then writes, in pool order, the pools above it and the lowest-numbered ties (a fifth pass). Its output is a function
// of the scores alone: the 512 best order keys, ties to the lower pool, in ascending pool order. This kernel computes
// the same set with one CTA a row:
//   1. one read: a histogram of the order keys' top 12 bits (shared memory, warp-aggregated atomics); the bin that
//      holds the 512th best key (b1), the count above it;
//   2. a second read: pools in bins above b1 marked selected (a bitmap), the keys and pools of bin b1 kept (shared
//      memory, up to CAP of them);
//   3. the threshold key among those candidates (a radix select of their low 20 bits, 10 a step), candidates above it
//      marked selected, those equal to it marked tied;
//   4. the bitmap compacted in pool order: every selected pool, then the first (512 - selected) tied ones, at their
//      positions by prefix counts.
// When bin b1 holds more than CAP pools (scores bunched, e.g. -inf rows), the low bits are resolved by two more
// reads (12 and 8 bits) and the row classified again; rows over 65,536 pools (no room for the bitmaps) take a final
// ordered read instead of the bitmaps. Every path gives the same pools: the set is defined by the order keys. Rows are
// read 16 bytes at a time when every row starts on 16 bytes (NP a multiple of 4), else 4 bytes (patch 0290).
//
// Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
// the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab.

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int TK = 512;                     // pools a row keeps (TOPK_POOLS)
constexpr int THREADS = 256;
constexpr int NB = 4096;                    // first-level bins: an order key's top 12 bits
constexpr int CAP = 1024;                   // candidates of the threshold bin kept in shared memory
constexpr int MAXW = 2048;                  // bitmap words in shared memory: rows of up to 65,536 pools

struct TkSmem {
    unsigned hist[NB];
    unsigned ckey[CAP];
    int cidx[CAP];
    unsigned sel[MAXW];                     // bit i of word w: pool 32 w + i selected outright
    unsigned tie[MAXW];                     // ... its key equals the threshold
    unsigned scan[THREADS / 32];
    int b1, above, ccount, thr, need;       // bin, keys above, candidates, the threshold key, ties to take
};

// _order_key: the bits of s + 0.0 (so -0 counts as +0), the sign bit flipped for positives, every bit for negatives
__device__ __forceinline__ unsigned order_key(float s) {
    float z;
    asm("add.rn.f32 %0, %1, 0f00000000;" : "=f"(z) : "f"(s));
    const unsigned u = __float_as_uint(z);
    return u ^ ((static_cast<unsigned>(static_cast<int>(u) >> 31)) | 0x80000000u);
}

// hist[bin] += 1 for every lane with ok, one shared atomic per distinct bin of the warp
__device__ __forceinline__ void hist_add(unsigned* hist, unsigned bin, bool ok) {
    const unsigned key = ok ? bin : 0xFFFFFFFFu;
    const unsigned peers = __match_any_sync(0xFFFFFFFFu, key);
    const int lane = threadIdx.x & 31;
    if (ok && lane == __ffs(peers) - 1) atomicAdd(&hist[bin], static_cast<unsigned>(__popc(peers)));
}

// Block-wide exclusive prefix sum of v (thread order), and the total
__device__ __forceinline__ unsigned block_excl(unsigned v, unsigned* scratch, unsigned& total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    unsigned x = v;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        const unsigned y = __shfl_up_sync(0xFFFFFFFFu, x, d);
        if (lane >= d) x += y;
    }
    if (lane == 31) scratch[warp] = x;
    __syncthreads();
    unsigned before = 0, all = 0;
#pragma unroll
    for (int w = 0; w < THREADS / 32; ++w) {
        const unsigned c = scratch[w];
        before += w < warp ? c : 0u;
        all += c;
    }
    __syncthreads();
    total = all;
    return before + x - v;
}

// The largest bin b of hist[0 .. nbins) with count(bins >= b) >= need; above = count(bins > b). Every thread gets them.
__device__ __forceinline__ void find_bin(const unsigned* hist, int nbins, unsigned need, unsigned* scratch, int& bin,
                                         unsigned& above, int* out_bin, int* out_above) {
    const int per = nbins / THREADS;                 // bins a thread, counted from the top
    const int hi = nbins - 1 - per * static_cast<int>(threadIdx.x);
    unsigned local = 0;
    for (int b = hi; b > hi - per; --b) local += hist[b];
    unsigned total;
    const unsigned excl = block_excl(local, scratch, total);   // keys in the bins above this thread's
    if (excl < need && excl + local >= need) {
        unsigned acc = excl;
        for (int b = hi; b > hi - per; --b) {
            if (acc + hist[b] >= need) {
                *out_bin = b;
                *out_above = static_cast<int>(acc);
                break;
            }
            acc += hist[b];
        }
    }
    __syncthreads();
    bin = *out_bin;
    above = static_cast<unsigned>(*out_above);
}

// V consecutive scores from pool i (zeros past NP, flagged): V = 4 takes 16-byte loads (every row 16-byte aligned:
// NP a multiple of 4 and an aligned base), V = 1 any row.
template <int V>
__device__ __forceinline__ void load_v(const float* __restrict__ row, int i, int NP, float (&e)[V]) {
    if constexpr (V == 4) {
        float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
        if (i < NP) v = *reinterpret_cast<const float4*>(row + i);
        e[0] = v.x;
        e[1] = v.y;
        e[2] = v.z;
        e[3] = v.w;
    } else {
        static_assert(V == 1, "4- or 1-wide loads");
        e[0] = i < NP ? row[i] : 0.f;
    }
}

template <int V>
__global__ void __launch_bounds__(THREADS) topk_rows_kernel(const float* __restrict__ scores, int64_t* __restrict__ out,
                                                           int NP) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    TkSmem& sm = *reinterpret_cast<TkSmem*>(smem_raw);
    const int tid = threadIdx.x;
    const float* row = scores + static_cast<size_t>(blockIdx.x) * NP;
    int64_t* dst = out + static_cast<size_t>(blockIdx.x) * TK;
    const int W = (NP + 31) / 32;
    const bool bitmap = W <= MAXW;
    constexpr int STEP = THREADS * V;                // pools a step of the CTA
    const int span = ((NP + STEP - 1) / STEP) * STEP;   // every lane runs every step

    // 1. the histogram of the top 12 bits
    for (int i = tid; i < NB; i += THREADS) sm.hist[i] = 0;
    if (bitmap)
        for (int i = tid; i < W; i += THREADS) sm.sel[i] = sm.tie[i] = 0;
    if (tid == 0) sm.ccount = 0;
    __syncthreads();
#pragma unroll 4
    for (int i = tid * V; i < span; i += STEP) {                      // (unrolled: several loads in flight)
        float e[V];
        load_v<V>(row, i, NP, e);
#pragma unroll
        for (int c = 0; c < V; ++c) hist_add(sm.hist, order_key(e[c]) >> 20, i + c < NP);
    }
    __syncthreads();
    int b1;
    unsigned above1;
    find_bin(sm.hist, NB, TK, sm.scan, b1, above1, &sm.b1, &sm.above);
    const unsigned need1 = TK - above1;              // keys of bin b1 to take (at least 1)

    // 2. classify: above b1 selected; b1's keys kept as candidates
#pragma unroll 2
    for (int i = tid * V; i < span; i += STEP) {
        float e[V];
        load_v<V>(row, i, NP, e);
#pragma unroll
        for (int c = 0; c < V; ++c) {
            const unsigned key = order_key(e[c]);
            const int b = static_cast<int>(key >> 20);
            const bool ok = i + c < NP;
            if (ok && b > b1 && bitmap) atomicOr(&sm.sel[(i + c) >> 5], 1u << ((i + c) & 31));
            const bool cand = ok && b == b1;
            const unsigned peers = __ballot_sync(0xFFFFFFFFu, cand);
            int base = 0;
            if ((tid & 31) == 0 && peers) base = atomicAdd(&sm.ccount, __popc(peers));
            base = __shfl_sync(0xFFFFFFFFu, base, 0);
            if (cand) {
                const int pos = base + __popc(peers & ((1u << (tid & 31)) - 1u));
                if (pos < CAP) {
                    sm.ckey[pos] = key;
                    sm.cidx[pos] = i + c;
                }
            }
        }
    }
    __syncthreads();
    const int C1 = sm.ccount;

    // 3. the threshold key and the ties to take
    unsigned thr;
    unsigned need;
    if (C1 <= CAP) {                                 // among the candidates in shared memory: bits 19..10, then 9..0
        for (int i = tid; i < 1024; i += THREADS) sm.hist[i] = 0;
        __syncthreads();
        for (int c = tid; c < C1; c += THREADS) atomicAdd(&sm.hist[(sm.ckey[c] >> 10) & 1023u], 1u);
        __syncthreads();
        int d1;
        unsigned above2;
        find_bin(sm.hist, 1024, need1, sm.scan, d1, above2, &sm.b1, &sm.above);
        const unsigned need2 = need1 - above2;
        for (int i = tid; i < 1024; i += THREADS) sm.hist[i] = 0;
        __syncthreads();
        for (int c = tid; c < C1; c += THREADS)
            if (((sm.ckey[c] >> 10) & 1023u) == static_cast<unsigned>(d1)) atomicAdd(&sm.hist[sm.ckey[c] & 1023u], 1u);
        __syncthreads();
        int d2;
        unsigned above3;
        find_bin(sm.hist, 1024, need2, sm.scan, d2, above3, &sm.b1, &sm.above);
        thr = (static_cast<unsigned>(b1) << 20) | (static_cast<unsigned>(d1) << 10) | static_cast<unsigned>(d2);
        need = need2 - above3;
        if (bitmap)
            for (int c = tid; c < C1; c += THREADS) {
                const unsigned k = sm.ckey[c];
                const int idx = sm.cidx[c];
                if (k > thr) atomicOr(&sm.sel[idx >> 5], 1u << (idx & 31));
                else if (k == thr) atomicOr(&sm.tie[idx >> 5], 1u << (idx & 31));
            }
    } else {                                         // bin b1 bunched: bits 19..8, then 7..0, from the row again
        for (int i = tid; i < NB; i += THREADS) sm.hist[i] = 0;
        __syncthreads();
        for (int i = tid * V; i < span; i += STEP) {
            float e[V];
            load_v<V>(row, i, NP, e);
#pragma unroll
            for (int c = 0; c < V; ++c) {
                const unsigned key = order_key(e[c]);
                hist_add(sm.hist, (key >> 8) & 4095u, i + c < NP && static_cast<int>(key >> 20) == b1);
            }
        }
        __syncthreads();
        int d1;
        unsigned above2;
        find_bin(sm.hist, NB, need1, sm.scan, d1, above2, &sm.b1, &sm.above);
        const unsigned need2 = need1 - above2;
        const unsigned pre = (static_cast<unsigned>(b1) << 12) | static_cast<unsigned>(d1);    // the key's top 24 bits
        for (int i = tid; i < 256; i += THREADS) sm.hist[i] = 0;
        __syncthreads();
        for (int i = tid * V; i < span; i += STEP) {
            float e[V];
            load_v<V>(row, i, NP, e);
#pragma unroll
            for (int c = 0; c < V; ++c) {
                const unsigned key = order_key(e[c]);
                hist_add(sm.hist, key & 255u, i + c < NP && (key >> 8) == pre);
            }
        }
        __syncthreads();
        int d2;
        unsigned above3;
        find_bin(sm.hist, 256, need2, sm.scan, d2, above3, &sm.b1, &sm.above);
        thr = (pre << 8) | static_cast<unsigned>(d2);
        need = need2 - above3;
        if (bitmap)
            for (int i = tid; i < NP; i += THREADS) {        // the threshold bin's pools: above / equal
                const unsigned key = order_key(row[i]);
                if (static_cast<int>(key >> 20) == b1) {
                    if (key > thr) atomicOr(&sm.sel[i >> 5], 1u << (i & 31));
                    else if (key == thr) atomicOr(&sm.tie[i >> 5], 1u << (i & 31));
                }
            }
    }
    __syncthreads();

    // 4. in pool order: the selected pools and the first `need` ties
    if (bitmap) {
        const int per = (W + THREADS - 1) / THREADS;
        const int w0 = min(W, tid * per), w1 = min(W, w0 + per);
        unsigned ns = 0, nt = 0;
        for (int w = w0; w < w1; ++w) ns += __popc(sm.sel[w]), nt += __popc(sm.tie[w]);
        unsigned tot;
        const unsigned sb = block_excl(ns, sm.scan, tot);
        const unsigned tb = block_excl(nt, sm.scan, tot);
        unsigned taken = min(tb, need);                // ties before this thread's words that are taken
        unsigned seen = tb;
        unsigned pos = sb + taken;
        for (int w = w0; w < w1; ++w) {
            const unsigned t = sm.tie[w];
            unsigned keep = 0;
            if (seen < need) {                           // the lowest (need - seen) ties of this word
                unsigned left = need - seen, bits = t;
                while (bits && left) {
                    const unsigned low = bits & (0u - bits);
                    keep |= low;
                    bits ^= low;
                    --left;
                }
            }
            seen += __popc(t);
            unsigned m = sm.sel[w] | keep;
            while (m) {
                const int b = __ffs(m) - 1;
                dst[pos++] = static_cast<int64_t>(32 * w + b);
                m &= m - 1;
            }
        }
    } else {                                         // no room for the bitmaps: the row once more, in order
        unsigned done_sel = 0, done_tie = 0;         // over the previous steps (the same in every thread)
        for (int i = tid * V; i < span; i += STEP) {
            float e[V];
            load_v<V>(row, i, NP, e);
            unsigned s4 = 0, t4 = 0;
#pragma unroll
            for (int c = 0; c < V; ++c) {
                const unsigned key = order_key(e[c]);
                const bool ok = i + c < NP;
                s4 |= (ok && key > thr) ? (1u << c) : 0u;
                t4 |= (ok && key == thr) ? (1u << c) : 0u;
            }
            unsigned ts, tt;
            const unsigned sb = block_excl(__popc(s4), sm.scan, ts);
            const unsigned tb = block_excl(__popc(t4), sm.scan, tt);
            unsigned seen = done_tie + tb;
            unsigned pos = done_sel + sb + min(seen, need);
#pragma unroll
            for (int c = 0; c < V; ++c) {
                const bool s = (s4 >> c) & 1u, t = (t4 >> c) & 1u;
                const bool take = s || (t && seen < need);
                if (t) ++seen;
                if (take) dst[pos++] = static_cast<int64_t>(i + c);
            }
            done_sel += ts;
            done_tie += tt;
        }
    }
}

}  // namespace

int topk_rows_smem() { return static_cast<int>(sizeof(TkSmem)); }

template <int V>
void topk_launch(const at::Tensor& scores, at::Tensor& out) {
    const int R = static_cast<int>(scores.size(0)), NP = static_cast<int>(scores.size(1));
    static bool configured = false;
    if (!configured) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(topk_rows_kernel<V>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            sizeof(TkSmem)));
        configured = true;
    }
    topk_rows_kernel<V><<<R, THREADS, sizeof(TkSmem), at::cuda::getCurrentCUDAStream()>>>(
        scores.data_ptr<float>(), out.data_ptr<int64_t>(), NP);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Any contiguous [R, NP >= 512] fp32 rows: 16-byte loads when every row starts on 16 bytes, else 4-byte loads
void topk_rows_cuda(const at::Tensor& scores, at::Tensor& out) {
    const bool v4 = scores.size(1) % 4 == 0 && reinterpret_cast<uintptr_t>(scores.data_ptr()) % 16 == 0;
    if (v4)
        topk_launch<4>(scores, out);
    else
        topk_launch<1>(scores, out);
}
