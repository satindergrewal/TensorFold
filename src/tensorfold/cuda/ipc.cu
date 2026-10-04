// One-shot all-gather between the GPUs of one host over CUDA IPC (PCIe or NVLink peer to peer), for decode-sized
// gathers. Same protocol as Mia's one-shot RoCE kernel (roce.cu: device epoch, two slots, acquire-polled flags, a
// timeout that poisons), without a host proxy: each rank writes its shard straight into every peer's slot. The IPC
// buffer handling follows b12x's PCIe one-shot collectives (https://github.com/local-inference-lab/b12x,
// b12x/comm/pcie, Apache License 2.0); no b12x source is copied.
//
// Every rank exports one region (ipc.h has the layout); a launch's grid and partition come from the byte count alone,
// so every rank's launch matches. Every launch starts the same way: each block reads the device epoch,
// seq = epoch + 1 (a poisoned runtime returns at once), and the last block to finish (a local arrival counter)
// advances the epoch, so CUDA graphs replay the exchange. Three ways to move block b's contiguous run of 16-byte packs:
//   flag protocol (SM stores, any size up to the slot):
//     1. push the run into slot seq & 1 of every peer (plain 16-byte peer-to-peer stores, no atomics: P2P atomics
//        need not be supported) and into the own place of the output;
//     2. the block meets and arrives on a local counter with a GPU-scope release; the last block to arrive fences
//        once at system scope and writes flag[slot][rank] = seq into every peer: ONE system fence a launch (a
//        system fence costs about half a microsecond on PCIe and they do not overlap, so a fence per warp or per
//        block made large gathers crawl);
//     3. every block polls its own flag[slot][peer] with relaxed system loads, then one acquire load (no fence);
//     4. the block copies run b of every peer's shard out of its own slots (L2 loads), rank by rank.
//   LL protocol (SM stores, small gathers; the idea of NCCL's LL protocol): every 4 payload bytes travel in one 8-byte
//   word beside a 32-bit tag from the sequence, (tag << 32) | payload, written and read as single-copy atomic 64-bit
//   elements, so a word that shows the tag carries its payload and no fence or separate flag is needed:
//     1. push the run as tagged words into slot seq & 1 of every peer, and into the own output;
//     2. each thread polls the words of its packs from every peer until all show the tag, copies the payload out
//        and clears the words (a word of a slot is written once between two clears, so a zero or an older tag never
//        passes for the awaited one).
//   copy-engine protocol (cudaMemcpyAsync peer copies, which graphs capture as memcpy nodes with fixed addresses, so
//   one slot a peer and an acknowledgement instead of two slots):
//     1. ce_begin (one block): wait until every peer acknowledged the last copy-engine launch (it has read that slot);
//     2. the host enqueues one peer copy a peer on side streams (parallel graph branches), joined back;
//     3. ce_finish: the copies have landed (stream order), so one thread fences once and writes ce_flag[rank] = seq
//        in every peer; every block waits for every peer's ce_flag (an acquire load) and copies the peers' shards
//        out; the blocks arrive with a GPU-scope release, and the last one fences once and acknowledges to every
//        peer (ce_ack[rank] = seq), recording the launch as the last copy-engine one.
// A wait past the limit records sequence, peer, block and time in mapped host memory and sets the poison word.
// The output layout is NCCL's (rank r's shard at r * nbytes) and the bytes are copied, never computed, so the result
// equals NCCL's all-gather bit for bit.
// Why two slots are enough for the SM protocols: a rank starts call n + 2 (slot n & 1 again) only after it has every
// peer's data of call n + 1, which a peer sends only after its launch n finished reading (and clearing) slot n & 1
// (stream order); a copy-engine launch between them keeps that true (its copies follow ce_begin, which follows the
// peer's launch n).

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "ipc.h"

namespace {

__device__ __forceinline__ unsigned long long ld_relaxed_gpu_u64(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned ld_relaxed_gpu_u32(const unsigned* p) {
    unsigned v;
    asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long ld_relaxed_sys_u64(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.relaxed.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ unsigned long long ld_acquire_sys_u64(const unsigned long long* p) {
    unsigned long long v;                       // a strong load and an L1 invalidation: no fence (sm_120 SASS)
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    return v;
}
// an L2 load (the L1 may hold an older copy of a slot the peers rewrote); a plain intrinsic, so the compiler keeps
// several in flight, and it stays after the __syncthreads that follows the acquire
__device__ __forceinline__ uint4 ld_cg_v4(const uint4* p) { return __ldcg(p); }
// two 64-bit elements: each a single-copy atomic access (PTX models a vector access as its element accesses)
__device__ __forceinline__ ulonglong2 ld_relaxed_sys_v2_u64(const unsigned long long* p) {
    ulonglong2 v;
    asm volatile("ld.relaxed.sys.global.v2.u64 {%0, %1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_relaxed_sys_v2_u64(unsigned long long* p, unsigned long long a,
                                                      unsigned long long b) {
    asm volatile("st.relaxed.sys.global.v2.u64 [%0], {%1, %2};" ::"l"(p), "l"(a), "l"(b) : "memory");
}
__device__ __forceinline__ void st_global_v4(void* p, uint4 v) {
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}
__device__ __forceinline__ void st_relaxed_sys_u64(unsigned long long* p, unsigned long long v) {
    asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned atom_inc_release_gpu(unsigned* p, unsigned limit) {
    unsigned old;                               // MEMBAR.GPU then the atomic: this block's work before its arrival
    asm volatile("atom.release.gpu.global.inc.u32 %0, [%1], %2;" : "=r"(old) : "l"(p), "r"(limit) : "memory");
    return old;
}
__device__ __forceinline__ void st_relaxed_sys_s64(long long* p, long long v) {
    asm volatile("st.relaxed.sys.global.s64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_relaxed_gpu_u64(unsigned long long* p, unsigned long long v) {
    asm volatile("st.relaxed.gpu.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ void st_release_gpu_u32(unsigned* p, unsigned v) {
    asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_acq_rel_sys() { asm volatile("fence.acq_rel.sys;" ::: "memory"); }
__device__ __forceinline__ unsigned long long globaltimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// a 16-byte pack i of a byte range [base, base + nbytes): whole and aligned in one vector access, else byte by byte
// (a tail pack, or a buffer that is not 16-byte aligned; only small gathers meet either)
__device__ __forceinline__ uint4 load_pack(const unsigned char* base, long long i, long long nbytes, bool aligned) {
    const long long off = i * PACK_BYTES;
    if (aligned && off + PACK_BYTES <= nbytes) return *reinterpret_cast<const uint4*>(base + off);
    unsigned long long lo = 0ull, hi = 0ull;                            // no indexed array: nothing on the stack
    const int n = (int)min((long long)PACK_BYTES, nbytes - off);
    for (int k = 0; k < n; ++k) {
        const unsigned long long byte = base[off + k];
        if (k < 8) lo |= byte << (8 * k);
        else hi |= byte << (8 * (k - 8));
    }
    return make_uint4((unsigned)lo, (unsigned)(lo >> 32), (unsigned)hi, (unsigned)(hi >> 32));
}
__device__ __forceinline__ void store_pack(unsigned char* base, long long i, long long nbytes, bool aligned, uint4 v) {
    const long long off = i * PACK_BYTES;
    if (aligned && off + PACK_BYTES <= nbytes) {
        *reinterpret_cast<uint4*>(base + off) = v;
        return;
    }
    const unsigned long long lo = ((unsigned long long)v.y << 32) | v.x, hi = ((unsigned long long)v.w << 32) | v.z;
    const int n = (int)min((long long)PACK_BYTES, nbytes - off);
    for (int k = 0; k < n; ++k) base[off + k] = (unsigned char)(k < 8 ? lo >> (8 * k) : hi >> (8 * (k - 8)));
}

// a source's slot in a receiver's region (the receiver keeps no slot for itself)
__device__ __forceinline__ long long slot_offset(long long base, int slot, int source, int receiver, int world,
                                                 long long slot_bytes) {
    const int index = source < receiver ? source : source - 1;
    return base + ((long long)slot * (world - 1) + index) * slot_bytes;
}

constexpr int LL_GROUP = 3;                     // peers whose words of a pack the LL receive loads at once

// a source's LL words in this rank's region (four 8-byte words a 16-byte pack)
__device__ __forceinline__ unsigned long long* ll_words(const Regions& regions, const Layout& layout, int slot,
                                                         int source, int rank, int world) {
    return reinterpret_cast<unsigned long long*>(
        regions.base[rank] + slot_offset(layout.ll_base, slot, source, rank, world, layout.ll_slot_bytes));
}

__device__ __forceinline__ unsigned long long* flag_at(char* region, int slot, int source) {
    return reinterpret_cast<unsigned long long*>(region + FLAG_OFFSET) + slot * MAX_WORLD + source;
}

// a wait past the limit: the record for the host (sequence last, after a fence), then the poison word
__device__ __forceinline__ void give_up(long long* err, unsigned* poison, int peer, int block,
                                        unsigned long long waited, unsigned long long seq) {
    st_relaxed_sys_s64(err + 1, peer);
    st_relaxed_sys_s64(err + 2, block);
    st_relaxed_sys_s64(err + 3, (long long)waited);
    fence_acq_rel_sys();
    st_relaxed_sys_s64(err + 0, (long long)seq);
    st_release_gpu_u32(poison, 1u);
}

// wait until *p == want (or >= want when at_least), at most timeout_ns; false when this wait gave up or another
// one did (the runtime is poisoned). Relaxed polls, then one acquire load of the value (a load and an L1
// invalidation, no fence): what the writer released before the value is visible after it.
__device__ __forceinline__ bool wait_for(const unsigned long long* p, unsigned long long want, bool at_least,
                                         const Control& ctl, long long timeout_ns, int peer, int block,
                                         unsigned long long seq) {
    unsigned long long v = ld_relaxed_sys_u64(p);
    if (at_least ? v < want : v != want) {
        const unsigned long long start = globaltimer();
        unsigned polls = 0;
        while (true) {
            v = ld_relaxed_sys_u64(p);
            if (at_least ? v >= want : v == want) break;
            if ((++polls & 255u) != 0u) continue;
            if (ld_relaxed_gpu_u32(ctl.poison) != 0u) return false;
            const unsigned long long waited = globaltimer() - start;
            if ((long long)waited > timeout_ns) {
                give_up(ctl.err, ctl.poison, peer, block, waited, seq);
                return false;
            }
        }
    }
    v = ld_acquire_sys_u64(p);                       // the value only moves forward: still satisfied
    return true;
}

// the launch's sequence, the same in every block (the epoch moves only after the last block arrived), or 0 when
// the runtime is poisoned
__device__ __forceinline__ unsigned long long begin(const Control& ctl) {
    __shared__ unsigned long long s_seq;
    if (threadIdx.x == 0) s_seq = ld_relaxed_gpu_u32(ctl.poison) != 0u ? 0ull : ld_relaxed_gpu_u64(ctl.epoch) + 1ull;
    __syncthreads();
    return s_seq;
}

// whether this block is the last of the launch to arrive on ``counter`` (the increment wraps it to 0 for the next
// launch); every thread gets the answer. With ``release``, the block's reads and writes come before its arrival
// (a GPU-scope release), and the last block acquires them.
__device__ __forceinline__ bool last_block(unsigned* counter, bool release) {
    __shared__ unsigned s_last;
    __syncthreads();
    if (threadIdx.x == 0) {
        const unsigned prior = release ? atom_inc_release_gpu(counter, gridDim.x - 1u)
                                       : atomicInc(counter, gridDim.x - 1u);
        s_last = prior == gridDim.x - 1u ? 1u : 0u;
    }
    __syncthreads();
    return s_last != 0u;
}

__device__ __forceinline__ void advance(const Control& ctl, unsigned long long seq) {
    if (threadIdx.x == 0 && ld_relaxed_gpu_u32(ctl.poison) == 0u) st_relaxed_gpu_u64(ctl.epoch, seq);
}

__device__ __forceinline__ unsigned long long* ce_flag_at(char* region, int source) {
    return reinterpret_cast<unsigned long long*>(region + CE_CTL_OFFSET) + source;
}
__device__ __forceinline__ unsigned long long* ce_ack_at(char* region, int reader) {
    return reinterpret_cast<unsigned long long*>(region + CE_CTL_OFFSET) + MAX_WORLD + reader;
}

__global__ void __launch_bounds__(MAX_THREADS) flag_gather_kernel(
        const unsigned char* __restrict__ in, unsigned char* __restrict__ out, long long nbytes, long long chunk,
        const __grid_constant__ Regions regions, Layout layout, Control ctl, long long timeout_ns, int world,
        int rank) {
    const unsigned long long seq = begin(ctl);
    if (seq == 0ull) return;                         // poisoned: later launches do nothing until the host raises
    const int tid = threadIdx.x, b = blockIdx.x, nt = blockDim.x;
    const int slot = (int)(seq & 1ull);
    const long long packs = (nbytes + PACK_BYTES - 1) / PACK_BYTES;
    const long long lo = (long long)b * chunk, hi = min(packs, lo + chunk);
    const bool in_aligned = (reinterpret_cast<uintptr_t>(in) & (PACK_BYTES - 1)) == 0;
    unsigned char* own = out + (long long)rank * nbytes;
    const bool own_aligned = (reinterpret_cast<uintptr_t>(own) & (PACK_BYTES - 1)) == 0;
    const bool own_copy = own != in;                                     // in place: the shard is already there

    // 1. push this block's run into every peer's slot (peers staggered by rank) and into the own output
    for (long long i = lo + tid; i < hi; i += nt) {
        const uint4 v = load_pack(in, i, nbytes, in_aligned);
        for (int k = 1; k < world; ++k) {
            const int p = (rank + k) % world;
            uint4* dst = reinterpret_cast<uint4*>(regions.base[p] +
                                                  slot_offset(DATA_OFFSET, slot, rank, p, world, layout.slot_bytes));
            st_global_v4(dst + i, v);                                    // a plain peer-to-peer write
        }
        if (own_copy) store_pack(own, i, nbytes, own_aligned, v);
    }
    // 2. every block's stores before the flags: each block arrives with a GPU-scope release; the last one fences
    //    once at system scope (covering every block's stores) and writes the flag into every peer
    if (last_block(ctl.stage, true) && tid == 0) {
        fence_acq_rel_sys();
        for (int k = 1; k < world; ++k) {
            const int p = (rank + k) % world;
            st_relaxed_sys_u64(flag_at(regions.base[p], slot, rank), seq);
        }
    }
    // 3. wait for every peer's flag (one thread a peer in every block)
    if (tid < world && tid != rank)
        wait_for(flag_at(regions.base[rank], slot, tid), seq, false, ctl, timeout_ns, tid, b, seq);
    __syncthreads();
    // 4. copy every peer's run out of the slots, in rank order
    if (ld_relaxed_gpu_u32(ctl.poison) == 0u) {
        const char* mine = regions.base[rank];
        for (int s = 0; s < world; ++s) {
            if (s == rank) continue;
            const uint4* src = reinterpret_cast<const uint4*>(
                mine + slot_offset(DATA_OFFSET, slot, s, rank, world, layout.slot_bytes));
            unsigned char* dst = out + (long long)s * nbytes;
            const bool dst_aligned = (reinterpret_cast<uintptr_t>(dst) & (PACK_BYTES - 1)) == 0;
            for (long long i = lo + tid; i < hi; i += nt) store_pack(dst, i, nbytes, dst_aligned, ld_cg_v4(src + i));
        }
    }
    if (last_block(ctl.arrive, false)) advance(ctl, seq);
}

__global__ void __launch_bounds__(MAX_THREADS) ll_gather_kernel(
        const unsigned char* __restrict__ in, unsigned char* __restrict__ out, long long nbytes, long long chunk,
        const __grid_constant__ Regions regions, Layout layout, Control ctl, long long timeout_ns, int world,
        int rank) {
    const unsigned long long seq = begin(ctl);
    if (seq == 0ull) return;
    const int tid = threadIdx.x, b = blockIdx.x, nt = blockDim.x;
    const int slot = (int)(seq & 1ull);
    const unsigned tag = (unsigned)seq != 0u ? (unsigned)seq : 1u;    // never 0, a cleared word
    const unsigned long long high = (unsigned long long)tag << 32;
    const long long packs = (nbytes + PACK_BYTES - 1) / PACK_BYTES;
    const long long lo = (long long)b * chunk, hi = min(packs, lo + chunk);
    const bool in_aligned = (reinterpret_cast<uintptr_t>(in) & (PACK_BYTES - 1)) == 0;
    unsigned char* own = out + (long long)rank * nbytes;
    const bool own_aligned = (reinterpret_cast<uintptr_t>(own) & (PACK_BYTES - 1)) == 0;
    const bool own_copy = own != in;

    // 1. push: a pack's four payload words as four tagged 64-bit words (32 bytes) into every peer's slot
    for (long long i = lo + tid; i < hi; i += nt) {
        const uint4 v = load_pack(in, i, nbytes, in_aligned);
        for (int k = 1; k < world; ++k) {
            const int p = (rank + k) % world;
            unsigned long long* dst = reinterpret_cast<unsigned long long*>(
                regions.base[p] + slot_offset(layout.ll_base, slot, rank, p, world, layout.ll_slot_bytes)) + 4 * i;
            st_relaxed_sys_v2_u64(dst, high | v.x, high | v.y);
            st_relaxed_sys_v2_u64(dst + 2, high | v.z, high | v.w);
        }
        if (own_copy) store_pack(own, i, nbytes, own_aligned, v);
    }
    // 2. receive: pack by pack, the four words of up to three peers at once (their loads in flight together), each
    //    waited for until every word shows the tag, then copied out and cleared
    bool stopped = false;
    for (long long i = lo + tid; i < hi && !stopped; i += nt) {
        for (int k0 = 1; k0 < world && !stopped; k0 += LL_GROUP) {
            ulonglong2 a[LL_GROUP], c[LL_GROUP];
#pragma unroll
            for (int j = 0; j < LL_GROUP; ++j) {
                if (k0 + j < world) {
                    const unsigned long long* w = ll_words(regions, layout, slot, (rank + world - k0 - j) % world, rank,
                                                           world) + 4 * i;
                    a[j] = ld_relaxed_sys_v2_u64(w);
                    c[j] = ld_relaxed_sys_v2_u64(w + 2);
                }
            }
#pragma unroll
            for (int j = 0; j < LL_GROUP; ++j) {
                if (k0 + j < world && !stopped) {
                    const int s = (rank + world - k0 - j) % world;
                    unsigned long long* w = ll_words(regions, layout, slot, s, rank, world) + 4 * i;
                    unsigned polls = 0;
                    unsigned long long start = 0ull;
                    while ((unsigned)(a[j].x >> 32) != tag || (unsigned)(a[j].y >> 32) != tag ||
                           (unsigned)(c[j].x >> 32) != tag || (unsigned)(c[j].y >> 32) != tag) {
                        if ((++polls & 255u) == 0u) {
                            if (ld_relaxed_gpu_u32(ctl.poison) != 0u) {   // another wait gave up first
                                stopped = true;
                                break;
                            }
                            const unsigned long long now = globaltimer();
                            if (start == 0ull) {
                                start = now;
                            } else if ((long long)(now - start) > timeout_ns) {
                                give_up(ctl.err, ctl.poison, s, b, now - start, seq);
                                stopped = true;
                                break;
                            }
                        }
                        a[j] = ld_relaxed_sys_v2_u64(w);
                        c[j] = ld_relaxed_sys_v2_u64(w + 2);
                    }
                    if (!stopped) {
                        unsigned char* dst = out + (long long)s * nbytes;
                        const bool dst_aligned = (reinterpret_cast<uintptr_t>(dst) & (PACK_BYTES - 1)) == 0;
                        store_pack(dst, i, nbytes, dst_aligned,
                                   make_uint4((unsigned)a[j].x, (unsigned)a[j].y, (unsigned)c[j].x, (unsigned)c[j].y));
                        // cleared for this slot's next use (a plain store: the peer writes these words again only
                        // after this launch has ended, see the header)
                        st_global_v4(w, make_uint4(0u, 0u, 0u, 0u));
                        st_global_v4(w + 2, make_uint4(0u, 0u, 0u, 0u));
                    }
                }
            }
        }
    }
    if (last_block(ctl.arrive, false)) advance(ctl, seq);
}

// copy-engine protocol, before the copies: every peer has read (acknowledged) the last copy-engine launch's slot
__global__ void ce_begin_kernel(const __grid_constant__ Regions regions, Control ctl, long long timeout_ns, int world,
                                int rank) {
    const unsigned long long seq = begin(ctl);
    if (seq == 0ull) return;
    const int t = threadIdx.x;
    if (t < world && t != rank) {
        const unsigned long long need = ld_relaxed_gpu_u64(ctl.last_ce);
        wait_for(ce_ack_at(regions.base[rank], t), need, true, ctl, timeout_ns, t, -1, seq);
    }
}

// copy-engine protocol, after the copies: flags, wait, copy-out, acknowledgements, epoch
__global__ void __launch_bounds__(MAX_THREADS) ce_finish_kernel(
        const unsigned char* __restrict__ in, unsigned char* __restrict__ out, long long nbytes, long long chunk,
        const __grid_constant__ Regions regions, Layout layout, Control ctl, long long timeout_ns, int world,
        int rank) {
    const unsigned long long seq = begin(ctl);
    if (seq == 0ull) return;
    const int tid = threadIdx.x, b = blockIdx.x, nt = blockDim.x;
    const long long packs = (nbytes + PACK_BYTES - 1) / PACK_BYTES;
    const long long lo = (long long)b * chunk, hi = min(packs, lo + chunk);
    // 1. this launch's copies have landed in every peer (they precede this kernel in stream order): say so, after
    //    one system fence
    if (b == 0 && tid == 0) {
        fence_acq_rel_sys();
        for (int k = 1; k < world; ++k) {
            const int p = (rank + k) % world;
            st_relaxed_sys_u64(ce_flag_at(regions.base[p], rank), seq);
        }
    }
    // the own shard meanwhile
    unsigned char* own = out + (long long)rank * nbytes;
    if (own != in) {
        const bool in_aligned = (reinterpret_cast<uintptr_t>(in) & (PACK_BYTES - 1)) == 0;
        const bool own_aligned = (reinterpret_cast<uintptr_t>(own) & (PACK_BYTES - 1)) == 0;
        for (long long i = lo + tid; i < hi; i += nt)
            store_pack(own, i, nbytes, own_aligned, load_pack(in, i, nbytes, in_aligned));
    }
    // 2. every peer's copy has landed here
    if (tid < world && tid != rank)
        wait_for(ce_flag_at(regions.base[rank], tid), seq, false, ctl, timeout_ns, tid, b, seq);
    __syncthreads();
    // 3. copy out, rank by rank
    if (ld_relaxed_gpu_u32(ctl.poison) == 0u) {
        const char* mine = regions.base[rank];
        for (int s = 0; s < world; ++s) {
            if (s == rank) continue;
            const uint4* src = reinterpret_cast<const uint4*>(
                mine + layout.ce_base + (long long)(s < rank ? s : s - 1) * layout.ce_slot_bytes);
            unsigned char* dst = out + (long long)s * nbytes;
            const bool dst_aligned = (reinterpret_cast<uintptr_t>(dst) & (PACK_BYTES - 1)) == 0;
            for (long long i = lo + tid; i < hi; i += nt) store_pack(dst, i, nbytes, dst_aligned, ld_cg_v4(src + i));
        }
    }
    // 4. the last block: every block has read its runs (each arrives with a GPU-scope release), so one system fence
    //    and an acknowledgement to every peer; the launch is the last copy-engine one
    if (last_block(ctl.arrive, true) && tid == 0 && ld_relaxed_gpu_u32(ctl.poison) == 0u) {
        fence_acq_rel_sys();
        for (int k = 1; k < world; ++k) {
            const int p = (rank + k) % world;
            st_relaxed_sys_u64(ce_ack_at(regions.base[p], rank), seq);
        }
        st_relaxed_gpu_u64(ctl.last_ce, seq);
        st_relaxed_gpu_u64(ctl.epoch, seq);
    }
}

}  // namespace

void launch_gather(cudaStream_t stream, int protocol, const at::Tensor& in, at::Tensor& out, long long nbytes,
                   long long chunk, int grid, int threads, const Regions& regions, const Layout& layout,
                   const Control& ctl, long long timeout_ns, int world, int rank) {
    const auto* src = reinterpret_cast<const unsigned char*>(in.data_ptr());
    auto* dst = reinterpret_cast<unsigned char*>(out.data_ptr());
    if (protocol == PROTO_LL) {
        ll_gather_kernel<<<(unsigned)grid, (unsigned)threads, 0, stream>>>(src, dst, nbytes, chunk, regions, layout,
                                                                          ctl, timeout_ns, world, rank);
    } else {
        flag_gather_kernel<<<(unsigned)grid, (unsigned)threads, 0, stream>>>(src, dst, nbytes, chunk, regions, layout,
                                                                            ctl, timeout_ns, world, rank);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_ce_begin(cudaStream_t stream, const Regions& regions, const Control& ctl, long long timeout_ns, int world,
                     int rank) {
    ce_begin_kernel<<<1, 32, 0, stream>>>(regions, ctl, timeout_ns, world, rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_ce_finish(cudaStream_t stream, const at::Tensor& in, at::Tensor& out, long long nbytes, long long chunk,
                      int grid, int threads, const Regions& regions, const Layout& layout, const Control& ctl,
                      long long timeout_ns, int world, int rank) {
    ce_finish_kernel<<<(unsigned)grid, (unsigned)threads, 0, stream>>>(
        reinterpret_cast<const unsigned char*>(in.data_ptr()), reinterpret_cast<unsigned char*>(out.data_ptr()), nbytes,
        chunk, regions, layout, ctl, timeout_ns, world, rank);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
