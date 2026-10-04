// Shared by ipc.cpp (the region, IPC handles, copies and launches) and ipc.cu (the one-shot all-gather kernels).
#pragma once

#include <ATen/ATen.h>
#include <cuda_runtime.h>

constexpr int IPC_ABI = 3;                      // ipc.py checks it; bump with any change to the layout below
constexpr int MAX_WORLD = 8;                    // ranks at most
constexpr int MAX_BLOCKS = 128;                 // blocks a launch at most
constexpr int MAX_THREADS = 1024;
constexpr int PACK_BYTES = 16;                  // the kernels move 16-byte packs

// the protocols (ipc.py's names): SM stores and a flag a block, SM stores of tagged words, copy-engine copies
constexpr int PROTO_FLAG = 0, PROTO_LL = 1, PROTO_CE = 2;

// One rank's exported region, in bytes (written by the peers, read here):
//   flag[2][MAX_WORLD] u64                  the flag protocol's flags: one a slot and source, for a whole launch
//   ce_flag[MAX_WORLD], ce_ack[MAX_WORLD]   the copy-engine protocol's flags and acknowledgements (u64)
//   data[2][world - 1][slot_bytes]          the flag protocol's slots (from DATA_OFFSET)
//   ll[2][world - 1][ll_slot_bytes]         the LL protocol's slots (none when its limit is 0)
//   ce[world - 1][ce_slot_bytes]            the copy-engine protocol's slots (none when it is off)
constexpr long long FLAG_OFFSET = 0;
constexpr long long CE_CTL_OFFSET = 2LL * MAX_WORLD * 8;
constexpr long long DATA_OFFSET = 4096;

// every rank's region as mapped in this process (this rank's own entry: the local region)
struct Regions {
    char* base[MAX_WORLD];
};

struct Layout {
    long long slot_bytes, ll_base, ll_slot_bytes, ce_base, ce_slot_bytes;
};

// this rank's device control words (not exported) and the mapped host error record
struct Control {
    unsigned long long* epoch;                  // the last finished launch's sequence
    unsigned* arrive;                           // blocks of the running launch that finished
    unsigned* poison;                           // set when a wait gave up: later launches do nothing
    unsigned long long* last_ce;                // the last copy-engine launch's sequence
    unsigned* stage;                            // blocks of the running flag-protocol launch that pushed their run
    long long* err;                             // {failed sequence, peer, block, waited ns} in mapped host memory
};

// a launch of the flag or LL protocol
void launch_gather(cudaStream_t stream, int protocol, const at::Tensor& in, at::Tensor& out, long long nbytes,
                   long long chunk, int grid, int threads, const Regions& regions, const Layout& layout,
                   const Control& ctl, long long timeout_ns, int world, int rank);
// the copy-engine protocol around its copies: before them (wait until every peer has read the last copy-engine
// slot), after them (flags, wait, copy-out, acknowledgements, epoch)
void launch_ce_begin(cudaStream_t stream, const Regions& regions, const Control& ctl, long long timeout_ns, int world,
                     int rank);
void launch_ce_finish(cudaStream_t stream, const at::Tensor& in, at::Tensor& out, long long nbytes, long long chunk,
                      int grid, int threads, const Regions& regions, const Layout& layout, const Control& ctl,
                      long long timeout_ns, int world, int rank);
