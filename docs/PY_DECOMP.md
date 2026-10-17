python round decomposition CAPTURED (its own TF_GLM_MULTI_PROFILE=25 profiler; 24 lines banked at docs/py_round_profile.txt)

The live lone-stream decode round of the python engine (4bpw, 8K,
NCCL-SHM, one-stream CUDA graphs), averaged over 25-round windows
during a 1200-token run at 147.7 tok/s:

  wall 14.2-14.7 ms  =  plan 0.0 + message 0.2 + propose 1.3 (DFlash2
  draft, host) + verify 0.2 (host) + sample 12.4-12.9 (includes the
  wait for the verify graph) + accept 0.0 + commit 0.1 + taps 0.1
  verify GPU 12.5-13.0 ms   <- the entire forward window
  1 stream, 3.2-3.8 rows, 1.84-2.20 tokens a round -> 147-150 tok/s
  startup tune: window ms by rows 1:9.01 ... 3:12.69 ... 8:17.78

The zig engine at the same widths (settled 1500-token run): wall
21.6 ms, 2.11 tokens a round -> 97.5 tok/s; GPU busy ~19 ms (nsys:
MoE experts 6.8 + NCCL allgather 3.2 + KDA chain 4.8 + attention ~0.3
+ misc).

CONCLUSION: python's ENTIRE verify GPU window (12.5-13.0 ms) is less
than the zig engine's GPU busy time by ~6 ms, and its host stages are
negligible (draft included: DFlash2 block 1.2 ms). The 1.5x-per-round
gap is inside the forward's kernels: MoE expert path and KDA chain.
Not transport (same SHM), not graphs (zig measured graph-neutral), not
prefetch (l2pf measured neutral), not KV dtype, not host overhead.

NEXT-WORK MAP (the port targets, in size order):
1. MoE expert path: python exl3_mm.routed + its launch table
   (docs: launch-table-both.json holds q4_dec split-k/f8_dec configs;
   our mmaMixed at 6.8 ms reads the same expert bytes) - compare the
   routed kernel's reads-per-item and tile mapping; the python table
   entry format is banked in the tune-table dump.
2. KDA chain 4.8 ms: python kda_step tune rows (warps 4, tr 16) are
   in the same table; our chain runs prep/step/out per layer.
3. The python engine remains bootable in one command (the start-prof4
   container recipe with TF_GLM_MULTI_PROFILE) as the live reference.
