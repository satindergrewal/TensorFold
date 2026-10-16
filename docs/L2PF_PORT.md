L2 prefetch port plan - the python engine's decode-round advantage (from its own source)

Extracted from the serving image (families/glm5_next/cuda/l2pf.py + l2pf.cu,
both banked beside this file; the mechanism originates in
jayleaton/glm53-tensorfold-spark patches/0460, Apache-2.0, (c) 2026 Jay
Leaton - credit him in the port's commit).

WHY THIS IS THE GAP: a decode layer alternates DRAM-bound matmuls with
phases that leave DRAM idle - the two all-gather waits, the hc glue,
router/top-k, the KDA recurrence, the latent attention core. The python
engine fires a small kernel on a SIDE STREAM at those idle phases that
prefetches the weights the main stream reads NEXT into L2
(cp.async.bulk.prefetch.L2 / evict_last lines / ld.global.cg touch). The
following matmul then finds up to 8 MiB of its bytes in L2 instead of
DRAM. It also reorders the MoE: the shared expert runs BEFORE the routed
experts so the routed ~100+MB stream does not evict it.

The zig engine has none of this, and its two biggest round costs are
exactly the attacked ones: MoE expert reads 6.8ms (DRAM-bound) + NCCL
all-gathers 3.2ms (DRAM-idle waits). The wavefront tried to overlap
compute with the gathers; l2pf overlaps PREFETCH with them - simpler and
proven at 155-158 tok/s in the python engine on this box.

PORT (zig):
1. kernels/cuda: port l2pf.cu (64 lines - the bulk/lines/touch
   prefetch kernel, a (address, bytes) pieces table on device).
2. cuda_forward.zig whole(): at each layer's mixer-site gather (site
   'a'): side-stream prefetch of the ffn hc weights, router, shared
   expert gate/up+down; at the ffn-site gather (site 'f'): next layer's
   hc + input norms + projections; after the last layer: final norm +
   head. Use the pieces side stream (already exists for the wavefront).
3. moe.zig: when the prefetcher is on, run the shared expert before the
   routed experts (same kernels, disjoint buffers - bit-identical).
4. Env TF_GLM_L2PF (off by default), TF_GLM_L2PF_MB=8, SITES=afo.
5. Gate: greedy sha ef4917015bea unchanged (prefetch only reads);
   then the A/B and round-time profile - expect the MoE's effective
   DRAM traffic to drop and rounds toward the python engine's 9-17ms.

Reference numbers on this box: python 155-158 single-stream; zig 92-102.
