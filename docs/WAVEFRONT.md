# Decode-round exchange pipelining (wavefront) — implementation spec

Status: designed and kernel-validated; NOT implemented. Everything below was
verified against the current tree by reading the kernels' signatures and
launch sites (session of 2026-10-17). This is the plan for the implementer.

## Why

A decode round pays ~90 serialized NCCL exchanges (2 per layer x 45), each a
~30 us round-trip synchronization between the two ranks — ~3 ms/round of pure
peer-wait that cannot be removed by protocol (RING, LL, one-shot SHM gather
and split send/recv all measured within noise of each other; split mode is
worse: it adds a second glued-row exchange per site). CUDA graphs batch the
launches but the waits remain (round-time neutral). The only cure is overlap:
one row-group's exchange running while the next group computes.

## The wall the design must respect

Rows are independent through the whole forward (row r's layer-j output feeds
only row r's layer-j+1 input), so a 2-group wavefront is mathematically
sound. The implementation constraint: the SOLO kernels index rows from the
pointers they are given (base-relative), but cache positions come from a
single device word `st.pos_dev` (`pos[0]`), with each row's cache slot and
causal limit derived as pos + row.

## Verified kernel facts (what makes it possible)

- glue.hcPre / hcPost: fully base-relative. Take x, gathered g, scratch part
  with row strides; grid.x = rows. Group calls: pass x + off*D*2, g + slot
  base + off*D*4, part + off*hcPartBytes(1) (check units: stride arg is s*d),
  rows = group size. No kernel changes needed.
- kda_block: body takes x + x_stride (offsettable); saves (save_k/v/g/b) and
  the proj staging are per-layer bases with row indexing inside — offset by
  the group's rows: saves.k = st.save_k + li*... + off*heads*dk*4 etc.
- dsa_block: body takes x + x_stride. Cache write position: pass a SECOND
  device word pos2 = pos + off for group B (kernels derive per-row positions
  and limits from the word they are handed — group B gets pos2 and computes
  pos+off+r' correctly for both writes and causal masking). pos2 is one u32
  device buffer, written once per round from host (eager) or via pinned
  staging if the round is graphed.
- The mixer's out slot: gathered + rank*FULL_ROWS*D*4 + off*D*4; the group's
  exchange then allgathers [group_rows x D] at that offset.
- MoE (ffn): x offsets fine; pick/wts/mlog offset by off*topk*4 etc. THE
  RACE: the plan kernel (experts.plan) is sequential state — two groups' plan
  kernels on one buffer race. Give group B its own plan region: plan buffers
  are sized for the full window's pairs; group B's region = base + the
  full-window offsets (members/items/counts/rank/hist each). The grouped
  expert kernel takes plan pointers per launch, so per-group pointers work.

## The wavefront loop (whole() restructured)

Split R rows into A = [0, a), B = [a, R) with a = R/2 (TF_GLM_WAVE_SPLIT to
tune; a = R rounds to eager for R <= 3).

Per layer j, using two streams (f.stream, p.side) + 4 events per layer site:

  pre(A); mixer(A, pos_dev);
  [fill A slot -> allgather A on side stream]
  || pre(B); mixer(B, pos2_dev);         -- B computes while A exchanges
  [allgather B on side]
  wait A -> post(A); pre_ffn(A); ffn(A);
  [allgather A2 on side]
  || wait B -> post(B); pre_ffn(B); ffn(B);
  [allgather B2 on side]
  wait A2, B2.

The gather helper becomes group-aware (rows, base offset in gathered). Events:
reuse the Pieces event arrays (filled/gathered) or add a small set.

## Gates

- Env TF_GLM_WAVE=1 enables; default off. Incompatible with TF_GLM_GRAPHS
  (different launch structure): if both set, wave wins and graphs are
  skipped for the round.
- Correctness gate: the solo 600-token lighthouse generation must reproduce
  sha ef4917015bea (greedy, eager reference). Then the 5-prompt battery.
- Expected: ~2.5-3 ms/round hidden; measure round ms from done-lines
  (tokens/rounds) before declaring anything.

## Follow-ons (same session if wave lands)

- KDA fb/gb fusion: the two b16.matmul calls in kda project() read adjacent
  columns of s.p ([fa,fa+dk) for fb, [fa+dk,fa+2dk) for gb) with independent
  outputs ka/kg — block-diagonal, so a fused batched kernel (two weight
  pointers, one launch) saves ~1 launch x 34 layers and one read of x.
  Also: the EXL3 qkv path's two copyRows can fold into the proj epilogue.
- The 160+ path: wave (~3 ms) + KDA fusion (~1.5 ms) + MoE bandwidth work
  (~6.8 ms today, near the 3.5bpw floor) lands ~15 ms rounds ~ 135-145 tok/s.
