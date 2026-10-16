# Decode-round exchange pipelining (wavefront) — implementation spec v2 (complete)

v2 adds the CLONE DESIGN (no kernel changes at all), every buffer's row
stride, the KDA parity rule for commit, and the remaining field lists.
Everything below was verified against the tree by reading signatures,
buffer-size tables and launch sites (2026-10-17 sessions).

## Design: clone Ranks, never touch kernels

Every kernel in the round is base-relative (it indexes rows from the pointers
it is handed). So a row-group is just a *cloned Rank whose Buffer/State
pointers are offset*. f.pre / f.mixer / f.post run UNCHANGED on the clone.
No kernel signatures change; no new CUDA is written.

Per round, with a = R/2, b = R - a, groups A = rows [0,a), B = [a,R):

  var bA = r.b.*;                       // gathered -> group A's scratch region
  var stA = r.st.*;                     // untouched fields, pointers shared
  var bB = r.b.*;                       // every row-sized field + its offsets
  var stB = r.st.*;                     // pos + a, pos_dev -> pos2, saves/proj
                                        // offset, cur -> flipped copy
  var rA = Rank{ .w, .st = &stA, .b = &bA };
  var rB = Rank{ .w, .st = &stB, .b = &bB };

Loop per layer (main stream = f.stream, side = pieces.side; events ev0..ev3):

  fA.pre(rA, attn_hc, in_norm, a); fA.mixer(rA, l, li, di, a, nchA, slot(rA, a));
  record ev0 (main);
  fB.pre(rB, ..., b); fB.mixer(rB, ..., b, nchB, slot(rB, b));   // B mixes while
  record ev2 (main);                                             // A will exchange
  side.wait(ev0); allgather A (scratchA slice, a*D f32); record ev1;
  side.wait(ev2); allgather B (scratchB slice, b*D f32); record ev3;
  main.wait(ev1); fA.post(rA, a);          // post reads the clone's gathered
  main.wait(ev3); fB.post(rB, b);
  // ffn site stays WHOLE on the real rank (v1: no MoE plan regions needed):
  f.pre(r0, ffn_hc, post_norm, R); f.ffn(r0, l, R, slot(r0, R));
  f.gather(R); f.post(r0, R);

slot(rA, a) automatically lands in the group scratch because the clone's
b.gathered points at it. KDA's per-row recurrence is safe: A and B launch
sequentially on the main stream, so B's chain sees A's ring writes.
The DSA group positions: stB.pos_dev = pos2 (a second device u32 written
per round as pos + a), stB.pos = pos + a; kernels derive per-row cache
slots and causal limits from the word they are handed.

## Row strides for the clone offsets (D = hidden 4096, S = hc 4, LL = kda layers)

Buffers (from the sizes table, one row each):
  x:        S*D*2      normed:  D*2        xs:   (D/64)*4
  post:     S*4        comb:    S*S*4      hcpart: 32*hc_blocks*4   (linear!)
  gathered: (per group: redirect to the scratch region, not offset)
  b.kda:  ka, kg: n*2 (n = LL*kda.dk); saves.out: n*2; wide.q: n*4; wide.y: n*2
          (p, mm_part shared: p is overridden from st; mm_part used sequentially)
  b.dsa:  dp: (q_lora+lw)*2   qr: q_lora*2        xs_qr: (q_lora/64)*4
          lat: lw*2           xs_lat: (lw/64)*4   q: heads*qk*2
          ikr: (i_dim+i_heads)*2   igr: i_dim*4   router_part: 128*4 (middle dim)
          qa, ol: heads*lw*2  vn: heads*dv*2       qi: i_heads*i_dim*2
          select.scores: np_max*4  ls, li: select_list*4
          select.pools: topk_pools*8  tokens: sparse.width*4  counts: 4
          partials (group B only): po + a*nchB*heads*lw*4, pm/pl likewise
State (group B): save_k: heads*dk*4   save_v: heads*dk*2   save_g: heads*dk*4
  save_b: heads*4   proj: proj_row*2 (after the per-layer base)   pos += a
  pos_dev -> pos2   cur -> a flipped []u1 copy (never mutate the real one)

Scratch: one device buffer, fp32 4*window_rows*D: group A's allgather region
at +0 ([2][a][D] in-place slice pattern), group B's at +2*a*D*4.

## The KDA parity rule (the one subtle invariant)

Eager: one mixer launch per KDA layer reads ring half[cur], writes half[1-cur];
commit replays from recAt(1 - cur) then flips st.cur.
Wave: A reads half[cur] -> half[1-cur]; B reads half[1-cur] -> half[cur].
After a wave round the NEWEST state is half[cur] — so commit must replay from
recAt(cur) and NOT flip. Give Forward a `wave` flag; commit becomes:

  const src = if (f.wave) cur else 1 - cur;   // replay layers read recAt(src)
  if (!f.wave) for (st.cur) |*x| x.* = 1 - x.*;

Transitions are consistent: eager->wave (cur points at newest, A reads it),
wave->eager (newest at half[cur], eager reads it, writes 1-cur, commit flips).

## Gates and invariants

- TF_GLM_WAVE=1 enables; whole() dispatches to wholeWave for decode windows
  (not prefill, no segs) with R >= 4; else the existing path.
- Wave rounds are EAGER (never graph-captured; if both envs set, wave wins and
  verify skips its graph branch for the round).
- pos2: one u32 device buffer, written per round (pageable 4B H2D, eager ok).
- Correctness gate: the 600-token lighthouse generation reproduces sha
  ef4917015bea; then the 5-prompt battery; then round-ms from done-lines.
- Expected: hides ~45 of the 90 waits in v1 (~1.3-1.5 ms/round); v2 extends
  the same clone trick to the ffn site (needs the MoE plan regions from v1
  spec: plan buffers are full-window-sized; group B's region sits past the
  full offsets; the grouped kernel takes plan pointers per launch) for the
  full ~2.5-3 ms.

## Follow-ons

- KDA fb/gb fusion (block-diagonal batched bmm, two weight pointers, one
  launch; fold the EXL3 qkv copyRows into the proj epilogue): ~1.5-2 ms.
- MoE bandwidth (6.8 ms, near the 3.5bpw floor) — last lever.
- Wave + graphs could coexist later (graphs of the wave structure, parity
  and pos2 through pinned staging) — only after wave is proven eager-exact.
