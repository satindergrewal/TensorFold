"""Wide verify windows for --parallel (TF_GLM_MULTI_WINDOW up to 256 rows): the captured widths, a round's padding to
them, and the window's bytes past the 64 rows the startup estimate always held.

At 16-32 streams a 63-row window leaves each stream one to four rows a round. A wider window lets every stream keep
its drafts: 32 streams of a pending token and seven drafts are 256 rows. The decode path takes such windows as they
are: its buffers are decode buffers sized by the window (no prompt kernel runs on them), and every kernel's choice by
rows (row tiles of the 4-bit, BF16 and FP8 matmuls and the router, heads a latent attention tile, rows a latent
absorb program, the routed experts' kernels) keeps each row's chain, so a row's bits never depend on the window.
tests/K3/test_wide_window_gpu.py checks that row by row at every row count 1-256.

CUDA graphs: every width up to 64 rows is captured as before. TF_GLM_MULTI_GRAPH_STEP (default 1: every width up to
the window) captures the widths past 64 every STEP rows only (and the window itself): fewer graphs, a shorter start.
A round whose rows fall between two captured widths then pads its window to the next one with extra draft rows,
repeating a stream's last token (streams without a grammar, the fewest rows first, within each stream's room).
Drafts only propose: a padded row is kept only where it equals the stream's own sample at that position, so replies
keep their bits; the draft policies and the reply's counts see the rows the drafters proposed, not the padding. A
round that cannot pad (every stream under a grammar, no room) runs its own width eagerly.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab (the multi-stream engine and its batched verify windows)."""

from __future__ import annotations

import os
from typing import Callable, Sequence

ALL_UP_TO = 64              # every window width up to this many rows is captured
SEG_MOST = 32               # rows a stream's padded window may reach (its index ring holds 64 rows and the 3 before)


def graph_step(value: str | None = None) -> int:
    """TF_GLM_MULTI_GRAPH_STEP: captured widths past ALL_UP_TO rows every this many rows (1 to 64; default 1)."""

    v = (os.environ.get("TF_GLM_MULTI_GRAPH_STEP", "") if value is None else value).strip()
    step = 1 if v == "" else int(v) if v.isdecimal() else -1
    if not 1 <= step <= 64:
        raise ValueError(f"TF_GLM_MULTI_GRAPH_STEP: 1 to 64 rows, not {v!r}")
    return step


def code() -> int:
    """The graph step for the ranks' start comparison (every rank must capture and pad alike)."""
    return graph_step()


def widths(window: int, step: int | None = None) -> list[int]:
    """The batched window widths that get a CUDA graph: 1 .. min(window, ALL_UP_TO), then every ``step`` rows to the
    window, and the window itself."""

    step = graph_step() if step is None else step
    out = list(range(1, min(window, ALL_UP_TO) + 1))
    out += list(range(ALL_UP_TO + step, window, step)) if window > ALL_UP_TO else []
    if window > ALL_UP_TO:
        out.append(window)
    return sorted(set(out))


def width_for(R: int, captured: Sequence[int]) -> int | None:
    """The smallest captured width of at least R rows (None: past every one)."""

    best = None
    for w in captured:
        if w >= R and (best is None or w < best):
            best = w
    return best


def plan(lengths: Sequence[int], eligible: Sequence[bool], room: Sequence[int], target: int) -> list[int] | None:
    """Rows to add to each segment so ``lengths`` sum to ``target``: one at a time to the eligible segment with the
    fewest rows (the earliest of equals), each within its ``room``. [] when nothing is needed; None when the target
    cannot be reached."""

    need = target - sum(lengths)
    if need <= 0:
        return []
    pads = [0] * len(lengths)
    while need > 0:
        best = None
        for k, ok in enumerate(eligible):
            if ok and pads[k] < room[k] and (best is None or lengths[k] + pads[k] < lengths[best] + pads[best]):
                best = k
        if best is None:
            return None
        pads[best] += 1
        need -= 1
    return pads


def pad(segments: list, eligible: Sequence[bool], room_of: Callable[[object], int], captured: Sequence[int],
        most: int = SEG_MOST) -> list[int]:
    """Pad a round's segments (``verify.Segment``: st, tokens) in place to the next captured width: each padded
    segment repeats its last token. Returns the rows added to each segment ([] when none were needed or the width
    cannot be reached, so the window runs as it is)."""

    lengths = [len(s.tokens) for s in segments]
    R = sum(lengths)
    target = width_for(R, captured)
    if target is None or target == R:
        return []
    room = [max(0, min(most - n, room_of(s))) for s, n in zip(segments, lengths)]
    pads = plan(lengths, eligible, room, target)
    if not pads:
        return []
    for s, p in zip(segments, pads):
        if p:
            s.tokens = list(s.tokens) + [s.tokens[-1]] * p
    return pads


def row_bytes(t: dict, world: int, share=None) -> int:
    """Bytes one more row of a batched verify window takes on a rank (``forward.Buffers`` with its expert scratch,
    the latent scratch, the window's KDA projections and replay scratch, the selection's tokens and split scratch;
    the selection's fp32 pool scores are counted with the pool), from the text config, rounded up generously."""

    from tensorfold.cuda.geometry import even_share, layer_counts

    share = even_share(world) if share is None else share
    D, S = int(t["hidden_size"]), int(t.get("hc_mult", 4))
    linear, _ = layer_counts(t)
    lin = t.get("linear_attn_config") or {}
    LL = share("lin", int(lin.get("num_heads", t.get("linear_num_heads", 64))))
    ld = int(lin.get("head_dim", t.get("linear_head_dim", 128)))
    HL = share("heads", int(t["num_attention_heads"]))
    qk = int(t["qk_nope_head_dim"]) + int(t.get("qk_rope_head_dim", 0))
    vd, lw = int(t["v_head_dim"]), int(t.get("kv_lora_rank", 512))
    slots = int(t["num_experts_per_tok"]) + 1
    ml = share("moe", int(t["moe_intermediate_size"]))
    dl = share("dense", int(t.get("intermediate_size", ml)))
    V = share("vocab", int(t["vocab_size"]))
    experts = int(t["n_routed_experts"])
    qi = int(t.get("index_n_heads", 32)) * int(t.get("index_head_dim", 128))
    buffers = (S * D * 2 + 8 * D * 2 + 3 * D * 4 + world * D * 4 + slots * D * 4 + 2 * slots * D * 2
               + 2 * slots * ml * 2 + slots * max(8 * ml, D) * 4 + V * 2 + 8 * 16384 * 4 + 5 * D * 2
               + HL * (2 * qk + vd) * 2 + (int(t.get("q_lora_rank", D)) + lw) * 4 + qi * 2 + 4 * LL * 128 * 2
               + 3 * dl * 2 + experts * 4 + 48 * 32 * 4 + 16384)
    latent = 7 * HL * (lw + 2) * 4 + 2 * HL * lw * 2
    kda = linear * ((3 * LL * ld + 2 * ld + LL) * 2 + LL * (ld * 2 + ld * 4 + ld * 2 + ld * 4 + 4))
    select = 2051 * 4 + 4 + (1024 + 64 * 257) * 4
    return int(1.1 * (buffers + latent + kda + select))


def extra_bytes(t: dict, world: int, window: int, share=None) -> int:
    """The window's bytes past the ALL_UP_TO rows the startup estimate's decode buffers always held."""

    return max(0, window - ALL_UP_TO) * row_bytes(t, world, share)
