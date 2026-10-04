"""TF_GLM_INDEX_SPLIT=1 (patch 0120): a prompt chunk's DSA token selection split between the ranks.

Without it every rank scores every row of a prompt chunk against every pooled key for all 32 index heads and selects
each row's 512 best pools (``sparse._select_prompt``), the same work four times over at TP4: the indexer is replicated
(``tp``'s layout), and it is the part of a prompt chunk whose cost grows with the context (a row scores pos / 4 pools).

Split, the chunk's selection blocks (``sparse.PromptSelect``: rows in blocks of SELECT_ROWS, 512) are dealt to the
ranks round-robin, block i to rank i % N: each rank computes its blocks' index queries (``qb``), scores and pools,
the ranks all-gather the chosen pools (int32 [512 rows, 512] a block, 1 MiB), and every rank writes every block's
tokens and counts from them. A rank therefore does about 1 / N of the indexer's matmul, scoring and selection.

Exactness: every block goes through the very launches ``_select_prompt`` makes for it (``PromptSelect.pools`` then
``PromptSelect.write_tokens``: the same kernels, grids, constexprs, the same NP, rows and device position), only on one
rank instead of on each; the ranks run one compiled kernel set on one GPU model, and a row's scores read nothing but
its own index query, head weights, position and the pooled keys (which every rank holds the same: the pool cache is
replicated and written by the same kernels). The pools cross the wire as int32 (values below 2^31: exact) and are
widened back to the int64 rows ``_tokens`` reads. So tokens and counts equal the unsplit ones bit for bit.
TF_GLM_INDEX_SPLIT_CHECK=1 also runs the unsplit selection and stops the server on any difference (a test setting:
it costs the full selection again and a device sync a layer).

When: prompt chunks only (eager, ``host_pos`` known), every row of the piece past the dense limit, at least two
blocks (R > SELECT_ROWS), the piece's first position at least TF_GLM_INDEX_SPLIT_FROM (default 16,384: shorter
contexts score few pools, where the gather costs more than it saves), two or more ranks. Every rank decides from the
same numbers, so they all take the same branch; the startup comparison holds the settings (``code``).

Where the gather runs: on the row split's second stream when a prompt chunk runs split with overlap
(``hcsplit.HcSplit``), so every NCCL call of the chunk stays on one stream in one order; else on the current stream."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


def _flag(env, name: str) -> bool:
    value = str(env.get(name, "") or "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name}: 0 or 1, not {value!r}")
    return value == "1"


@dataclass(frozen=True)
class IndexSplit:
    on: bool = False
    start: int = 16384             # TF_GLM_INDEX_SPLIT_FROM: the first position a split piece may start at
    check: bool = False            # TF_GLM_INDEX_SPLIT_CHECK: compare with the unsplit selection (tests)

    @classmethod
    def from_env(cls, env=None) -> "IndexSplit":
        env = os.environ if env is None else env
        on = _flag(env, "TF_GLM_INDEX_SPLIT")
        raw = str(env.get("TF_GLM_INDEX_SPLIT_FROM", "") or "16384").strip()
        if not raw.isdecimal():
            raise ValueError(f"TF_GLM_INDEX_SPLIT_FROM: a token position (0 or more), not {raw!r}")
        check = _flag(env, "TF_GLM_INDEX_SPLIT_CHECK")
        if check and not on:
            raise ValueError("TF_GLM_INDEX_SPLIT_CHECK=1 checks the split selection: it needs TF_GLM_INDEX_SPLIT=1")
        return cls(on, int(raw), check)

    def code(self) -> list[int]:
        return [int(self.on), self.start if self.on else 0, int(self.check)]

    def describe(self) -> str:
        what = f"prompt chunks' DSA token selection split between the ranks from position {self.start}"
        return what + (" (checked against the unsplit selection)" if self.check else "")


SETTINGS: IndexSplit | None = None


def settings() -> IndexSplit:
    global SETTINGS
    if SETTINGS is None:
        SETTINGS = IndexSplit.from_env()
    return SETTINGS


def applies(w, b, R: int, host_pos: int | None, all_sparse: bool, sparse_np: int | None) -> bool:
    """Whether a piece of R rows from ``host_pos`` selects split (the same answer on every rank)."""

    from .sparse import SELECT_ROWS

    s = settings()
    world = int(getattr(w, "world", 1))
    return (s.on and world > 1 and getattr(w, "comm", None) is not None and b.prefill and sparse_np is None
            and host_pos is not None and all_sparse and host_pos >= s.start and R > SELECT_ROWS)


def _side_stream(b):
    """The row split's second stream while a split prompt chunk runs (its NCCL calls go there), else None."""

    sp = getattr(b, "split", None)
    if sp is not None and getattr(sp, "active", False) and getattr(sp, "stream", None) is not None:
        return sp.stream
    return None


def deal(blocks: int, world: int, rank: int) -> tuple[list[int], int]:
    """Block i goes to rank i % world, into its slot i // world: this rank's blocks and the slots a rank holds."""

    return list(range(rank, blocks, world)), -(-blocks // world)


def select(w, b, qr: torch.Tensor, xs_qr: torch.Tensor, qb, qi: torch.Tensor, wts: torch.Tensor, pk: torch.Tensor,
           host_pos: int, R: int, np_max: int, pos_dev: torch.Tensor, mm) -> tuple[torch.Tensor, torch.Tensor]:
    """``sparse.select_tokens`` for a prompt piece of R rows (``applies``), split: ``qr`` / ``xs_qr`` the rows' query
    latents, ``qb`` the indexer's query projection, ``qi`` the rows' index query buffer (written for this rank's
    blocks only, or every row when checking), ``wts`` their head weights, ``mm(x, q, xs, out)`` the forward's
    projection call. Returns every row's tokens and counts, as the unsplit selection."""

    from .sparse import PromptSelect, _select_prompt

    s = settings()
    world, rank = int(w.world), int(w.rank)
    sel = PromptSelect(qi, wts, pk, host_pos, R, np_max, pos_dev)
    starts = list(sel.blocks())
    mine, slots = deal(len(starts), world, rank)
    if s.check:
        mm(qr[:R], qb, xs_qr[:R], qi[:R])
    else:
        for i in mine:                       # the index queries of this rank's rows only (row-independent matmul)
            a = starts[i]
            n = sel.rows(a)
            mm(qr[a:a + n], qb, xs_qr[a:a + n], qi[a:a + n])
    B = sel.B
    send = torch.empty((slots, B, 512), dtype=torch.int32, device=qi.device)
    for j, i in enumerate(mine):
        a = starts[i]
        n = sel.rows(a)
        send[j, :n].copy_(sel.pools(a, sel.at(a)))                                    # int64 -> int32: exact
    recv = torch.empty((world, slots, B, 512), dtype=torch.int32, device=qi.device)
    comm = getattr(w.comm, "nccl", w.comm)
    side = _side_stream(b)
    if side is None:
        comm.all_gather(send.reshape(-1), recv.reshape(-1))
    else:
        main = torch.cuda.current_stream()
        ready = torch.cuda.Event()
        ready.record(main)
        with torch.cuda.stream(side):
            side.wait_event(ready)
            comm.all_gather(send.reshape(-1), recv.reshape(-1))
            done = torch.cuda.Event()
            done.record(side)
        send.record_stream(side)
        recv.record_stream(side)
        main.wait_event(done)
    tokens, counts = sel.outputs()
    for i, a in enumerate(starts):
        n = sel.rows(a)
        pools = recv[i % world, i // world, :n].to(torch.int64)                        # contiguous, as top_pools'
        sel.write_tokens(a, pools, sel.at(a), tokens, counts)
    if s.check:
        ref_tokens, ref_counts = _select_prompt(qi[:R], wts, pk, host_pos, R, np_max, pos_dev)
        same_counts = torch.equal(counts, ref_counts)
        # rows with count 0 attend densely and nothing reads their tokens (``_select_prompt``): compare counted rows
        live = ref_counts > 0
        same_tokens = torch.equal(tokens[live], ref_tokens[live])
        if not (same_counts and same_tokens):
            bad = (counts != ref_counts) | ((tokens != ref_tokens).any(dim=1) & live)
            rows = bad.nonzero().flatten()[:8].tolist()
            raise RuntimeError(f"TF_GLM_INDEX_SPLIT_CHECK: rank {rank}: the split selection differs from the unsplit "
                               f"one at position {host_pos} ({R} rows; first rows {rows})")
    return tokens, counts
