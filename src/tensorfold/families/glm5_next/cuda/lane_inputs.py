"""TF_GLM_LANE_INPUTS (patch 0183): prompt lanes exchange a row-parallel output projection's bf16 inputs instead of its
fp32 partials, and each rank computes every rank's partial of its own rows itself.

Where: prompt chunks in two lanes (TF_GLM_PREFILL_LANES=2) on the row split (TF_GLM_HC_SPLIT=1) with the copy-engine
exchange (TF_GLM_HC_EXCHANGE=ce). At a lane's exchange site after an output projection whose input columns are split
between the ranks (KDA's and DSA's o_proj, the dense MLP's down_proj), every rank today computes its fp32 partial of
the projection for all of the lane's rows (its input columns times its slice of the weight) and sends each peer that
peer's rows of it (16 KiB a row: 4,096 fp32); the peer sums the N partials of its own rows rank 0 first (hc_post).
With the setting each rank sends each peer the peer's rows of its bf16 INPUT instead (at four ranks KDA 2,048 columns
= 4 KiB a row, DSA 4,096 = 8 KiB, the MLP's down 3,072 = 6 KiB), and the peer computes every rank's partial of its own
rows from them, with that rank's slice of the weight, which every rank holds (gathered from its owner at start, byte
for byte; never quantized again). The main stream computes only its own rows' partial (a quarter of today's projection
at four ranks); the lanes' exchange stream runs the peers' slot projections after the copies, beside the other lane's
block. The normed rows' exchange after the glue is unchanged.

Value: a comma-separated list of kda, dsa, mlp; "all" for the three; unset, "0", "off" or "none": none (the default,
today's code path). Every rank must be given the same value (the engine compares it at start).

Exact: slot q of a rank's staging block today holds rank q's prompt matmul of all the lane's rows, restricted to this
rank's rows; with the setting it holds the prompt matmul of those rows alone with rank q's weight bytes. The prompt
matmul gives a row the same bits in any row range and any tile (``prefill_matmul``: weights rounded once to bf16, one
fp32 chain over K in order; qmm_prefill.cu), so the two are equal byte for byte; hc_post reads the same slots in the
same order, and everything after it is unchanged. Hence it needs 4-bit projections on the prompt matmul (an MLX
checkpoint, or TF_GLM_DENSE=q4 on an EXL3 one) of the same shape on every rank: the start refuses anything else.

TF_GLM_LANE_DIRECT=1 (same patch): the lanes' swapped normed rows (and TF_GLM_MOE_GLUE rowsplit's picks and weights)
are written by the peers' copy engines straight into this rank's row buffers, which then live in the copy-engine arena,
instead of into its staging and from there by three local copies a step on the exchange stream (13,830 copies of
4 MiB a rank in a cold 100K prompt, 54 GiB of local traffic; the copies the host blocked on when the exchange stream's
launch queue was full). Exact: the same bytes in the same places; a peer's copies of a step start after this rank's
record of the step before, and every read of those rows' previous contents (the lane's block on the main stream, its
shared expert on the MoE glue's stream, its plan and combine) is ordered before the lane's next partials or inputs
step, which waits for the block's fill or input rows (tests/k4/test_lane_inputs_cpu.py models the peers' writes).

Memory: every rank holds the other ranks' slices of the chosen projections. GLM-5.3-Flash at four ranks with 4-bit
projections: KDA 34 layers x 3 x 4.5 MiB = 459 MiB, DSA 11 x 3 x 9 MiB = 297 MiB, MLP down 3 x 3 x 6.75 MiB = 61 MiB.
The startup estimate counts them (``weights``: the pool shrinks by what they take); the copy-engine arena grows by
(N - 1) x H x 2 bytes x the widest input (H = a chunk's rows a rank: 24 MiB at 4,096-row chunks with dsa).

Licensed under the Apache License, Version 2.0. Builds on TensorFold's row split, prompt lanes and copy-engine
exchange (patches 0122, 0123) and its 4-bit prompt matmul."""

from __future__ import annotations

import os
import re

import torch

KINDS = ("kda", "dsa", "mlp")
OFF = ("", "0", "off", "none")
_ENV = "TF_GLM_LANE_INPUTS"
# a checkpoint tensor of an output projection the setting may hold whole: (layer, which, part)
_NAME = re.compile(r"(?:^|\.)layers\.(\d+)\.(self_attn\.o_proj|mlp\.down_proj)\.(weight|scales|biases)$")


def parse(value: str | None) -> frozenset:
    v = (value or "").strip().lower()
    if v in OFF:
        return frozenset()
    if v == "all":
        return frozenset(KINDS)
    names = [p.strip() for p in v.split(",") if p.strip()]
    bad = [n for n in names if n not in KINDS]
    if bad or not names:
        raise ValueError(f"{_ENV}: a comma-separated list of {', '.join(KINDS)}, or all, or 0 (off), not {value!r}")
    return frozenset(names)


def wanted() -> frozenset:
    return parse(os.environ.get(_ENV, ""))


def mask() -> int:
    """The settings as a bit mask: KINDS, then TF_GLM_LANE_DIRECT (ValueError when one is not valid); the engine
    compares it across ranks."""

    kinds = wanted()
    return sum(1 << i for i, n in enumerate(KINDS) if n in kinds) + (8 if direct_wanted() else 0)


def direct_wanted() -> bool:
    v = (os.environ.get("TF_GLM_LANE_DIRECT", "") or "0").strip()
    if v not in ("0", "1"):
        raise ValueError(f"TF_GLM_LANE_DIRECT: 0 (off) or 1, not {v!r}")
    return v == "1"


def direct(world: int) -> bool:
    """TF_GLM_LANE_DIRECT asked for and usable (``usable``, two ranks or more); never raises (the start refuses a
    value that is not valid)."""

    try:
        return direct_wanted() and int(world) > 1 and usable()
    except ValueError:
        return False


def usable(env=None) -> bool:
    """Prompt lanes on the row split with the copy-engine exchange (settings only: the same answer on every rank)."""

    env = os.environ if env is None else env
    return ((env.get("TF_GLM_PREFILL_LANES", "") or "1").strip() == "2"
            and (env.get("TF_GLM_HC_SPLIT", "") or "0").strip() == "1"
            and (env.get("TF_GLM_HC_EXCHANGE", "") or "p2p").strip() == "ce")


def used(world: int) -> frozenset:
    """The kinds that run: the wanted ones where they can (``usable``, two ranks or more). Never raises: a value that
    is not valid runs none here, and the start's comparison refuses it on every rank."""

    try:
        kinds = wanted()
    except ValueError:
        return frozenset()
    return kinds if kinds and int(world) > 1 and usable() else frozenset()


def describe(world: int, added: int = 0) -> str:
    kinds = [n for n in KINDS if n in wanted()]
    if not kinds:                                   # TF_GLM_LANE_DIRECT alone
        return ("prompt lanes' swapped rows written straight into the peers' row buffers (TF_GLM_LANE_DIRECT)"
                + ("" if direct(world) else " [not used: it needs TF_GLM_PREFILL_LANES=2, TF_GLM_HC_SPLIT=1 and "
                                             "TF_GLM_HC_EXCHANGE=ce]"))
    what = {"kda": "KDA's", "dsa": "DSA's", "mlp": "the dense MLP's down"}
    text = (f"prompt lanes exchange {', '.join(what[n] for n in kinds)} output projection inputs (bf16) instead of "
            f"their fp32 partials, each rank computing every rank's partial of its rows (TF_GLM_LANE_INPUTS)")
    if not used(world):
        return text + " [not used: it needs TF_GLM_PREFILL_LANES=2, TF_GLM_HC_SPLIT=1 and TF_GLM_HC_EXCHANGE=ce]"
    text += f"; every rank's slices of them on each rank (+{added / 2 ** 20:.0f} MiB)"
    return text + ("; swapped rows written straight into the peers' row buffers (TF_GLM_LANE_DIRECT)"
                   if direct(world) else "")


def chosen(cfg, kinds, index: int, which: str) -> bool:
    """Whether layer ``index``'s projection ``which`` (self_attn.o_proj or mlp.down_proj) is one the kinds hold whole
    (the MTP layer's never: it runs no prompt lane)."""

    if not 0 <= index < int(cfg.layers):
        return False
    if which == "self_attn.o_proj":
        return cfg.kinds[index] in kinds
    return "mlp" in kinds and cfg.mlp_kinds[index] == "dense"


def weights(base, cfg, world: int):
    """A split_weights transform for the startup estimate: the chosen projections' tensors whole (every rank's slice,
    ``world`` times this rank's share: the start refuses shares that differ)."""

    kinds = used(world)
    if not kinds:
        return base

    def transform(name: str, info: dict) -> tuple[int, int]:
        total, extra = base(name, info)
        m = _NAME.search(name)
        if total and m and chosen(cfg, kinds, int(m.group(1)), m.group(2)):
            return total * int(world), extra
        return total, extra
    return transform


def targets(w, kinds) -> list[tuple[int, str, object]]:
    """(layer, kind, projection) of every projection the kinds cover, in layer order (the same list on every rank)."""

    out = []
    for layer in w.layers:
        if layer.kind in kinds:
            out.append((layer.index, layer.kind, (layer.kda if layer.kind == "kda" else layer.dsa).o))
        if "mlp" in kinds and layer.mlp is not None:
            out.append((layer.index, "mlp", layer.mlp.down))
    return out


def _signature(q) -> list[int]:
    """What every rank's copy of a projection must agree on: its shapes, group size and the tensors' sizes."""

    return [int(q.n), int(q.k), int(q.gs), q.weight.numel(), q.scales.numel(), q.biases.numel()]


def gather(w) -> int:
    """Every rank's slice of the chosen projections on every rank, as ``q.ranks`` (rank r's ``Q4``; None at this
    rank's own index, whose slice is ``q`` itself). A collective: every rank calls it at start after the weights
    load; it raises on every rank alike when a projection is not a 4-bit prompt-matmul one or the ranks' shapes
    differ. Each slice is its owner's bytes (an all-gather), checked against the owner's copy of its own. Returns the
    bytes it added on this rank."""

    from .qmm import Q4

    world, rank = int(w.world), int(w.rank)
    kinds = used(world)
    if not kinds:
        return 0
    comm = getattr(w.comm, "nccl", w.comm)
    mats = targets(w, kinds)
    # every rank's signatures (fixed length: a row a layer), to refuse together
    width = 7
    sig = torch.zeros((len(w.layers) * 2, width), dtype=torch.int64)
    for j, (index, kind, q) in enumerate(mats):
        slot = 2 * index + (kind == "mlp")
        sig[slot, 0] = 1 if isinstance(q, Q4) else -1
        if isinstance(q, Q4):
            sig[slot, 1:] = torch.tensor(_signature(q), dtype=torch.int64)
    mine = sig.view(-1).to(w.device)
    got = torch.empty((world * mine.numel(),), dtype=torch.int64, device=w.device)
    comm.all_gather(mine, got)
    every = got.view(world, -1, width).cpu()
    if bool((every[:, :, 0] == -1).any()):
        raise RuntimeError(f"{_ENV}: the chosen output projections must be 4-bit (the prompt matmul's: an MLX "
                           "checkpoint, or TF_GLM_DENSE=q4 for an EXL3 one); a rank holds them as BF16 / FP8")
    if not bool((every == every[0]).all()):
        raise RuntimeError(f"{_ENV}: the ranks' slices of the chosen projections differ in shape (shares that are not "
                           "equal, e.g. three ranks): the setting needs equal shares")
    added = 0
    for index, kind, q in mats:
        parts = []
        for t in (q.weight, q.scales, q.biases):
            src = t.contiguous().view(-1)
            whole = torch.empty((world * src.numel(),), dtype=src.dtype, device=src.device)
            comm.all_gather(src, whole)
            whole = whole.view(world, *t.shape)
            if not torch.equal(whole[rank], t):
                raise RuntimeError(f"{_ENV}: layer {index}'s gathered slice of this rank is not its own")
            parts.append(whole)
        ranks = []
        for r in range(world):
            if r == rank:
                ranks.append(None)
                continue
            wt, sc, bi = (p[r].clone() for p in parts)
            ranks.append(Q4(wt, sc, bi, q.n, q.k, q.gs))
            added += sum(t.numel() * t.element_size() for t in (wt, sc, bi))
        q.ranks = tuple(ranks)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return added


def row_bytes(w) -> int:
    """Bytes a row of the widest chosen input (bf16): what each sender's staging row in the copy-engine arena holds
    (0 when nothing is chosen or gathered). The same on every rank (``gather`` checked the shapes)."""

    widest = 0
    for _, _, q in targets(w, KINDS):
        if getattr(q, "ranks", None) is not None:
            widest = max(widest, int(q.k))
    return widest * 2


def inputs_of(q):
    """The other ranks' slices of a projection (``q.ranks``) when the setting holds it, else None."""

    return getattr(q, "ranks", None)
