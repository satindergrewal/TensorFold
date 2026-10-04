"""K12 prompt kernels for GLM's routed EXL3 experts (TF_GLM_EXPERT_PROMPT_KERNEL=k12..., exl3_k12.cu): exl3.cu's
prompt kernels' bits (the same items, blocks, fp32 chains and epilogue formulas) with a main loop built to keep the
tensor pipe fed (one basic block a stage, compiled for the item's 8-member column blocks; the next k tile's decode
between the current k tile's mma; the neighbour word from shared memory; the codebook mask in a register).

Settings:
- TF_GLM_EXPERT_PROMPT_KERNEL: unset / ``0`` (off: exl3.cu's prompt kernels);
  - ``k12``: register-direct (every warp decodes its own tiles), four cp.async stages;
  - ``k12-s3``: the same, three stages;
  - ``k12-ws``: warp-specialised: two producer warps decode each k tile's 16 tiles once into shared memory, eight
    consumer warps only load fragments and run the mma (items of up to 32 members run register-direct);
  - ``k12-m128``: blocks of up to 128 members (an expert's items paired), so a decoded tile feeds up to 16 mma: gate
    and up as a cluster of two blocks (8 m tiles of one matrix each, the epilogue's rows exchanged through
    distributed shared memory), down in 128-column x 128-member blocks; ``k12-g128`` / ``k12-d128``: only the gate/up
    / only the down kernel of k12-m128 (the other k12's);
  - ``k12-p80``: k12's kernels on blocks of up to 80 members (an expert of up to 80 rows in one block, no small
    remainder item; a decoded tile feeds up to 10 mma);
  all exact: every output has the prompt kernels' bits (the same fp32 chain of mma over the ascending k tiles, the
  same epilogue arithmetic). Values of other kernels (not starting with ``k12``) are left to them.

Used for prompt windows (64-member plans) of layers whose experts share one gate/up suh (one rotated row a token),
4-bit trellis words [E, K/16, N/16, 32], D % 256 == 0 and NI % 128 == 0; anything else keeps exl3.cu's kernels."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

SETTING = "TF_GLM_EXPERT_PROMPT_KERNEL"
# name -> configuration index of exl3_k12.cu (exl3_k12_prompt_cuda's cfg)
CONFIGS = {"k12": 0, "k12-s3": 1, "k12-ws": 2, "k12-m128": 3, "k12-g128": 4, "k12-d128": 5, "k12-p80": 6}
BLOCKS = (3, 4, 5, 6)               # configurations on blocks of 128 or 80 members (a scratch of 4 max_items + 2 int32)


def setting(value: str | None = None) -> int | None:
    """The K12 configuration TF_GLM_EXPERT_PROMPT_KERNEL names, or None (off, or another kernel's value)."""

    v = (os.environ.get(SETTING, "") if value is None else value).strip().lower()
    if not v.startswith("k12"):
        return None
    if v not in CONFIGS:
        raise ValueError(f"{SETTING}: K12 configurations are {', '.join(CONFIGS)}; not {v!r}")
    return CONFIGS[v]


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_exl3_k12_v1", sources=[str(here / "exl3_k12.cpp"), str(here / "exl3_k12.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


_NOTES: set[str] = set()


def note(msg: str) -> None:
    """One line a process for each distinct message (the setting in use, or why a layer keeps exl3.cu's kernels)."""

    if msg not in _NOTES:
        _NOTES.add(msg)
        print(f"[tensorfold] {SETTING}={os.environ.get(SETTING, '')}: {msg}", flush=True)


def unused_reason(ex, D: int, NI: int, shx: bool, tile: int) -> str | None:
    """None when the K12 kernels take this prompt window, else why exl3.cu's prompt kernels run it."""

    if not shx:
        return "exl3.cu's prompt kernels for this layer (its rows are not rotated once a token: TF_GLM_EXL3_ROWROT=0 " \
               "or experts without one shared gate/up suh)"
    if tile != 64:
        return f"exl3.cu's prompt kernels (plans of {tile} members, TF_GLM_EXL3_PASS; K12 takes 64)"
    if not supported(ex, D, NI):
        return f"exl3.cu's prompt kernels for this layer (D {D}, width {NI} or the trellis layout not supported)"
    return None


def supported(ex, D: int, NI: int) -> bool:
    """Shapes and layout the K12 kernels take (else the caller keeps exl3.cu's prompt kernels)."""

    if ex.shared_suh is None or D % 256 or NI % 128 or NI < 128:
        return False
    E = ex.count
    for t, k, n in ((ex.gt, D, NI), (ex.ut, D, NI), (ex.dt, NI, D)):
        if t.dtype != torch.int32 or tuple(t.shape) != (E, k // 16, n // 16, 32) or not t.is_contiguous():
            return False
    return all(s.dtype == torch.float16 and s.is_contiguous() for s in (ex.svh_g, ex.svh_u, ex.suh_d, ex.svh_d))


def routed(x: torch.Tensor, plan, ex, s, y: torch.Tensor, R: int, limit: float, cfg: int) -> None:
    """Xd (s.xd) and Y (fp32 [pairs, D]) of a 64-member prompt plan's routed pairs from the normed bf16 rows x [R, D]:
    the rows rotated once (exl3.cu rot_rows into s.xg), then the K12 gate/up and down kernels."""

    from tensorfold.cuda import experts as grouped

    from .exl3_mm import _ext as _exl3_ext
    from .exl3_mm import item_order

    if plan.tile != 64:
        raise ValueError(f"K12 takes 64-member prompt plans, not {plan.tile}")
    D, NI = ex.dims, ex.width
    cap = grouped.max_items(R * s.slots, plan.experts, plan.tile)
    blocks = None
    if cfg in BLOCKS:                                # kept on the Scratch, grown to the largest window's bound
        blocks = getattr(s, "k12_blocks", None)
        if blocks is None or blocks.numel() < 4 * cap + 2:
            full = grouped.max_items(s.rows * s.slots, plan.experts, plan.tile)
            blocks = torch.zeros((4 * max(cap, full) + 2,), dtype=torch.int32, device=s.xg.device)
            s.k12_blocks = blocks
    _exl3_ext().rot_rows(x, x.stride(0), ex.shared_suh, s.xg, R, D)
    _ext().prompt(s.xg[:R], ex.gt, ex.ut, ex.dt, plan.items, plan.counts, plan.members, ex.svh_g, ex.svh_u, ex.suh_d,
                  ex.svh_d, s.xd, y, D, NI, ex.count, cap, float(limit), s.slots, s.order if item_order() else None,
                  blocks, cfg)
