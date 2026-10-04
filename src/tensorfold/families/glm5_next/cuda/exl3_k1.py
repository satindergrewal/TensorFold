"""K1 prompt kernels for GLM's routed EXL3 experts (TF_GLM_EXPERT_PROMPT_KERNEL, exl3_k1.cu): decoded once a block of
up to 128 members, gate/up and down in one persistent launch.

Settings:
- TF_GLM_EXPERT_PROMPT_KERNEL: unset / ``0`` (off: exl3.cu's prompt kernels);
  - ``k1`` (8 warps, 2 m tiles each, blocks of up to 128 members, one block an SM), ``k1-64`` (blocks of up to 64, two
    blocks an SM) or ``k1-w16`` (16 warps, 1 m tile each): exl3.cu's prompt arithmetic, the prompt kernels' bits;
  - ``k1q``, ``k1q-64``, ``k1q-w16``: QUALITY-CHANGING. The gate/up matmuls in int8 (the rotated rows with one scale a
    token, the codebook values as round(31.75 v)) with exact int32 sums, then the same epilogue; the down matmul as
    K1's. Deterministic and batch-invariant (a row's scale is its own), but not the prompt kernels' bits: rebuild the
    gate references and run the quality checks before serving with it.
- TF_GLM_K1_OB: down output blocks (256 columns) a down tile runs in one pipeline: 1, 2, 4 (default), 8 or 16.
- TF_GLM_K1_LAG: list positions between a block's gate/up tiles and its down tiles; ``auto`` (default: three grids'
  worth) or a number (a large one puts every down tile after every gate/up tile).
- TF_GLM_K1_CTAS: at most this many blocks in the persistent grid (default 0: every SM's), to leave SMs to work on
  another stream beside the routed experts (TF_GLM_MOE_GLUE's side stream).
TF_GLM_K1_OB, TF_GLM_K1_LAG and TF_GLM_K1_CTAS change scheduling only (the same bits for every value).

Used for prompt windows of layers whose experts share one gate/up suh (one rotated row a token), 4-bit trellis words
[E, K/16, N/16, 32], D % 256 == 0 and NI % 128 == 0; anything else keeps exl3.cu's prompt kernels."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

SETTING = "TF_GLM_EXPERT_PROMPT_KERNEL"
# name -> (configuration index of exl3_k1.cu, int8 gate/up)
CONFIGS = {"k1": (0, False), "k1-64": (1, False), "k1-w16": (2, False),
           "k1q": (0, True), "k1q-64": (1, True), "k1q-w16": (2, True)}
OB_VALUES = (1, 2, 4, 8, 16)


def setting(value: str | None = None) -> tuple[int, bool] | None:
    """(configuration index, int8 gate/up) of TF_GLM_EXPERT_PROMPT_KERNEL, or None when off or set to another
    kernel's value (``k12...``: exl3_k12.py's, when its patch is applied too)."""

    v = (os.environ.get(SETTING, "") if value is None else value).strip().lower()
    if v in ("", "0", "off", "none") or v.startswith("k12"):
        return None
    if v not in CONFIGS:
        raise ValueError(f"{SETTING}: 0 (off), {', '.join(CONFIGS)}; not {v!r}")
    return CONFIGS[v]


def down_blocks(value: str | None = None) -> int:
    """TF_GLM_K1_OB: 256-column output blocks a down tile runs (the same bits for every value)."""

    v = (os.environ.get("TF_GLM_K1_OB", "") if value is None else value).strip() or "4"
    if not v.isdecimal() or int(v) not in OB_VALUES:
        raise ValueError(f"TF_GLM_K1_OB: one of {OB_VALUES}, not {v!r}")
    return int(v)


def max_ctas(value: str | None = None) -> int:
    """TF_GLM_K1_CTAS: 0 (every SM's blocks) or a cap on the persistent grid (the same bits for every value)."""

    v = (os.environ.get("TF_GLM_K1_CTAS", "") if value is None else value).strip() or "0"
    if not v.isdecimal():
        raise ValueError(f"TF_GLM_K1_CTAS: a non-negative number, not {v!r}")
    return int(v)


def lag(value: str | None = None) -> int:
    """TF_GLM_K1_LAG: -1 (auto) or a non-negative number of list positions (the same bits for every value)."""

    v = (os.environ.get("TF_GLM_K1_LAG", "") if value is None else value).strip().lower() or "auto"
    if v == "auto":
        return -1
    if not v.isdecimal():
        raise ValueError(f"TF_GLM_K1_LAG: auto or a non-negative number, not {v!r}")
    return int(v)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_exl3_k1_v1", sources=[str(here / "exl3_k1.cpp"), str(here / "exl3_k1.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def supported(ex, D: int, NI: int) -> bool:
    """Shapes and layout the K1 kernels take (else the caller keeps exl3.cu's prompt kernels)."""

    if ex.shared_suh is None or D % 256 or NI % 128 or NI < 128 or D > 8192:
        return False
    E = ex.count
    for t, k, n in ((ex.gt, D, NI), (ex.ut, D, NI), (ex.dt, NI, D)):
        if t.dtype != torch.int32 or tuple(t.shape) != (E, k // 16, n // 16, 32) or not t.is_contiguous():
            return False
    return all(s.dtype == torch.float16 and s.is_contiguous() for s in (ex.svh_g, ex.svh_u, ex.suh_d, ex.svh_d))


def max_blocks(pairs: int, experts: int, members: int) -> int:
    """Blocks of up to ``members`` a plan of ``pairs`` can need: one an expert in use plus one a full block."""

    return min(pairs, experts) + pairs // members


class K1Scratch:
    """The tile list's buffers for a window of ``rows`` x ``slots`` pairs (a few KB), and for K1q the int8 rows and
    their scales (rows x D bytes)."""

    def __init__(self, rows: int, slots: int, experts: int, D: int, NI: int, cfg: int, q: bool, ob: int,
                 device) -> None:
        ext = _ext()
        self.cfg, self.q, self.ob = cfg, q, ob
        pairs = rows * slots
        self.members = int(ext.block_members(cfg, q))
        cap = max_blocks(pairs, experts, self.members)
        if cap > int(ext.max_blocks()):
            raise ValueError(f"K1: a window of {pairs} pairs can need {cap} member blocks, more than "
                             f"{int(ext.max_blocks())}; use smaller prompt chunks")
        ncb, nog = NI // 128, (D // 256) // ob
        i32 = dict(dtype=torch.int32, device=device)
        self.blocks = torch.zeros((cap, 3), **i32)
        self.tiles = torch.zeros((cap * (ncb + nog),), **i32)
        self.ctrl = torch.zeros((4,), **i32)
        self.done = torch.zeros((cap,), **i32)
        self.xq = torch.zeros((rows, D), dtype=torch.int8, device=device) if q else None
        self.xs = torch.zeros((rows,), dtype=torch.float32, device=device) if q else None

    def nbytes(self) -> int:
        ts = (self.blocks, self.tiles, self.ctrl, self.done, self.xq, self.xs)
        return sum(t.numel() * t.element_size() for t in ts if t is not None)


def scratch(s, ex, cfg: int, q: bool, ob: int) -> K1Scratch:
    """The K1 scratch kept on an exl3_mm.Scratch (made on first use, remade when the configuration changes)."""

    k1 = getattr(s, "k1", None)
    key = (cfg, q, ob, ex.count, ex.dims, ex.width)
    if k1 is None or k1.key != key:
        k1 = K1Scratch(s.rows, s.slots, ex.count, ex.dims, ex.width, cfg, q, ob, s.xg.device)
        k1.key = key
        s.k1 = k1
    return k1


def routed(x: torch.Tensor, plan, ex, s, y: torch.Tensor, R: int, limit: float, config: tuple[int, bool]) -> None:
    """Xd (s.xd) and Y (fp32 [pairs, D]) of the plan's routed pairs from the normed bf16 rows x [R, D]: K1 rotates the
    rows as rot_rows (s.xg), K1q quantises them (its scratch's xq / xs)."""

    from .exl3_mm import _ext as _exl3_ext

    cfg, q = config
    ob = down_blocks()
    k1 = scratch(s, ex, cfg, q, ob)
    D = ex.dims
    ext = _ext()
    if q:
        ext.rot_quant(x, x.stride(0), ex.shared_suh, k1.xq, k1.xs, R, D)
    else:
        _exl3_ext().rot_rows(x, x.stride(0), ex.shared_suh, s.xg, R, D)
    ext.run(s.xg[:R], ex.gt, ex.ut, ex.dt, ex.svh_g, ex.svh_u, ex.suh_d, ex.svh_d, s.xd, y, plan.items, plan.counts,
            plan.members, None, k1.blocks, k1.tiles, k1.ctrl, k1.done, D, ex.width, s.slots, float(limit), cfg, ob,
            lag(), q, k1.xq, k1.xs, max_ctas())
