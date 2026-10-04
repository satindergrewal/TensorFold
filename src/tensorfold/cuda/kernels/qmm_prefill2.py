"""More launch configurations of the 4-bit prompt matmul (patch 0195, qmm_prefill2.cu), every output with qmm_prefill's
bits: other CTA and warp tiles, deeper pipelines, two 64-input groups a stage, and a tail split that covers the columns
of a last, mostly idle wave of CTAs with smaller tiles in the same launch (a prompt lane of 2,048 rows leaves 30-65% of
the GPU idle in its last wave with 128 x 128 tiles on most of GLM-5.3-Flash's TP4 shapes), and (patch 0197) 16- and
32-row tiles for calls of a few hundred rows (a prompt chunk's 256-row pieces: a 1,024-column projection runs 8 CTAs of
128 x 256 on 188 SMs).

Why the bits are qmm_prefill's: every output is one chain of m16n8k16 tensor-core steps over K in order, from the
same fragments (the same shared-memory layouts, the same pair() / fma.rn.bf16x2 weight decode), rounded once in the
same epilogue; the per-group code is qmm_prefill's verbatim (tests/k5/test_q4p2_cpu.py compares the text), and the
accumulator chains are fixed by data dependence whatever the tile, stage or warp. What changes is only which CTA
computes an output and when its inputs arrive.

How it is chosen: a launch table (TF_GLM_TUNE, kernel q4_prefill) names variant v as tile 32 + v, per weight shape and
row bucket; tools/glm_autotune.py offers them beside the stock tiles 0-11 and keeps one only if its outputs equal the
stock tile's bit for bit at every tested row count. On the GPU, the first call of each variant for a weight shape and
output type also compares it with the stock tile 0 byte for byte (the call's own weights; random inputs over a wide
range of magnitudes at the call's rows and at 1 and 777 rows); a difference turns that variant off for that shape
for the process (a line says so) and the stock tile runs. Groups of 64 only (other groups run the stock tiles).

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

BASE = 32            # a launch table's tile id of variant 0
COUNT = 24           # variants (qmm_prefill2.cu's Q4P2_VARIANTS; checked against the extension at the first build)
DESCRIBE = {
    0: "128x128, 2 stages (tile 9's launch), fp32 outputs in pairs",
    1: "128x128 + 64x64 tail split, 2 stages",
    2: "128x128 + 128x64 tail split, 2 stages",
    3: "128x128, 2 stages of 2 groups",
    4: "128x128 + 64x64 tail split, 2 stages of 2 groups",
    5: "128x128, 3 stages",
    6: "128x128 + 64x64 tail split, 3 stages",
    7: "64x128 + 64x64 tail split, 3 stages",
    8: "64x64, 4 stages",
    9: "128x64 + 64x64 tail split, 3 stages",
    10: "64x256 (4 warps side by side), 2 stages",
    11: "64x256 (4 warps side by side), 3 stages",
    12: "128x256 (8 warps) + 64x128 tail split, 2 stages",
    13: "128x256 (8 warps) + 64x128 tail split, 3 stages",
    14: "256x128 (8 warps) + 128x64 tail split, 2 stages",
    15: "64x256 (4 warps side by side) + 64x64 tail split, 2 stages",
    16: "32x64, 4 stages",
    17: "32x128, 4 stages",
    18: "16x64, 4 stages",
    19: "16x128, 4 stages",
    20: "32x256 (4 warps side by side), 3 stages",
    21: "32x64, 6 stages",
    22: "64x64 + 32x64 tail split, 4 stages",
    23: "64x128 + 32x64 tail split, 3 stages",
}
_checked: dict[tuple, bool] = {}


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    ext = load(name="tensorfold_qmm_prefill2_v2", sources=[str(here / "qmm_prefill2.cpp"),
                                                          str(here / "qmm_prefill2.cu")],
               extra_cuda_cflags=["-O3"], verbose=False)
    if ext.variants() != COUNT:
        raise RuntimeError(f"qmm_prefill2: the extension has {ext.variants()} variants, this module {COUNT}")
    return ext


def is_variant(tile: int) -> bool:
    return BASE <= int(tile) < BASE + COUNT


def describe(tile: int) -> str:
    return DESCRIBE.get(int(tile) - BASE, "?")


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.view(torch.uint8), b.view(torch.uint8))


def _wide(rows: int, k: int, gen, device) -> torch.Tensor:
    e = torch.randint(-8, 9, (rows, k), device=device, generator=gen).to(torch.float32)
    return (torch.randn((rows, k), device=device, generator=gen) * torch.exp2(e)).to(torch.bfloat16)


def check(variant: int, q, f32: bool, rows: int, device) -> bool:
    """Variant ``variant`` against the stock tile 0 on ``q`` (the call's weights), byte for byte, at ``rows``, 1 and 777
    rows of random inputs over a wide range of magnitudes."""
    from .qmm import _ext as stock

    gen = torch.Generator(device=device).manual_seed(195 + variant)
    dtype = torch.float32 if f32 else torch.bfloat16
    for m in sorted({max(1, int(rows)), 1, 777}):
        x = _wide(m, q.k, gen, device)
        a = torch.full((m, q.n), 3.0, dtype=dtype, device=device)
        b = torch.full((m, q.n), 5.0, dtype=dtype, device=device)
        stock().qmm_prefill(x, q.weight, q.scales, q.biases, a, q.n, q.gs, f32, 0)
        _ext().qmm_prefill2(x, q.weight, q.scales, q.biases, b, q.n, q.gs, f32, variant)
        if not _same(a, b):
            return False
    return True


def matmul(x: torch.Tensor, q, f32: bool, out: torch.Tensor, tile: int) -> bool:
    """qmm_prefill's output by variant tile - BASE into ``out``; False (nothing written) when the variant does not take
    this call (groups other than 64, inside a CUDA graph capture before its check) or its check failed on this GPU:
    the caller then runs the stock tile."""
    if q.gs != 64 or not is_variant(tile):
        return False
    v = int(tile) - BASE
    key = (v, q.n, q.k, bool(f32))
    ok = _checked.get(key)
    if ok is None:
        if torch.cuda.is_current_stream_capturing():
            return False
        try:
            ok = check(v, q, f32, x.shape[0], x.device)
            why = "" if ok else ": its outputs differ from the stock tile's on this GPU"
        except Exception as exc:  # noqa: BLE001 - a variant that cannot run here is off; the stock tile runs
            ok, why = False, f": {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
        _checked[key] = ok
        if not ok:
            print(f"[tensorfold] 4-bit prompt matmul tile {tile} ({describe(tile)}) for {q.n}x{q.k}"
                  f"{' fp32' if f32 else ''}: off{why}", flush=True)
    if not ok:
        return False
    _ext().qmm_prefill2(x, q.weight, q.scales, q.biases, out, q.n, q.gs, f32, v)
    return True


def tiles(variant: int, M: int, N: int, K: int, resident: int) -> dict:
    """The CTA tiles a launch covers (no GPU needed once built): {nbig, cbig, nsmall, tiles: [(row, col, rows, cols)]}."""
    t = _ext().tiles(variant, M, N, K, resident)
    return {"nbig": t[0], "cbig": t[1], "nsmall": t[2],
            "tiles": [tuple(t[i:i + 4]) for i in range(3, len(t), 4)]}
