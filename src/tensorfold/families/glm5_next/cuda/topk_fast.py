"""TF_GLM_TOPK_FAST=1 (patch 0199): a prompt chunk's top-512 pools a row (``sparse.top_pools`` on its ``_select_rows``
path) by a CUDA kernel that reads the row's scores twice instead of five times (``sparse_topk.cu``).

_select_rows' output is a function of the scores alone: each row's 512 best order keys (``_order_key``: -0 counted as
+0), ties to the lower pool, in ascending pool order. The kernel finds the same set by a 12-bit histogram, the
threshold bin's keys in shared memory and a bitmap compacted in pool order (the kernel's header has the steps and the
paths for bunched scores and long rows), so every pool is _select_rows' (int64, the same order). On the GPU the first
call of a process compares it with _select_rows byte for byte (rows of normal, tied, -inf-heavy and signed-zero
scores, 1,024 to 131,072 pools); a difference turns the setting off for the process and _select_rows runs.
TF_GLM_TOPK_FAST_CHECK=1 compares every call (a test setting).

Applies to prompt chunks' rows (``PromptSelect.pools``): contiguous fp32 rows of at least 512 pools, any length and
alignment (16-byte loads when every row starts on 16 bytes, else 4-byte ones: patch 0290). ``launch`` runs the
reference (``top_pools``) for anything else and says so once a kind of shape, so a call never fails on its shape.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab."""

from __future__ import annotations

import os

import torch

K = 512


def _flag(name: str) -> bool:
    value = (os.environ.get(name, "") or "0").strip()
    if value not in ("0", "1"):
        raise ValueError(f"{name}: 0 or 1, not {value!r}")
    return value == "1"


ENABLED = _flag("TF_GLM_TOPK_FAST")
CHECK = _flag("TF_GLM_TOPK_FAST_CHECK")
_decided: bool | None = None


def applies(scores: torch.Tensor) -> bool:
    """The kernel's shapes: contiguous fp32 [R, NP] rows on the GPU, NP from 512 to 2^30 (any alignment)."""
    return (scores.is_cuda and scores.dtype == torch.float32 and scores.dim() == 2 and scores.is_contiguous()
            and K <= scores.shape[1] <= 1 << 30)


_FELL_BACK: set = set()


def launch(scores: torch.Tensor) -> torch.Tensor:
    """The rows' 512 best pools, ascending (int64 [R, 512]): _select_rows' output. A shape the kernel does not take
    runs the reference (``sparse.top_pools``) with one line a kind of shape; it never raises on a shape."""
    if not applies(scores):
        why = (str(scores.device.type), str(scores.dtype), scores.dim(), bool(scores.is_contiguous()),
               scores.shape[-1] >= K if scores.dim() else False)
        if why not in _FELL_BACK:
            _FELL_BACK.add(why)
            print(f"[tensorfold] fast top-512 pools: scores {tuple(scores.shape)} {scores.dtype} "
                  f"{'contiguous' if scores.is_contiguous() else 'strided'}: not the kernel's shape, the reference "
                  "runs", flush=True)
        from . import sparse

        return sparse.top_pools(scores, K)
    from .attn_fast import _ws

    out = torch.empty((scores.shape[0], K), dtype=torch.int64, device=scores.device)
    _ws().topk_rows(scores, out)
    return out


def reference(scores: torch.Tensor) -> torch.Tensor:
    from . import sparse

    out = torch.empty((scores.shape[0], K), dtype=torch.int64, device=scores.device)
    R, NP = scores.shape
    sparse._select_rows[(R,)](scores, out, NP, scores, K=K, BLOCK=1024, VIS=False, num_warps=4)
    return out


def self_check(device) -> bool:
    """The kernel against _select_rows on rows of every kind and length, byte for byte."""
    gen = torch.Generator(device=device).manual_seed(199)
    # the engine's pool counts (multiples of 1,024; the capacity when a prompt reaches it: any count), rows of 16-byte
    # and 4-byte alignment, 1 to 64 rows
    for NP in (512, 1024, 4096, 4226, 25600, 25601, 32768, 65536, 65537, 70656, 131072, 131075):
        R = (64 if NP <= 32768 else 16) - (NP % 4)
        rows = []
        base = torch.randn((R, NP), device=device, generator=gen)
        rows.append(base)                                                       # distinct scores
        rows.append(torch.round(base * 4) / 4)                                  # many ties
        heavy = base.clone()
        heavy[:, NP // 3:] = float("-inf")                                      # past the visible pools
        rows.append(heavy)
        zeros = torch.where(base > 0.5, base, torch.zeros_like(base))           # a bunched threshold bin
        zeros[:, ::7] = -0.0
        rows.append(zeros)
        e = torch.randint(-30, 31, (R, NP), device=device, generator=gen).float()
        rows.append(base * torch.exp2(e))                                       # wide magnitudes
        for s in rows:
            s = s.contiguous()
            if not torch.equal(launch(s), reference(s)):
                return False
    return True


def on(scores: torch.Tensor) -> bool:
    """Whether to run the kernel for this call: the setting, the shape, and its check passed on this GPU."""
    global _decided
    if not ENABLED or not applies(scores):
        return False
    if _decided is None:
        try:
            ok = self_check(scores.device)
            why = "" if ok else ": its pools differ from _select_rows' on this GPU"
        except Exception as exc:  # noqa: BLE001 - a kernel that cannot run here is off; _select_rows runs
            ok, why = False, f": {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
        _decided = ok
        print(f"[tensorfold] fast top-512 pools: {'on (checked byte for byte)' if ok else 'off' + why}", flush=True)
    return _decided


def top512(scores: torch.Tensor) -> torch.Tensor:
    """launch(), and with TF_GLM_TOPK_FAST_CHECK=1 _select_rows too, compared (stops on a difference)."""
    out = launch(scores)
    if CHECK and not torch.equal(out, reference(scores)):
        raise RuntimeError("TF_GLM_TOPK_FAST_CHECK: the fast top-512 pools differ from _select_rows'")
    return out
