"""TF_GLM_LANE_PARTIALS=bf16 (patch 0293): every rank's partial of a row-parallel projection is rounded to bf16 before
the ranks exchange it, and the partials are added in fp32 in rank order as before. A NEW REFERENCE (it changes bits);
off by default.

Today each of a forward's 90 exchange sites sums the ranks' fp32 partials rank 0 first in fp32 and rounds the sum to
bf16 (hc_post: branch = bf16(((g0 + g1) + g2) + g3)). With the setting each partial is rounded to bf16 first (round to
nearest even, the conversion every GPU and torch use) and the sum is the same fp32 chain of the rounded values:
branch = bf16(((r(g0) + r(g1)) + r(g2)) + r(g3)), r(g) = fp32(bf16(g)). That is what a tensor-parallel all-reduce in
bf16 does (vLLM's TP all-reduce sums bf16 partials too), with a fixed order instead of a ring's.

What moves (four ranks, bytes a rank sends a row of a site):
  prompt lanes on the copy engines (TF_GLM_HC_EXCHANGE=ce): the partials 3 x 16 KiB -> 3 x 8 KiB (the copy engines move
      bf16 rows; the receiver widens them into the fp32 staging block hc_post reads). With TF_GLM_LANE_INPUTS the
      sites it covers exchange inputs as before and only round the partials they compute.
  the row split's other exchanges (first chunks, multi-prompt chunks: copy engines in pieces): bf16 the same way;
      on NCCL (TF_GLM_HC_EXCHANGE p2p / gather): fp32 rows that hold bf16 values (the same sums, no bytes saved).
  decode windows and unsplit prompt chunks (one all-gather a site): decode buffers all-gather bf16 rows and widen them
      (half the bytes); prompt buffers' rare unsplit chunks all-gather fp32 rows that hold bf16 values.
  TF_GLM_TWOSHOT_ROWS windows: the scatter carries fp32 rows that hold bf16 values (its second shot is bf16 already).

Deterministic and batch-invariant: the rounding is per element and the sum's order is fixed (rank order, fp32), so a
row's bits never depend on the rows beside it, its chunk, its window or its batch: prompt chunks stay chunk-invariant
(every prompt path rounds alike), drafted rounds equal serial ones and concurrent streams equal solo ones, under the
new references. The rounding is placed with the exchange that already reads those rows, on its stream (the rows a
rank sends right before it sends them, its own rows where hc_post's caller reads them), so it adds no ordering.

Value: unset, "0", "off" or "fp32": off (today's fp32 partials, unchanged); "bf16": on. Every rank must be given the
same value (the engine compares it at start).

Licensed under the Apache License, Version 2.0. Builds on TensorFold's row split, prompt lanes and copy-engine
exchange (patches 0122, 0123, 0126) and the two-shot decode exchange (0261)."""

from __future__ import annotations

import os

import torch

OFF = ("", "0", "off", "fp32", "none")
_ENV = "TF_GLM_LANE_PARTIALS"
_on: bool | None = None


def parse(value: str | None) -> bool:
    v = (value or "").strip().lower()
    if v in OFF:
        return False
    if v == "bf16":
        return True
    raise ValueError(f"{_ENV}: bf16, or 0 / fp32 (off), not {value!r}")


def on() -> bool:
    """The setting (read once; ``reset`` for tests). Never raises: a value that is not valid is off here and the
    start's comparison (``mask``) refuses it on every rank."""

    global _on
    if _on is None:
        try:
            _on = parse(os.environ.get(_ENV, ""))
        except ValueError:
            _on = False
    return _on


def reset(value: str | None = None) -> None:
    global _on
    if value is not None:
        os.environ[_ENV] = value
    _on = None


def mask() -> int:
    """The setting as an integer for the start's comparison across ranks (ValueError when it is not valid)."""

    return int(parse(os.environ.get(_ENV, "")))


def describe() -> str:
    return ("rank partials rounded to bf16 before every exchange, summed in fp32 in rank order (TF_GLM_LANE_PARTIALS="
            "bf16: a new reference; prompt lanes and decode all-gathers move bf16 rows)")


def to_bf16(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst`` (bf16) = ``src`` (fp32) rounded to nearest even, element by element."""

    dst.copy_(src)


def widen(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst`` (fp32) = ``src`` (bf16), exactly."""

    dst.copy_(src)


def round_(t: torch.Tensor, scratch: torch.Tensor | None = None) -> None:
    """fp32 rows rounded to bf16 values in place (``scratch``: bf16 rows of the same shape, else a temporary)."""

    if scratch is None:
        t.copy_(t.to(torch.bfloat16))
        return
    to_bf16(scratch, t)
    widen(t, scratch)
