"""Mixed-rate EXL3 checkpoints through the tuned kernels where they apply.

A mixed k3/k4 checkpoint routes every token's experts over both rates. The tuned
prompt/decode kernels (``exl3_mm``) read stacked uniform trellises, so they take the
k4 (width-64) experts only; the k3 (width-48) experts stay on the universal per-expert
path. This module holds the two groups and the id remap, and runs a window: the slow
group first (its slice loop assigns every routed slot; foreign slots hold stale scratch
bytes), then the fast group, whose kernels overwrite exactly its own slots - the two
writes are disjoint, so no masking copy is needed and the combine sees every slot
written exactly once, by its own rate's kernels.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tensorfold.cuda.exl3.experts import Exl3RoutedExperts


@dataclass
class PairedExperts:
    """One MoE layer's routed experts split by trellis width (the loader builds it)."""

    fast: object                   # exl3_mm.Exl3Experts: the uniform width-64 group
    slow: Exl3RoutedExperts        # the universal path's object for the width-48 group
    lut_fast: torch.Tensor         # int32 [E+1]: the fast group's local id, -1 elsewhere
    lut_slow: torch.Tensor         # int32 [E+1]: the slow group's local id, -1 elsewhere
    count: int                     # E, the router's expert count
    width: int                     # NI on this rank (both groups share the geometry)
    dims: int                      # D

    def nbytes(self) -> int:
        return int(self.fast.nbytes()) + int(self.slow.nbytes_read(range(self.count)))


def _pick_group(pick: torch.Tensor, lut: torch.Tensor, sentinel: int) -> torch.Tensor:
    """The group's local picks: -1 entries (foreign experts, the shared slot) become ``sentinel``."""

    local = lut[pick.to(torch.long)]
    return torch.where(local >= 0, local, torch.full_like(local, sentinel))
