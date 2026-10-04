"""Tensor-parallel split helpers (this fork: the 2-rank subset of MiaAI-Lab's tp-n layout, patch 0054, Apache-2.0).

split_sizes(total, world, unit): each rank's share of a family's total - halves at two ranks whenever the total
is even (the two-rank engine's split, whatever the unit), else whole ``unit``s as even as they go.
"""

from __future__ import annotations

UNITS = {"heads": 1, "lin": 1, "dense": 128, "moe": 128, "shared": 128, "vocab": 64}
SIZES = (2, 3, 4)                     # the engine's tensor-parallel sizes


def parts(total: int, world: int, unit: int = 1) -> list[int]:
    """Each rank's share of ``total`` (a multiple of ``unit``) in rank order."""

    total, world, unit = int(total), int(world), int(unit)
    if world < 1 or unit < 1:
        raise ValueError(f"shares of {total} over {world} ranks in units of {unit}")
    if total % unit:
        raise ValueError(f"{total} is not a whole number of {unit}-wide units")
    units = total // unit
    if units < world:
        raise ValueError(f"{total} ({units} units of {unit}) cannot give each of {world} ranks a unit")
    base, extra = divmod(units, world)
    return [(base + (r < extra)) * unit for r in range(world)]


def split_sizes(total: int, world: int, unit: int) -> list[int]:
    """Each rank's share of a family's ``total``: halves at two ranks whenever the total is even."""

    if world == 2 and total % 2 == 0:
        return [total // 2, total // 2]
    return parts(total, world, unit)
