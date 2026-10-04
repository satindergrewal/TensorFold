"""TF_GLM_DRAFT_PROMPT_SKIP=1 (patch 0124): a prompt's DFlash2 context rows that no later draft reads are not
computed, and a prompt chunk's context updates queue without a host sync each.

DFlash2's attention is a sliding window (its config's sliding_window: 2,048 for GLM-5.3-Flash's): a block pass at
context end P reads the keys and values of positions P - window - 1 .. P - 1 only (``dflash2._dattn_kernel``; a kept
state copies exactly those, ``decode._drafter_views`` / ``_ring_slots``). A prompt's context is read at its end (the
first drafts) and at every point where a state is kept (its window rows are copied then); a row before every such
point's window is never read again. Without this setting every prompt row still went through the drafter's context
update: the fc projection of the five tapped layers (5 x 4,096 -> 4,096, replicated: about 170 MFLOP a row on every
rank) and the KV projections of the drafter's layers, in launches of up to 64 rows; and with --parallel each launch
first waited on the host for the previous launch's table to leave its one pinned buffer, which, right after a prompt
chunk's forward, meant waiting for the whole forward, so the next chunk could only be queued once the chunk's launches
had gone through one at a time.

With it, the rows of a prompt chunk before the earliest window a later read point needs are skipped (``skip_rows``;
the drafter's context end advances over them, ``skip_taps``), and the rows from there on go through the same calls as
before. Each row's update is row-independent (the drafter's matmuls fix their K splits by shape; its norm, rotary and
cache writes are per row), so every row any draft reads holds the same bits and the drafts are the same. With
--parallel the context updates take their tables from a ring of pinned buffers, each one waited on only when it comes
round again (``dflash2_multi.MultiDrafter``).

Exact: drafts only propose, and here they do not even change; replies are the same, token for token. The prompt-state
digests (TF_GLM_PREFILL_DIGEST) include the drafter's window at each prompt's end."""

from __future__ import annotations

import os
from typing import Iterable

MARGIN = 64          # rows computed before the earliest window a read point needs, past what it reads


def enabled(value: str | None = None) -> bool:
    value = (os.environ.get("TF_GLM_DRAFT_PROMPT_SKIP", "") if value is None else value).strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"TF_GLM_DRAFT_PROMPT_SKIP: 0 or 1, not {value!r}")
    return value == "1"


ON = enabled()


def code() -> list[int]:
    return [int(ON)]


def describe() -> str:
    return ("prompt chunks skip the DFlash2 context rows no later draft reads (outside every read point's sliding "
            "window), context updates queue without a host sync each")


def skip_rows(drafter, start: int, rows: int, reads: Iterable[int]) -> int:
    """How many leading rows of prompt rows start .. start + rows no draft will read: rows before the earliest
    window (window + 1 rows and MARGIN) of the read points past ``start`` (``reads``: the prompt's end and every point
    where its state is kept). 0 when the setting is off, the drafter attends without a window, or no read point
    lies past ``start``."""

    if not ON or drafter is None:
        return 0
    window = int(getattr(drafter, "window", -1))
    if window < 0:
        return 0
    later = [int(p) for p in reads if int(p) > start]
    if not later:
        return 0
    first = min(later) - window - 1 - MARGIN
    return max(0, min(int(rows), first - int(start)))
