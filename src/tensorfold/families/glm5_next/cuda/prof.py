"""TF_GLM_PROFILE: per block kind of every prompt chunk, printed after each prefill; never inside decode or capture.

``1``: each block timed with device syncs, and the memory each block added (the original profile). The syncs serialize
the streams (the prefill lanes, the row split's exchanges) and expose every launch and every cross-rank wait, so short
blocks read far longer than they run: read shares of long blocks only.
``events`` (patch 0191): each block timed by CUDA events recorded on the stream that issues it, read once after the
prompt; nothing waits, so streams keep overlapping and the prompt runs at speed. A block's time is its span on its
issuing stream (its own kernels, plus any wait for another stream queued inside it); work it queues on another
stream (an exchange) is not in it. Nested blocks ("dsa: x" inside "dsa (total)") are timed alike; shares count the
outer kinds only. No memory figures in this mode (they need syncs)."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager

import torch


def _mode(value: str | None = None) -> str:
    value = (os.environ.get("TF_GLM_PROFILE", "0") if value is None else value).strip() or "0"
    if value not in ("0", "1", "events"):
        raise ValueError(f"TF_GLM_PROFILE: 0 (off), 1 (synced) or events, not {value!r}")
    return value


MODE = _mode()
ENABLED = MODE != "0"
EVENTS = MODE == "events"
active = False                     # set only around prefill chunks
totals: dict[str, float] = {}
counts: dict[str, int] = {}
peaks: dict[str, int] = {}         # the most memory a block allocated on top of what was live before it
host: dict[str, float] = {}        # events mode: host seconds spent issuing each block (Python, launches, host waits)
_marks: list[tuple[str, object, object]] = []     # events mode: (name, start, end) of every block since the report
_pool: list = []                   # events mode: recorded events, reused once read
_used = 0


def _event():
    global _used
    if _used == len(_pool):
        _pool.append(torch.cuda.Event(enable_timing=True))
    e = _pool[_used]
    _used += 1
    return e


@contextmanager
def timed(name: str, events_only: bool = False):
    """Time the block as ``name``. ``events_only``: a block only the events mode times (one on a side stream, e.g. a
    lane's exchange and glue), so the synced mode's figures stay what they were."""
    if not (ENABLED and active) or (events_only and not EVENTS):
        yield
        return
    if EVENTS:
        stream = torch.cuda.current_stream()
        a = _event()
        a.record(stream)
        t = time.perf_counter()
        try:
            yield
        finally:
            host[name] = host.get(name, 0.0) + time.perf_counter() - t      # the host's time issuing the block
            b = _event()
            b.record(torch.cuda.current_stream())     # the block left the stream it entered on
            _marks.append((name, a, b))
        return
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    yield
    torch.cuda.synchronize()
    totals[name] = totals.get(name, 0.0) + time.perf_counter() - t
    counts[name] = counts.get(name, 0) + 1
    peaks[name] = max(peaks.get(name, 0), torch.cuda.max_memory_allocated() - before)


_wall = [0.0]                      # events mode: first block start to last block end, summed over the chunks reported


def _collect() -> None:
    """Events mode: wait for the recorded events once and add their spans to the totals (and the wall time from the
    first block's start to the last block's end, every stream: events share one timeline)."""

    global _used
    if not _marks:
        return
    torch.cuda.synchronize()
    first = _marks[0][1]
    last = 0.0
    for name, a, b in _marks:
        totals[name] = totals.get(name, 0.0) + a.elapsed_time(b) / 1000.0
        counts[name] = counts.get(name, 0) + 1
        last = max(last, first.elapsed_time(b) / 1000.0)
    _wall[0] += last
    _marks.clear()
    _used = 0


def report(tokens: int) -> None:
    if not ENABLED:
        return
    if EVENTS:
        _collect()
    if not totals:
        return
    total = sum(v for k, v in totals.items() if ":" not in k)          # "dsa: x" parts are inside "dsa (total)"
    if EVENTS:
        parts = ", ".join(f"{k} {v:.3f}s ({100 * v / total:.0f}%, {counts.get(k, 0)}x {1e3 * v / max(1, counts.get(k, 0)):.3f}ms)"
                          for k, v in sorted(totals.items(), key=lambda kv: -kv[1]))
        issued = sum(v for k, v in host.items() if ":" not in k)
        print(f"[tensorfold] prefill profile (events: GPU time on the issuing stream, streams overlapping), {tokens} "
              f"tokens, first block to last {_wall[0]:.2f}s, outer blocks {total:.2f}s "
              f"({100 * total / max(_wall[0], 1e-9):.0f}% of it), host issuing them {issued:.2f}s: {parts}", flush=True)
        slow = ", ".join(f"{k} {v:.3f}s" for k, v in sorted(host.items(), key=lambda kv: -kv[1])[:8])
        print(f"[tensorfold] prefill profile host time issuing (most): {slow}", flush=True)
        _wall[0] = 0.0
        host.clear()
    else:
        parts = ", ".join(f"{k} {v:.2f}s ({100 * v / total:.0f}%)" for k, v in sorted(totals.items(), key=lambda kv: -kv[1]))
        print(f"[tensorfold] prefill profile, {tokens} tokens in {total:.1f}s timed ({tokens / total:.0f} tok/s): {parts}",
              flush=True)
        calls = ", ".join(f"{k} {counts.get(k, 0)}" for k in sorted(totals, key=lambda k: -totals[k]))
        print(f"[tensorfold] prefill profile calls: {calls}", flush=True)
        big = ", ".join(f"{k} {v / 2**30:.2f}" for k, v in sorted(peaks.items(), key=lambda kv: -kv[1])[:6])
        print(f"[tensorfold] prefill memory (GiB): allocated {torch.cuda.memory_allocated() / 2**30:.1f}, reserved "
              f"{torch.cuda.memory_reserved() / 2**30:.1f}; most a block added: {big}", flush=True)
    totals.clear()
    counts.clear()
    peaks.clear()
