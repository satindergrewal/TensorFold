"""The per-round stream cap of GLM-5.3-Flash's concurrent decode rounds (--parallel): a round's verify window holds
at most ``rows`` rows (TF_GLM_MULTI_WINDOW) and every decoding stream brings at least its pending token, so a round
takes at most ``rows`` streams; the others wait for the next rounds.

Before the cap a round took every decoding stream and ``multi.trim`` cut drafts only, never a stream's pending row:
with more decoding streams than window rows the batched verify refused the window (its buffers hold ``rows``), the
round failed on every rank, and the ``broken`` latch failed every live stream until the ranks restarted. A startup
refusal of --parallel above the window stood in for this cap until it existed.

The streams wait in a queue: a stream joins it at the back when it starts decoding and goes to the back again each
time a round serves it (streams a round serves keep their order among themselves). When a round cannot hold every
stream it may hold (``RoundCap.take``), it takes them in this order:

1. the streams a round left out and that no round has served since, the longest waiting first whatever their class,
   so a stream left out goes first in the next round that may hold it: with 2 x rows streams or fewer it never waits
   two rounds in a row, and with more it waits at most ceil(streams / rows) rounds;
2. the sooner class when the priority lanes are on (realtime, interactive, normal, background): the room the left
   out streams leave goes to the sooner classes first; with the lanes off (TENSORFOLD_PRIORITY=0) every stream counts
   alike;
3. the queue's order: the longest waiting first.

The round holds the chosen streams in admission order. A round whose streams all fit takes them all, as before, so a
setting whose window holds --parallel rows (the default recipe's 40 streams in 63 rows) runs exactly the rounds it ran
without the cap. With one class, 40 streams in a 32-row window each get 4 rounds of every 5, and 64 streams every
other round.

The priority lanes' round gap (TENSORFOLD_PRIORITY_ROUND_GAP_MS: decoding streams get a round at least this often,
even while realtime heads hold rounds of their own) counts from the end of the last round that served every
decoding stream. Under the cap that takes a cycle of rounds of every stream: the first such round after the gap's
mark opens it with the streams it could hold, each later one ticks off the streams it serves, and the cycle closes
(``settled``) when every one was served; a stream waiting for room, or gone, leaves the cycle, as a round of every
stream always counted the streams it paused.

Exactness (bit-identical): a stream's tokens never depend on which round it lands in. The batched verify gives each
stream's rows the bits of its own window alone, sampling draws by position, and a stream's drafts (which never change
its reply) and its depth policy advance only in rounds it is in. Rank 0 decides the rounds and sends each round's
streams; rank 1 applies them, so this planner runs on rank 0 only and reaches no request's arithmetic.

Pure Python (no torch): the tests drive it directly."""

from __future__ import annotations

from typing import Any, Sequence

NORMAL = 2                                   # cuda.priority.NORMAL: the class of a stream that names none


def klass(lane: Any) -> int:
    """A stream's priority class (``lane.s.priority``; normal when it has none)."""

    s = getattr(lane, "s", None)
    k = getattr(s, "priority", NORMAL)
    return NORMAL if k is None or k < 0 else int(k)


class RoundCap:
    """Rank 0's planner of which decoding streams each round holds (see the module docstring).

    ``rows``: the verify window's rows (a round holds at most that many streams); ``classes``: order by priority
    class (the priority lanes are on). Streams are duck-typed: ``sid``, ``order`` and ``s.priority``."""

    def __init__(self, rows: int, classes: bool = True) -> None:
        if int(rows) < 1:
            raise ValueError(f"a round of {rows} rows")
        self.rows = int(rows)
        self.classes = bool(classes)
        self.round = 0                       # rounds planned
        self.seq = 0                         # the queue's last place given
        self.place: dict[int, int] = {}      # sid -> its place in the queue (lower: waiting longer)
        self.owed: set[int] = set()          # streams a round left out and no round has served since
        self.cover: set[int] | None = None   # the open cycle's streams not served yet (None: no cycle open)
        self.counts = {"capped": 0, "left_out": 0, "cycles": 0}

    def _back(self) -> int:
        self.seq += 1
        return self.seq

    def take(self, candidates: Sequence[Any], decoding: Sequence[Any] | None = None, full: bool = True) -> list:
        """The streams this round holds, at most ``rows``, in admission order.

        ``candidates``: the streams the round may hold (every decoding stream, or a realtime reply's heads, with the
        ones waiting for room left out), in admission order; ``decoding``: every decoding stream (default the
        candidates), so a stream that left is forgotten and one that just started joins the queue; ``full``: the
        round is a round of every decoding stream (it opens or continues the round gap's cycle; a heads round does
        neither)."""

        self.round += 1
        everyone = list(candidates) if decoding is None else list(decoding)
        live = {l.sid for l in everyone} | {l.sid for l in candidates}
        for sid in [sid for sid in self.place if sid not in live]:
            del self.place[sid]
        for l in list(everyone) + list(candidates):     # a stream that just started: at the queue's back
            if l.sid not in self.place:
                self.place[l.sid] = self._back()
        self.owed &= live
        ranked = sorted(candidates, key=self._key)
        if len(ranked) <= self.rows:
            chosen, left = list(candidates), []
        else:
            ranked = ranked[:self.rows]
            keep = {id(l) for l in ranked}
            chosen = [l for l in candidates if id(l) in keep]
            left = [l for l in candidates if id(l) not in keep]
            self.counts["capped"] += 1
            self.counts["left_out"] += len(left)
        took = {l.sid for l in chosen}
        self.owed = (self.owed | {l.sid for l in left}) - took
        if full:
            # the round gap's cycle: opened by this round when none is open, then every stream it may hold ticked
            # off as served; a stream waiting for room (not a candidate) leaves it
            here = {l.sid for l in candidates}
            opened = self.cover is None
            cover = (here if opened else self.cover & here) - took
            if opened and cover:
                self.counts["cycles"] += 1
            self.cover = cover or None
        elif self.cover is not None:
            self.cover = (self.cover & live) or None
        for l in ranked:                                 # served: to the queue's back, in the order they were taken
            self.place[l.sid] = self._back()
        return chosen

    def _key(self, lane: Any) -> tuple:
        if lane.sid in self.owed:                        # left out before: the longest waiting first, any class
            return (0, 0, self.place[lane.sid], lane.order)
        return (1, klass(lane) if self.classes else 0, self.place[lane.sid], lane.order)

    def forget(self, sid: int) -> None:
        """A stream left (finished, cancelled, given back): no place in the queue, no turn, out of the open cycle."""

        self.place.pop(sid, None)
        self.owed.discard(sid)
        if self.cover is not None:
            self.cover.discard(sid)
            self.cover = self.cover or None

    def settled(self) -> bool:
        """Whether no cycle is open: every stream the last cycle's rounds could hold has been served."""

        return self.cover is None

    def describe(self) -> str:
        return (f"rounds hold at most {self.rows} streams (one pending row each); the others take the next rounds, "
                f"those left out first, then {'by class, then ' if self.classes else ''}the longest waiting")
