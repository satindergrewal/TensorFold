"""Priority lanes for concurrent CUDA engines (patch 0141): request classes, and the rules that order admission,
prompt chunks and decode rounds by them. Pure Python (no torch): rank 0's scheduler asks, and every rule takes its
clock as an argument, so tests drive it on a simulated clock.

A request names its class with the ``priority`` field (or, when the body has none, the ``X-Priority`` header):
``realtime`` (latency-critical, for example speech), ``interactive`` (alias ``high``: a person waiting on the
reply), ``normal`` (``default``) or ``background`` (``low``, ``batch``), or an integer as vLLM reads its field (lower
is sooner): 0 normal, -1 interactive, -2 or less realtime, 1 or more background. A request naming none gets
TENSORFOLD_PRIORITY_DEFAULT (normal); session-title requests stay background, as before.

What a class decides (rank 0 decides everything; the other ranks apply rank 0's messages, so nothing here reaches
the arithmetic of any request):

- admission: the waiting request of the soonest class goes first (a normal or background request moves one class
  sooner for every TENSORFOLD_PRIORITY_ADMIT_AGE_S it has waited, up to interactive: nothing ages into realtime),
  then arrival order. Slots are kept in tiers: TENSORFOLD_PRIORITY_RESERVE_REALTIME slots only realtime requests
  take, TENSORFOLD_PRIORITY_RESERVE_INTERACTIVE more only realtime and interactive ones (by default one each from 4
  and 8 slots on), each tier while a request of its class arrived within TENSORFOLD_PRIORITY_TIER_HOLD_S (a server
  whose clients name no class keeps every slot). A tier counts only the streams of the classes it keeps out, so no
  slot idles while its own class has room. When no slot a waiting request may take is free, the youngest decoding
  streams of a later class that is normal or background (never realtime or interactive) give their slots back and
  replay later, as background streams always did: exactly as many as the request needs, or none when that many
  cannot yield; at most TENSORFOLD_PRIORITY_MAX_YIELDS times a request, and a replay waits as a new arrival.
- prompt chunks: the filling stream that advances next is a realtime one if any (the fewest rows left first), else
  one that has not advanced for TENSORFOLD_PRIORITY_STARVE_MS (the longest waiting first; never two such chunks in a
  row), else the soonest class, then the fewest prompt rows left (a short prompt never waits behind a long one), then
  admission order.
- chunk or round: while streams decode, a prompt chunk of a sooner class than every decoding stream goes before the
  rounds (a round still runs at least every TENSORFOLD_PRIORITY_ROUND_GAP_MS), one of a later class takes
  TENSORFOLD_PRIORITY_DOWN_SHARE of the iterations, equal classes alternate by the engine's own share as before, and
  a chunk of at most TENSORFOLD_PRIORITY_SHORT_ROWS rows (no later class than the decoding streams) does not wait for
  its turn.
- chunk size: while realtime requests arrive (within TENSORFOLD_PRIORITY_TIER_HOLD_S), a prompt chunk of a later class
  is at most TENSORFOLD_PRIORITY_REALTIME_ROWS rows (a chunk in flight is what a realtime arrival waits for), so
  larger prompt chunks (TF_GLM_PREFILL_ROWS) keep their speed when no realtime client is about.
- a realtime reply's head: while a realtime stream has written fewer than TENSORFOLD_PRIORITY_REALTIME_HEAD tokens,
  rounds hold only the realtime streams in their heads and no prompt chunk of a later class runs (unless starving),
  so the first sentence of a latency-critical reply decodes at a lone stream's speed; the other streams wait those
  few rounds.

TENSORFOLD_PRIORITY=0 restores the scheduler as it was (arrival order, background last)."""

from __future__ import annotations

import itertools
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

REALTIME, INTERACTIVE, NORMAL, BACKGROUND = 0, 1, 2, 3
CLASSES = ("realtime", "interactive", "normal", "background")
NAMES = {"realtime": REALTIME, "interactive": INTERACTIVE, "high": INTERACTIVE, "normal": NORMAL, "default": NORMAL,
         "background": BACKGROUND, "low": BACKGROUND, "batch": BACKGROUND}
HEADER = "X-Priority"                   # read when the body has no ``priority`` field


def _from_int(value: int) -> int:
    return min(BACKGROUND, max(REALTIME, NORMAL + int(value)))


def parse(value: Any, default: int = NORMAL) -> int:
    """A request's ``priority`` (a class name, an alias or a vLLM-style integer) as a class; ``default`` when absent.
    ValueError for anything else."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, bool):
        pass
    elif isinstance(value, int):
        return _from_int(value)
    elif isinstance(value, float) and value.is_integer():
        return _from_int(int(value))
    elif isinstance(value, str):
        text = value.strip().lower()
        if text in NAMES:
            return NAMES[text]
        try:
            return _from_int(int(text))
        except ValueError:
            pass
    raise ValueError(f"priority: realtime, interactive (high), normal (default) or background (low, batch), or an "
                     f"integer where lower is sooner (0 normal, -1 interactive, -2 realtime, 1 background), not "
                     f"{value!r}")


def name(klass: int) -> str:
    return CLASSES[min(BACKGROUND, max(REALTIME, int(klass)))]


def effective(klass: int, waited_s: float, age_s: float) -> int:
    """A waiting request's class for admission: a normal or background one moves one class sooner for every
    ``age_s`` seconds waited (0: never), up to interactive (realtime is never reached by waiting)."""

    if age_s <= 0 or waited_s <= 0 or klass <= INTERACTIVE:
        return klass
    return max(INTERACTIVE, klass - int(waited_s // age_s))


# -- settings ---------------------------------------------------------------------------------------------------
def _number(env: Mapping[str, str], key: str, default: float, low: float, high: float) -> float:
    raw = env.get(key, "").strip()
    try:
        value = default if raw == "" else float(raw)
    except ValueError:
        value = low - 1
    if not low <= value <= high:
        raise ValueError(f"{key}: a number from {low:g} to {high:g}, not {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    """The priority rules' settings (TENSORFOLD_PRIORITY_*; ``from_env``)."""

    # each field's setting is TENSORFOLD_PRIORITY_<NAME> (the module docstring); -1 reserves: by --parallel
    enabled: bool = True                # TENSORFOLD_PRIORITY: 0 restores the order as before
    default: int = NORMAL               # _DEFAULT: the class of a request that names none
    reserve_rt: int = -1                # _RESERVE_REALTIME: slots only realtime takes (auto: 1 from 4 slots)
    reserve_int: int = -1               # _RESERVE_INTERACTIVE: more for realtime and interactive (auto: 1 from 8)
    starve_s: float = 2.0               # _STARVE_MS: a filling stream this long without a chunk goes first
    round_gap_s: float = 1.0            # _ROUND_GAP_MS: decoding streams get a round at least this often
    short_rows: int = 256               # _SHORT_ROWS: a chunk this short does not wait for its turn (0: off)
    down_share: float = 0.25            # _DOWN_SHARE: iterations a later class's chunk takes while sooner decode
    up_share: float = 1.0               # _UP_SHARE: iterations a sooner class's chunk takes while later decode
    admit_age_s: float = 10.0           # _ADMIT_AGE_S: seconds waited that move a request one class sooner
    max_yields: int = 2                 # _MAX_YIELDS: times one request may give its slot back
    tier_hold_s: float = 900.0          # _TIER_HOLD_S: a tier holds while its class arrived this recently
    rt_head: int = 32                   # _REALTIME_HEAD: a realtime reply's tokens decoded in rounds of its own
    rt_rows: int = 2048                 # _REALTIME_ROWS: later classes' chunk rows while realtime arrives (0: off)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        on = env.get("TENSORFOLD_PRIORITY", "").strip() or "1"
        if on not in ("0", "1"):
            raise ValueError(f"TENSORFOLD_PRIORITY: 0 or 1, not {on!r}")
        if on == "0":                                 # off: the other settings are not read
            return cls(enabled=False)
        default = parse(env.get("TENSORFOLD_PRIORITY_DEFAULT", ""), NORMAL)

        def slots(key: str) -> int:
            raw = env.get(key, "").strip().lower()
            if raw in ("", "auto"):
                return -1
            if raw.isdecimal() and int(raw) <= 31:
                return int(raw)
            raise ValueError(f"{key}: auto or 0 to 31 slots, not {raw!r}")

        return cls(enabled=on == "1", default=default, reserve_rt=slots("TENSORFOLD_PRIORITY_RESERVE_REALTIME"),
                   reserve_int=slots("TENSORFOLD_PRIORITY_RESERVE_INTERACTIVE"),
                   starve_s=_number(env, "TENSORFOLD_PRIORITY_STARVE_MS", 2000, 50, 600000) / 1000.0,
                   round_gap_s=_number(env, "TENSORFOLD_PRIORITY_ROUND_GAP_MS", 1000, 20, 600000) / 1000.0,
                   short_rows=int(_number(env, "TENSORFOLD_PRIORITY_SHORT_ROWS", 256, 0, 65536)),
                   down_share=_number(env, "TENSORFOLD_PRIORITY_DOWN_SHARE", 0.25, 0.01, 1.0),
                   up_share=_number(env, "TENSORFOLD_PRIORITY_UP_SHARE", 1.0, 0.01, 1.0),
                   admit_age_s=_number(env, "TENSORFOLD_PRIORITY_ADMIT_AGE_S", 10, 0, 86400),
                   max_yields=int(_number(env, "TENSORFOLD_PRIORITY_MAX_YIELDS", 2, 0, 100)),
                   tier_hold_s=_number(env, "TENSORFOLD_PRIORITY_TIER_HOLD_S", 900, 0, 31536000),
                   rt_head=int(_number(env, "TENSORFOLD_PRIORITY_REALTIME_HEAD", 32, 0, 1 << 20)),
                   rt_rows=int(_number(env, "TENSORFOLD_PRIORITY_REALTIME_ROWS", 2048, 0, 1 << 20)))

    def tiers(self, max_streams: int, active: tuple[bool, bool] = (True, True)) -> tuple[int, int]:
        """(slots only realtime takes, more slots only realtime and interactive take):
        TENSORFOLD_PRIORITY_RESERVE_REALTIME (by default 1 from 4 slots on) and TENSORFOLD_PRIORITY_RESERVE_INTERACTIVE
        (by default 1 from 8 slots on), each only while ``active`` (a request of its class arrived lately), cut so
        normal and background requests keep at least one slot (the interactive tier first)."""

        rt = (1 if max_streams >= 4 else 0) if self.reserve_rt < 0 else self.reserve_rt
        it = (1 if max_streams >= 8 else 0) if self.reserve_int < 0 else self.reserve_int
        rt, it = (rt if active[0] else 0), (it if active[1] else 0)
        rt = max(0, min(rt, max_streams - 1))
        it = max(0, min(it, max_streams - 1 - rt))
        return rt, it

    def slots_reserved(self, max_streams: int, active: tuple[bool, bool] = (True, True)) -> int:
        return sum(self.tiers(max_streams, active))

    def short(self, klass: int, live: Sequence[int], max_streams: int, active: tuple[bool, bool] = (True, True)) -> int:
        """How many live streams must leave before a request of ``klass`` may be admitted; ``live``: the classes the
        live streams were admitted at. A free slot; for a request past realtime also fewer than all but the realtime
        tier held by streams past realtime; for a normal or background request also fewer than all but both tiers
        held by normal and background streams. Each tier counts only the classes it keeps out, so a stream of a
        tier's own class never makes it keep a second slot free."""

        rt, it = self.tiers(max_streams, active)
        need = len(live) - (max_streams - 1)
        if klass > REALTIME:
            need = max(need, sum(1 for k in live if k > REALTIME) - (max_streams - rt - 1))
        if klass > INTERACTIVE:
            need = max(need, sum(1 for k in live if k > INTERACTIVE) - (max_streams - rt - it - 1))
        return max(0, need)

    def cap(self, klass: int, max_streams: int, active: tuple[bool, bool] = (True, True)) -> int:
        """The most slots streams of ``klass`` and later classes may hold: realtime every one; interactive all but the
        realtime tier; normal and background all but both tiers (``active``: which tiers hold, ``tiers``)."""

        rt, it = self.tiers(max_streams, active)
        return max_streams if klass <= REALTIME else max_streams - rt if klass <= INTERACTIVE else max_streams - rt - it

    def describe(self, max_streams: int) -> str:
        if not self.enabled:
            return "priority lanes off (TENSORFOLD_PRIORITY=0): arrival order, background last"
        rt, it = self.tiers(max_streams)
        return (f"priority lanes realtime > interactive > normal > background (default {name(self.default)}): "
                f"of {max_streams} slots {rt} kept for realtime and {it} more for interactive, "
                f"starve {self.starve_s * 1e3:.0f} ms, round gap {self.round_gap_s * 1e3:.0f} ms, short chunk "
                f"{self.short_rows} rows, shares up {self.up_share:g} / down {self.down_share:g}, admission age "
                f"{self.admit_age_s:g} s, at most {self.max_yields} yields a request, tiers hold "
                f"{self.tier_hold_s:g} s, realtime head {self.rt_head} tokens, chunks of later classes at most "
                f"{self.rt_rows or 'any'} rows while realtime requests arrive")


def enabled(env: Mapping[str, str] | None = None) -> bool:
    env = os.environ if env is None else env
    return (env.get("TENSORFOLD_PRIORITY", "").strip() or "1") != "0"


def request_class(body: Mapping[str, Any], default: int | None = None, *, title: bool = False) -> int:
    """A request body's class (``title``: a session-title request, background as before)."""

    if title:
        return BACKGROUND
    if default is None:
        default = parse(os.environ.get("TENSORFOLD_PRIORITY_DEFAULT", ""), NORMAL)
    return parse(body.get("priority"), default)


# -- prompt chunks: which filling stream advances, and whether a chunk or a round goes next --------------------------
@dataclass(frozen=True)
class Fill:
    """A filling stream as the rules see it: its class, prompt rows left, admission order and the clock time of its
    last chunk (or of its admission)."""

    klass: int
    rows: int
    order: int
    since: float


def fill_key(f: Fill, now: float, starve_s: float) -> tuple:
    """Sort key, soonest first: starving streams (oldest progress first), then class, rows left, admission order."""

    if now - f.since >= starve_s:
        return (0, f.since, f.order)
    return (1, f.klass, f.rows, f.order)


def pick_fill(fills: Sequence[Fill], now: float, starve_s: float, starved_last: bool = False) -> tuple[int, bool]:
    """(index of the filling stream that advances next, whether it goes for starving): a realtime one if any (the
    fewest rows left first), else, unless the last chunk went to a starving stream (``starved_last``), the one that
    has waited longest past ``starve_s``, else by class, rows left and admission order."""

    if not fills:
        raise ValueError("no filling stream to pick")
    rt = [i for i, f in enumerate(fills) if f.klass <= REALTIME]
    if rt:
        return min(rt, key=lambda i: (fills[i].rows, fills[i].order)), False
    if starved_last:
        return min(range(len(fills)), key=lambda i: fill_key(fills[i], now, float("inf"))), False
    k = min(range(len(fills)), key=lambda i: fill_key(fills[i], now, starve_s))
    return k, now - fills[k].since >= starve_s


class Arbiter:
    """Rank 0's choice, each iteration while streams decode, between the next prompt chunk and a decode round."""

    def __init__(self, settings: Settings, share: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.s, self.share, self.clock = settings, float(share), clock
        self.credit = 0.0
        self.last_round: float | None = None
        self.counts = {"chunks": 0, "rounds": 0, "starved": 0, "overdue": 0, "short": 0, "held": 0}

    def choose(self, fill_class: int, fill_rows: int, starving: bool, round_class: int, hold: bool = False) -> bool:
        """True: the chunk goes now; False: a decode round. ``round_class``: the soonest decoding stream's class;
        ``hold``: a realtime reply's head decodes (no chunk of a later class unless starving)."""

        s = self.s
        now = self.clock()
        if self.last_round is not None and now - self.last_round >= s.round_gap_s:
            self.counts["overdue"] += 1
            return self._round()
        if starving:
            self.counts["starved"] += 1
            return self._chunk()
        if hold and fill_class > round_class:
            self.counts["held"] += 1
            return self._round()
        if s.short_rows and fill_rows <= s.short_rows and fill_class <= round_class:
            self.counts["short"] += 1
            return self._chunk()
        share = (s.up_share if fill_class < round_class else s.down_share if fill_class > round_class
                 else self.share)
        self.credit = min(self.credit + share, 1.0 + share)
        if self.credit < 1.0:
            return self._round()
        self.credit -= 1.0
        return self._chunk()

    def _chunk(self) -> bool:
        self.counts["chunks"] += 1
        return True

    def _round(self) -> bool:
        self.counts["rounds"] += 1
        return False

    def rounded(self) -> None:
        """A decode round just ran."""

        self.last_round = self.clock()


# -- admission ------------------------------------------------------------------------------------------------------
def preemptible(s: Any) -> bool:
    """A stream that can give its slot back and replay: no grammar or images (they cannot replay), not finished."""

    return (not getattr(s, "done", False) and getattr(s, "constraint", None) is None
            and getattr(s, "vision", None) is None and len(getattr(s, "out", ())) < getattr(s, "count", 0))


def victims(streams: Sequence[Any], waiter_class: int, max_yields: int, need: int) -> list:
    """``need`` streams that give their slots to a waiting request of ``waiter_class`` (``victim`` in turn), or none
    when fewer can (a request is never given a slot it would still have to wait for)."""

    pool, out = list(streams), []
    for _ in range(max(0, need)):
        v = victim(pool, waiter_class, max_yields)
        if v is None:
            return []
        out.append(v)
        pool = [s for s in pool if s is not v]
    return out


def victim(streams: Sequence[Any], waiter_class: int, max_yields: int) -> Any | None:
    """The decoding stream (``streams`` in admission order) that gives its slot to a waiting request of
    ``waiter_class``: a later class that is normal or background, replayable, given back fewer than ``max_yields``
    times; the latest class first, then the youngest. A stream counts at the class it was admitted at
    (``admitted_as``: its aged class, so a request let in by waiting is not sent back by the next arrival)."""

    best, best_key = None, None
    for order, s in enumerate(streams):
        klass = getattr(s, "admitted_as", getattr(s, "priority", NORMAL))
        if klass <= waiter_class or klass < NORMAL or getattr(s, "yields", 0) >= max_yields or not preemptible(s):
            continue
        key = (klass, order)
        if best_key is None or key > best_key:
            best, best_key = s, key
    return best


class PriorityWaiting:
    """Waiting (stream, box) pairs: the soonest effective class first (``effective``: aged by the time waited since
    the request arrived, ``stream.arrived`` on ``clock``), then arrival. ``queue.Queue``'s put / get / get_nowait,
    plus peeking (``best``), removing a peeked item and waiting for any item without taking it."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, age_s: float = 0.0) -> None:
        self.cv = threading.Condition()
        self.items: list[tuple[int, tuple]] = []
        self.seq = itertools.count()
        self.clock, self.age_s = clock, age_s

    def put(self, item: tuple, block: bool = True, timeout: float | None = None) -> None:
        with self.cv:
            self.items.append((next(self.seq), item))
            self.cv.notify_all()

    def rank(self, stream: Any, now: float | None = None) -> int:
        """The stream's effective class now."""

        now = self.clock() if now is None else now
        return effective(getattr(stream, "priority", NORMAL), now - getattr(stream, "arrived", now), self.age_s)

    def _best(self) -> tuple[int, tuple] | None:
        if not self.items:
            return None
        now = self.clock()
        return min(self.items, key=lambda e: (self.rank(e[1][0], now), e[0]))

    def best(self) -> tuple | None:
        """The item ``get`` would take, left waiting (None: nothing waits)."""

        with self.cv:
            entry = self._best()
            return entry[1] if entry is not None else None

    def remove(self, item: tuple) -> None:
        with self.cv:
            self.items = [e for e in self.items if e[1] is not item]

    def get(self, block: bool = True, timeout: float | None = None) -> tuple:
        with self.cv:
            if block:
                end = None if timeout is None else time.monotonic() + timeout
                while not self.items:
                    left = None if end is None else end - time.monotonic()
                    if left is not None and left <= 0:
                        raise queue.Empty
                    self.cv.wait(left)
            entry = self._best()
            if entry is None:
                raise queue.Empty
            self.items.remove(entry)
            return entry[1]

    def get_nowait(self) -> tuple:
        return self.get(False)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until an item waits (or ``timeout`` seconds pass): whether one does."""

        with self.cv:
            if not self.items and (timeout is None or timeout > 0):
                self.cv.wait_for(lambda: bool(self.items), timeout)
            return bool(self.items)

    def foreground(self) -> bool:
        """Whether a request of a class before background waits."""

        with self.cv:
            return any(getattr(e[1][0], "priority", NORMAL) < BACKGROUND for e in self.items)

    def qsize(self) -> int:
        with self.cv:
            return len(self.items)

    def empty(self) -> bool:
        return self.qsize() == 0

    def counts(self) -> dict[str, int]:
        with self.cv:
            out = dict.fromkeys(CLASSES, 0)
            for _, (s, _box) in self.items:
                out[name(getattr(s, "priority", NORMAL))] += 1
            return out
