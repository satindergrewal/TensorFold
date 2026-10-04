"""TF_GLM_SEGPROF=K: where a batched decode round's GPU time goes, by kind of segment, every K-th batched round
(0, the default: off). Graph-safe: no host sync inside the round, nothing changes in what any kernel computes.

How: with the setting on, ``verify.BatchedVerify.capture`` captures every window width twice, the plain graph as
always and a profiled one in which every segment boundary is a CUDA event-record node (``torch.cuda.Event(
enable_timing=True, external=True)`` recorded during the capture: an external event, which a replay records, instead
of a cross-stream dependency). Every K-th batched round replays the profiled graph (an eager window records the same
events live); after the round's own sync (the sampler's), rank 0 reads the time from each event to the next and adds
it to the kind that ran there. The other rounds replay the plain graphs, as without the setting. At start, each
checked width's profiled graph is replayed against its plain graph on the same window: the logits must be the same
bit for bit (else the profile turns itself off with a line), and the replay times give the profile's cost.

Kinds (TF_GLM_SEGPROF_DETAIL=kinds, the default): ``experts`` (routed experts), ``shared`` (the shared expert),
``dsa`` (DSA attention: latent write, indexer, token selection, absorb, attention, expand), ``kda`` (the KDA
recurrence), ``dense`` (projections: KDA's and DSA's in and out projections, the dense MLP layers), ``head`` (the
vocabulary projection), ``exchange`` (the ranks' all-gathers), ``glue`` (everything else: hyper-connections, norms,
routing, combine, embedding). TF_GLM_SEGPROF_DETAIL=layers: fewer events, by layer kind (``kda layer``, ``dsa layer``,
``moe layer``, ``mlp layer``, ``head``, ``glue``). Spans outside the window's graph, on the GPU timeline (they include
time the GPU waits for the host there): ``sampler`` and ``commit``.

The boundaries come from the forward's existing ``prof.timed`` blocks, ``forward.mm`` (a projection called outside
any timed block is ``dense``; the one writing the logits is ``head``) and ``forward.gather`` (``exchange``): this
module wraps those three functions while the setting is on (``install``) and they pass straight through when no
profiled window is being recorded. The ranks must agree on the setting (compared at start): the start check replays
graphs whose all-gathers every rank joins.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab (the multi-stream engine, its batched verify windows and
their CUDA graphs, the forward's timed blocks)."""

from __future__ import annotations

import os
import time
import zlib
from contextlib import ExitStack, contextmanager

KINDS = ("experts", "shared", "dsa", "kda", "dense", "head", "exchange", "glue")
LAYER_KINDS = ("kda layer", "dsa layer", "moe layer", "mlp layer", "head", "glue")
SPANS = ("sampler", "commit")
END = "end"

# the forward's prof.timed blocks -> kind (detail "kinds"); blocks not listed keep the kind they run in
BLOCKS = {
    "kda: projections": "dense", "kda: recurrence": "kda",
    "dsa: latent write": "dsa", "dsa: indexer update": "dsa", "dsa: select tokens": "dsa", "dsa: absorb": "dsa",
    "dsa: attention": "dsa", "dsa: dense attention": "dsa", "dsa: sparse attention": "dsa", "dsa: expand": "dsa",
    "moe: shared expert": "shared", "moe: routed (exl3)": "experts", "moe: gate/up": "experts", "moe: down": "experts",
    "moe: combine": "glue", "moe: route": "glue", "moe: all-gather": "exchange", "hc": "glue", "hc (layer end)": "glue",
}
# detail "layers": the layer blocks only
LAYER_BLOCKS = {"kda": "kda layer", "dsa (total)": "dsa layer", "moe (total)": "moe layer", "mlp": "mlp layer"}
CHECK_EVERY = 64            # the start check's widths: every captured width up to 64 that is a power of two, then 64s
REPORT_DEFAULT = 20


def settings(env=None) -> tuple[int, int, str]:
    """(every, report, detail) from TF_GLM_SEGPROF (0 off, else profile every K-th batched round, K <= 1,000,000),
    TF_GLM_SEGPROF_REPORT (profiled rounds a report line, default 20) and TF_GLM_SEGPROF_DETAIL (kinds or layers).
    ValueError on a value that is not valid."""

    env = os.environ if env is None else env

    def number(name: str, default: int, least: int, most: int) -> int:
        v = (env.get(name, "") or str(default)).strip()
        if not v.isdigit() or not least <= int(v) <= most:
            raise ValueError(f"{name}: {least} to {most}, not {v!r}")
        return int(v)

    every = number("TF_GLM_SEGPROF", 0, 0, 1_000_000)
    report = number("TF_GLM_SEGPROF_REPORT", REPORT_DEFAULT, 1, 1_000_000)
    detail = (env.get("TF_GLM_SEGPROF_DETAIL", "") or "kinds").strip().lower()
    if detail not in ("kinds", "layers"):
        raise ValueError(f"TF_GLM_SEGPROF_DETAIL: kinds or layers, not {detail!r}")
    return every, report, detail


def code() -> int:
    """The settings for the ranks' start comparison (ValueError when not valid)."""

    every, report, detail = settings()
    return zlib.crc32(f"{every},{report},{detail}".encode()) & 0x3FFFFFFF


class Prof:
    """One process's segment profile: the event pool, each captured width's event kinds, the sums."""

    def __init__(self, every: int, report: int = REPORT_DEFAULT, detail: str = "kinds", reader: bool = True,
                 log=print, event=None) -> None:
        import torch

        self.every, self.report_every, self.detail, self.reader, self.log = every, report, detail, reader, log
        self.make = event or (lambda: torch.cuda.Event(enable_timing=True, external=True))
        self.span_make = event or (lambda: torch.cuda.Event(enable_timing=True))
        self.kinds_all = KINDS if detail == "kinds" else LAYER_KINDS
        self.pool: list = []                   # events, shared by every log: a log's i-th boundary is pool[i]
        self.kinds: list[str] | None = None    # the log being recorded: kinds[i] runs from event i to event i + 1
        self.cur: str | None = None
        self.logs: dict[int, list[str]] = {}   # captured width -> its kinds
        self.pending: list[str] | None = None  # this round's log, read after the round
        self.pending_rows = 0
        self.spans: dict[str, tuple] = {}      # name -> (start, end) events of this round
        self.queue: list = []                  # spans read once their end event has completed (the commit's later)
        self.n = 0                             # batched rounds seen
        self.off: str | None = None            # why the profile turned itself off (the start check)
        self._reset()

    def _reset(self) -> None:
        self.rounds = 0
        self.rows = 0
        self.total = 0.0
        self.sums = dict.fromkeys(self.kinds_all, 0.0)
        self.span_sums = dict.fromkeys(SPANS, 0.0)
        self.span_counts = dict.fromkeys(SPANS, 0)
        self.events = 0
        self.t_report = time.perf_counter()

    # -- recording -----------------------------------------------------------------------------------------------------
    def kind_of(self, block: str) -> str | None:
        return (BLOCKS if self.detail == "kinds" else LAYER_BLOCKS).get(block)

    def _event(self, kind: str) -> None:
        k = self.kinds
        if k is None or kind == self.cur:
            return
        i = len(k)
        if i == len(self.pool):
            self.pool.append(self.make())
        self.pool[i].record()
        k.append(kind)
        self.cur = kind

    @contextmanager
    def seg(self, kind: str):
        """GPU work issued inside runs under ``kind``; the kind before resumes after."""

        prev = self.cur
        self._event(kind)
        try:
            yield
        finally:
            if prev is not None:
                self._event(prev)

    def start(self) -> None:
        self.kinds, self.cur = [], None
        self._event("glue")

    def stop(self) -> list[str]:
        self._event(END)
        k, self.kinds, self.cur = self.kinds or [], None, None
        return k

    @contextmanager
    def recording(self):
        """Record one forward's boundaries (inside a capture: as event-record nodes of the graph)."""

        self.start()
        out: list[str] = []
        try:
            yield out
        finally:
            out.extend(self.stop())

    # -- rounds --------------------------------------------------------------------------------------------------------
    def want(self) -> bool:
        """Whether this batched round is a profiled one (every K-th; never once the start check turned it off)."""

        self.n += 1
        return self.off is None and self.every > 0 and self.n % self.every == 0

    def replayed(self, width: int, rows: int) -> None:
        """A profiled graph of ``width`` rows ran for a round of ``rows`` rows: read its events after the round."""

        self.pending, self.pending_rows = self.logs.get(width), rows

    @contextmanager
    def eager(self, rows: int):
        """A profiled round's eager window (no graph of its width): record its events live."""

        with self.recording() as log:
            yield
        self.pending, self.pending_rows = log, rows

    @contextmanager
    def span(self, name: str):
        """A span outside the window's graph (the sampler, the commit) of a profiled round, on the GPU timeline."""

        if self.pending is None or not self.reader:
            yield
            return
        a, b = self.span_make(), self.span_make()
        a.record()
        try:
            yield
        finally:
            b.record()
            self.spans[name] = (a, b)

    def collect(self) -> None:
        """At the end of every batched round (after its sampler's sync): rank 0 adds a profiled round's event
        intervals to their kinds, and each span once its end event has completed (the commit's, a round later at
        most); every rank forgets the round. A report line every TF_GLM_SEGPROF_REPORT profiled rounds."""

        log, spans = self.pending, self.spans
        self.pending, self.spans = None, {}
        if not self.reader:
            return
        self.queue.extend(spans.items())
        if log is not None and len(log) >= 2:
            ev = self.pool
            t = [0.0] + [ev[0].elapsed_time(ev[i]) for i in range(1, len(log))]
            for i in range(len(log) - 1):
                k = log[i]
                self.sums[k] = self.sums.get(k, 0.0) + (t[i + 1] - t[i])
            self.total += t[-1]
            self.rounds += 1
            self.rows += self.pending_rows
            self.events += len(log)
        waiting = []
        for name, (a, b) in self.queue:
            if b.query():
                self.span_sums[name] = self.span_sums.get(name, 0.0) + a.elapsed_time(b)
                self.span_counts[name] = self.span_counts.get(name, 0) + 1
            else:
                waiting.append((name, (a, b)))
        self.queue = waiting
        if self.rounds >= self.report_every:
            self.report()

    def report(self) -> None:
        n = max(self.rounds, 1)
        total = self.total / n
        parts = " + ".join(f"{k} {self.sums[k] / n:.2f} ({100 * self.sums[k] / max(self.total, 1e-9):.1f}%)"
                           for k in self.kinds_all if k in self.sums)
        spans = ", ".join(f"{k} span {self.span_sums[k] / max(self.span_counts[k], 1):.2f}" for k in SPANS
                          if self.span_counts.get(k))
        self.log(f"[tensorfold] segment profile, {self.rounds} rounds (1 in {self.every} batched rounds), "
                 f"{self.rows / n:.1f} rows a round, {self.events / n:.0f} events a forward: ms a round: verify GPU "
                 f"{total:.2f} = {parts}" + (f"; {spans}" if spans else ""), flush=True)
        self._reset()

    # -- the start check -----------------------------------------------------------------------------------------------
    def check(self, v, reps: int = 3) -> str:
        """Every rank, after ``v.capture``: each checked width's profiled graph against its plain graph on the
        capture's window (slot 0, timing tokens): the logits bit for bit, and the fastest of ``reps`` replays each.
        Turns the profile off when any width differs. The same replays on every rank (their all-gathers)."""

        import torch

        from .sparse import SPARSE_FROM
        from .verify import timing_tokens

        widths = sorted(set(w for w in v.pgraphs if (w <= CHECK_EVERY and w & (w - 1) == 0) or w % CHECK_EVERY == 0)
                        | {max(v.pgraphs)}) if v.pgraphs else []
        index = v.e.caches.index is not None
        tokens = timing_tokens(v.w.cfg.vocab, v.rows_max)
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        bad, costs = [], []
        for W in widths:
            v._stage(tokens[:W])
            v.seg_rows.set([v._span(v.e.home, 0, W)], sparse_from=SPARSE_FROM if index else None)
            v._write_table([(W, 0, 0, 0, W)])
            best = {}
            outs = {}
            for _ in range(reps + 1):
                for name, g in (("plain", v.graphs[W]), ("profiled", v.pgraphs[W])):
                    a.record()
                    g.replay()
                    b.record()
                    b.synchronize()
                    best[name] = min(best.get(name, float("inf")), a.elapsed_time(b))
                    outs[name] = v.b.logits[:W].clone()
            if not torch.equal(outs["plain"].view(torch.uint8), outs["profiled"].view(torch.uint8)):
                bad.append(W)
            costs.append((W, best["plain"], best["profiled"], len(self.logs.get(W, ()))))
        if bad:
            self.off = f"the profiled windows' logits differ from the plain ones at widths {bad}"
        cost = ", ".join(f"{W}: +{p - q:.2f} ms ({100 * (p - q) / max(q, 1e-9):+.1f}%, {e} events)"
                         for W, q, p, e in costs)
        return (f"[tensorfold] segment profile (TF_GLM_SEGPROF={self.every}, {self.detail}): "
                + (f"OFF: {self.off}" if self.off else
                   f"the profiled windows' logits equal the plain ones bit for bit at widths {widths}")
                + f"; replay cost (fastest of {reps}) by width: {cost}")


ACTIVE: Prof | None = None


def setup(rank: int) -> Prof | None:
    """The process's profile from the settings (None when off), its hooks installed (``install``)."""

    global ACTIVE
    every, report, detail = settings()
    if every <= 0:
        ACTIVE = None
        return None
    ACTIVE = Prof(every, report, detail, reader=rank == 0)
    install()
    return ACTIVE


_installed = False


def install() -> None:
    """Wrap ``prof.timed``, ``forward.mm`` and ``forward.gather`` (module attributes the forward looks up at each
    call): while a profiled window is recorded they mark its segment boundaries, else they pass straight through."""

    global _installed
    if _installed:
        return
    from . import forward, prof

    timed0, mm0, gather0 = prof.timed, forward.mm, forward.gather

    def timed(name: str, *args, **kwargs):
        p = ACTIVE
        kind = p.kind_of(name) if p is not None and p.kinds is not None else None
        if kind is None:
            return timed0(name, *args, **kwargs)
        stack = ExitStack()
        stack.enter_context(p.seg(kind))
        stack.enter_context(timed0(name, *args, **kwargs))
        return stack

    def mm(b, x, q, xs, out, f32: bool = False):
        p = ACTIVE
        if p is None or p.kinds is None or p.cur != "glue":
            return mm0(b, x, q, xs, out, f32)
        logits = getattr(b, "logits", None)
        kind = "head" if logits is not None and out.data_ptr() == logits.data_ptr() else (
            "dense" if p.detail == "kinds" else None)
        if kind is None:
            return mm0(b, x, q, xs, out, f32)
        with p.seg(kind):
            return mm0(b, x, q, xs, out, f32)

    def gather(w, b, R: int):
        p = ACTIVE
        if p is None or p.kinds is None or p.detail != "kinds":
            return gather0(w, b, R)
        with p.seg("exchange"):
            return gather0(w, b, R)

    prof.timed, forward.mm, forward.gather = timed, mm, gather
    _installed = True
