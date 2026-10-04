"""TF_GLM_PREFILL_LANES=2 (patch 0123): a prompt chunk's rows in two lanes, interleaved block by block, so each lane's
row-split exchange and glue run while the other lane computes.

Without lanes every one of a chunk's 90 exchange sites waits for its own exchange: the row split (``hcsplit``) hides a
site's exchange only behind the little row-independent work around it (the own rows' partial, the glue, the next
block's front on own rows). With two lanes, lane A (rows 0 .. RA) and lane B (rows RA .. R) take turns on the main
stream: A's attention block, B's attention block, A's MLP / MoE block, B's MLP / MoE block, and so on; each lane's
site (its partials' exchange, hc_post, the taps and stream mean, the next hc_pre, the rows' exchange) runs on the
exchange stream while the other lane's block runs. The exchange then costs only what the other lane's compute does not
cover (with TF_GLM_HC_EXCHANGE=ce the copies take no SMs from it).

Exact: each lane is a chunk of its own rows from its own position (RA a multiple of the 64-row prompt grid), run
through the calls a chunk makes (``forward._mixer`` and ``_ffn`` on the lane's rows, the row split's glue kernels and
exchanges on the lane's rows). At every layer lane A's block runs before lane B's on the main stream, so lane B reads
lane A's KDA state, conv window, latents, index keys and pooled keys of that layer as the next chunk would; nothing of
a lane depends on the other lane's later layers (A's rows come first; a layer's attention reads only that layer's
caches). TensorFold gives a prompt row the same bits in any chunking on the prompt grid (a resumed prompt equals a
fresh one), so every row gets the unsplit chunk's bits; the glue's hc_post / hc_pre / stream means are
row-independent kernels, run on the exchange stream instead of the main one (events order the two).

When: chunks that run split (TF_GLM_HC_SPLIT=1), of one prompt (multi-prompt chunks run as before), with every row
past the dense limit (a prompt's first 2,051 tokens run as before), two lanes of at least TF_GLM_LANE_MIN_ROWS rows
(default 512) that each split into whole shares a rank. Pair it with 4,096-row chunks (TF_GLM_PREFILL_ROWS): each
lane then has the rows of a 2,048-row chunk and the index split (TF_GLM_INDEX_SPLIT) still gives every rank a block.

TF_GLM_KDA_OVERLAP (patch 0128, off by default): a KDA layer's chunked recurrence runs one block a head (16 blocks a
rank at TP4) through the lane's 32-row sub-chunks in order, so the GPU is mostly idle while it runs, and lane B's
recurrence waits for lane A's (it starts from the state lane A leaves). With the setting, the recurrences run on a
stream of their own: lane A's beside lane B's input projections, lane B's beside lane A's output projection (1), and
with 2 also lane B's output projection follows on that stream while the main stream runs lane A's MLP / MoE block.
The same calls on the same rows and states in the same order on every buffer (``kda_lanes``): the same bits.

TF_GLM_LANE_INPUTS (patch 0183, ``lane_inputs``): at the sites after an output projection that every rank holds whole,
a lane swaps the projection's bf16 inputs instead of its fp32 partials; the current stream computes only this rank's
rows' partial and the exchange stream every peer's (``LaneSplit._inputs``): the same partials, the same bits."""

from __future__ import annotations

import copy
import math
import os
from types import SimpleNamespace

import torch

from . import glue, moe_glue, prof
from .hcsplit import HcSplit, SplitSettings

# Buffers attributes whose first axis is the window's rows: a lane views its rows of each
ROWS = ("ids", "ids_host", "hin", "x", "normed", "xs", "post", "comb", "hcpart", "ka", "kg", "xs_fa", "xs_ga", "kxs",
        "dp", "qr", "xs_qr", "lat", "xs_lat", "q", "kn", "vn", "xs_ao", "ikr", "igr", "qi", "gu", "act", "xs_act",
        "mlog", "pick", "wts", "ey", "sgu", "sact", "sxs", "sy", "part", "partb", "hidden", "fnormed", "fxs", "me",
        "mcat", "mxs", "mx")
# handled one by one below
SPECIAL = ("eact", "kproj", "taps", "kscratch", "lat_s", "gath", "split", "rows", "alloc_rows", "exl3")


def lane_count(value: str | None = None) -> int:
    value = (os.environ.get("TF_GLM_PREFILL_LANES", "") if value is None else value).strip() or "1"
    if value not in ("1", "2"):
        raise ValueError(f"TF_GLM_PREFILL_LANES: 1 (off) or 2, not {value!r}")
    return int(value)


def lane_min_rows(value: str | None = None) -> int:
    value = (os.environ.get("TF_GLM_LANE_MIN_ROWS", "") if value is None else value).strip() or "512"
    if not value.isdecimal() or not 64 <= int(value) <= 8192:
        raise ValueError(f"TF_GLM_LANE_MIN_ROWS: rows from 64 to 8,192, not {value!r}")
    return int(value)


def kda_overlap(value: str | None = None) -> int:
    """TF_GLM_KDA_OVERLAP (patch 0128): 0 (off, the default); 1: a KDA layer's recurrence runs on a stream of its own,
    lane A's beside lane B's input projections and lane B's beside lane A's output projection; 2: as 1, and lane B's
    output projection follows its recurrence on that stream, beside lane A's MLP / MoE block. Same bits (scheduling
    only); used where the lanes run (TF_GLM_PREFILL_LANES=2)."""

    value = (os.environ.get("TF_GLM_KDA_OVERLAP", "") if value is None else value).strip() or "0"
    if value not in ("0", "1", "2"):
        raise ValueError(f"TF_GLM_KDA_OVERLAP: 0 (off), 1 or 2, not {value!r}")
    return int(value)


LANES = lane_count()
MIN_ROWS = lane_min_rows()
KDA_OVERLAP = kda_overlap()
GRID = 64                       # lanes start on a multiple of this and of the prompt grid (TF_GLM_PROMPT_GRID)


def code() -> list[int]:
    return [LANES, MIN_ROWS if LANES > 1 else 0, KDA_OVERLAP if LANES > 1 else 0]


def describe() -> str:
    what = (f"prompt chunks in two lanes interleaved block by block (each lane's exchange and glue beside the other "
            f"lane's compute; lanes of {MIN_ROWS}+ rows)")
    if KDA_OVERLAP:
        what += ("; KDA recurrences on a stream of their own beside the projections"
                 + (" and the other lane's MLP / MoE block" if KDA_OVERLAP == 2 else "") + " (TF_GLM_KDA_OVERLAP)")
    return what


def cut(R: int, world: int, grid: int = 0) -> int | None:
    """Lane A's rows of a chunk of R: about half, on a multiple of 64 and of the prompt grid; None when the chunk does
    not take lanes (a lane under MIN_ROWS, or a lane that does not split into whole shares a rank)."""

    unit = math.lcm(GRID, int(grid or 0) or GRID)
    ra = -(-(R // 2) // unit) * unit
    rb = R - ra
    if ra < MIN_ROWS or rb < MIN_ROWS or ra % world or rb % world:
        return None
    return ra


def applies(w, b, R: int, host_pos: int | None, sparse_np, mixer) -> bool:
    """Whether this chunk runs in lanes (the same answer on every rank: settings and the chunk's numbers only)."""

    from . import latent

    sp = b.split
    return (LANES > 1 and latent.ENABLED and b.prefill and sp is not None and mixer is None and sparse_np is None
            and host_pos is not None and host_pos >= w.cfg.dense_limit and sp.applies(R)
            and cut(R, int(w.world), w.meta.get("prompt_grid", 0)) is not None)


def rows_view(b, lo: int, hi: int):
    """Rows lo .. hi of a prompt buffer set (``forward.Buffers``) as a buffer set of its own: every row-indexed tensor
    sliced, the scratch the main stream uses one block at a time shared. A row-indexed tensor this module does not
    know stops it (so a new buffer can never be shared unsliced by mistake)."""

    n = hi - lo
    alloc = getattr(b, "alloc_rows", b.rows)
    v = SimpleNamespace()
    for name, value in vars(b).items():
        if name in ROWS:
            if value is None:                    # a row buffer a setting did not allocate (TF_GLM_LANE_PARTIALS)
                setattr(v, name, None)
                continue
            if value.shape[0] != alloc:
                raise RuntimeError(f"prefill lanes: buffer {name} has {value.shape[0]} rows, not {alloc}")
            setattr(v, name, value[lo:hi])
        elif name in SPECIAL:
            continue
        elif isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == alloc and alloc > 1:
            raise RuntimeError(f"prefill lanes: buffer {name} looks row-indexed but has no lane view")
        else:
            setattr(v, name, value)
    v.rows = v.alloc_rows = n
    slots = b.eact.shape[0] // alloc
    v.eact = b.eact[lo * slots:hi * slots]
    v.exl3 = _exl3_view(b, lo, hi, slots)
    v.kproj = b.kproj[:, lo:hi]
    v.taps = [t[lo:hi] for t in b.taps]
    s = b.kscratch
    v.kscratch = SimpleNamespace(out=s.out[lo:hi], k=s.k[lo:hi], v=s.v[lo:hi], g=s.g[lo:hi], b=s.b[lo:hi])
    lat = copy.copy(b.lat_s)
    lat.qa, lat.ol, lat.rows = b.lat_s.qa[lo:hi], b.lat_s.ol[lo:hi], n
    v.lat_s = lat
    d = b.part.shape[1]
    # the split's staging in the all-gather buffer: lane A's from its start, lane B's past every rank's whole partial
    # of lane A's rows (N x RA rows: the gather exchange's layout; the p2p blocks take N x HA = RA rows), so the two
    # never overlap; world x rows x D holds N x RA + N x RB rows
    v.gath = b.gath[int(b.world) * lo * d:]
    v.split = None
    return v


def _exl3_view(b, lo: int, hi: int, slots: int):
    """The EXL3 expert scratch a lane of rows lo .. hi uses. GLM's own kernels (``exl3_mm``) take the routed outputs
    as an argument (the lane's rows of ``ey``) and share the scratch, as every other scratch the main stream uses one
    block at a time. The universal route (``weights.exl3_route``; no plan) writes its outputs into its scratch's
    ``y``, of which ``ey`` is a view: the lane gets the lane's rows of ``y`` (its routed experts then land in its own
    rows of ``ey``, where its combine reads them), the rest of that scratch shared."""

    ex = getattr(b, "exl3", None)
    if ex is None or getattr(b, "plan", None) is not None or getattr(ex, "y", None) is None:
        return ex
    part = copy.copy(ex)
    part.y = ex.y[lo * slots:hi * slots]
    part.rows = hi - lo
    return part


class LaneSplit(HcSplit):
    """A lane's row split: one piece, its partials' exchange, glue and rows' exchange all on the lanes' exchange
    stream (``glue_async``), the main stream waiting only where the lane's next block needs its rows (``wait``)."""

    def __init__(self, w, view, parent: HcSplit, stream, ce_at: int) -> None:
        s = parent.settings
        super().__init__(w, view, SplitSettings(True, False, 1, s.min_rows, s.exchange, False))
        self.stream = stream
        self.ev_fill = [torch.cuda.Event()]
        self.ev_done = torch.cuda.Event()
        self.ev_front = torch.cuda.Event()       # TF_GLM_KDA_OVERLAP: the lane's KDA projections written (main)
        self.ev_rec = torch.cuda.Event()         # ... and its recurrence's read-outs (the recurrence stream)
        self.ev_in = torch.cuda.Event()          # TF_GLM_LANE_INPUTS: an output projection's input rows written
        self.ce = parent.ce
        self.ce_at = ce_at
        # TF_GLM_LANE_DIRECT: the peers write the lane's swapped rows that live in the arena in place (lane_inputs)
        self.direct = bool(getattr(self.ce, "home_off", None))

    def partial(self, fill) -> None:
        """The lane's whole partial on the main stream, then its exchange on the exchange stream. A fill that left
        its last rows to another stream says so in ``fill.ready`` (an event: TF_GLM_MOE_GLUE defer); the exchange
        waits for it too. An output projection that every rank holds whole exchanges its inputs instead
        (TF_GLM_LANE_INPUTS: ``_inputs``)."""

        if getattr(fill, "inputs", None) is not None:
            self._inputs(fill)
            return
        main = torch.cuda.current_stream()
        fill(0, self.R)
        self.ev_fill[0].record(main)
        ready = getattr(fill, "ready", None)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ev_fill[0])
            if ready is not None:
                self.stream.wait_event(ready)
            with prof.timed("lane: partials exchange", events_only=True):
                self._swap_partial(0)

    def _inputs(self, fill) -> None:
        """TF_GLM_LANE_INPUTS (``lane_inputs``): an output projection's partials from its inputs. The current stream
        records that the projection's input rows are written (``ev_in``) and computes this rank's rows' partial with
        its own slice (``fill`` on those rows: b.part, as before); the exchange stream, from that record on, swaps the
        input rows (``_swap_inputs``), computes every peer's partial of this rank's rows with that peer's slice into
        the peer's slot of the staging block (``_slot_partials``: where the peer's copy landed before), then waits for
        the own rows' partial; ``glue_async`` sums the slots rank 0 first as before. The input rows stay unwritten
        until the lane's next block, which waits for the lane's glue (``wait``), which follows the swap. For a fill
        with inputs (``forward.out_proj`` sets them: ``fill.inputs``, ``fill.ranks``)."""

        x, ranks = fill.inputs, fill.ranks
        if self.ce is None or self.ce.inp_row < x.shape[1] * 2:
            raise RuntimeError("TF_GLM_LANE_INPUTS: the copy-engine arena has no input staging for this projection")
        cur = torch.cuda.current_stream()
        lo = self.mine()
        self.ev_in.record(cur)
        fill(lo, lo + self.H)
        self.ev_fill[0].record(cur)
        ready = getattr(fill, "ready", None)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ev_in)
            self._swap_inputs(x)
            self._slot_partials(x, ranks)
            self.stream.wait_event(self.ev_fill[0])
            if ready is not None:
                self.stream.wait_event(ready)

    def _swap_inputs(self, x) -> None:
        """The lane's input rows of a projection: this rank's rows of each peer out, the peers' rows of this rank's
        rows in (one copy-engine step)."""

        self.ce.inputs(x, self.H, 0, self.H, at=self.ce_at)

    def _slot_partials(self, x, ranks) -> None:
        """Each peer's fp32 partial of this rank's rows: its staged input rows times its slice of the weight (the
        prompt matmul, as the peer's own ``fill`` runs it), into its slot of the staging block."""

        from . import lane_partials
        from .forward import mm

        block = self.got[0]
        k = x.shape[1]
        bf16 = lane_partials.on()
        for p in self.peers:
            mm(self.b, self.ce.staged(p, k, self.ce_at, self.H), ranks[p], None, block[p], f32=True)
            if bf16:                     # TF_GLM_LANE_PARTIALS=bf16: the peer's partial rounded as the peer would
                lane_partials.round_(block[p], self.b.partb[p * self.H:(p + 1) * self.H])

    def glue_async(self, hc=None, norm=None, taps: tuple[int, ...] = (), final: bool = False, moe=None) -> None:
        """``HcSplit.glue`` of the lane's own rows (one piece), all on the exchange stream after its partials'
        exchange: hc_post, the taps' and final stream means, the next hc_pre, the rows' exchange. ``moe``: the layer
        whose MoE reads these rows (TF_GLM_MOE_GLUE rowsplit routes them here and swaps their picks and weights with
        them; ``moe_glue.lane_route``)."""

        b, c = self.b, self.w.cfg
        lo, hi = self.mine(), self.mine() + self.H
        outs = [b.taps[slot] for slot in taps] + ([b.hidden] if final else []) + ([b.normed] if hc is not None else [])
        with torch.cuda.stream(self.stream), prof.timed("lane: glue + rows exchange", events_only=True):
            self._post(0, lo, hi)
            x = b.x[lo:hi]
            for slot in taps:
                glue.stream_mean(x, b.taps[slot][lo:hi])
            if final:
                glue.stream_mean(x, b.hidden[lo:hi])
            if hc is not None:
                glue.hc_pre(x, hc.fn, hc.base, hc.scale, norm, b.normed[lo:hi], b.xs[lo:hi], b.post[lo:hi],
                            b.comb[lo:hi], b.hcpart[lo:hi], c.eps, c.hc_eps, c.hc_iters, prompt=True)
            narrow = moe_glue.lane_route(self, moe, lo, hi) if moe is not None and hc is not None else []
            if narrow:
                self._swap_rows(0, outs, narrow)
            else:
                self._swap_rows(0, outs)
            self.ev_done.record(self.stream)

    def wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.ev_done)


def _lanes(w, b, R: int):
    """The chunk's two lanes (row views, splits), cached on the prompt buffers by their row ranges."""

    sp = b.split
    ra = cut(R, int(w.world), w.meta.get("prompt_grid", 0))
    cache = getattr(sp, "_lanes", None)
    if cache is None:
        cache = sp._lanes = {}
        sp._lane_stream = sp.stream if sp.stream is not None else torch.cuda.Stream()
    key = (ra, R)
    if key not in cache:
        made = []
        for lo, hi in ((0, ra), (ra, R)):
            view = rows_view(b, lo, hi)
            ls = LaneSplit(w, view, sp, sp._lane_stream, lo // int(w.world))
            view.split = ls
            made.append((lo, hi, view, ls))
        cache[key] = made
    return cache[key], sp._lane_stream


def _rec_stream(sp):
    """The KDA recurrences' stream (TF_GLM_KDA_OVERLAP), made once a split. High priority: a recurrence keeps one
    block a head busy for the lane's whole length, so its blocks should start as soon as an SM frees up."""

    rec = getattr(sp, "_rec_stream", None)
    if rec is None:
        rec = sp._rec_stream = torch.cuda.Stream(priority=-1)
    return rec


def kda_lanes(layer, w, run, main, rec, mode: int) -> None:
    """A KDA layer's block for both lanes with its recurrences on ``rec`` (TF_GLM_KDA_OVERLAP = ``mode``).

    Main stream: lane A's input projections, lane B's, then each lane's output projection (mode 1), or lane A's only
    (mode 2: lane B's runs on ``rec`` after its recurrence); each output projection's partials and glue then go to the
    exchange stream as ``_mixer`` leaves them. Recurrence stream: lane A's recurrence after its projections, then lane
    B's after its own (lane B starts from the state and conv window lane A's leaves, as in ``lane_layers``).

    Same bits: the calls of ``forward.kda_block`` on the same rows and states in the same order on each buffer; only
    the streams differ, and events order every buffer one call writes and another reads. The projections write the
    lane's kproj / ka / kg / xs rows, which its recurrence reads (ev_front); its recurrence writes the lane's read-outs
    (kscratch rows), which its output projection reads (ev_rec in mode 1, stream order in mode 2), and the layer's KDA
    state and conv window, which lane B's recurrence reads next (stream order). The lane's next projections, of a later
    layer, come after its output projection on the main stream (mode 1) or after the main stream waited for its glue,
    which waited for that projection's partials (mode 2), so they never overwrite rows a recurrence still reads."""

    from .forward import kda_front, kda_rows, out_proj

    outs = []
    for view, ls, lst, n, nch in run:
        ls.wait()                                  # the lane's normed rows (its last glue)
        with prof.timed("kda"):                    # (TF_GLM_PROFILE's timers sync: they serialize the streams)
            with prof.timed("kda: projections"):
                kda_front(layer, view, 0, n)
            ls.ev_front.record(main)
            rec.wait_event(ls.ev_front)
            with torch.cuda.stream(rec):
                outs.append(kda_rows(layer, w, lst, view, 0, n))
                ls.ev_rec.record(rec)
    first = run[0][1]
    for (view, ls, lst, n, nch), out in zip(run, outs):
        on_rec = mode == 2 and ls is not first
        if not on_rec:
            main.wait_event(ls.ev_rec)
        with torch.cuda.stream(rec if on_rec else main):
            with prof.timed("kda"), prof.timed("kda: out + all-gather"):
                out_proj(w, view, out, layer.kda.o, None, n, site=(layer.index, "a"))
            ls.glue_async(layer.ffn_hc, layer.post_norm, **moe_glue.glue_args(layer))
        moe_glue.lane_front(layer, w, view, ls, n)   # TF_GLM_MOE_GLUE side: the FFN front once the rows are in


def _lane_state(st, lo: int):
    """The chunk's state seen from row ``lo`` (lane B): the same caches, KDA states, conv windows and parity list,
    its first row's position on the host and on the device."""

    if lo == 0:
        return st
    v = copy.copy(st)
    v.pos = st.pos + lo
    v.pos_dev = st.pos_dev + lo
    return v


def lane_layers(w, st, b, R: int, host_pos: int) -> None:
    """Every layer of a prompt chunk in two lanes (``applies``); leaves b.normed's, the taps' and b.hidden's rows
    as ``forward.split_layers`` does."""

    from .forward import _ffn, _mixer, chunks_for

    c = w.cfg
    if st.pos != host_pos:
        raise RuntimeError(f"prefill lanes: the state is at {st.pos}, the chunk at {host_pos}")
    lanes, stream = _lanes(w, b, R)
    main = torch.cuda.current_stream()
    run = []
    for lo, hi, view, ls in lanes:
        ls.begin(hi - lo)
        lst = _lane_state(st, lo)
        run.append((view, ls, lst, hi - lo, chunks_for(lst, hi - lo)))
    rec = _rec_stream(b.split) if KDA_OVERLAP else None
    stream.wait_stream(main)
    if rec is not None:
        rec.wait_stream(main)                 # the states and conv windows as the main stream left them
    try:
        layers = w.layers
        first, h = layers[0], layers[0].attn_hc
        glue.hc_pre(b.x[:R], h.fn, h.base, h.scale, first.in_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                    b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters, prompt=True)
        for i, layer in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            if rec is not None and layer.kind == "kda":
                kda_lanes(layer, w, run, main, rec, KDA_OVERLAP)
            else:
                for view, ls, lst, n, nch in run:     # attention blocks: lane A's, then lane B's (reads A's caches)
                    ls.wait()
                    _mixer(layer, w, lst, view, n, nch, lst.pos, None, None)
                    ls.glue_async(layer.ffn_hc, layer.post_norm, **moe_glue.glue_args(layer))
                    moe_glue.lane_front(layer, w, view, ls, n)    # TF_GLM_MOE_GLUE side (moe_glue.py)
            for view, ls, lst, n, nch in run:     # MLP / MoE blocks
                ls.wait()
                _ffn(layer, w, view, n, None)
                ls.glue_async(nxt.attn_hc if nxt is not None else None, nxt.in_norm if nxt is not None else None,
                              taps=tuple(b.tap_at.get(layer.index, ())), final=nxt is None)
        for view, ls, lst, n, nch in run:
            ls.wait()
    finally:
        main.wait_stream(stream)
        if rec is not None:
            main.wait_stream(rec)
        for view, ls, lst, n, nch in run:
            ls.active = False
        ce = getattr(b.split, "ce", None)
        if ce is not None:                       # TF_GLM_CE_COUNT=1: the chunk's copy-engine bytes
            ce.report(f"lane chunk of {R} rows")
