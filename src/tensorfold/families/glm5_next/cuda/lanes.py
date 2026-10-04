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
lane then has the rows of a 2,048-row chunk and the index split (TF_GLM_INDEX_SPLIT) still gives every rank a block."""

from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import torch

from . import glue
from .hcsplit import HcSplit, SplitSettings

# Buffers attributes whose first axis is the window's rows: a lane views its rows of each
ROWS = ("ids", "ids_host", "hin", "x", "normed", "xs", "post", "comb", "hcpart", "ka", "kg", "xs_fa", "xs_ga", "kxs",
        "dp", "qr", "xs_qr", "lat", "xs_lat", "q", "kn", "vn", "xs_ao", "ikr", "igr", "qi", "gu", "act", "xs_act",
        "mlog", "pick", "wts", "ey", "sgu", "sact", "sxs", "sy", "part", "hidden", "fnormed", "fxs", "me", "mcat",
        "mxs", "mx")
# handled one by one below
SPECIAL = ("eact", "kproj", "taps", "kscratch", "lat_s", "gath", "split", "rows", "alloc_rows")


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


LANES = lane_count()
MIN_ROWS = lane_min_rows()
GRID = 64                       # lanes start on the prompt grid (TF_GLM_PROMPT_GRID's 64; a multiple of 32 and 16)


def code() -> list[int]:
    return [LANES, MIN_ROWS if LANES > 1 else 0]


def describe() -> str:
    return (f"prompt chunks in two lanes interleaved block by block (each lane's exchange and glue beside the other "
            f"lane's compute; lanes of {MIN_ROWS}+ rows)")


def cut(R: int, world: int) -> int | None:
    """Lane A's rows of a chunk of R: about half, on the 64-row grid; None when the chunk does not take lanes (a lane
    under MIN_ROWS, or a lane that does not split into whole shares a rank)."""

    ra = -(-(R // 2) // GRID) * GRID
    rb = R - ra
    if ra < MIN_ROWS or rb < MIN_ROWS or ra % world or rb % world:
        return None
    return ra


def applies(w, b, R: int, host_pos: int | None, sparse_np, mixer) -> bool:
    """Whether this chunk runs in lanes (the same answer on every rank: settings and the chunk's numbers only)."""

    sp = b.split
    return (LANES > 1 and b.prefill and sp is not None and mixer is None and sparse_np is None and host_pos is not None
            and host_pos >= w.cfg.dense_limit and sp.applies(R) and cut(R, int(w.world)) is not None)


def rows_view(b, lo: int, hi: int):
    """Rows lo .. hi of a prompt buffer set (``forward.Buffers``) as a buffer set of its own: every row-indexed tensor
    sliced, the scratch the main stream uses one block at a time shared. A row-indexed tensor this module does not
    know stops it (so a new buffer can never be shared unsliced by mistake)."""

    n = hi - lo
    alloc = getattr(b, "alloc_rows", b.rows)
    v = SimpleNamespace()
    for name, value in vars(b).items():
        if name in ROWS:
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
    v.kproj = b.kproj[:, lo:hi]
    v.taps = [t[lo:hi] for t in b.taps]
    s = b.kscratch
    v.kscratch = SimpleNamespace(out=s.out[lo:hi], k=s.k[lo:hi], v=s.v[lo:hi], g=s.g[lo:hi], b=s.b[lo:hi])
    lat = copy.copy(b.lat_s)
    lat.qa, lat.ol, lat.rows = b.lat_s.qa[lo:hi], b.lat_s.ol[lo:hi], n
    v.lat_s = lat
    d = b.part.shape[1]
    v.gath = b.gath[lo * d:]          # the split's staging blocks: lane A's N x HA rows first, then lane B's
    v.split = None
    return v


class LaneSplit(HcSplit):
    """A lane's row split: one piece, its partials' exchange, glue and rows' exchange all on the lanes' exchange
    stream (``glue_async``), the main stream waiting only where the lane's next block needs its rows (``wait``)."""

    def __init__(self, w, view, parent: HcSplit, stream, ce_at: int) -> None:
        s = parent.settings
        super().__init__(w, view, SplitSettings(True, False, 1, s.min_rows, s.exchange, False))
        self.stream = stream
        self.ev_fill = [torch.cuda.Event()]
        self.ev_done = torch.cuda.Event()
        self.ce = parent.ce
        self.ce_at = ce_at

    def partial(self, fill) -> None:
        """The lane's whole partial on the main stream, then its exchange on the exchange stream."""

        main = torch.cuda.current_stream()
        fill(0, self.R)
        self.ev_fill[0].record(main)
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ev_fill[0])
            self._swap_partial(0)

    def glue_async(self, hc=None, norm=None, taps: tuple[int, ...] = (), final: bool = False) -> None:
        """``HcSplit.glue`` of the lane's own rows (one piece), all on the exchange stream after its partials'
        exchange: hc_post, the taps' and final stream means, the next hc_pre, the rows' exchange."""

        b, c = self.b, self.w.cfg
        lo, hi = self.mine(), self.mine() + self.H
        outs = [b.taps[slot] for slot in taps] + ([b.hidden] if final else []) + ([b.normed] if hc is not None else [])
        with torch.cuda.stream(self.stream):
            self._post(0, lo, hi)
            x = b.x[lo:hi]
            for slot in taps:
                glue.stream_mean(x, b.taps[slot][lo:hi])
            if final:
                glue.stream_mean(x, b.hidden[lo:hi])
            if hc is not None:
                glue.hc_pre(x, hc.fn, hc.base, hc.scale, norm, b.normed[lo:hi], b.xs[lo:hi], b.post[lo:hi],
                            b.comb[lo:hi], b.hcpart[lo:hi], c.eps, c.hc_eps, c.hc_iters, prompt=True)
            self._swap_rows(0, outs)
            self.ev_done.record(self.stream)

    def wait(self) -> None:
        torch.cuda.current_stream().wait_event(self.ev_done)


def _lanes(w, b, R: int):
    """The chunk's two lanes (row views, splits), cached on the prompt buffers by their row ranges."""

    sp = b.split
    ra = cut(R, int(w.world))
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
    stream.wait_stream(main)
    try:
        layers = w.layers
        first, h = layers[0], layers[0].attn_hc
        glue.hc_pre(b.x[:R], h.fn, h.base, h.scale, first.in_norm, b.normed[:R], b.xs[:R], b.post[:R], b.comb[:R],
                    b.hcpart[:R], c.eps, c.hc_eps, c.hc_iters, prompt=True)
        for i, layer in enumerate(layers):
            nxt = layers[i + 1] if i + 1 < len(layers) else None
            for view, ls, lst, n, nch in run:     # attention blocks: lane A's, then lane B's (it reads A's caches)
                ls.wait()
                _mixer(layer, w, lst, view, n, nch, lst.pos, None, None)
                ls.glue_async(layer.ffn_hc, layer.post_norm)
            for view, ls, lst, n, nch in run:     # MLP / MoE blocks
                ls.wait()
                _ffn(layer, w, view, n, None)
                ls.glue_async(nxt.attn_hc if nxt is not None else None, nxt.in_norm if nxt is not None else None,
                              taps=tuple(b.tap_at.get(layer.index, ())), final=nxt is None)
        for view, ls, lst, n, nch in run:
            ls.wait()
    finally:
        main.wait_stream(stream)
        for view, ls, lst, n, nch in run:
            ls.active = False
