"""TF_GLM_MOE_GLUE: the MoE layer's work around the routed experts (router, top-k, plan, shared expert, combine) in
fewer kernels and off the critical path, with the bits of today's kernels.

A MoE layer runs, for a window or a prompt chunk of R rows: the router matmul (8 K slices), their sum, the sigmoid /
bias top-k with its normalization, the plan (the (row, slot) pairs grouped by expert), the routed experts, the shared
expert (gate/up, SwiGLU, down) and the combine (the routed slots' weighted sum, the shared expert's row last, a rank's
fp32 partial). The parts, each on its own in the setting:

  side     the shared expert on a second stream beside the route and the routed experts (it reads only the layer's
           normed rows). Prompt chunks in two lanes (TF_GLM_PREFILL_LANES=2) start it as soon as the lane's normed rows
           exist (right after the lane's glue, beside the other lane's attention block); decode windows fork it inside
           their CUDA graphs. The same kernels on the same rows: scheduling only.
  defer    (with side, prompt lanes) the combine on that stream too, so the main stream goes on with the other lane's
           route and routed experts while it runs; the lane's exchange waits for it. Scheduling only.
  route    one kernel for the router matmul, the slice sums, the top-k, the weights and the plan (moe_glue.cu):
           6 launches of a prompt chunk (router, sum, top-k, three plan passes) and 3 of a decode window (fused
           router + top-k, plan) become 1, with no host round trip.
  combine  the combine as a CUDA kernel (4 columns a thread, every slot's load in flight at once, streaming loads).
  shared   a prompt chunk's shared-expert gate/up matmul with SwiGLU in its epilogue (one launch, no gate/up round trip).
  rowsplit (prompt lanes on the copy-engine exchange, TF_GLM_HC_EXCHANGE=ce) each rank routes only its own quarter of
           a lane's rows, on the exchange stream right after their glue (the fused kernel's route-only mode, or the
           replaced kernels on those rows), and their picks and weights (72 bytes a row) travel with the normed rows in
           the same copy-engine step; the main stream then builds the plan alone (the plan-only mode). Every rank
           decides it from its settings alone (the swap must carry the same buffers everywhere).

Value: unset, "0", "off" or "none": none of it (the default: today's code path, unchanged); "all", "1" or "on":
every part; or a comma-separated list of names. TF_GLM_MOE_SHARED_TILE (0-3, default 0) picks the shared gate/up
kernel's tile (0: 128 rows x 64 columns, 8 warps; 1: 64 x 64, 4 warps; 2: 128 x 64, 8 warps along M; 3: 64 x 128);
TF_GLM_MOE_SIDE_PRIORITY (0, the default, or -1: high) the second stream's priority. Every rank must be given the same
values (the engine compares them at start).

Exactness. side and defer change no kernel: each runs on the same rows with the same inputs, events ordering every
buffer a call writes and another reads (tests/k4/test_side_streams_cpu.py proves it on the real lane schedule). route,
combine and shared reproduce the replaced kernels' arithmetic instruction for instruction: every logit is
glue._router_part's chain of mma.sync m16n8k16 steps over its K slice in ascending k, the slices added in order; the
selection is glue._topk's (sub.f32, mul.f32 by log2(e), ex2.approx.f32, add.f32, div.full.f32; the largest choice,
the lowest index among equal ones, NaN never; weights in pick order over their sum + 1e-20, then times the scale);
the plan is experts.cu's (an expert's pairs in pair order, items of T pairs in expert order); the combine is
glue._combine's fma.rn.f32 chain in slot order; the shared gate/up is the prompt matmul's chain (bf16(fma(q, s, b))
weights, one fp32 chain over K), rounded to bf16 as it stores them, then glue._swiglu's operations
(tests/k4/ptx_contract.py reads those instructions from the Triton kernels' sm_120 PTX). On top, each kernel is
compared byte for byte with the kernels it replaces on this GPU at its first use outside a CUDA graph capture (random
rows over many magnitudes, ties, the layer's own weights); any difference turns that part off for the process (a line
says so) and the replaced kernels run. Every bit of every reply is therefore today's.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (the Triton glue kernels, the grouped-expert
plan and the 4-bit prompt matmul, whose arithmetic it reproduces) and on the GLM-5.3-Flash recipe and patches
0001-0056 by MiaAI-Lab."""

from __future__ import annotations

import os
from functools import lru_cache

import torch

from . import prof

PARTS = ("side", "defer", "route", "combine", "shared", "rowsplit")
CHECKED = ("route", "combine", "shared")          # kernels compared bit for bit on this GPU before their first use
OFF = ("", "0", "off", "none")
TOPK = 8                                          # the fused route's experts a token (GLM-5.3-Flash's)
MAX_ROWS = 10240                                  # the fused route's rows a call at most (its plan table's shared memory)

_wanted: frozenset | None = None
_decided: dict[str, bool] = {}
# The EXL3 expert route this process serves (weights.exl3_route, set by the loader): the glue is GLM's own route's
# (its fused route builds GLM's plan and its MoE block calls GLM's kernels, exl3_mm); on the universal route
# (0.6.1's any-width path: no plan) TF_GLM_MOE_GLUE asks for nothing and the MoE blocks run as without it.
_route: str | None = None


def parse(value: str | None) -> frozenset:
    """TF_GLM_MOE_GLUE's value as the set of parts it asks for (ValueError on an unknown name)."""

    v = (value or "").strip().lower()
    if v in OFF:
        return frozenset()
    if v in ("all", "1", "on"):
        return frozenset(PARTS)
    names = {n.strip() for n in v.split(",") if n.strip()}
    bad = names - set(PARTS)
    if bad:
        raise ValueError(f"TF_GLM_MOE_GLUE: off, all or names among {', '.join(PARTS)}; not {', '.join(sorted(bad))}")
    if "defer" in names and "side" not in names:
        raise ValueError("TF_GLM_MOE_GLUE: defer runs the combine on side's stream: name side as well")
    return frozenset(names)


def rowsplit_on() -> bool:
    """rowsplit asked for and usable: prompt lanes on the copy-engine exchange (settings only, the same on every
    rank). Read where the arena is made (``narrow_bytes``) and where a lane's glue runs (``lane_route``)."""

    if not asks("rowsplit"):
        return False
    env = os.environ
    return ((env.get("TF_GLM_HC_EXCHANGE", "") or "p2p").strip() == "ce"
            and (env.get("TF_GLM_PREFILL_LANES", "") or "1").strip() == "2")


def asked() -> frozenset:
    """TF_GLM_MOE_GLUE's parts as set (ValueError when not valid), whatever the route."""

    global _wanted
    if _wanted is None:
        _wanted = parse(os.environ.get("TF_GLM_MOE_GLUE", ""))
    return _wanted


def wanted() -> frozenset:
    """The parts that run: those asked for, none on the universal EXL3 route (``exl3_route``)."""

    parts = asked()
    return frozenset() if _route == "universal" else parts


def exl3_route(route: str) -> None:
    """The loader's EXL3 expert route (``glm``, ``universal`` or "" for other formats), before any buffer is made."""

    global _route
    _route = route or None


def not_used() -> str | None:
    """A line for the start when TF_GLM_MOE_GLUE asks for parts the universal route does not run, else None."""

    if _route == "universal" and asked():
        return ("TF_GLM_MOE_GLUE: not used on the universal EXL3 route (GLM's own EXL3 kernels only); the MoE blocks "
                "run as without it")
    return None


def reset(value: str | None = None) -> None:
    """Forget the setting and the kernel decisions (tests); ``value``: set TF_GLM_MOE_GLUE first."""

    global _wanted, _side_stream, _route
    if value is not None:
        os.environ["TF_GLM_MOE_GLUE"] = value
    _wanted = None
    _route = None
    _side_stream = None
    _decided.clear()
    _states.clear()


def on() -> bool:
    return bool(wanted())


def asks(name: str) -> bool:
    return name in wanted()


def shared_tile() -> int:
    v = (os.environ.get("TF_GLM_MOE_SHARED_TILE", "") or "0").strip()
    if v not in ("0", "1", "2", "3"):
        raise ValueError(f"TF_GLM_MOE_SHARED_TILE: 0, 1, 2 or 3, not {v!r}")
    return int(v)


def side_priority() -> int:
    v = (os.environ.get("TF_GLM_MOE_SIDE_PRIORITY", "") or "0").strip()
    if v not in ("0", "-1"):
        raise ValueError(f"TF_GLM_MOE_SIDE_PRIORITY: 0 or -1, not {v!r}")
    return int(v)


def narrow_bytes() -> int:
    """Bytes a row the copy-engine arena keeps beside the swapped rows for rowsplit's picks and weights (int32 and fp32
    [rows, 9]); 0 without rowsplit (the arena as before). Settings only: every rank gets the same arena."""

    return 2 * (TOPK + 1) * 4 if rowsplit_on() else 0


def mask() -> int:
    """TF_GLM_MOE_GLUE as a bit mask over PARTS (ValueError when the value is not valid)."""

    return sum(1 << i for i, n in enumerate(PARTS) if n in wanted())


def code() -> list[int]:
    """The settings as integers (the engine compares mask, shared_tile and side_priority across ranks: every rank
    must run the same calls)."""

    return [mask(), shared_tile(), side_priority()]


def describe() -> str:
    parts = [n for n in PARTS if n in wanted()]
    what = {"side": "the shared expert on a second stream (prompt lanes: from the lane's glue on)",
            "defer": "prompt lanes' combine on that stream",
            "route": "router, top-k and plan in one kernel",
            "combine": "a CUDA combine",
            "shared": f"the shared expert's gate/up with SwiGLU in one prompt kernel (tile {shared_tile()})",
            "rowsplit": "prompt lanes' routes split between the ranks (own rows on the exchange stream, the picks "
                        "swapped with the rows)" + ("" if rowsplit_on() else
                                                   " [not used: it needs TF_GLM_HC_EXCHANGE=ce and "
                                                   "TF_GLM_PREFILL_LANES=2]")}
    return "MoE glue (TF_GLM_MOE_GLUE): " + "; ".join(what[n] for n in parts) + \
        (" (kernels checked bit for bit at first use)" if set(parts) & set(CHECKED) else "")


@lru_cache(maxsize=1)
def _ext():
    from pathlib import Path

    from tensorfold.cuda.build import CLUSTERS, load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_moe_glue_v1", sources=[str(here / "moe_glue.cpp"), str(here / "moe_glue.cu")],
                need=CLUSTERS, extra_cuda_cflags=["-O3"], verbose=False)


def kernel_on(name: str, check) -> bool:
    """Whether to run kernel part ``name`` now: asked for and its check passed (run here, on its first call outside a
    CUDA graph capture; inside a capture before it: False, the replaced kernels run). Per process."""

    if name not in wanted():
        return False
    got = _decided.get(name)
    if got is not None:
        return got
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    try:
        ok = bool(check())
        why = "" if ok else ": its outputs differ from the replaced kernels' on this GPU"
    except Exception as exc:  # noqa: BLE001 - a part that cannot run here is off, the replaced kernels run
        ok, why = False, f": {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
    _decided[name] = ok
    print(f"[tensorfold] MoE glue {name}: {'on (checked bit for bit)' if ok else 'off' + why}", flush=True)
    return ok


# -- per buffer set state -------------------------------------------------------------------------------------------
class State:
    """One buffer set's (``forward.Buffers`` or a prompt lane's view) second stream, events and route scratch."""

    def __init__(self, b) -> None:
        self.b = b                                   # held: the key is id(b)
        self.stream = None
        self.ev_fork = self.ev_side = self.ev_routed = self.ev_part = self.ev_early = None
        self.early: dict[int, object] = {}           # layer index -> the event after its front ran on the stream
        self.split_routed: dict[int, bool] = {}      # layer index -> its rows' picks came from the rows' owners
        self.rank = self.hist = self.tick = None

    def side(self):
        """The second stream (one for the process: prompt chunks and decode windows run one at a time on the main
        stream, so their second-stream work is in host order there too) and this buffer set's events."""

        global _side_stream
        if self.stream is None:
            if _side_stream is None:
                _side_stream = torch.cuda.Stream(priority=side_priority())
            self.stream = _side_stream
            self.ev_fork, self.ev_side = torch.cuda.Event(), torch.cuda.Event()
            self.ev_routed, self.ev_part, self.ev_early = torch.cuda.Event(), torch.cuda.Event(), torch.cuda.Event()
        return self.stream

    def scratch(self, R: int, slots: int, experts: int, device):
        """The fused route's scratch, sized once for every window this buffer set holds (its rows with any pad rows):
        never reallocated, since captured CUDA graphs keep its addresses. The ticket counters start at zero and the
        kernel leaves them at zero."""

        if self.rank is None:
            rows = max(int(getattr(self.b, "alloc_rows", 0) or 0), int(self.b.rows))
            blocks = int(_ext().route_blocks(rows))
            self.rank = torch.zeros((rows * slots,), dtype=torch.int32, device=device)
            self.hist = torch.zeros((blocks * experts,), dtype=torch.int32, device=device)
            self.tick = torch.zeros((blocks + 1,), dtype=torch.int32, device=device)
        if self.rank.numel() < R * slots:
            raise ValueError(f"moe_glue: a window of {R} rows past the buffers' {self.rank.numel() // slots}")
        return self.rank, self.hist, self.tick


_states: dict[int, State] = {}
_side_stream = None


def state(b) -> State:
    """``b``'s state, made on first use (a dict by identity: a prompt lane's view copies its parent's attributes,
    so nothing of this lives on the buffer object itself)."""

    s = _states.get(id(b))
    if s is None or s.b is not b:
        s = _states[id(b)] = State(b)
    return s


# -- route ----------------------------------------------------------------------------------------------------------
def _block(experts: int) -> int:
    """glue.select's BLOCK: the next power of two above the routed experts (the shared slot's id among them)."""

    n = 1
    while n < experts + 1:
        n *= 2
    return n


def route_ok(x: torch.Tensor, router: torch.Tensor, b, c) -> bool:
    """Whether the fused route takes this call: glue.router's sliced path (8 K slices: the bits it reproduces),
    8 experts a token, at most 511 routed experts, unit-stride 16-byte rows."""

    from . import glue

    return (glue.ROUTER_KS == 8 and c.top_k == TOPK and b.plan.slots == TOPK + 1 and x.dim() == 2
            and x.shape[1] % 512 == 0 and x.stride(1) == 1 and (x.shape[0] <= 1 or x.stride(0) % 8 == 0)
            and x.data_ptr() % 16 == 0 and router.is_contiguous() and router.dtype == torch.bfloat16
            and router.data_ptr() % 16 == 0 and router.shape == (c.experts, x.shape[1]) and _block(c.experts) <= 512
            and b.plan.experts == c.experts + 1 and b.mlog.is_contiguous() and x.shape[0] <= b.plan.rows
            and x.shape[0] <= MAX_ROWS)


def _route_fits(rows: int, experts: int) -> bool:
    """The fused route's shared memory for ``rows`` rows fits this GPU (else the replaced kernels run). Asked after
    the part's check passed (the extension is loaded then); never raises."""

    try:
        most = _smem_optin()
        return most > 0 and int(_ext().route_bytes(rows, experts)) <= most
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def _smem_optin() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(getattr(props, "shared_memory_per_block_optin", 0) or 0)


def route(b, layer, c, R: int) -> None:
    """The layer's route of rows 0 .. R: logits, picks and weights into b.mlog / b.pick / b.wts and the plan into
    b.plan (with its item size: a prompt plan's own, a decode plan's experts.TILE)."""

    from tensorfold.cuda import experts as grouped

    from . import forward as F
    from . import glue

    m = layer.moe
    x = b.normed[:R]
    tile = b.plan.tile if b.plan.prefill else grouped.TILE
    if state(b).split_routed.pop(layer.index, False):
        # rowsplit: every row's picks and weights are in (this rank's own rows routed by ``lane_route``, the others
        # swapped in with the normed rows): the plan alone
        if asks("route") and route_ok(x, m.router, b, c) and kernel_on(
                "route", lambda: check_route(m.router, m.bias, c, b.plan.tile, x.device)) and _route_fits(
                R, c.experts + 1):
            rank, hist, tick = state(b).scratch(R, b.plan.slots, c.experts + 1, x.device)
            b.plan.tile = tile
            _ext().plan(b.pick[:R], b.plan.members, b.plan.items, b.plan.counts, rank, hist, tick, c.experts + 1,
                        tile)
        else:
            grouped.route(b.pick[:R], b.plan, b.plan.tile)
        return
    if (asks("route") and route_ok(x, m.router, b, c)
            and kernel_on("route", lambda: check_route(m.router, m.bias, c, b.plan.tile, x.device))
            and _route_fits(R, c.experts + 1)):
        rank, hist, tick = state(b).scratch(R, b.plan.slots, c.experts + 1, x.device)
        b.plan.tile = tile
        _ext().route(x, m.router, m.bias, b.mlog[:R], b.pick[:R], b.wts[:R], b.plan.members, b.plan.items,
                     b.plan.counts, rank, hist, tick, float(c.routed_scale), bool(c.norm_topk), c.experts + 1, tile,
                     _block(c.experts), True)
        return
    # the replaced kernels, as forward.moe_block calls them
    if not b.prefill and F._fused("router", glue.router_select_ok(x, m.router, b.mlog[:R]), lambda: F._fuse().check_router(
            x, m.router, m.bias, c.top_k, c.experts, c.routed_scale, c.norm_topk)):
        glue.router_select(x, m.router, b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts,
                           c.routed_scale, c.norm_topk)
    else:
        glue.router(x, m.router, b.mlog[:R])
        glue.select(b.mlog[:R], m.bias, b.pick[:R], b.wts[:R], c.top_k, c.experts, c.routed_scale, c.norm_topk)
    grouped.route(b.pick[:R], b.plan, b.plan.tile)


# -- combine --------------------------------------------------------------------------------------------------------
def combine_ok(y: torch.Tensor, sy: torch.Tensor | None, wts: torch.Tensor, out: torch.Tensor) -> bool:
    return (y.dtype == torch.float32 and y.dim() == 3 and y.shape[1] == TOPK + 1 and y.is_contiguous()
            and wts.is_contiguous() and wts.shape[1] == TOPK + 1 and out.is_contiguous() and out.shape[1] % 4 == 0
            and y.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0
            and (sy is None or (sy.dtype == torch.float32 and sy.is_contiguous() and sy.data_ptr() % 16 == 0)))


def combine_rows(b, c, lo: int, hi: int) -> None:
    """Rows lo .. hi of this rank's fp32 MoE share into b.part: the routed slots' weighted sum, the shared expert's
    row (b.sy) last with its weight (forward.moe_block's combine)."""

    from . import glue

    if hi <= lo:
        return
    with_sy = b.prefill or os.environ.get("TF_GLM_COMBINE_SY", "1") != "0"
    y, sy, wts, out = b.ey[lo:hi], b.sy[lo:hi], b.wts[lo:hi], b.part[lo:hi]
    if (asks("combine") and combine_ok(y, sy, wts, out)
            and kernel_on("combine", lambda: check_combine(c.hidden, y.device))):
        # the shared row read from sy where it is: the same terms in the same order as the copy into the last slot
        _ext().combine(y, sy, wts, out)
        return
    if with_sy:
        glue.combine_shared(y, sy, wts, out)
    else:
        b.ey[lo:hi, c.top_k].copy_(sy)
        glue.combine(y, wts, out)


# -- the shared expert ----------------------------------------------------------------------------------------------
def shared_ok(b, s) -> bool:
    """A prompt chunk's shared expert the fused gate/up kernel takes: 4-bit gate/up of groups of 64 (TF_GLM_DENSE=q4)
    on the prompt matmul's packing, an intermediate width of 128s."""

    from . import qmm

    gu = s.gu
    return (b.prefill and isinstance(gu, qmm.Q4) and gu.gs == 64 and gu.n % 256 == 0 and gu.k % 64 == 0
            and gu.scales.dim() == 2 and gu.scales.shape[0] == gu.k // 64 and gu.scales.shape[1] >= gu.n
            and all(t.data_ptr() % 16 == 0 for t in (gu.weight, gu.scales, gu.biases)))


def shared_rows(layer, w, b, lo: int, hi: int) -> None:
    """The MoE's shared expert of rows lo .. hi into b.sy (forward.shared_front, or the fused gate/up kernel)."""

    from . import forward as F

    if hi <= lo:
        return
    s, c = layer.moe.shared, w.cfg
    if (asks("shared") and shared_ok(b, s)
            and kernel_on("shared", lambda: check_shared(s.gu, c.limit, b.normed.device))):
        ni = s.gu.n // 2
        with prof.timed("moe: shared expert"):
            _ext().shared_gu(b.normed[lo:hi], s.gu.weight, s.gu.scales, s.gu.biases, b.sact[lo:hi], ni,
                             float(c.limit), shared_tile())
            # b.sxs (the act's group sums) is not written: a prompt chunk's down projection never reads it
            F.mm(b, b.sact[lo:hi], s.down, b.sxs[lo:hi], b.sy[lo:hi], f32=True)
        return
    F.shared_front(layer, w, b, lo, hi)


def side_ok(b, layer) -> bool:
    """Whether the shared expert may run beside the main stream here: a decode window (its projections keep their
    split-K partials in clusters, or in b.sk, which nothing on the main stream uses meanwhile); a prompt chunk only
    with 4-bit projections (the prompt matmul keeps no partials: BF16 / FP8 ones would share b.sk with the other
    lane's blocks)."""

    from . import qmm

    s = layer.moe.shared
    if s is None:
        return False
    if not b.prefill:
        return True
    return isinstance(s.gu, qmm.Q4) and isinstance(s.down, qmm.Q4)


def _lane_split(b) -> bool:
    sp = getattr(b, "split", None)
    return sp is not None and getattr(sp, "active", False) and type(sp).__name__ == "LaneSplit"


# -- the blocks ------------------------------------------------------------------------------------------------------
def ffn(layer, w, b, R: int, done=None):
    """forward._ffn under TF_GLM_MOE_GLUE: the dense MLP (its front perhaps run by ``lane_front``) or the MoE."""

    from . import forward as F

    early = state(b).early.pop(layer.index, None)
    if layer.mlp is not None:
        if early is not None:
            torch.cuda.current_stream().wait_event(early)
            return F.mlp_block(layer, w, b, R, (0, R))
        return F.mlp_block(layer, w, b, R, done)
    return moe_block(layer, w, b, R, done, early)


def _after_route(b, m, R: int) -> None:
    """Calls that follow the routing plan in forward.moe_block: builder K3's routed-expert L2 prefetch
    (exl3_stream.prefetch_experts, TF_GLM_L2PF_EXPERT_MB) when that module is present and the prefetcher runs."""

    from . import l2pf

    if l2pf.ACTIVE is None or getattr(b, "plan", None) is None:
        return
    k3 = _k3_stream()
    if k3 is not None:
        k3.prefetch_experts(l2pf.ACTIVE, b.plan, m.experts, R)


@lru_cache(maxsize=1)
def _k3_stream():
    """Builder K3's exl3_stream module when it is installed, else None (looked up once; a module that is there but
    fails to import raises, as forward.moe_block's import would)."""

    import importlib
    import importlib.util

    name = __name__.rsplit(".", 1)[0] + ".exl3_stream"
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


def lane_front(layer, w, b, ls, rows: int) -> None:
    """A prompt lane's FFN front (the MoE's shared expert, or the dense MLP's gate/up) on the second stream as soon as
    the lane's normed rows exist: right after its glue (``ls.glue_async``, which records ls.ev_done on the exchange
    stream once the rows are swapped in), beside the other lane's attention block. ``ffn`` then waits for it where its
    output is read. (TF_GLM_MOE_GLUE side; prompt lanes, patch 0123.)"""

    from . import forward as F

    if not asks("side") or rows <= 0:
        return
    if layer.mlp is not None:
        if not _q4_mlp(layer):
            return
        front = lambda lo, hi: F.mlp_front(layer, w, b, lo, hi)  # noqa: E731
    elif side_ok(b, layer):
        front = lambda lo, hi: shared_rows(layer, w, b, lo, hi)  # noqa: E731
    else:
        return
    st = state(b)
    stream = st.side()
    with torch.cuda.stream(stream):
        stream.wait_event(ls.ev_done)
        front(0, rows)
        st.ev_early.record(stream)
    st.early[layer.index] = st.ev_early


def glue_args(layer) -> dict:
    """lanes' FFN-side ``glue_async`` call's extra argument: the layer, for rowsplit's own-rows route ({} without
    rowsplit: the call as before)."""

    return {"moe": layer} if rowsplit_on() else {}


def lane_route(ls, layer, lo: int, hi: int) -> list:
    """rowsplit: a prompt lane's route of this rank's own rows lo .. hi, called by ``lanes.LaneSplit.glue_async`` on
    the exchange stream right after their hc_pre; returns the row buffers (picks, weights) its rows' swap carries beside
    the normed rows ([] when rowsplit does not run: the swap as before). The main stream's ``route`` then builds the
    plan alone. Decided from the settings alone (every rank's swap must carry the same buffers); the fused kernel's
    route-only mode when its check passed, else glue.router + glue.select on those rows: the same bits."""

    from . import glue

    if layer is None or getattr(layer, "moe", None) is None or not rowsplit_on() or getattr(ls, "ce", None) is None:
        return []
    b, c, m = ls.b, ls.w.cfg, layer.moe
    if c.top_k != TOPK:                             # (the model's config: the same on every rank)
        return []
    if hi > lo:
        x = b.normed[lo:hi]
        if (asks("route") and route_ok(x, m.router, b, c)
                and kernel_on("route", lambda: check_route(m.router, m.bias, c, b.plan.tile, x.device))
                and _route_fits(hi - lo, c.experts + 1)):
            rank, hist, tick = state(b).scratch(hi - lo, b.plan.slots, c.experts + 1, x.device)
            _ext().route(x, m.router, m.bias, b.mlog[lo:hi], b.pick[lo:hi], b.wts[lo:hi], b.plan.members,
                         b.plan.items, b.plan.counts, rank, hist, tick, float(c.routed_scale), bool(c.norm_topk),
                         c.experts + 1, b.plan.tile, _block(c.experts), False)
        else:
            glue.router(x, m.router, b.mlog[lo:hi])
            glue.select(b.mlog[lo:hi], m.bias, b.pick[lo:hi], b.wts[lo:hi], c.top_k, c.experts, c.routed_scale,
                        c.norm_topk)
    state(b).split_routed[layer.index] = True
    return [b.pick, b.wts]


def _q4_mlp(layer) -> bool:
    from . import qmm

    return isinstance(layer.mlp.gu, qmm.Q4)


def moe_block(layer, w, b, R: int, done=None, early=None):
    """forward.moe_block under TF_GLM_MOE_GLUE. ``done``: rows whose shared expert already ran on the main stream
    (TF_GLM_PREFILL_OVERLAP=2); ``early``: the second stream's event after ``lane_front`` ran it on every row."""

    from tensorfold.cuda import experts as grouped

    from . import exl3_mm, l2pf
    from . import forward as F

    c, m = w.cfg, layer.moe
    main = torch.cuda.current_stream()
    st = state(b)
    done0 = done                                    # the caller's (forward.moe_block's conditions read it)
    ready = None                                    # the second stream's event once the shared rows are in b.sy
    if m.shared is not None:
        if early is not None:
            ready, done = early, (0, R)
        elif asks("side") and side_ok(b, layer) and any(z > a for a, z in F.undone(R, done)):
            stream = st.side()
            st.ev_fork.record(main)
            with torch.cuda.stream(stream):
                stream.wait_event(st.ev_fork)
                for a, z in F.undone(R, done):
                    shared_rows(layer, w, b, a, z)
                st.ev_side.record(stream)
            ready, done = st.ev_side, (0, R)
    with prof.timed("moe: route"):
        route(b, layer, c, R)
    if m.shared is not None and done0 is None:
        _after_route(b, m, R)                       # integration point: work that needs the plan, before the experts
    if m.shared is not None and ready is None and l2pf.ACTIVE is not None and done is None:
        # TF_GLM_L2PF (as forward.moe_block): the shared expert before the routed experts while L2 holds what site "a"
        # prefetched for it
        shared_rows(layer, w, b, 0, R)
        done = (0, R)
    if m.shared is not None:
        with prof.timed("moe: routed (exl3)"):
            exl3_mm.routed(b.normed[:R], b.pick, b.plan, m.experts, b.exl3, b.ey.view(-1, c.hidden), R, c.limit)
        defer = asks("defer") and ready is not None and _lane_split(b)

        def fill(lo: int, hi: int) -> None:
            for a, z in F.undone(hi, done):       # rows lo .. hi the front has not run on
                if max(a, lo) < z:
                    shared_rows(layer, w, b, max(a, lo), z)
            if defer and (lo, hi) == (0, R):
                # the lane's whole partial at once (lanes.LaneSplit.partial): the combine after the shared rows on the
                # second stream (stream order) and after the routed experts (this event); the lane's exchange waits
                # for ``fill.ready`` (lanes.py)
                st.ev_routed.record(main)
                stream = st.side()
                with torch.cuda.stream(stream):
                    stream.wait_event(st.ev_routed)
                    with prof.timed("moe: combine"):
                        combine_rows(b, c, lo, hi)
                    st.ev_part.record(stream)
                fill.ready = st.ev_part
                return
            if ready is not None:
                main.wait_event(ready)
            with prof.timed("moe: combine"):
                combine_rows(b, c, lo, hi)

        return F.partials(w, b, R, fill, "moe: all-gather", site=(layer.index, "f"))
    with prof.timed("moe: gate/up"):
        grouped.gate_up(b.normed[:R], m.experts, b.plan, b.eact, R)
    with prof.timed("moe: down"):
        grouped.down(b.eact, m.experts, b.plan, b.ey.view(-1, c.hidden), R)

    def fill(lo: int, hi: int) -> None:
        from . import glue

        with prof.timed("moe: combine"):
            glue.combine(b.ey[lo:hi], b.wts[lo:hi], b.part[lo:hi])

    return F.partials(w, b, R, fill, "moe: all-gather", site=(layer.index, "f"))


# -- the checks (on this GPU, before first use) ---------------------------------------------------------------------
def _same(*pairs) -> bool:
    for a, b in pairs:
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)):
            return False
    return True


def _wide(shape, gen, device, spread: int) -> torch.Tensor:
    """Normal values times 2^e, e uniform in [-spread, spread]: sums of them round differently in any other order."""

    e = torch.randint(-spread, spread + 1, shape, device=device, generator=gen).to(torch.float32)
    return torch.randn(shape, device=device, generator=gen) * torch.exp2(e)


CHECK_ROUTE_ROWS = (1, 5, 16, 17, 32, 33, 64, 65, 300, 1100)


def route_inputs(R: int, D: int, seed: int, device, draw: str) -> torch.Tensor:
    """Router input rows of a check: ``wide`` (magnitudes 2^-4 .. 2^4), ``ties`` (every 4th row zero, so every logit is
    0 and every choice is its bias: ties wherever biases tie; the others repeat 3 rows, so equal rows tie exactly)."""

    g = torch.Generator(device=device).manual_seed(seed)
    x = (_wide((R, D), g, device, 4) * 0.05).to(torch.bfloat16)
    if draw == "ties":
        x[::4] = 0
        if R > 3:
            x[1::4] = x[1].clone()          # x[1::4] holds row 1 itself: copy it first (torch refuses aliased writes)
    return x


def route_reference(x, router, bias, c, tile, prefill: bool):
    """glue.router + glue.select + experts.route on fresh buffers: (logits, pick, wts, plan)."""

    from tensorfold.cuda import experts as grouped

    from . import glue

    R, dev = x.shape[0], x.device
    logits = torch.empty((R, c.experts), dtype=torch.float32, device=dev)
    pick = torch.empty((R, TOPK + 1), dtype=torch.int32, device=dev)
    wts = torch.empty((R, TOPK + 1), dtype=torch.float32, device=dev)
    plan = grouped.Plan(R, TOPK + 1, c.experts + 1, dev, prefill=prefill, tile=tile)
    glue.router(x, router, logits)
    glue.select(logits, bias, pick, wts, c.top_k, c.experts, c.routed_scale, c.norm_topk)
    grouped.route(pick, plan, plan.tile)
    return logits, pick, wts, plan


def route_fused(x, router, bias, c, tile, prefill: bool):
    """The fused route on fresh buffers: (logits, pick, wts, plan)."""

    from tensorfold.cuda import experts as grouped

    R, dev = x.shape[0], x.device
    logits = torch.full((R, c.experts), float("nan"), dtype=torch.float32, device=dev)
    pick = torch.full((R, TOPK + 1), -7, dtype=torch.int32, device=dev)
    wts = torch.full((R, TOPK + 1), float("nan"), dtype=torch.float32, device=dev)
    plan = grouped.Plan(R, TOPK + 1, c.experts + 1, dev, prefill=prefill, tile=tile)
    plan.members.fill_(-7)
    plan.items.fill_(-7)
    T = plan.tile if prefill else grouped.TILE
    blocks = int(_ext().route_blocks(R))
    rank = torch.zeros((R * (TOPK + 1),), dtype=torch.int32, device=dev)
    hist = torch.zeros((blocks * (c.experts + 1),), dtype=torch.int32, device=dev)
    tick = torch.zeros((blocks + 1,), dtype=torch.int32, device=dev)
    _ext().route(x, router, bias, logits, pick, wts, plan.members, plan.items, plan.counts, rank, hist, tick,
                 float(c.routed_scale), bool(c.norm_topk), c.experts + 1, T, _block(c.experts), True)
    if int(tick.abs().sum()) != 0:
        raise RuntimeError("the route kernel left its ticket counters non-zero")
    return logits, pick, wts, plan


def same_route(ref, got) -> bool:
    """Logits, picks and weights byte for byte; the plans' counts, items (as many as counts[0]) and members."""

    l0, p0, w0, a = ref
    l1, p1, w1, b = got
    if not _same((l0, l1), (p0, p1), (w0, w1), (a.counts, b.counts)):
        return False
    n = int(a.counts[0])
    P = p0.numel()
    return _same((a.items[:n], b.items[:n]), (a.members[:P], b.members[:P]))


def check_route(router, bias, c, tile, device) -> bool:
    """The fused route against the replaced kernels on random rows (the layer's router weights and bias, and a
    rounded bias with ties), at row counts through every tile of the kernel and several row blocks."""

    tied = torch.round(bias * 4) / 4                    # coarse biases: many choices tie exactly
    for i, R in enumerate(CHECK_ROUTE_ROWS):
        for prefill in (False, True):
            for draw, bb in (("wide", bias), ("ties", tied)):
                x = route_inputs(R, router.shape[1], 1000 + i, device, draw)
                ref = route_reference(x, router, bb, c, tile, prefill)
                if not same_route(ref, route_fused(x, router, bb, c, tile, prefill)):
                    return False
                if draw == "wide" and R in (5, 300, 1100) and not same_modes(ref, x, router, bb, c, tile, prefill):
                    return False
    return True


def same_modes(ref, x, router, bias, c, tile, prefill: bool) -> bool:
    """The route-only mode's logits, picks and weights, and the plan-only mode's plan of the reference's picks,
    against the replaced kernels (rowsplit runs both)."""

    from tensorfold.cuda import experts as grouped

    R, dev = x.shape[0], x.device
    T = tile if prefill else grouped.TILE
    blocks = int(_ext().route_blocks(R))
    rank = torch.zeros((R * (TOPK + 1),), dtype=torch.int32, device=dev)
    hist = torch.zeros((blocks * (c.experts + 1),), dtype=torch.int32, device=dev)
    tick = torch.zeros((blocks + 1,), dtype=torch.int32, device=dev)
    logits = torch.full((R, c.experts), float("nan"), device=dev)
    pick = torch.full((R, TOPK + 1), -7, dtype=torch.int32, device=dev)
    wts = torch.full((R, TOPK + 1), float("nan"), device=dev)
    plan = grouped.Plan(R, TOPK + 1, c.experts + 1, dev, prefill=prefill, tile=tile)
    _ext().route(x, router, bias, logits, pick, wts, plan.members, plan.items, plan.counts, rank, hist, tick,
                 float(c.routed_scale), bool(c.norm_topk), c.experts + 1, T, _block(c.experts), False)
    if not _same((ref[0], logits), (ref[1], pick), (ref[2], wts)):
        return False
    plan.members.fill_(-7)
    plan.items.fill_(-7)
    _ext().plan(ref[1], plan.members, plan.items, plan.counts, rank, hist, tick, c.experts + 1, T)
    if int(tick.abs().sum()) != 0:
        return False
    return same_route(ref, (ref[0], ref[1], ref[2], plan))


def check_combine(D: int, device) -> bool:
    """The CUDA combine against glue.combine_shared (and glue.combine with the shared row in the last slot)."""

    from . import glue

    for i, R in enumerate((1, 3, 64, 257)):
        g = torch.Generator(device=device).manual_seed(77 + i)
        y = _wide((R, TOPK + 1, D), g, device, 12)
        sy = _wide((R, D), g, device, 12)
        wts = torch.rand((R, TOPK + 1), device=device, generator=g) * 0.5
        a, b2, c2 = (torch.empty((R, D), device=device) for _ in range(3))
        glue.combine_shared(y, sy, wts, a)
        _ext().combine(y, sy, wts, b2)
        y2 = y.clone()
        y2[:, TOPK].copy_(sy)
        d = torch.empty((R, D), device=device)
        glue.combine(y2, wts, d)
        _ext().combine(y2, None, wts, c2)
        if not _same((a, b2), (d, c2), (a, d)):
            return False
    return True


def check_shared(gu, limit: float, device) -> bool:
    """The fused gate/up + SwiGLU against the prompt matmul and glue.swiglu (the layer's own weights)."""

    from tensorfold.cuda.kernels import qmm as shared

    from . import glue, qmm

    ni = gu.n // 2
    for i, R in enumerate((1, 37, 128, 300)):
        g = torch.Generator(device=device).manual_seed(55 + i)
        x = (_wide((R, gu.k), g, device, 3) * 0.3).to(torch.bfloat16)
        if i == 3:
            x[::7] *= 64                                 # some rows past the SwiGLU clip
        ref = torch.empty((R, gu.n), dtype=torch.bfloat16, device=device)
        shared.prefill_matmul(x, gu, f32=False, out=ref, tile=qmm.prefill_tile(gu.n, gu.k, R))
        act0 = torch.empty((R, ni), dtype=torch.bfloat16, device=device)
        xs = torch.empty((R, ni // 64), dtype=torch.float32, device=device)
        glue.swiglu(ref, act0, xs, limit)
        act1 = torch.full((R, ni), float("nan"), dtype=torch.bfloat16, device=device)
        _ext().shared_gu(x, gu.weight, gu.scales, gu.biases, act1, ni, float(limit), shared_tile())
        if not _same((act0, act1)):
            return False
    return True


__all__ = ["CHECKED", "PARTS", "asks", "check_combine", "check_route", "check_shared", "code", "combine_rows",
           "describe", "ffn", "lane_front", "moe_block", "on", "parse", "reset", "route", "shared_rows", "state",
           "wanted"]
