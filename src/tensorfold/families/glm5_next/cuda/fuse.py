"""Decode-window kernel fusions that keep every output's bits (TF_GLM_FUSE).

A decode forward of GLM-5.3-Flash launches about 1,800 kernels; on a large discrete GPU each launch costs about as
much as a small kernel's work. These fusions run two or three adjacent kernels as fewer launches with the same
per-element operations, the same reduction trees and the same roundings:

  hc      a site's hyper-connection post with the next site's pre (hc_post + hc_partial + _hc_finish: 3 launches to
          2, at the 84 of the 90 sites that are not followed by a tap or the final mean; glue.hc_post_pre)
  router  the MoE router's slice sum with its top-k selection (2 to 1, every MoE layer; glue.router_select)
  gates   the KDA gates' two 64-input group sums (2 to 1, every KDA layer; qmm.group_sums_split)
  kdaout  the KDA read-out's group sums inside its output kernel (2 to 1, every KDA layer; kda.chain(xs=...))

Setting: TF_GLM_FUSE unset, "1" or "on": router and gates (Triton kernels holding the unfused kernels' source
verbatim); "all": kdaout and hc as well, which reproduce in CUDA what Triton 3.7 compiles for the unfused kernels
(_group_sums' reduction tree for these rows, _hc_post's per-element operation order, read from their PTX), so they
are exact by construction for that compiler and checked again on the GPU (another Triton version may compile the
unfused kernels otherwise: the check then turns the fusion off); "0" or "off": none (the off switch); a
comma-separated list of names: those. Prompt chunks never take them. TF_GLM_FUSE_HC_ROWS (default 8): windows of
more rows keep hc unfused (its blocks recompute their K block of the new streams from all four old ones, extra L2
reads that grow with the rows while the launch it saves does not; both ways give the same bits, so the limit is a
speed choice). The engine reads both settings at start on every rank and refuses a value that is not valid, or
values that differ between ranks, on every rank together.

Each fusion is used only after it gave the unfused kernels' bits on this GPU: on its first call outside a CUDA graph
capture, both run on random inputs of the call's widths (with the call's own weights; hundreds to a thousand rows,
several seeds; values spread over a wide range of magnitudes and, for hc, a cancelling draw as well, so that another
order of operations shows) and their outputs are compared byte for byte; any difference turns that fusion off for
the process (a line says so) and the unfused kernels run, as they do inside a capture before the check. The engine's
graphs are captured after eager warm-up rounds, which run the checks. Decisions are per process (per rank); a fused
and an unfused rank compute the same bits.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab."""

from __future__ import annotations

import os
from typing import Callable

import torch

NAMES = ("hc", "router", "gates", "kdaout")
DEFAULT = ("router", "gates")
CHECK_ROWS = 1024                  # rows the checks run (fewer where a row is large)
CHECK_SEEDS = 3

_decided: dict[str, bool] = {}
_wanted: frozenset | None = None


def wanted() -> frozenset:
    """The fusions TF_GLM_FUSE asks for (read once)."""

    global _wanted
    if _wanted is None:
        # this fork: fusions default OFF - the hand-merged Aevonix kernel sources are not yet first-use-verified
        # on the mixed path; set TF_GLM_FUSE=on/all to try them (their checks gate each fusion either way)
        v = (os.environ.get("TF_GLM_FUSE", "") or "off").strip().lower()
        if v in ("1", "on"):
            _wanted = frozenset(DEFAULT)
        elif v == "all":
            _wanted = frozenset(NAMES)
        elif v in ("0", "off", "none"):
            _wanted = frozenset()
        else:
            names = {n.strip() for n in v.split(",") if n.strip()}
            bad = names - set(NAMES)
            if bad:
                raise ValueError(f"TF_GLM_FUSE: on, off or names among {', '.join(NAMES)}; "
                                 f"not {', '.join(sorted(bad))}")
            _wanted = frozenset(names)
    return _wanted


def reset(value: str | None = None) -> None:
    """Forget the decisions (tests): TF_GLM_FUSE is read again, or ``value`` is used."""

    global _wanted, _hc_rows
    _decided.clear()
    _wanted = None
    _hc_rows = None
    if value is not None:
        os.environ["TF_GLM_FUSE"] = value


def code() -> int:
    """The requested fusions as a bit mask (for logs and tests; ranks may differ in what their checks allowed)."""
    return sum(1 << i for i, n in enumerate(NAMES) if n in wanted())


_hc_rows: int | None = None


def hc_rows() -> int:
    """TF_GLM_FUSE_HC_ROWS: the most rows a window may have for the hc fusion (read once)."""

    global _hc_rows
    if _hc_rows is None:
        v = int(os.environ.get("TF_GLM_FUSE_HC_ROWS", "") or 8)
        if v < 0:
            raise ValueError("TF_GLM_FUSE_HC_ROWS: 0 (never) or more rows")
        _hc_rows = v
    return _hc_rows


def asks(name: str) -> bool:
    """Whether the setting asks for ``name`` (before its check)."""
    return name in wanted()


def on(name: str, check: Callable[[], bool]) -> bool:
    """Whether to run fusion ``name`` now: asked for, and its check passed (run here on its first call outside a
    capture); inside a capture before the check: False."""

    if name not in wanted():
        return False
    got = _decided.get(name)
    if got is not None:
        return got
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return False
    try:
        ok = bool(check())
        why = "" if ok else ": its outputs differ from the unfused kernels' on this GPU"
    except Exception as exc:  # noqa: BLE001 - a fusion that cannot run here is off, the unfused kernels run
        ok, why = False, f": {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
    _decided[name] = ok
    print(f"[tensorfold] decode fusion {name}: {'on (checked bit for bit)' if ok else 'off' + why}", flush=True)
    return ok


def _same(*pairs) -> bool:
    for a, b in pairs:
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)):
            return False
    return True


def _gen(seed: int, device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)


def _wide(shape, gen, device, spread: int = 12, scale: float = 1.0) -> torch.Tensor:
    """Normal values times 2^e, e uniform in [-spread, spread] per element: sums of such values round differently
    under any other order of operations (a fp32 tensor)."""
    e = torch.randint(-spread, spread + 1, shape, device=device, generator=gen).to(torch.float32)
    return torch.randn(shape, device=device, generator=gen) * torch.exp2(e) * scale


def check_hc(x: torch.Tensor, g: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor,
             norm_w: torch.Tensor, eps: float, hc_eps: float, iters: int) -> bool:
    """hc_post + hc_pre against glue.hc_post_pre on random streams, partials, post and comb (the call's weights):
    CHECK_SEEDS draws over a wide range of magnitudes (another order of the partials' adds shows) and one cancelling
    draw (``hc_cancelling``), on which another rounding order of the first two products shows."""

    dev, R = x.device, 256
    world, _, d = g.shape
    for seed in range(CHECK_SEEDS + 1):
        if not _check_hc_once(dev, R, world, d, x.shape[1], seed, fn, base, scale, norm_w, eps, hc_eps, iters,
                              cancel=seed == CHECK_SEEDS):
            return False
    return True


def hc_cancelling(x: torch.Tensor, g: torch.Tensor, comb: torch.Tensor) -> None:
    """Make a draw cancelling, in place: stream 1 = -stream 0, stream 1's comb weights 1 + 2^-20 times stream 0's,
    streams 2 and 3 and the gathered partials 2^-36 smaller. A new stream's x0 c0 + x1 c1 then nearly cancels, so the
    rounding of whichever product is rounded first (an fp32 ulp of x0 c0) is a large part of the result and shows in
    its bf16 value for most elements (about 92%); on draws of independent values it shows for a few in a million."""

    d = g.shape[-1]
    x[:, d:2 * d] = -x[:, :d]
    x[:, 2 * d:] *= 2.0 ** -36
    g *= 2.0 ** -36
    comb[:, 4:8] = comb[:, 0:4] * (1 + 2.0 ** -20)


def _check_hc_once(dev, R, world, d, wide, seed, fn, base, scale, norm_w, eps, hc_eps, iters, cancel=False) -> bool:
    from . import glue

    gen = _gen(10 + seed, dev)
    xs0 = _wide((R, wide), gen, dev, 6).to(torch.bfloat16)
    gg = _wide((world, R, d), gen, dev, 6)
    post0 = torch.rand((R, 4), device=dev, generator=gen) * 2.0
    comb0 = torch.rand((R, 16), device=dev, generator=gen) * 0.5
    if cancel:
        hc_cancelling(xs0, gg, comb0)
    outs = []
    for fused in (False, True):
        xx, post, comb = xs0.clone(), post0.clone(), comb0.clone()
        out = torch.empty((R, d), dtype=torch.bfloat16, device=dev)
        xs = torch.empty((R, d // 64), dtype=torch.float32, device=dev)
        part = torch.zeros((R, glue.HC_BLOCKS, 32), dtype=torch.float32, device=dev)
        if fused:
            xn = torch.empty_like(xx)
            glue.hc_post_pre(xx, gg, fn, base, scale, norm_w, out, xs, post, comb, part, eps, hc_eps, iters, xn)
        else:
            glue.hc_post(xx, xx, gg, post, comb)
            glue.hc_pre(xx, fn, base, scale, norm_w, out, xs, post, comb, part, eps, hc_eps, iters)
        outs.append((xx, out, xs, post, comb, part[:, :, :25].contiguous()))
    return _same(*zip(*outs))


def check_router(x: torch.Tensor, w: torch.Tensor, bias: torch.Tensor, top_k: int, experts: int, scale: float,
                 norm: bool) -> bool:
    """router + select against glue.router_select on random rows (the call's router weights and bias)."""

    from . import glue

    dev, R = x.device, CHECK_ROWS
    xx = _wide((R, x.shape[1]), _gen(2, dev), dev, 4).to(x.dtype)
    if w.shape[0] != experts:
        return False
    outs = []
    for fused in (False, True):
        logits = torch.empty((R, experts), dtype=torch.float32, device=dev)
        pick = torch.empty((R, top_k + 1), dtype=torch.int32, device=dev)
        wts = torch.empty((R, top_k + 1), dtype=torch.float32, device=dev)
        if fused:
            glue.router_select(xx, w, logits, bias, pick, wts, top_k, experts, scale, norm)
        else:
            glue.router(xx, w, logits)
            glue.select(logits, bias, pick, wts, top_k, experts, scale, norm)
        outs.append((logits, pick, wts))
    return _same(*zip(*outs))


def check_gates(width: int, fa_off: int, device) -> bool:
    """Two qmm.group_sums against qmm.group_sums_split on random projection rows."""

    from . import qmm

    R = CHECK_ROWS
    p = _wide((R, width), _gen(3, device), device, 12).to(torch.bfloat16)
    a1 = torch.empty((R, 2), dtype=torch.float32, device=device)
    b1 = torch.empty_like(a1)
    a2, b2 = torch.empty_like(a1), torch.empty_like(a1)
    qmm.group_sums(p[:, fa_off:fa_off + 128], a1)
    qmm.group_sums(p[:, fa_off + 128:fa_off + 256], b1)
    qmm.group_sums_split(p[:, fa_off:fa_off + 256], 128, a2, b2)
    return _same((a1, a2), (b1, b2))


def check_kdaout(heads: int, device) -> bool:
    """The wide KDA chain plus qmm.group_sums of its output against the chain with xs, on random inputs."""

    for seed in range(CHECK_SEEDS):
        if not _check_kdaout_once(heads, device, 192, seed):
            return False
    return True


def _check_kdaout_once(heads: int, device, R: int, seed: int) -> bool:
    from . import kda, qmm

    H = heads
    gen = _gen(40 + seed, device)
    C = 3 * H * kda.DK
    b_off = C + 256
    p = (torch.randn((R, b_off + H + 32), device=device, generator=gen) * 0.5).to(torch.bfloat16)
    a = torch.randn((R, H * kda.DK), device=device, generator=gen).to(torch.bfloat16)
    # gates and norm weights over many magnitudes: the read-outs then span enough exponents that a 64-wide sum rounds
    # differently under any other order (sums of a narrow range of bf16 values are exact in every order)
    gate = (_wide((R, H * kda.DV), gen, device, 2) * 6).to(torch.bfloat16)
    cs = (torch.randn((3, C), device=device, generator=gen) * 0.5).to(torch.bfloat16)
    cw = (torch.randn((C, 4), device=device, generator=gen) * 0.5).to(torch.bfloat16)
    state = torch.randn((H, kda.DV, kda.DK), device=device, generator=gen) * 0.1
    a_log = torch.rand(H, device=device, generator=gen) * 2 - 1
    dt = torch.randn(H * kda.DK, device=device, generator=gen) * 0.1
    nw = torch.exp2(torch.randint(-8, 9, (kda.DV,), device=device, generator=gen).float()).to(torch.bfloat16)
    outs = []
    for fused in (False, True):
        sc = kda.KDAScratch(R, H, device)
        so = torch.empty_like(state)
        xs = torch.empty((R, H * kda.DV // 64), dtype=torch.float32, device=device)
        out = kda.chain(p, b_off, a, gate, cs, cw, state, a_log, dt, nw, 1e-5, -5.0, R, sc, so, wide=True,
                        xs=xs if fused else None)
        if not fused:
            qmm.group_sums(out, xs)
        outs.append((out.clone(), so, xs))
    return _same(*zip(*outs))


__all__ = ["NAMES", "asks", "check_gates", "check_hc", "check_kdaout", "check_router", "code", "on", "reset",
           "wanted"]
