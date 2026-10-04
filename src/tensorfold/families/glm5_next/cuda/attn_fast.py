"""TF_GLM_SPARSE_FAST=1 (patch 0194): a prompt chunk's DSA sparse attention (latent.sparse_onepass) by a kernel that
gives every output latent._sparse_onepass' bits, faster.

Why: sparse attention is the largest DSA stage of a prompt chunk (a row's 16 heads over its 2,051 selected latents:
about 67 MFLOP a row, 137 GFLOP a 2,048-row lane a layer). _sparse_onepass runs a row on eight warps over 32-key
tiles; its 16 x 32 score tile is two n8 tiles fewer than eight warps, so warps 4-7 compute warps 0-3's scores again
(a third of its tensor-core work), and every warp loads the row's whole 16 x 512 query from shared memory for every
32 keys.

How: the same program per row, eight warps, and the same online softmax, but the scores come 64 keys at a time: one
16 x 64 product, one n8 tile a warp, nothing computed twice and the query loaded half as often; its two 32-key halves
go through shared memory into _sparse_onepass' own 16 x 32 layout, and each half then runs _sparse_onepass' tile code
(fp8 scale, query scale, mask, running maximum, rescale, probabilities, the PV product into the rescaled output, the
row sum) on its 32 keys, in key order. The key rows for the next 64 keys are gathered into registers while the
current ones are used (as _sparse_onepass gathers the next tile's).

Exact: every score is one mma chain over the 512-wide latent (32 m16n8k16 steps, operands of k width 2, the same
order), so it does not depend on which warp computes it or how many columns the product has; everything after the
scores is _sparse_onepass' code in _sparse_onepass' TTGIR layouts (mma v2, warps [1, 8], 16 x 8; k width 2; its
shared-memory swizzles), so the softmax's maximum and sum trees and the PV chains are its own (tests/k5/check_ptx.py
--sparse compares the PTX float work: the score chains and every tile's softmax-and-PV unit). A second half with no
valid key is skipped (the reference never runs that tile). On the GPU the outputs are compared with
_sparse_onepass' byte for byte on the first call of a process (random rows over a wide range of magnitudes, token
lists of every length class); a difference turns the setting off for the process and the reference runs.
TF_GLM_SPARSE_FAST_CHECK=1 compares every call (a test setting).

TF_GLM_SPARSE_FAST=2 (patch 0199): the same bits by a CUDA kernel with warp roles (``sparse_ws.cu``): four score warps
hold the row's query in registers and compute each tile's scores (each one 8-key block's chain), four PV warps run the
tile's softmax and PV product on their 128 output columns and gather the next tiles into a second buffer, so a tile's
scores are computed while the PV warps run the one before and nothing is computed twice. Every operation of the
reference's per-tile code is done on the same values in the same order with the same PTX instructions (the file's
header lists them); the first call compares it with the reference as above.

Applies where _sparse_onepass would run with its fixed tile and launch: TF_GLM_SPARSE_ONEPASS on (the default),
16-head tiles (a rank's DSA heads at TP4), 32-key tiles, one latent slice, prefetch, eight warps.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab."""

from __future__ import annotations

import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

# _sparse_onepass' layouts on sm_120 (Triton 3.7: HBT 16, KTT 32, LW 512, eight warps), read from its TTGIR
_BLK = gl.constexpr(gl.BlockedLayout([1, 16], [1, 32], [8, 1], [1, 0]))     # key rows (fp8 codes, bf16)
_BLK1 = gl.constexpr(gl.BlockedLayout([1, 8], [1, 32], [4, 2], [1, 0]))     # the query and the output
_MMA = gl.constexpr(gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 8], instr_shape=[16, 8]))
_DA = gl.constexpr(gl.DotOperandLayout(operand_index=0, parent=_MMA, k_width=2))
_DB = gl.constexpr(gl.DotOperandLayout(operand_index=1, parent=_MMA, k_width=2))
_SH = gl.constexpr(gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]))     # query, key rows
_SH2 = gl.constexpr(gl.SwizzledSharedLayout(vec=8, per_phase=2, max_phase=4, order=[1, 0]))    # probabilities
_SHF = gl.constexpr(gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[1, 0]))    # scores (fp32)
WARPS = 8
KC = 128                            # the score product's k step a launch (speed only: the k chain is the same)
LAUNCH = (8, 1, True)               # (warps, stages, prefetch) _sparse_onepass must run with for this kernel


def _flag(name: str, values=("0", "1")) -> str:
    value = (os.environ.get(name, "") or "0").strip()
    if value not in values:
        raise ValueError(f"{name}: {' or '.join(values)}, not {value!r}")
    return value


MODE = _flag("TF_GLM_SPARSE_FAST", ("0", "1", "2"))      # 1: the Gluon kernel (0194), 2: the CUDA one (0199)
ENABLED = MODE != "0"
CHECK = _flag("TF_GLM_SPARSE_FAST_CHECK") == "1"
_decided: bool | None = None
_WS = None


def _ws():
    """The warp-role CUDA kernel's extension (TF_GLM_SPARSE_FAST=2, and topk_fast's TF_GLM_TOPK_FAST=1; built at
    first use)."""
    global _WS
    if _WS is None:
        from pathlib import Path

        from tensorfold.cuda.build import load

        here = Path(__file__).parent
        _WS = load(name="tensorfold_glm_dsa_ws_v1", sources=[str(here / "sparse_ws.cpp"), str(here / "sparse_ws.cu"),
                                                            str(here / "sparse_topk.cu")],
                   extra_cuda_cflags=["-O3"], verbose=False)
    return _WS


@gluon.jit
def _ids(row, t0, n, KTT: gl.constexpr, layout: gl.constexpr):
    """Token ids t0 .. t0 + KTT of the row's list (0 past n), int64, in ``layout``."""
    idx = t0 + gl.arange(0, KTT, layout=layout)
    return gl.load(row + idx, mask=idx < n, other=0).to(gl.int64)


@gluon.jit
def _rows(LC, tok, t0, n, RS: gl.constexpr, LW: gl.constexpr, KTT: gl.constexpr):
    """The tokens' latent rows [KTT, LW] (fp8 codes or bf16, RS elements apart); rows past n are zeros."""
    k = gl.arange(0, LW, layout=gl.SliceLayout(0, _BLK))
    ok = t0 + gl.arange(0, KTT, layout=gl.SliceLayout(1, _BLK)) < n
    return gl.load(LC + tok[:, None] * RS + k[None, :], mask=ok[:, None], other=0.0)


@gluon.jit
def _scales(LS, tok, t0, n, LW: gl.constexpr, RS: gl.constexpr, KTT: gl.constexpr, FP8: gl.constexpr):
    """The tokens' fp8 row scales [KTT] (ones past n; ones for a bf16 cache, unread)."""
    if FP8:
        ok = t0 + gl.arange(0, KTT, layout=gl.SliceLayout(0, _MMA)) < n
        return gl.load(LS + tok * (RS // 4) + LW // 4, mask=ok, other=1.0)
    return gl.full([KTT], 1.0, gl.float32, gl.SliceLayout(0, _MMA))


@gluon.jit
def _tile(scores, ks, valid, kv, ps, m, l, o, SCALE: gl.constexpr, FP8: gl.constexpr, KTT: gl.constexpr):
    """latent._tile from its scores on: ``scores`` the 16 x KTT dots (fp32, _MMA), ``kv`` the tile's key rows in
    shared memory [KTT, LW], ``ps`` the probabilities' shared buffer."""
    if FP8:
        scores = scores * ks[None, :]
    scores = scores * SCALE
    scores = gl.where(valid[None, :], scores, float("-inf"))
    tile_m = gl.max(scores, 1)
    active = tile_m != float("-inf")
    next_m = gl.where(active, gl.maximum(m, tile_m), m)
    alpha = gl.where(active, gl.where(m == float("-inf"), 0.0, gl.exp(m - next_m)), 1.0)
    p = gl.where(valid[None, :] & active[:, None], gl.exp(scores - next_m[:, None]), 0.0)
    if FP8:
        pb = (p * ks[None, :]).to(gl.bfloat16)
    else:
        pb = p.to(gl.bfloat16)
    ps.store(pb)
    o = mma_v2(ps.load(_DA), kv.load(_DB), o * alpha[:, None])
    l = l * alpha + gl.sum(p, 1)
    return next_m, l, o


@gluon.jit
def _onepass_fast(QA, LC, LS, TOK, CNT, OUT, W: gl.constexpr, H: gl.constexpr, LW: gl.constexpr,
                  SCALE: gl.constexpr, HBT: gl.constexpr, KTT: gl.constexpr, RS: gl.constexpr, FP8: gl.constexpr,
                  KC: gl.constexpr):
    """Program (row, head group): _sparse_onepass with one latent slice, its scores 2 x KTT keys at a time."""
    r = gl.program_id(0)
    hb = gl.program_id(1)
    n = gl.load(CNT + r)
    if n > 0:
        hh = hb * HBT + gl.arange(0, HBT, layout=gl.SliceLayout(1, _BLK1))
        hok = hh < H
        k = gl.arange(0, LW, layout=gl.SliceLayout(0, _BLK1))
        q = gl.load(QA + (r * H + hh[:, None]) * LW + k[None, :], mask=hok[:, None], other=0).to(gl.bfloat16)
        qs = gl.allocate_shared_memory(gl.bfloat16, [HBT, LW], _SH, q)
        kvs = gl.allocate_shared_memory(gl.bfloat16, [2 * KTT, LW], _SH)
        scs = gl.allocate_shared_memory(gl.float32, [HBT, 2 * KTT], _SHF)
        ps = gl.allocate_shared_memory(gl.bfloat16, [HBT, KTT], _SH2)
        m = gl.full([HBT], float("-inf"), gl.float32, gl.SliceLayout(1, _MMA))
        l = gl.zeros([HBT], gl.float32, gl.SliceLayout(1, _MMA))
        o = gl.zeros([HBT, LW], gl.float32, _MMA)
        row = TOK + r * W
        KK: gl.constexpr = 2 * KTT
        # the two 32-key halves' rows (registers), scales and the next ids, as _sparse_onepass keeps a tile's
        ta = _ids(row, 0, n, KTT, gl.SliceLayout(1, _BLK))
        tb = _ids(row, KTT, n, KTT, gl.SliceLayout(1, _BLK))
        kva = _rows(LC, ta, 0, n, RS, LW, KTT)
        kvb = _rows(LC, tb, KTT, n, RS, LW, KTT)
        ksa = _scales(LS, _ids(row, 0, n, KTT, gl.SliceLayout(0, _MMA)), 0, n, LW, RS, KTT, FP8)
        ksb = _scales(LS, _ids(row, KTT, n, KTT, gl.SliceLayout(0, _MMA)), KTT, n, LW, RS, KTT, FP8)
        ta = _ids(row, KK, n, KTT, gl.SliceLayout(1, _BLK))
        tb = _ids(row, KK + KTT, n, KTT, gl.SliceLayout(1, _BLK))
        for t0 in range(0, n, KK):
            kvs.slice(0, KTT, dim=0).store(kva.to(gl.bfloat16))
            kvs.slice(KTT, KTT, dim=0).store(kvb.to(gl.bfloat16))
            # the next 64 keys' rows into registers (their ids are here), and the ids after them
            kva = _rows(LC, ta, t0 + KK, n, RS, LW, KTT)
            kvb = _rows(LC, tb, t0 + KK + KTT, n, RS, LW, KTT)
            ksa_next = _scales(LS, _ids(row, t0 + KK, n, KTT, gl.SliceLayout(0, _MMA)), t0 + KK, n, LW, RS, KTT, FP8)
            ksb_next = _scales(LS, _ids(row, t0 + KK + KTT, n, KTT, gl.SliceLayout(0, _MMA)), t0 + KK + KTT, n, LW, RS,
                               KTT, FP8)
            ta = _ids(row, t0 + 2 * KK, n, KTT, gl.SliceLayout(1, _BLK))
            tb = _ids(row, t0 + 2 * KK + KTT, n, KTT, gl.SliceLayout(1, _BLK))
            # the 512-wide product in four 128-wide steps (the same k chain: steps 0 .. 31 in order, each step's
            # accumulator the last one's), so only a quarter of the query operand is live at a time
            kt = kvs.permute([1, 0])
            dots = gl.zeros([HBT, KK], gl.float32, _MMA)
            for c in gl.static_range(LW // KC):
                dots = mma_v2(qs.slice(c * KC, KC, dim=1).load(_DA), kt.slice(c * KC, KC, dim=0).load(_DB), dots)
            scs.store(dots)
            va = t0 + gl.arange(0, KTT, layout=gl.SliceLayout(0, _MMA)) < n
            m, l, o = _tile(scs.slice(0, KTT, dim=1).load(_MMA), ksa, va, kvs.slice(0, KTT, dim=0), ps, m, l, o,
                            SCALE, FP8, KTT)
            if t0 + KTT < n:
                vb = t0 + KTT + gl.arange(0, KTT, layout=gl.SliceLayout(0, _MMA)) < n
                m, l, o = _tile(scs.slice(KTT, KTT, dim=1).load(_MMA), ksb, vb, kvs.slice(KTT, KTT, dim=0), ps, m, l,
                                o, SCALE, FP8, KTT)
            ksa = ksa_next
            ksb = ksb_next
        out = (o / l[:, None]).to(gl.bfloat16)
        gl.store(OUT + (r * H + hh[:, None]) * LW + k[None, :], gl.convert_layout(out, _BLK1), mask=hok[:, None])


def applies(qa: torch.Tensor, tile, launch) -> bool:
    """The reference's call is the one this kernel replaces: 16-head programs (a rank's 16 heads: the tile's head
    count capped as _sparse_onepass caps it), 32-key tiles, one latent slice, the 512-wide latent, its fixed launch."""
    from . import latent

    hbt, kt, split = tile or latent.ONEPASS_TILE
    H = qa.shape[1]
    hbt = max(16, min(hbt, triton.next_power_of_2(H)))
    return (hbt == 16 and H == 16 and kt == 32 and split == 1 and qa.shape[2] == 512
            and tuple(launch or latent.ONEPASS_LAUNCH) == LAUNCH)


def launch(qa: torch.Tensor, cache: torch.Tensor, tokens: torch.Tensor, counts: torch.Tensor, out: torch.Tensor,
           scale: float, kernel: str | None = None) -> None:
    """latent.sparse_onepass(qa, cache, tokens, counts, out, scale) for this kernel's shapes (``applies``); ``kernel``
    "1" (Gluon, 0194) or "2" (warp roles, CUDA, 0199), default the setting's."""
    from .latent import _cache_parts

    R, H, LW = qa.shape
    if not (qa.is_contiguous() and out.is_contiguous() and tokens.is_contiguous() and cache.is_contiguous()):
        raise ValueError("sparse attention: expected contiguous queries, output, token lists and cache")
    vals, scl, rs, fp8 = _cache_parts(cache, LW)
    if R == 0:
        return
    if (kernel or MODE) == "2":         # patch 0199: the warp-role CUDA kernel (the cache as stored: codes or bf16)
        _ws().sparse_ws(qa, cache, tokens, counts, out, float(scale))
        return
    _onepass_fast[(R, triton.cdiv(H, 16))](qa, vals, scl, tokens, counts, out, W=tokens.shape[1], H=H, LW=LW,
                                           SCALE=scale, HBT=16, KTT=32, RS=rs, FP8=fp8, KC=KC, num_warps=WARPS)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))


def self_check(kind: str, device, kernel: str | None = None) -> bool:
    """Both kernels on random queries and caches over a wide range of magnitudes, token lists of every length class
    (0, 1, 31, 32, 33, 63, 64, 65, 2,048, 2,051 tokens, random and repeated ids), compared byte for byte."""
    from . import kv8, latent

    gen = torch.Generator(device=device).manual_seed(194)

    def wide(shape, spread):
        e = torch.randint(-spread, spread + 1, shape, device=device, generator=gen).float()
        return torch.randn(shape, device=device, generator=gen) * torch.exp2(e)

    cap = 6000
    cache = kv8.zeros(cap, 512, kind, device)
    rows = wide((cap, 512), 3)
    cache.copy_(kv8.quantize_rows(rows) if kind == "fp8" else rows.to(torch.bfloat16))
    lens = [0, 1, 31, 32, 33, 63, 64, 65, 96, 127, 2048, 2051, 2051, 2049, 500, 1000]
    R = len(lens)
    W = 2051
    tokens = torch.full((R, W), -1, dtype=torch.int32, device=device)
    for i, n in enumerate(lens):
        if n:
            ids = torch.randint(0, cap, (n,), device=device, generator=gen) if i % 2 else \
                torch.sort(torch.randperm(cap, device=device, generator=gen)[:n]).values
            tokens[i, :n] = ids.to(torch.int32)
    counts = torch.tensor(lens, dtype=torch.int32, device=device)
    for spread in (2, 6):
        qa = wide((R, 16, 512), spread).to(torch.bfloat16)
        a = torch.full((R, 16, 512), 7.0, device=device).to(torch.bfloat16)
        b = a.clone()
        latent._sparse_onepass_ref(qa, cache, tokens, counts, a, 256 ** -0.5)
        launch(qa, cache, tokens, counts, b, 256 ** -0.5, kernel)
        if not _same(a, b):
            return False
    return True


def on(qa: torch.Tensor, cache: torch.Tensor, tile, launch_) -> bool:
    """Whether to run this kernel for this call: the setting, its shapes, and its check passed on this GPU."""
    global _decided
    if not ENABLED or not applies(qa, tile, launch_):
        return False
    if _decided is None:
        from . import kv8

        try:
            ok = self_check(kv8.kind_of(cache), qa.device)
            why = "" if ok else ": its outputs differ from the reference kernel's on this GPU"
        except Exception as exc:  # noqa: BLE001 - a kernel that cannot run here is off; the reference runs
            ok, why = False, f": {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
        _decided = ok
        kind = "warp roles, CUDA" if MODE == "2" else "Gluon"
        print(f"[tensorfold] fast DSA sparse attention ({kind}): {'on (checked bit for bit)' if ok else 'off' + why}",
              flush=True)
    return _decided


def run(qa, cache, tokens, counts, out, scale) -> None:
    """launch(), and with TF_GLM_SPARSE_FAST_CHECK=1 the reference on a copy, compared (stops on a difference)."""
    if CHECK:
        from . import latent

        ref = out.clone()
        latent._sparse_onepass_ref(qa, cache, tokens, counts, ref, scale)
    launch(qa, cache, tokens, counts, out, scale)
    if CHECK and not _same(ref, out):
        raise RuntimeError("TF_GLM_SPARSE_FAST_CHECK: the fast sparse attention differs from the reference's")
