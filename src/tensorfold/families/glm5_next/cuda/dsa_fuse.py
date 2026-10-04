"""TF_GLM_DSA_FUSE (patch 0193): DSA's small per-row stages in fewer launches, every output's bits kept, in prompt
chunks and in decode windows (single-stream and segmented):

  writes  the latent write, the indexer's key and gate write and the pool keys in one launch (3 to 1)
  norms   the query and latent RMSNorms of the DSA front in one launch (2 to 1)

Exact by construction: each part is the unfused kernel's source with its shapes, warps and arguments, so it
compiles to the same operations in the same order (tests/k5/check_ptx.py --dsa compares the PTX float work part by
part). writes runs one warp a program, as _index_write and _pool_keys do; the latent row, written by four warps
before, has no sum (its fp8 scale is a maximum), so its bytes do not depend on the warps. A pool whose four tokens
fall in this window computes their keys and gates as _index_write stores them (the same LayerNorm, rounded to bf16)
instead of reading them back from the ring, so no program reads what another program of the launch writes; tokens
of earlier windows are read from the ring as before (the ring holds the window's rows and the three before them, so
this launch's writes never cover them). norms runs each norm as its own program row of one grid with the norm's own
block and warps; each part takes its own row pointer, so the loads specialize as the unfused calls' do.

Each fusion is used after its outputs equalled the unfused kernels' byte for byte on this GPU (first call outside a
CUDA graph capture; random rows over a wide range of magnitudes, positions that start and end windows inside pools,
fp8 and bf16 caches); a difference turns it off for the process (a line says so) and the unfused kernels run.

Setting: TF_GLM_DSA_FUSE 0 / off (default), 1 / on / all, or a comma list of writes, norms.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab."""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from . import kv8

NAMES = ("writes", "norms")
_wanted: frozenset | None = None
_decided: dict[str, bool] = {}


def wanted() -> frozenset:
    global _wanted
    if _wanted is None:
        v = (os.environ.get("TF_GLM_DSA_FUSE", "") or "0").strip().lower()
        if v in ("0", "off", "none"):
            _wanted = frozenset()
        elif v in ("1", "on", "all"):
            _wanted = frozenset(NAMES)
        else:
            names = {n.strip() for n in v.split(",") if n.strip()}
            bad = names - set(NAMES)
            if bad:
                raise ValueError(f"TF_GLM_DSA_FUSE: 0, 1 or names among {', '.join(NAMES)}; not {', '.join(sorted(bad))}")
            _wanted = frozenset(names)
    return _wanted


def code() -> int:
    return sum(1 << i for i, n in enumerate(NAMES) if n in wanted())


def reset(value: str | None = None) -> None:
    global _wanted
    _decided.clear()
    _wanted = None
    if value is not None:
        os.environ["TF_GLM_DSA_FUSE"] = value


def on(name: str, check) -> bool:
    """Run fusion ``name`` now: asked for and its check passed on this GPU (run on its first call outside a capture)."""
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
    except Exception as exc:  # noqa: BLE001 - a fusion that cannot run here is off; the unfused kernels run
        ok, why = False, f": {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
    _decided[name] = ok
    print(f"[tensorfold] DSA fusion {name}: {'on (checked bit for bit)' if ok else 'off' + why}", flush=True)
    return ok


# -- writes: latent + index key / gate + pool keys -------------------------------------------------------------------
@triton.jit
def _key_row(KR, k_stride, LNW, LNB, r, eps, D: tl.constexpr):
    """_index_write's LayerNorm of key row r (fp32, before its bf16 store)."""
    d = tl.arange(0, D)
    x = tl.load(KR + r * k_stride + d).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    xc = x - mean
    var = tl.sum(xc * xc, axis=0) / D
    return xc / tl.sqrt(var + eps) * tl.load(LNW + d).to(tl.float32) + tl.load(LNB + d).to(tl.float32)


@triton.jit
def _token(t, P, KR, k_stride, GR, LNW, LNB, IK, IG, RING, eps, D: tl.constexpr):
    """Token t's index key and gate as their bf16 ring rows: computed when t is a row of this window (row t - P, as
    _index_write stores it), else read from the ring (an earlier window wrote it). bf16 here, widened where
    _pool_keys widens its loads, so the compiler folds the same conversions into the same adds."""
    d = tl.arange(0, D)
    if t >= P:
        r = t - P
        k = _key_row(KR, k_stride, LNW, LNB, r, eps, D).to(tl.bfloat16)
        g = tl.load(GR + r * D + d).to(tl.bfloat16)
    else:
        s = t % RING
        k = tl.load(IK + s * D + d)
        g = tl.load(IG + s * D + d)
    return k, g


@triton.jit
def _pool(k0, g0, k1, g1, k2, g2, k3, g3, APE, D: tl.constexpr):
    """_pool_keys' arithmetic on four tokens' bf16 keys and gates."""
    d = tl.arange(0, D)
    l0 = g0.to(tl.float32) + tl.load(APE + 0 * D + d).to(tl.float32)
    l1 = g1.to(tl.float32) + tl.load(APE + 1 * D + d).to(tl.float32)
    l2 = g2.to(tl.float32) + tl.load(APE + 2 * D + d).to(tl.float32)
    l3 = g3.to(tl.float32) + tl.load(APE + 3 * D + d).to(tl.float32)
    m = tl.maximum(tl.maximum(l0, l1), tl.maximum(l2, l3))
    e0 = tl.exp(l0 - m)
    e1 = tl.exp(l1 - m)
    e2 = tl.exp(l2 - m)
    e3 = tl.exp(l3 - m)
    s = ((e0 + e1) + e2) + e3
    t0 = ((e0 / s).to(tl.bfloat16).to(tl.float32) * k0.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    t1 = ((e1 / s).to(tl.bfloat16).to(tl.float32) * k1.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    t2 = ((e2 / s).to(tl.bfloat16).to(tl.float32) * k2.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    t3 = ((e3 / s).to(tl.bfloat16).to(tl.float32) * k3.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    return ((t0 + t1) + t2) + t3


@triton.jit
def _dsa_writes(LAT, lat_stride, LC, LS, KR, k_stride, GR, LNW, LNB, IK, IG, APE, PK, PKS, POS, R, RING, eps,
                LW: tl.constexpr, RSL: tl.constexpr, FP8L: tl.constexpr, D: tl.constexpr, RSP: tl.constexpr,
                FP8P: tl.constexpr):
    """Programs 0 .. R - 1: row r's latent (_lat_write) and index key and gate (_index_write); programs R .. R + R // 4
    + 1: pool i (_pool_keys), its in-window tokens computed here."""
    g = tl.program_id(0)
    P = tl.load(POS).to(tl.int64)
    if g < R:
        r = g
        k = tl.arange(0, LW)
        kv8.store_row(LC, LS, P + r, tl.load(LAT + r * lat_stride + k).to(tl.float32), LW, RSL, FP8L)
        d = tl.arange(0, D)
        y = _key_row(KR, k_stride, LNW, LNB, r, eps, D)
        slot = (P + r) % RING
        tl.store(IK + slot * D + d, y.to(tl.bfloat16))
        tl.store(IG + slot * D + d, tl.load(GR + r * D + d).to(tl.bfloat16))
    else:
        p = P // 4 + (g - R)
        if 4 * p + 3 <= P + R - 1:
            k0, g0 = _token(4 * p + 0, P, KR, k_stride, GR, LNW, LNB, IK, IG, RING, eps, D)
            k1, g1 = _token(4 * p + 1, P, KR, k_stride, GR, LNW, LNB, IK, IG, RING, eps, D)
            k2, g2 = _token(4 * p + 2, P, KR, k_stride, GR, LNW, LNB, IK, IG, RING, eps, D)
            k3, g3 = _token(4 * p + 3, P, KR, k_stride, GR, LNW, LNB, IK, IG, RING, eps, D)
            kv8.store_row(PK, PKS, p, _pool(k0, g0, k1, g1, k2, g2, k3, g3, APE, D), D, RSP, FP8P)


def writes(lat: torch.Tensor, cache: torch.Tensor, k_raw: torch.Tensor, gate: torch.Tensor, ln_w: torch.Tensor,
           ln_b: torch.Tensor, ape: torch.Tensor, ik: torch.Tensor, ig: torch.Tensor, pk: torch.Tensor,
           pos: torch.Tensor) -> None:
    """latent.latent_write(lat, cache, pos) and sparse.index_update(k_raw, gate, ...) in one launch."""
    R = lat.shape[0]
    ring = ik.shape[0]
    if ik.dtype != torch.bfloat16 or ig.shape != ik.shape or not ik.is_contiguous() or not ig.is_contiguous():
        raise ValueError("dsa writes: bf16 key and gate rings of one shape, contiguous")
    if lat.shape[1] != kv8.width(cache) or kv8.width(pk) != 128 or k_raw.shape[0] != R:
        raise ValueError("dsa writes: rows, latent width or pooled-key width do not match")
    vals, scl, rsl, fp8l = kv8.parts(cache)
    pkv, pks, rsp, fp8p = kv8.parts(pk)
    _dsa_writes[(R + R // 4 + 2,)](lat, lat.stride(0), vals, scl, k_raw, k_raw.stride(0), gate, ln_w, ln_b, ik, ig, ape,
                                   pkv, pks, pos, R, ring, 1e-6, LW=lat.shape[1], RSL=rsl, FP8L=fp8l, D=128, RSP=rsp,
                                   FP8P=fp8p, num_warps=1)


@triton.jit
def _seg_token(t, q, r, SEG, KR, k_stride, GR, LNW, LNB, IK, IG, IB, RING, eps, D: tl.constexpr):
    """_token for a segmented window (bf16 rows): token t of row r's stream (row r at position q) is this window's row r - (q - t)
    when that row is in r's segment, else the stream's ring row IB + t % RING."""
    d = tl.arange(0, D)
    rr = r - (q - t)
    inside = rr >= 0
    if inside:
        inside = tl.load(SEG + rr) == tl.load(SEG + r)
    if inside:
        k = _key_row(KR, k_stride, LNW, LNB, rr, eps, D).to(tl.bfloat16)
        g = tl.load(GR + rr * D + d).to(tl.bfloat16)
    else:
        s = IB + t % RING
        k = tl.load(IK + s * D + d)
        g = tl.load(IG + s * D + d)
    return k, g


@triton.jit
def _seg_dsa_writes(LAT, lat_stride, LC, LS, KR, k_stride, GR, LNW, LNB, IK, IG, APE, PK, PKS, POS, BASE, IBASE, SEG,
                    RING, eps, LW: tl.constexpr, RSL: tl.constexpr, FP8L: tl.constexpr, D: tl.constexpr,
                    RSP: tl.constexpr, FP8P: tl.constexpr):
    """Program r: _seg_lat_write, _seg_index_write and _seg_pool_keys of row r."""
    r = tl.program_id(0)
    q = tl.load(POS + r).to(tl.int64)
    B = tl.load(BASE + r).to(tl.int64)
    IB = tl.load(IBASE + r).to(tl.int64)
    k = tl.arange(0, LW)
    kv8.store_row(LC, LS, B + q, tl.load(LAT + r * lat_stride + k).to(tl.float32), LW, RSL, FP8L)
    d = tl.arange(0, D)
    y = _key_row(KR, k_stride, LNW, LNB, r, eps, D)
    row = IB + q % RING
    tl.store(IK + row * D + d, y.to(tl.bfloat16))
    tl.store(IG + row * D + d, tl.load(GR + r * D + d).to(tl.bfloat16))
    if q % 4 == 3:
        t = (q // 4) * 4
        k0, g0 = _seg_token(t + 0, q, r, SEG, KR, k_stride, GR, LNW, LNB, IK, IG, IB, RING, eps, D)
        k1, g1 = _seg_token(t + 1, q, r, SEG, KR, k_stride, GR, LNW, LNB, IK, IG, IB, RING, eps, D)
        k2, g2 = _seg_token(t + 2, q, r, SEG, KR, k_stride, GR, LNW, LNB, IK, IG, IB, RING, eps, D)
        k3, g3 = _seg_token(t + 3, q, r, SEG, KR, k_stride, GR, LNW, LNB, IK, IG, IB, RING, eps, D)
        kv8.store_row(PK, PKS, B // 4 + q // 4, _pool(k0, g0, k1, g1, k2, g2, k3, g3, APE, D), D, RSP, FP8P)


def seg_writes(lat: torch.Tensor, cache: torch.Tensor, k_raw: torch.Tensor, gate: torch.Tensor, ln_w: torch.Tensor,
               ln_b: torch.Tensor, ape: torch.Tensor, ik: torch.Tensor, ig: torch.Tensor, pk: torch.Tensor,
               rows) -> None:
    """latent.seg_latent_write and sparse.seg_index_update (``rows``: segments.SegRows) in one launch."""
    R = lat.shape[0]
    if lat.shape[1] != kv8.width(cache) or kv8.width(pk) != 128 or k_raw.shape[0] != R:
        raise ValueError("dsa writes: rows, latent width or pooled-key width do not match")
    if cache.stride(-1) != 1 or cache.stride(0) != cache.shape[1]:
        raise ValueError("dsa writes: a cache of contiguous rows")
    vals, scl, rsl, fp8l = kv8.parts(cache)
    pkv, pks, rsp, fp8p = kv8.parts(pk)
    _seg_dsa_writes[(R,)](lat, lat.stride(0), vals, scl, k_raw, k_raw.stride(0), gate, ln_w, ln_b, ik, ig, ape, pkv, pks,
                          rows.pos, rows.base, rows.ibase, rows.seg, rows.ring_rows, 1e-6, LW=lat.shape[1], RSL=rsl,
                          FP8L=fp8l, D=128, RSP=rsp, FP8P=fp8p, num_warps=1)


# -- norms: the DSA front's query and latent RMSNorms ----------------------------------------------------------------
@triton.jit
def _rms_part(X, x_stride, W, OUT, o_stride, XS, eps, r, D: tl.constexpr, BLOCK: tl.constexpr, SUMS: tl.constexpr):
    """glue._rmsnorm's body for row r."""
    d = tl.arange(0, BLOCK)
    ok = d < D
    x = tl.load(X + r * x_stride + d, mask=ok, other=0.0).to(tl.float32)
    rinv = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / D + eps)
    w = tl.load(W + d, mask=ok, other=0.0).to(tl.float32)
    y = (w * (x * rinv).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    tl.store(OUT + r * o_stride + d, y, mask=ok)
    if SUMS:
        g = tl.sum(tl.reshape(tl.where(ok, y.to(tl.float32), 0.0), (BLOCK // 64, 64)), axis=1)
        gi = tl.arange(0, BLOCK // 64)
        tl.store(XS + r * (D // 64) + gi, g, mask=gi < D // 64)


@triton.jit
def _norm_pair(XA, xa_stride, WA, OA, oa_stride, XSA, XB, xb_stride, WB, OB, ob_stride, XSB, eps,
               DA: tl.constexpr, BA: tl.constexpr, DB: tl.constexpr, BB: tl.constexpr, SUMS: tl.constexpr):
    """Program (r, 0): the first norm of row r; (r, 1): the second; each glue._rmsnorm's body with its own block."""
    r = tl.program_id(0)
    if tl.program_id(1) == 0:
        _rms_part(XA, xa_stride, WA, OA, oa_stride, XSA, eps, r, DA, BA, SUMS)
    else:
        _rms_part(XB, xb_stride, WB, OB, ob_stride, XSB, eps, r, DB, BB, SUMS)


def norm_pair_ok(xa: torch.Tensor, xb: torch.Tensor) -> bool:
    """Both norms take the same warps as glue.rmsnorm would (4: blocks up to 2,048) and write group sums."""
    return (xa.shape[0] == xb.shape[0] and triton.next_power_of_2(xa.shape[1]) <= 2048
            and triton.next_power_of_2(xb.shape[1]) <= 2048)


def norm_pair(xa, wa, oa, xsa, xb, wb, ob, xsb, eps: float) -> None:
    """glue.rmsnorm(xa, wa, eps, oa, xsa) and glue.rmsnorm(xb, wb, eps, ob, xsb) in one launch (``norm_pair_ok``)."""
    rows = xa.shape[0]
    if oa.stride(-1) != 1 or ob.stride(-1) != 1:
        raise ValueError("norm pair: rows of the outputs must be contiguous")
    _norm_pair[(rows, 2)](xa, xa.stride(0), wa, oa, oa.stride(0), xsa, xb, xb.stride(0), wb, ob, ob.stride(0), xsb,
                          eps, DA=xa.shape[1], BA=triton.next_power_of_2(xa.shape[1]), DB=xb.shape[1],
                          BB=triton.next_power_of_2(xb.shape[1]), SUMS=True, num_warps=4)


# -- checks ---------------------------------------------------------------------------------------------------------
def _same(*pairs) -> bool:
    for a, b in pairs:
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if not torch.equal(a.contiguous().view(-1).view(torch.uint8), b.contiguous().view(-1).view(torch.uint8)):
            return False
    return True


def _wide(shape, gen, device, spread: int) -> torch.Tensor:
    e = torch.randint(-spread, spread + 1, shape, device=device, generator=gen).to(torch.float32)
    return torch.randn(shape, device=device, generator=gen) * torch.exp2(e)


def check_writes(kind: str, device) -> bool:
    """writes / seg_writes against latent_write + index_update (and their segmented forms) on random rows: windows of
    1 .. 4,096 rows starting at every position mod 4, rings that wrap, fp8 and bf16 caches."""
    from . import latent, sparse

    gen = torch.Generator(device=device).manual_seed(193)
    bf = torch.bfloat16
    for P, R in ((0, 4096), (8191, 2048), (5, 1), (6, 3), (7, 2), (4097, 64), (130, 512), (12345, 37)):
        cap = P + R + 64
        ring = min(cap, -(-(R + 3) // 64) * 64)
        lat = _wide((R, 512), gen, device, 6).to(bf)
        kraw = _wide((R, 160), gen, device, 6).to(bf)[:, :128]
        gate = _wide((R, 128), gen, device, 6)
        ln_w = _wide((128,), gen, device, 2).to(bf)
        ln_b = _wide((128,), gen, device, 2).to(bf)
        ape = _wide((4, 128), gen, device, 2).to(bf)
        ik0 = _wide((ring, 128), gen, device, 4).to(bf)
        ig0 = _wide((ring, 128), gen, device, 4).to(bf)
        lc0 = kv8.zeros(cap, 512, kind, device)
        pk0 = kv8.zeros(cap // 4 + 2, 128, kind, device)
        pos = torch.tensor([P], dtype=torch.int32, device=device)
        outs = []
        for fused in (False, True):
            lc, pk, ik, ig = lc0.clone(), pk0.clone(), ik0.clone(), ig0.clone()
            if fused:
                writes(lat, lc, kraw, gate, ln_w, ln_b, ape, ik, ig, pk, pos)
            else:
                latent.latent_write(lat, lc, pos)
                sparse.index_update(kraw, gate, ln_w, ln_b, ape, ik, ig, pk, pos)
            outs.append((lc, pk, ik, ig))
        if not _same(*zip(*outs)):
            return False
    return _check_seg_writes(kind, device, gen)


def _check_seg_writes(kind: str, device, gen) -> bool:
    from . import latent, sparse
    from .segments import SegRows

    bf = torch.bfloat16
    ring = 128
    for segs in (((0, 7, 5), (4096, 2, 1), (8192, 1023, 9)), ((0, 3, 1),), ((2048, 0, 4), (6144, 6, 6)),
                 ((0, 100, 32), (4096, 201, 31))):
        rows = SegRows(64, device, max_segs=8, ring=ring)
        R = rows.set([(b, p, n, i * ring) for i, (b, p, n) in enumerate(segs)])
        cap = 12288
        lat = _wide((R, 512), gen, device, 6).to(bf)
        kraw = _wide((R, 160), gen, device, 6).to(bf)[:, :128]
        gate = _wide((R, 128), gen, device, 6)
        ln_w = _wide((128,), gen, device, 2).to(bf)
        ln_b = _wide((128,), gen, device, 2).to(bf)
        ape = _wide((4, 128), gen, device, 2).to(bf)
        ik0 = _wide((8 * ring, 128), gen, device, 4).to(bf)
        ig0 = _wide((8 * ring, 128), gen, device, 4).to(bf)
        lc0 = kv8.zeros(cap, 512, kind, device)
        pk0 = kv8.zeros(cap // 4 + 2, 128, kind, device)
        outs = []
        for fused in (False, True):
            lc, pk, ik, ig = lc0.clone(), pk0.clone(), ik0.clone(), ig0.clone()
            if fused:
                seg_writes(lat, lc, kraw, gate, ln_w, ln_b, ape, ik, ig, pk, rows)
            else:
                latent.seg_latent_write(lat, lc, rows)
                sparse.seg_index_update(kraw, gate, ln_w, ln_b, ape, ik, ig, pk, rows)
            outs.append((lc, pk, ik, ig))
        if not _same(*zip(*outs)):
            return False
    return True


def check_norms(qa: int, kb: int, device) -> bool:
    """norm_pair against two glue.rmsnorm calls on random rows of the DSA front's widths (the call's own)."""
    from . import glue

    gen = torch.Generator(device=device).manual_seed(194)
    bf = torch.bfloat16
    for R in (1, 7, 64, 513):
        dp = _wide((R, qa + kb), gen, device, 6).to(bf)
        wa = _wide((qa,), gen, device, 3).to(bf)
        wb = _wide((kb,), gen, device, 3).to(bf)
        outs = []
        for fused in (False, True):
            oa = torch.empty((R, qa), dtype=bf, device=device)
            ob = torch.empty((R, kb), dtype=bf, device=device)
            xa = torch.empty((R, qa // 64), dtype=torch.float32, device=device)
            xb = torch.empty((R, kb // 64), dtype=torch.float32, device=device)
            if fused:
                norm_pair(dp[:, :qa], wa, oa, xa, dp[:, qa:], wb, ob, xb, 1e-5)
            else:
                glue.rmsnorm(dp[:, :qa], wa, 1e-5, oa, xa)
                glue.rmsnorm(dp[:, qa:], wb, 1e-5, ob, xb)
            outs.append((oa, ob, xa, xb))
        if not _same(*zip(*outs)):
            return False
    return True
