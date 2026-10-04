"""TF_GLM_SELECT_FAST=1 (patch 0192): a prompt chunk's DSA pool scores (the indexer's s_p = sum_h w_h relu(scale *
qi_h . pool_p) a row and complete pool) by a kernel that gives every score the bits of ``sparse._scores``, faster.

Why: at long contexts the indexer scores are the token selection's main cost (a row scores pos / 4 pools, 32 heads of
128: 8 KFLOP a pool, at 100K about 50 GFLOP a 512-row block a layer), and ``_scores`` spends most of its time
outside the tensor cores: a program scores RB rows x 64 pools, so for every 64 pools each row's 32 x 128 index query
is loaded again from memory and staged through shared memory, and the 64 x 128 key tile is masked, staged and loaded
again for each row.

How: a program keeps RB rows' index queries in shared memory and walks pool blocks (blocks cs, cs + CS, .. of the
chunk's NP pools); a block's keys go through shared memory once and, where every pool of the block is visible to every
row of the program (all but the blocks at the rows' own positions), its tensor-core operand is loaded once for all RB
rows. The blocks at the rows' positions take ``_scores``' per-row masking.

Exact: written in Gluon with ``_scores``' layouts spelled out, as Triton 3.7 compiles ``_scores`` for sm_120: the
32 x 64 dot as two-by-one m16n8k16 tiles a warp, warps side by side over the pools, and its operands' k width, which
orders a fragment's k elements inside each mma step and so sets the bits: 4 with fp8 pooled keys (Triton widens the
upcast keys' fragments), 2 with bf16 keys (the swizzle follows, 16- or 8-wide). So a row's 64 dots, its fp8 scale
and query scale, its ReLU, its head weights and its sum over the 32 heads are ``_scores``' operations in ``_scores``'
order on the same operands: ``tests/k5/check_ptx.py`` compares both kernels' dot operand layouts in their TTGIR and
their PTX float instructions one for one, for each key format. A key the mask would zero is never read where the mask
is all-true. Rows, pools and the -inf fill are ``_scores``'. On the GPU the scores are compared with ``_scores``' byte
for byte on the first call of a process for its key format (random rows over a wide range of magnitudes) and, with
TF_GLM_SELECT_FAST_CHECK=1, on every call (a test setting); a difference turns the setting off for the process (a line
says so) and ``_scores`` runs.

Settings: TF_GLM_SELECT_FAST 0 (default: off) or 1; TF_GLM_SELECT_FAST_RB 1, 2 or 4 (rows a program, default 4) and
TF_GLM_SELECT_FAST_GRID (programs a call aims for, default 4 x the GPU's SMs) change speed only.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab."""

from __future__ import annotations

import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

# _scores' layouts on sm_120 (Triton 3.7, num_warps 4, a 32 x 128 query tile, a 64 x 128 key tile): read from its TTGIR.
# The dot's operands depend on the pooled keys' format: fp8 keys (upcast to bf16 in registers) give k width 4 and a
# 16-wide swizzle, bf16 keys k width 2 and an 8-wide swizzle; the k width orders a fragment's k elements inside each
# mma step, so it sets the bits (the swizzle only the addresses). FP8_LAYOUTS / BF16_LAYOUTS: (A, B, shared).
_BLK = gl.constexpr(gl.BlockedLayout([1, 1], [1, 32], [1, 4], [1, 0]))
_BLK1 = gl.constexpr(gl.BlockedLayout([1], [32], [4], [0]))
_MMA = gl.constexpr(gl.NVMMADistributedLayout(version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8]))
FP8_LAYOUTS = (gl.DotOperandLayout(operand_index=0, parent=_MMA.value, k_width=4),
               gl.DotOperandLayout(operand_index=1, parent=_MMA.value, k_width=4),
               gl.SwizzledSharedLayout(vec=16, per_phase=1, max_phase=4, order=[1, 0]))
BF16_LAYOUTS = (gl.DotOperandLayout(operand_index=0, parent=_MMA.value, k_width=2),
                gl.DotOperandLayout(operand_index=1, parent=_MMA.value, k_width=2),
                gl.SwizzledSharedLayout(vec=8, per_phase=1, max_phase=8, order=[1, 0]))
# loads only (a tile's register layout before it goes to shared memory sets no bit): 16-byte vectors
_KV = gl.constexpr(gl.BlockedLayout([1, 16], [4, 8], [4, 1], [1, 0]))      # a 64 x 128 fp8 key tile
_QV = gl.constexpr(gl.BlockedLayout([1, 8], [2, 16], [4, 1], [1, 0]))      # a 32 x 128 bf16 query tile
WARPS = 4


def _flag(name: str, default: str, allowed: tuple[str, ...]) -> str:
    value = (os.environ.get(name, "") or default).strip()
    if value not in allowed:
        raise ValueError(f"{name}: {' or '.join(allowed)}, not {value!r}")
    return value


ENABLED = _flag("TF_GLM_SELECT_FAST", "0", ("0", "1")) == "1"
CHECK = _flag("TF_GLM_SELECT_FAST_CHECK", "0", ("0", "1")) == "1"
RB = int(_flag("TF_GLM_SELECT_FAST_RB", "4", ("1", "2", "4")))
GRID = int(os.environ.get("TF_GLM_SELECT_FAST_GRID", "0") or 0)

_decided: bool | None = None


def code() -> list[int]:
    """The settings every rank must share (the startup comparison): on, check."""
    return [int(ENABLED), int(CHECK)]


def describe() -> str:
    return ("prompt chunks' DSA pool scores by the fast kernel (the same bits, checked on this GPU"
            + (", every call)" if CHECK else ")"))


@gluon.jit
def _stage_q(QS, QI, i, r, R, H: gl.constexpr, HP: gl.constexpr, D: gl.constexpr):
    """Row r's index query [HP, D] into shared slot i, zeros past the R rows."""
    hh = gl.arange(0, HP, layout=gl.SliceLayout(1, _QV))
    dd = gl.arange(0, D, layout=gl.SliceLayout(0, _QV))
    q = gl.load(QI + (r * H + hh[:, None]) * D + dd[None, :], mask=(hh < H)[:, None] & (r < R), other=0.0)
    QS.index(i).store(q.to(gl.bfloat16))


@gluon.jit
def _row_out(OUT, W, w_stride, r, R, NP, dots, ks, hh_m, hok_m, p_m, p_o, npool, scale, wscale, FP8: gl.constexpr):
    """_scores' epilogue for row r: fp8 scales, query scale, ReLU, head weights, the sum over the heads, -inf past the
    row's pools, the store (its head weights loaded as _scores loads them)."""
    if FP8:
        dots = dots * ks[None, :]
    w = gl.load(W + r * w_stride + hh_m, mask=hok_m & (r < R), other=0.0).to(gl.float32) * wscale
    sc = gl.sum(w[:, None] * gl.maximum(dots * scale, 0.0), axis=0)
    sc = gl.where(p_m < npool, sc, float("-inf"))
    gl.store(OUT + r * NP + p_o, gl.convert_layout(sc, _BLK1), mask=(p_o < NP) & (r < R))


@gluon.jit
def _scores_fast(QI, W, w_stride, PK, PKS, OUT, POS, R, NP, scale, wscale, H: gl.constexpr, HP: gl.constexpr,
                 D: gl.constexpr, BP: gl.constexpr, RB: gl.constexpr, RS: gl.constexpr, FP8: gl.constexpr,
                 CS: gl.constexpr, DA: gl.constexpr, DB: gl.constexpr, SH: gl.constexpr):
    """Program (RB rows, column split cs): pool blocks cs, cs + CS, .. of the NP scored, each through _scores'
    arithmetic for every row of the program."""
    rb = gl.program_id(0)
    cs = gl.program_id(1)
    P = gl.load(POS)
    r0 = rb * RB
    d_b = gl.arange(0, D, layout=gl.SliceLayout(0, _KV))
    hh_m = gl.arange(0, HP, layout=gl.SliceLayout(1, _MMA))
    hok_m = hh_m < H
    QS = gl.allocate_shared_memory(gl.bfloat16, [RB, HP, D], SH)             # RB buffers of SH
    for i in gl.static_range(RB):
        _stage_q(QS, QI, i, r0 + i, R, H, HP, D)
    ksm = gl.allocate_shared_memory(gl.bfloat16, [BP, D], SH)
    visible = (P + gl.minimum(r0 + RB, R)) // 4           # pools the program's last row sees
    first = (P + r0 + 1) // 4                              # pools its first row sees
    kmax = (P + r0 + RB) // 4                              # _scores' key mask (rb * RB + RB)
    nblk = (NP + BP - 1) // BP
    for pb in range(cs, nblk, CS):
        p_m = pb * BP + gl.arange(0, BP, layout=gl.SliceLayout(0, _MMA))
        p_b = pb * BP + gl.arange(0, BP, layout=gl.SliceLayout(1, _KV))
        p_o = pb * BP + gl.arange(0, BP, layout=_BLK1)
        if pb * BP >= visible:
            ninf = gl.full([BP], float("-inf"), gl.float32, _BLK1)
            for i in gl.static_range(RB):
                gl.store(OUT + (r0 + i) * NP + p_o, ninf, mask=(p_o < NP) & (r0 + i < R))
        else:
            k = gl.load(PK + p_b[:, None].to(gl.int64) * RS + d_b[None, :], mask=(p_b < kmax)[:, None],
                        other=0.0).to(gl.bfloat16)
            if FP8:
                ks = gl.load(PKS + p_m.to(gl.int64) * (RS // 4) + D // 4, mask=p_m < kmax, other=1.0)
            else:
                ks = gl.full([BP], 1.0, gl.float32, gl.SliceLayout(0, _MMA))
            z = gl.zeros([HP, BP], gl.float32, _MMA)
            if pb * BP + BP <= first:                      # every pool of the block seen by every row: no mask
                ksm.store(k)
                kb = ksm.permute([1, 0]).load(DB)
                for i in gl.static_range(RB):
                    r = r0 + i
                    dots = mma_v2(QS.index(i).load(DA), kb, z)
                    _row_out(OUT, W, w_stride, r, R, NP, dots, ks, hh_m, hok_m, p_m, p_o, (P + r + 1) // 4, scale,
                             wscale, FP8)
            else:                                          # the rows' own block: _scores' mask, row by row
                for i in gl.static_range(RB):
                    r = r0 + i
                    if r < R:
                        npool = (P + r + 1) // 4
                        ksm.store(gl.where((p_b < npool)[:, None], k, 0.0))
                        dots = mma_v2(QS.index(i).load(DA), ksm.permute([1, 0]).load(DB), z)
                        _row_out(OUT, W, w_stride, r, R, NP, dots, ks, hh_m, hok_m, p_m, p_o, npool, scale, wscale,
                                 FP8)


def _sms(device) -> int:
    if device.type != "cuda":
        return 4                                    # the interpreter (tests)
    return torch.cuda.get_device_properties(device).multi_processor_count


def _grid(n: int, NP: int, rb: int, device) -> tuple[int, int]:
    """(row blocks, column splits): splits enough for about GRID programs (speed only)."""
    blocks = -(-n // rb)
    target = GRID or 4 * _sms(device)
    cs = max(1, min(-(-NP // 64), -(-target // blocks)))
    return blocks, cs


def launch(sel, a: int, at: torch.Tensor, scores: torch.Tensor, rb: int | None = None, cs: int | None = None) -> None:
    """The scores of PromptSelect ``sel``'s block at row a (n rows over its NP pools) into ``scores`` [n, NP]."""
    n, NP = sel.rows(a), sel.NP
    rb = rb or RB
    blocks, split = _grid(n, NP, rb, scores.device)
    split = cs or split
    da, db, sh = FP8_LAYOUTS if sel.fp8 else BF16_LAYOUTS
    _scores_fast[(blocks, split)](sel.qi[a:a + n], sel.wts[a:a + n], sel.wts.stride(0), sel.pkv, sel.pks, scores, at,
                                  n, NP, sel.D ** -0.5, sel.wscale, H=sel.H, HP=max(16, triton.next_power_of_2(sel.H)),
                                  D=sel.D, BP=64, RB=rb, RS=sel.rs, FP8=sel.fp8, CS=split, DA=da, DB=db, SH=sh,
                                  num_warps=WARPS)


def reference(sel, a: int, at: torch.Tensor, scores: torch.Tensor) -> None:
    """``sparse._scores``' launch for the same block (what PromptSelect.pools runs without the setting)."""
    from .sparse import _scores

    n, NP = sel.rows(a), sel.NP
    _scores[(triton.cdiv(n, sel.rb), triton.cdiv(NP, 64))](
        sel.qi[a:a + n], sel.wts[a:a + n], sel.wts.stride(0), sel.pkv, sel.pks, scores, at, n, NP, sel.D ** -0.5,
        sel.wscale, H=sel.H, HP=max(16, triton.next_power_of_2(sel.H)), D=sel.D, BP=64, RB=sel.rb, RS=sel.rs,
        FP8=sel.fp8, num_warps=4)


def _same(x: torch.Tensor, y: torch.Tensor) -> bool:
    return x.shape == y.shape and torch.equal(x.contiguous().view(torch.int32), y.contiguous().view(torch.int32))


def self_check(sel, device) -> bool:
    """Both kernels on random index queries, head weights and pooled keys of the call's kind (fp8 or bf16 keys), over
    a wide range of magnitudes, at several positions and row counts, compared byte for byte."""
    from . import kv8
    from .sparse import PromptSelect

    gen = torch.Generator(device=device).manual_seed(192)

    def wide(shape, spread):
        e = torch.randint(-spread, spread + 1, shape, device=device, generator=gen).float()
        return torch.randn(shape, device=device, generator=gen) * torch.exp2(e)

    kind = "fp8" if sel.fp8 else "bf16"
    for pos, R in ((2048, 512), (8191, 129), (20480, 512), (40000, 7)):
        cap = pos + R + 8
        pk = kv8.zeros(cap // 4 + 2, sel.D, kind, device)
        keys = wide((cap // 4 + 2, sel.D), 4)
        if kind == "fp8":
            pk.copy_(kv8.quantize_rows(keys))
        else:
            pk.copy_(keys.to(torch.bfloat16))
        qi = wide((R, sel.H * sel.D), 6).to(torch.bfloat16)
        wbuf = wide((R, sel.H + 128), 6).to(torch.bfloat16)
        wts = wbuf[:, 128:128 + sel.H]
        pos_dev = torch.tensor([pos], dtype=torch.int32, device=device)
        other = PromptSelect(qi, wts, pk, pos, R, pk.shape[0] - 2, pos_dev, block=R)
        n, NP = other.rows(0), other.NP
        a_ = torch.full((n, NP), 7.0, device=device)
        b_ = torch.full((n, NP), 3.0, device=device)
        reference(other, 0, pos_dev, a_)
        for rb in (1, 2, 4):
            b_.fill_(3.0)
            launch(other, 0, pos_dev, b_, rb=rb)
            if not _same(a_, b_):
                return False
    return True


def on(sel) -> bool:
    """Whether to score ``sel``'s blocks with the fast kernel: the setting, and the check passed on this GPU (run on
    the first call of the process)."""

    global _decided
    if not ENABLED:
        return False
    if _decided is None:
        try:
            ok = self_check(sel, sel.qi.device)
            why = "" if ok else ": its scores differ from the reference kernel's on this GPU"
        except Exception as exc:  # noqa: BLE001 - a kernel that cannot run here is off; the reference runs
            ok, why = False, f": {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
        _decided = ok
        print(f"[tensorfold] fast DSA pool scores: {'on (checked bit for bit)' if ok else 'off' + why}", flush=True)
    return _decided


def scores(sel, a: int, at: torch.Tensor, out: torch.Tensor) -> None:
    """The block's scores into ``out`` [n, NP] (``on(sel)`` true); TF_GLM_SELECT_FAST_CHECK=1 compares every call
    with the reference kernel and stops on a difference."""
    launch(sel, a, at, out)
    if CHECK:
        ref = torch.empty_like(out)
        reference(sel, a, at, ref)
        if not _same(out, ref):
            bad = (out.view(torch.int32) != ref.view(torch.int32)).any(dim=1).nonzero().flatten()[:8].tolist()
            raise RuntimeError(f"TF_GLM_SELECT_FAST_CHECK: the fast pool scores differ from the reference's (rows "
                               f"{bad} of the block at row {a})")
