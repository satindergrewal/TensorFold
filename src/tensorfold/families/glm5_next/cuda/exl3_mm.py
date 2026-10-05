"""EXL3 expert kernels fix K splits and warps by shape so each routed pair's output is independent of the window's other rows."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import os
from pathlib import Path

import torch

from tensorfold.cuda import experts as grouped

from . import tune

# a rank's expert widths the decode kernel (``dec``) takes: two ranks' 1024, four ranks' 512, three ranks' 768 / 640
# (its down projection's K / 64 k steps a warp: 16, 8, 12, 10); others run the grouped kernel (the same bits)
DEC_WIDTHS = (1024, 768, 640, 512)
# Gate/up and down tile counts, warps, and K splits stay fixed across row counts to preserve each row's bits.
GATEUP_CFG = (8, 4, 4)
DOWN_CFG = (8, 4, 1)
# Prompt chunks launch an item's n blocks next to each other (TF_GLM_EXL3_NFIRST=0: the decode order), so they read
# its rows while L2 still holds them: the same kernel and arithmetic, 2.3-2.5x faster at 4,096 rows.
PROMPT_ROWS = 64          # rows from which a window takes the prompt order


def _pf() -> bool:
    """TF_GLM_EXL3_PF=0: no register prefetch in the expert kernel (the same bits either way)."""
    import os

    return os.environ.get("TF_GLM_EXL3_PF", "1") != "0"


def _nfirst() -> bool:
    import os

    return os.environ.get("TF_GLM_EXL3_NFIRST", "1") != "0"


def dec_settings() -> tuple[bool, int, bool]:
    """Decode windows' EXL3 path. TF_GLM_EXL3_DEC (default on): ``dec``, grouped_kernel's sums with the down
    projection's epilogue fused into it (one launch less a layer, no Z round trip); TF_GLM_EXL3_FUSE: 1 (default) the
    down epilogue, 2 gate/up's too (the last of an item's blocks runs it: measured slower), 0 neither;
    TF_GLM_EXL3_XROW (default on): a layer whose experts share one gate/up suh rotates each row's input once
    (rot_rows: the same fp16 inputs). Every setting gives the same bits; TF_GLM_EXL3_DEC=0: the grouped kernel and
    its epilogue kernels."""
    on = os.environ.get("TF_GLM_EXL3_DEC", "1") != "0"
    fuse = int(os.environ.get("TF_GLM_EXL3_FUSE", "1"))
    if fuse not in (0, 1, 2, 3):
        raise ValueError(f"TF_GLM_EXL3_FUSE is 0 to 3, not {fuse}")
    return on, fuse, os.environ.get("TF_GLM_EXL3_XROW", "1") != "0"


LOADS = {"0": 0, "ldg": 0, "1": 1, "nc": 1, "nc1": 1, "nc2": 2, "nc4": 3}


def dec_loads() -> int:
    """TF_GLM_EXL3_LOADS: the decode kernel's trellis loads (the same bits for every value). ``0`` / ``ldg``
    (default): 32-bit loads one k step ahead; ``nc`` / ``nc1``, ``nc2``, ``nc4``: 16-byte ld.global.nc loads 1, 2 or
    4 k steps ahead through a staging area, the item read with its count in one round trip and the first steps'
    words issued before the member rows arrive (after jayleaton/glm53-tensorfold-spark patches/0580, Apache-2.0)."""

    v = (os.environ.get("TF_GLM_EXL3_LOADS", "") or "0").strip().lower()
    if v not in LOADS:
        raise ValueError(f"TF_GLM_EXL3_LOADS: 0, nc, nc2 or nc4, not {v!r}")
    return LOADS[v]


def _seq() -> bool:
    """TF_GLM_EXL3_SEQ (default on): prompt chunks sum a matmul's K splits in one program (same bits)."""
    import os

    return os.environ.get("TF_GLM_EXL3_SEQ", "1") != "0"


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_exl3_v21", sources=[str(here / "exl3.cpp"), str(here / "exl3.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def dec_config(kind: str, K: int, N: int, R: int) -> dict:
    """The decode kernel's launch for gate/up (``kind`` "gu": K = D, N = NI) or down ("dn": K = NI, N = D) at R rows:
    the environment's (TF_GLM_EXL3_LOADS / _FUSE / _XROW, one warp along N), then a launch table's entry for the shape
    and rows (TF_GLM_TUNE, ``tune``). Every value gives the same bits (``dec_kernel``'s notes)."""

    _, fuse, xrow = dec_settings()
    if kind == "gu":
        cfg = {"ld": dec_loads(), "wn": 1, "fuse": 2 if fuse & 2 else 0, "xrow": int(xrow)}
    elif kind == "dn":
        cfg = {"ld": dec_loads(), "wn": 1, "fuse": 1 if fuse & 1 else 0}
    else:
        raise ValueError(f"dec_config: gu or dn, not {kind!r}")
    t = tune.pick("exl3_dec_" + kind, tune.shape(K, N), R)
    if t:
        cfg.update(t)
    if cfg["wn"] == 4 and cfg["ld"] != 0:            # (tune refuses such an entry; a merge must not reach the kernel)
        cfg["wn"] = 1
    return cfg


def dec_gateup(x: torch.Tensor, pick: torch.Tensor, plan: grouped.Plan, ex: "Exl3Experts", s: "Scratch", R: int,
               limit: float, cfg: dict, items: int | None = None) -> None:
    """Decode windows' gate/up: the input rotation, ``dec`` on gate and up (4 K splits) and the SwiGLU / down
    rotation epilogue (fused into the last block, or its own kernel) -> s.xd, under launch ``cfg`` (``dec_config``)."""

    ext = _ext()
    slots, P = s.slots, s.rows * s.slots
    D, NI = ex.dims, ex.width
    if items is None:
        items = grouped.max_items(R * slots, plan.experts)
    xrow = bool(cfg["xrow"]) and ex.shared_suh is not None
    if xrow:                                         # one rotated input a row, read by all its pairs' gate and up
        ext.rot_rows(x, x.stride(0), ex.shared_suh, s.xg, R, D)
        xg = xu = s.xg
    else:
        ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
        xg, xu = s.xg, s.xu
    fused = 2 if int(cfg["fuse"]) == 2 else 0
    ext.dec(xg, xu, ex.gt, ex.ut, plan.items, plan.counts, plan.members, s.z, 2, D, NI, P, 4, items, slots,
            fused, xrow, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, float(limit), s.done, int(cfg["ld"]), int(cfg["wn"]))
    if not fused:
        ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, NI, 4, slots, float(limit))


def dec_down(pick: torch.Tensor, plan: grouped.Plan, ex: "Exl3Experts", s: "Scratch", y: torch.Tensor, R: int,
             cfg: dict, items: int | None = None) -> None:
    """Decode windows' down projection from s.xd -> y (fp32 [pairs, D]) under launch ``cfg`` (``dec_config``)."""

    ext = _ext()
    slots, P = s.slots, s.rows * s.slots
    D, NI = ex.dims, ex.width
    if items is None:
        items = grouped.max_items(R * slots, plan.experts)
    fused = 1 if int(cfg["fuse"]) == 1 else 0
    ext.dec(s.xd, s.xd, ex.dt, ex.dt, plan.items, plan.counts, plan.members, s.z, 1, NI, D, P, 1, items, slots,
            fused, False, ex.svh_d, ex.svh_d, ex.svh_d, y, 0.0, s.done, int(cfg["ld"]), int(cfg["wn"]))
    if not fused:
        ext.down_epilogue(s.z, pick, ex.svh_d, y, R, P, D, 1, slots)


@dataclass
class Exl3Experts:
    """One layer's routed experts on this rank: trellis words [E, K/16, N/16, 32] int32 and scales per expert."""

    gt: torch.Tensor          # gate trellis [E, D/16, NI/16, 32]
    ut: torch.Tensor          # up trellis
    dt: torch.Tensor          # down trellis [E, NI/16, D/16, 32]
    suh_g: torch.Tensor       # [E, D] fp16
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # [E, NI] fp16 (this rank's columns)
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # [E, NI] fp16 (this rank's rows of down)
    svh_d: torch.Tensor       # [E, D]
    count: int
    width: int                # NI: this rank's share of the expert width
    dims: int                 # D
    # [D] fp16 when every expert's gate and up share one suh bitwise (found at construction; GLM-5.3-Flash's
    # EXL3 checkpoint, every MoE layer): prompt chunks then rotate each row's input once (``rot_rows``)
    shared_suh: torch.Tensor | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.shared_suh = shared_suh(self.suh_g, self.suh_u)

    def nbytes(self) -> int:
        extra = () if self.shared_suh is None else (self.shared_suh,)
        return sum(t.numel() * t.element_size() for t in (self.gt, self.ut, self.dt, self.suh_g, self.suh_u,
                                                         self.svh_g, self.svh_u, self.suh_d, self.svh_d, *extra))


def shared_suh(suh_g: torch.Tensor, suh_u: torch.Tensor) -> torch.Tensor | None:
    """The one input scale vector [D] if every expert's gate and up suh equal it bit for bit (-0 is not +0), else
    None. Every pair's rotated input x_row * suh @ H is then the row's alone, whatever expert or matrix it feeds."""

    if suh_g.dim() != 2 or suh_g.shape != suh_u.shape or suh_g.dtype != torch.float16 or suh_u.dtype != suh_g.dtype:
        return None
    g, u = suh_g.view(torch.int16), suh_u.view(torch.int16)
    if not (bool(torch.equal(g, g[:1].expand_as(g))) and bool(torch.equal(u, g))):
        return None
    return suh_g[0].clone().contiguous()


def words(trellis: torch.Tensor) -> torch.Tensor:
    """A trellis (int16 [..., 64], 4 bits) as the kernels read it: int32 [..., 32], the same bytes."""

    if trellis.dtype != torch.int16 or trellis.shape[-1] not in (48, 64):
        raise ValueError("only 3-bit or 4-bit EXL3 trellises (int16 [..., 48] / [..., 64]) are supported")
    return trellis.contiguous().view(torch.int32)


def prompt_pass() -> int:
    """TF_GLM_EXL3_PASS: members a prompt kernel pass takes (64 or 128, the same bits; 128 measured slower on real
    routing: most passes are small, and its one 16-warp block an SM keeps fewer experts in flight). A launch table's
    ``exl3_prompt`` entry "all" (TF_GLM_TUNE) takes precedence."""
    import os

    t = tune.pick("exl3_prompt", "all")
    value = int(t["passm"]) if t else int(os.environ.get("TF_GLM_EXL3_PASS", "64"))
    if value not in (64, 128):
        raise ValueError(f"TF_GLM_EXL3_PASS is 64 or 128, not {value}")
    return value


@lru_cache(maxsize=1)
def _dump_counts_path() -> str:
    """TF_GLM_DUMP_COUNTS=<path>: a diagnostic (off by default). Every prompt window's routed call appends one line
    to <path>.<pid>: its rows, then each expert's member count (the routing histogram the expert kernels see;
    tests/K13/bench_k13.py --replay runs those calls). It syncs the GPU each call: for one-off dumps only."""

    return os.environ.get("TF_GLM_DUMP_COUNTS", "")


def _dump_counts(pick: torch.Tensor, R: int, E: int) -> None:
    counts = torch.bincount(pick[:R].reshape(-1).long(), minlength=E + 1)[:E].tolist()   # the shared id E left out
    with open(f"{_dump_counts_path()}.{os.getpid()}", "a") as f:
        f.write(f"{R} " + " ".join(map(str, counts)) + "\n")


@lru_cache(maxsize=1)
def _k12_config():
    """TF_GLM_EXPERT_PROMPT_KERNEL's K12 configuration (exl3_k12.setting), read once a process."""

    from . import exl3_k12

    return exl3_k12.setting()


def prompt_stages() -> int:
    """The prompt kernels' cp.async pipeline depth: 3 (GB10's), or a launch table's ``exl3_prompt`` entry "all"
    (TF_GLM_TUNE): 2 to 4, loads only, the same bits."""

    t = tune.pick("exl3_prompt", "all")
    return int(t.get("stages", 3)) if t else 3


def prompt_kernels() -> bool:
    """TF_GLM_EXL3_PROMPT (default on): prompt chunks run the Y^T = W^T X^T kernels on a 64-pair plan (one fp32
    chain a row: other bits than the decode kernels', independent of the chunking); 0: the decode kernels."""
    import os

    return os.environ.get("TF_GLM_EXL3_PROMPT", "1") != "0"


def item_order() -> bool:
    """TF_GLM_EXL3_ORDER (default on): prompt kernels run the items of single-item experts (DRAM-bound) and of
    multi-item experts (mma-bound, weights shared through L2) interleaved; 0: plan order. The same bits either way."""
    import os

    return os.environ.get("TF_GLM_EXL3_ORDER", "1") != "0"


def row_rotation() -> bool:
    """TF_GLM_EXL3_ROWROT (default on): prompt chunks of layers whose experts share one gate/up suh rotate each row's
    input once and load it once for gate and up (the same fp16 inputs, so the same bits); 0: per pair, as decode."""
    import os

    return os.environ.get("TF_GLM_EXL3_ROWROT", "1") != "0"


class Scratch:
    """Per-window buffers for up to ``rows`` rows of ``slots`` slots (the last slot is the shared expert's); a
    prompt window's (``prompt``) kernels keep no split partials."""

    def __init__(self, rows: int, slots: int, dims: int, width: int, device, *, prompt: bool = False) -> None:
        P = rows * slots
        self.xg = torch.zeros((P, dims), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, dims), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, width), dtype=torch.float16, device=device)
        # a decode window's fp32 partials: gate/up's [2][splits][P][width] or down's [splits][P][dims] (the decode
        # and grouped kernels), or the streamed kernel's fine slabs' (exl3_stream.fine_room); scratch only
        from . import exl3_stream

        zn = 1 if prompt else max(P * max(2 * GATEUP_CFG[2] * width, DOWN_CFG[2] * dims),
                                  exl3_stream.fine_room(rows, slots, dims, width))
        self.z = torch.zeros((zn,), dtype=torch.float32, device=device)
        # decode windows' fused gate/up epilogue: a finished-block count per (item, 128 columns), reset by the kernel
        self.done = torch.zeros((1 if prompt else (P + P // grouped.TILE + 1) * max(1, width // 128),),
                                dtype=torch.int32, device=device)
        self.rows, self.slots = rows, slots
        if prompt and _k12_config() is not None:     # TF_GLM_EXPERT_PROMPT_KERNEL read (and checked) at start
            from . import exl3_k12
            exl3_k12.note("K12 prompt expert kernels (exact: the prompt kernels' bits)")
        # a prompt plan's launch order (item_order): room for any plan of up to 1,024 experts in items of >= 16 pairs
        self.order = torch.zeros((P // 16 + 1024 if prompt else 1,), dtype=torch.int32, device=device)
        # decode windows' streamed kernel (TF_GLM_EXL3_STREAM, exl3_stream): its work queue, counts and flags
        from . import exl3_stream

        self.stream = None if prompt else exl3_stream.state(rows, slots, dims, device)


@lru_cache(maxsize=1)
def _k1_config() -> tuple[int, bool] | None:
    """TF_GLM_EXPERT_PROMPT_KERNEL (exl3_k1.py): the K1 prompt kernels' (configuration, int8 gate/up), or None (off,
    the default)."""

    from . import exl3_k1

    return exl3_k1.setting()


def routed(x: torch.Tensor, pick: torch.Tensor, plan: grouped.Plan, ex: Exl3Experts, s: Scratch, y: torch.Tensor,
           R: int, limit: float) -> None:
    """Y[row * slots + slot] (fp32) for rows 0..R-1's routed slots, from normed bf16 x and ``experts.route``'s plan."""

    ext = _ext()
    slots, P = s.slots, s.rows * s.slots
    D, NI = ex.dims, ex.width
    if plan.tile in (64, 128):                       # a prompt window: the Y^T kernels, an item a pass
        if _dump_counts_path():                      # TF_GLM_DUMP_COUNTS: the call's per-expert member counts
            _dump_counts(pick, R, ex.count)
        shx = ex.shared_suh is not None and row_rotation()
        k1 = _k1_config() if shx else None
        if k1 is not None:                           # TF_GLM_EXPERT_PROMPT_KERNEL (exl3_k1.py)
            from . import exl3_k1

            if exl3_k1.supported(ex, D, NI):
                exl3_k1.routed(x, plan, ex, s, y, R, limit, k1)
                return
        k12 = _k12_config()
        if k12 is not None:                          # TF_GLM_EXPERT_PROMPT_KERNEL=k12... (exl3_k12.py): the same bits
            from . import exl3_k12
            why = exl3_k12.unused_reason(ex, D, NI, shx, plan.tile)
            if why is None:
                exl3_k12.routed(x, plan, ex, s, y, R, limit, k12)
                return
            exl3_k12.note(why)
        if shx:                                      # one rotated input a row, read by all its pairs' gate and up
            ext.rot_rows(x, x.stride(0), ex.shared_suh, s.xg, R, D)
            xg = xu = s.xg
        else:
            ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
            xg, xu = s.xg, s.xu
        ext.prompt(xg, xu, ex.gt, ex.ut, ex.dt, plan.items, plan.counts, plan.members, ex.svh_g, ex.svh_u,
                   ex.suh_d, ex.svh_d, s.xd, y, D, NI, ex.count, grouped.max_items(R * slots, plan.experts,
                                                                                   plan.tile), float(limit),
                   plan.tile, slots, shx, s.order if item_order() else None, prompt_stages())
        return
    if plan.tile != grouped.TILE:
        raise ValueError("the EXL3 kernel takes items of 16 pairs")
    items = grouped.max_items(R * slots, plan.experts)
    dec, _, _ = dec_settings()
    if dec and R < PROMPT_ROWS and GATEUP_CFG == (8, 4, 4) and DOWN_CFG == (8, 4, 1) and D == 4096 and NI in DEC_WIDTHS:
        dec_gateup(x, pick, plan, ex, s, R, limit, dec_config("gu", D, NI, R), items)
        dec_down(pick, plan, ex, s, y, R, dec_config("dn", NI, D, R), items)
        return
    ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
    nfirst = R >= PROMPT_ROWS and _nfirst()
    nt, w, sk = GATEUP_CFG
    seq = nfirst and sk > 1 and _seq()        # the splits summed in one program: an SK = 1 epilogue, the same bits
    ext.grouped(s.xg, s.xu, ex.gt, ex.ut, plan.items, plan.counts, plan.members, s.z, 2, D, NI, P, sk, items, nt,
                w, nfirst, _pf(), seq)
    ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, NI, 1 if seq else sk, slots,
                        float(limit))
    nt, w, sk = DOWN_CFG
    ext.grouped(s.xd, s.xd, ex.dt, ex.dt, plan.items, plan.counts, plan.members, s.z, 1, NI, D, P, sk, items, nt,
                w, nfirst, _pf(), nfirst and sk > 1 and _seq())
    ext.down_epilogue(s.z, pick, ex.svh_d, y, R, P, D, sk, slots)
