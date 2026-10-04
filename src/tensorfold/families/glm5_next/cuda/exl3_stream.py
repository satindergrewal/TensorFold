"""Routed EXL3 experts of decode windows streamed slab by slab (TF_GLM_EXL3_STREAM): the decode kernel's bits.

A verify window's routed experts are mostly a read of their trellis: 3 MB an expert on a rank at TP4, 8 experts at one
row (25 MB a layer) and 100-240 at 16-64 rows (0.3-0.75 GB a layer). The decode kernel (``exl3_mm.dec_*``, exl3.cu
``dec_kernel``) gives every block an (item, 128 columns, K split) and its warps read 1 KB of each k row: thousands of
short strided streams; the box's launch-table harness measured it at 36% of the card's bandwidth at one row and 51-62%
at 2-63 rows. Here (exl3s.cu) ONE persistent launch a layer streams each routed expert's trellis once in long
contiguous runs: a block takes a slab, whole 4 KB k rows of one expert matrix (coarse: the 64 rows of one K split of
gate or up, 256 KB back to back, or a 4 KB column slice of every k row of down; fine, for small windows: one 16- or
8-row chain of those, 4x the slabs so every multiprocessor has work), brings it in through a shared-memory pipeline of
16-byte cp.async copies several stages deep and decodes it with the decode kernel's arithmetic. The last gate/up slab
of an item runs the item's gate/up epilogue; the item's down slabs wait for it; the last block out resets the work
queue, so CUDA graphs replay it.

Settings (read once a process; every rank must give the same values, the engine compares them at start):

- ``TF_GLM_EXL3_STREAM``: ``0`` / ``off`` / unset (default): the decode kernels as before; ``on``: decode windows of
  ``DEFAULT_ROWS`` rows and more; a number N (1 to 256): windows of N rows and more (``1``: every decode window).
- ``TF_GLM_EXL3_STREAM_SLICE``: the column slice a block streams of every k row by window rows, ``rows:slice`` pairs
  (slice ``1k``, ``2k`` or ``4k``: blocks of 4, 8 or 16 warps, four, two or one a multiprocessor); a window takes the
  last pair whose rows it reaches; default ``DEFAULT_SLICE``.
- ``TF_GLM_EXL3_STREAM_FINE``: windows of fewer rows than this take the fine slabs (default ``DEFAULT_FINE``: never);
  where the partials fit the window's scratch.
- ``TF_GLM_EXL3_STREAM_PIPE``: rows a stage x stages, ``4x3`` (default), ``4x4``, ``2x4`` or ``2x6``.
- ``TF_GLM_EXL3_STREAM_BLOCKS``: the persistent grid (default 0: as many blocks as fit at once).
- ``TF_GLM_EXL3_STREAM_PDL``: ``1`` launches it as a programmatic dependent launch (its blocks wait for the previous
  launch on the device instead of after it); ``0`` (default): a plain launch.
- ``TF_GLM_L2PF_EXPERT_MB``: with the L2 prefetcher on (TF_GLM_L2PF), right after a decode window's routing its side
  stream brings up to this many MiB of the routed experts' trellis into L2 (in the order the experts are read: gate
  and up item by item, then down; an expert once) while the main stream runs the shared expert and the input
  rotation (default 0: off). Independent of TF_GLM_EXL3_STREAM (the decode kernels read the same bytes). It only reads.

Exactness: every output is the decode kernel's float chain (each warp's mma chain over the decode kernel's K range of
that warp, from zero; the chains added in the decode kernel's warp order, in registers (coarse) or by the epilogue from
the chains' partials (fine): the same fp32 adds in the same order; the splits added from zero in split order by the
same epilogue arithmetic); the inputs and fragments are the same values and mma keeps rows independent. Only data
movement and which block computes what change, and no choice depends on what else is in the window: a window of any
rows gives each row the bits it gets alone, and every choice by rows (which kernel, which slabs) changes nothing.
Windows of 64-256 rows (wide windows) get the grouped kernel's bits, which are the decode kernel's. The first call
outside a CUDA graph capture checks it on this GPU against the engine's own path, byte for byte, on the call's own
weights and input form (random inputs, several row counts and routings, both slab sizes); any difference turns it off
for the process (a line says so) and the decode kernels run, as they do in a capture before the check. The GPU tests
(tests/K3) cover the other forms (per-pair inputs, two ranks' widths, every row count to 256, graph replays).

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and on
the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab (the EXL3 decode kernels this reproduces; the trellis
format after ExLlamaV3, MIT, Copyright (c) 2025 Turboderp)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

PIPES = {"4x3": (4, 3), "4x4": (4, 4), "2x4": (2, 4), "2x6": (2, 6)}
DEFAULT_PIPE = "4x3"        # the box's best for 4 KB slices at 3-256 rows (K35's timing table)
SLICES = {"1k": 4, "2k": 8, "4k": 16}                    # slice -> warps a block
DEFAULT_SLICE = "1:1k,160:4k"
DEFAULT_ROWS = 1            # ``on``: every decode window (the README's timing table may set a floor)
DEFAULT_FINE = 0            # windows below this many rows take the fine slabs (0: none)
MOST_ROWS = 256             # the widest decode window it takes (wide windows' scratch: 128, 192, 256 rows)
CHECK_ROWS = (1, 5, 16, 33, 63, 64, 128, 256)
CHECK_SEEDS = 2


@dataclass(frozen=True)
class Settings:
    rows: int                # 0: off; else decode windows of at least this many rows take the streamed kernel
    fine: int                # windows below this many rows take the fine slabs
    rs: int                  # k rows a pipeline stage
    stages: int
    blocks: int              # persistent grid (0: as many as fit)
    pdl: bool
    prefetch_mb: int         # TF_GLM_L2PF_EXPERT_MB
    slices: tuple = ((1, 16),)   # (from rows, warps a block), rows ascending, the first from 1

    def code(self) -> list[int]:
        return [self.rows, self.fine, self.rs * 16 + self.stages, self.blocks, int(self.pdl), self.prefetch_mb,
                *[v for r, w in self.slices for v in (r, w)]]

    def warps(self, R: int) -> int:
        """A block's warps for a window of R rows (the last slice whose rows R reaches)."""
        out = self.slices[0][1]
        for r, w in self.slices:
            if R >= r:
                out = w
        return out


_settings: Settings | None = None


def read(env=None) -> Settings:
    """The settings from the environment (ValueError on a value that is not valid)."""

    env = os.environ if env is None else env

    def number(name: str, default: int, most: int) -> int:
        v = (env.get(name, "") or str(default)).strip()
        if not v.isdigit() or int(v) > most:
            raise ValueError(f"{name}: 0 to {most}, not {v!r}")
        return int(v)

    def flag(name: str, default: str) -> bool:
        v = (env.get(name, "") or default).strip().lower()
        if v not in ("0", "1", "on", "off"):
            raise ValueError(f"{name}: 1 or 0, not {v!r}")
        return v in ("1", "on")

    v = (env.get("TF_GLM_EXL3_STREAM", "") or "0").strip().lower()
    if v in ("0", "off", "no", "false"):
        rows = 0
    elif v in ("on", "yes", "true"):
        rows = DEFAULT_ROWS
    elif v.isdigit() and 1 <= int(v) <= MOST_ROWS:
        rows = int(v)                            # a number is always the fewest rows ("1": every decode window)
    else:
        raise ValueError(f"TF_GLM_EXL3_STREAM: off, on or the fewest rows a window takes it (1 to {MOST_ROWS}), "
                         f"not {v!r}")
    p = (env.get("TF_GLM_EXL3_STREAM_PIPE", "") or DEFAULT_PIPE).strip().lower()
    if p not in PIPES:
        raise ValueError(f"TF_GLM_EXL3_STREAM_PIPE: {', '.join(PIPES)}, not {p!r}")
    rs, stages = PIPES[p]
    return Settings(rows, number("TF_GLM_EXL3_STREAM_FINE", DEFAULT_FINE, MOST_ROWS + 1), rs, stages,
                    number("TF_GLM_EXL3_STREAM_BLOCKS", 0, 1 << 20), flag("TF_GLM_EXL3_STREAM_PDL", "0"),
                    number("TF_GLM_L2PF_EXPERT_MB", 0, 1024), slices(env.get("TF_GLM_EXL3_STREAM_SLICE", "")))


def slices(value: str | None) -> tuple:
    """TF_GLM_EXL3_STREAM_SLICE: ``rows:slice`` pairs, rows ascending from 1 (``1k`` alone: every window)."""

    v = (value or DEFAULT_SLICE).strip().lower()
    out = []
    for part in v.split(","):
        part = part.strip()
        r, _, sl = part.rpartition(":") if ":" in part else ("1", "", part)
        if not r.strip().isdigit() or sl.strip() not in SLICES:
            raise ValueError(f"TF_GLM_EXL3_STREAM_SLICE: rows:slice pairs (slice 1k, 2k or 4k), not {v!r}")
        out.append((int(r), SLICES[sl.strip()]))
    if not out or out[0][0] != 1 or any(b[0] <= a[0] for a, b in zip(out, out[1:])) or out[-1][0] > MOST_ROWS:
        raise ValueError(f"TF_GLM_EXL3_STREAM_SLICE: rows from 1, ascending, at most {MOST_ROWS}, not {v!r}")
    return tuple(out)


def settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = read()
    return _settings


def code() -> int:
    """The settings as one int for the engine's start comparison (a CRC of every value; ValueError when not valid)."""

    import zlib

    return zlib.crc32(",".join(map(str, settings().code())).encode()) & 0x3FFFFFFF


_decided: bool | None = None
_why: str | None = None


def reset(value: str | None = None, **more: str) -> None:
    """Forget the settings and the check (tests): the environment is read again, ``value`` / ``PIPE=..`` set first."""

    global _settings, _decided, _why
    if value is not None:
        os.environ["TF_GLM_EXL3_STREAM"] = value
    for k, v in more.items():
        os.environ["TF_GLM_L2PF_EXPERT_MB" if k == "prefetch_mb" else f"TF_GLM_EXL3_STREAM_{k.upper()}"] = v
    _settings, _decided, _why = None, None, None


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_exl3s_v3", sources=[str(here / "exl3s.cpp"), str(here / "exl3s.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def capacity(rows: int, slots: int) -> int:
    """The most items a decode plan of ``rows`` x ``slots`` pairs can hold (``grouped.max_items``, any experts)."""

    pairs = rows * slots
    return pairs + pairs // 16


def state(rows: int, slots: int, dims: int, device) -> torch.Tensor:
    """The kernel's queue, counts and flags for windows of up to ``rows`` rows (model width ``dims``): zeros, and
    zeros again after each launch."""

    return torch.zeros((2 + (2 + max(1, dims // 128)) * capacity(rows, slots),), dtype=torch.int32, device=device)


def fits(ex) -> bool:
    """Shapes the kernel takes: 4-bit trellis words, D and the rank's expert width multiples of 512, whole chains a
    pipeline stage, the decode kernels' K splits (GLM-5.3-Flash at two and four ranks: widths 1024 and 512)."""

    from . import exl3_mm

    st = settings()
    D, NI = ex.dims, ex.width
    if exl3_mm.GATEUP_CFG != (8, 4, 4) or exl3_mm.DOWN_CFG != (8, 4, 1):
        return False
    if D % 512 or NI % 512 or (D // 16) % 16 or (NI // 16) % 4:
        return False
    if ((D // 16) // 16) % st.rs or ((NI // 16) // 4) % st.rs:
        return False
    return (ex.gt.dtype == torch.int32 and ex.gt.dim() == 4 and tuple(ex.gt.shape[1:]) == (D // 16, NI // 16, 32)
            and ex.dt.dim() == 4 and tuple(ex.dt.shape[1:]) == (NI // 16, D // 16, 32))


_bypass = False             # the check's reference runs: the engine's path without this kernel


def wanted(R: int, ex, s=None) -> bool:
    """Whether a decode window of R rows asks for the streamed kernel (before its check)."""

    rows = settings().rows
    if _bypass or rows == 0 or R < rows or R > MOST_ROWS or not fits(ex):
        return False
    return s is None or (getattr(s, "stream", None) is not None and R <= s.rows)


def fine_room(rows: int, slots: int, dims: int, width: int) -> int:
    """fp32 partials the fine slabs of windows of up to ``rows`` rows take (``exl3_mm.Scratch`` holds at least
    these): every chain's gate/up and down partials of the widest window below TF_GLM_EXL3_STREAM_FINE; 0 when the
    streamed kernel is off or its settings are not valid (the engine's start check reports those)."""

    try:
        st = settings()
    except ValueError:
        return 0
    if st.rows == 0 or st.fine <= 1:
        return 0
    return min(rows, st.fine - 1) * slots * (32 * width + 4 * dims)


def fine_slabs(R: int, ex, s) -> bool:
    """Whether a window of R rows takes the fine slabs: fewer rows than TF_GLM_EXL3_STREAM_FINE and every chain's
    partials (gate/up and down) fit the scratch's Z."""

    P = R * s.slots
    return R < settings().fine and 32 * P * ex.width + 4 * P * ex.dims <= s.z.numel()


def prefetch_experts(pf, plan, ex, R: int) -> None:
    """TF_GLM_L2PF_EXPERT_MB: fork the L2 prefetcher's side stream (``pf``: l2pf.ACTIVE, which joins it before the
    forward returns) right after the routing plan, and bring the plan's routed experts' first MiB into L2 there."""

    from tensorfold.cuda import experts as grouped

    mb = settings().prefetch_mb
    side = getattr(pf, "side", None)
    if mb <= 0 or side is None or not fits(ex):
        return
    items = grouped.max_items(R * plan.slots, plan.experts)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _ext().prefetch(plan.items, plan.counts, ex.gt, ex.ut, ex.dt, items, mb << 20)
    pf.forked = True


def launch(x, pick, plan, ex, s, y, R: int, limit: float, items: int, fine: bool | None = None,
           warps: int | None = None) -> None:
    """``exl3_mm.routed``'s decode branch through the streamed kernel: Xd (s.xd) and Y for rows 0..R-1's routed pairs
    (``fine``: the slab size, by default by rows; ``warps``: a block's warps, by default the slice for R rows)."""

    from . import exl3_mm

    st = settings()
    ext = exl3_mm._ext()
    D, NI, slots = ex.dims, ex.width, s.slots
    xrow = ex.shared_suh is not None
    if xrow:                                     # one rotated input a row, read by all its pairs' gate and up
        ext.rot_rows(x, x.stride(0), ex.shared_suh, s.xg, R, D)
        x0 = x1 = s.xg
    else:
        ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots)
        x0, x1 = s.xg, s.xu
    if fine is None:
        fine = fine_slabs(R, ex, s)
    _ext().stream(x0, x1, ex.gt, ex.ut, ex.dt, plan.items, plan.counts, plan.members, ex.svh_g, ex.svh_u, ex.suh_d,
                  ex.svh_d, s.z, s.xd, y, s.stream, D, NI, R * slots, slots, items, float(limit), xrow, bool(fine),
                  st.rs, st.stages, st.blocks, st.pdl, int(warps or st.warps(R)))


def on(ex, s) -> bool:
    """Whether to run the streamed kernel now: asked for, and its check passed (run here on the first call outside a
    capture); inside a capture before the check: False."""

    global _decided, _why
    if _decided is not None:
        return _decided
    if torch.cuda.is_current_stream_capturing():
        return False
    try:
        bad = check(ex, s)
    except Exception as exc:  # noqa: BLE001 - a kernel that fails to build or run is not used
        bad = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200] if str(exc) else ''}"
    _decided, _why = bad is None, bad
    st = settings()
    if bad is None:
        sl = ", ".join(f"{k} from {r} rows" for r, w in st.slices for k, v in SLICES.items() if v == w)
        print(f"[tensorfold] streamed EXL3 experts: on for decode windows of {st.rows} to {MOST_ROWS} rows (slices "
              f"{sl}; fine slabs below {st.fine} rows, pipeline {st.rs}x{st.stages}, pdl {int(st.pdl)}; "
              "checked bit for bit against the decode kernels)", flush=True)
    else:
        print(f"[tensorfold] streamed EXL3 experts: off ({bad})", flush=True)
    return _decided


def check(ex, s, rows: tuple[int, ...] = CHECK_ROWS, seeds: int = CHECK_SEEDS) -> str | None:
    """The streamed kernel (both slab sizes where they fit) against the engine's own path (``exl3_mm.routed``
    without it: the decode kernel, the grouped kernel from 64 rows) on this GPU with ``ex``'s weights, in ``s``'s
    buffers (which the caller's window overwrites next): Y and Xd of every routed pair compared byte for byte. None,
    or what differed."""

    global _bypass
    from tensorfold.cuda import experts as grouped

    from . import exl3_mm

    E, D, slots = ex.count, ex.dims, s.slots
    K8 = slots - 1
    dev = ex.gt.device
    top = min(max(rows), s.rows, MOST_ROWS)
    plan = grouped.Plan(top, slots, E + 1, dev)
    pick = torch.full((top, slots), E, dtype=torch.int32, device=dev)
    ys = [torch.empty((top * slots, D), dtype=torch.float32, device=dev) for _ in range(2)]
    g = torch.Generator(device=dev)
    for R in sorted({min(r, top) for r in rows}):
        for seed in range(seeds):
            g.manual_seed(1000 * seed + R)
            x = (torch.randn((R, D), device=dev, generator=g) * (0.5 + 2.0 * seed)).to(torch.bfloat16)
            pool = E if seed % 2 == 0 else min(E, K8 + 3)     # odd seeds: few experts, items of 16 pairs and more
            pick.fill_(E)
            pick[:R, :K8] = torch.rand((R, pool), device=dev, generator=g).argsort(dim=1)[:, :K8].int()
            grouped.route(pick[:R].contiguous(), plan, plan.tile)
            items = grouped.max_items(R * slots, plan.experts)
            routed = torch.arange(R * slots, device=dev) % slots != slots - 1
            ref = None
            for mode in ("ref", "coarse", "fine"):
                if mode == "fine" and 32 * R * slots * ex.width + 4 * R * slots * D > s.z.numel():
                    continue
                ys[0 if mode == "ref" else 1].view(torch.uint8).fill_(0x7F)
                s.xd.view(torch.uint8).fill_(0x7F)
                if mode == "ref":
                    _bypass = True
                    try:
                        exl3_mm.routed(x, pick, plan, ex, s, ys[0], R, 10.0)
                    finally:
                        _bypass = False
                    ref = (ys[0][:R * slots][routed].clone(), s.xd[:R * slots][routed].clone())
                    continue
                launch(x, pick, plan, ex, s, ys[1], R, 10.0, items, fine=mode == "fine")
                got = (ys[1][:R * slots][routed], s.xd[:R * slots][routed])
                for name, a, b in (("Y", ref[0], got[0]), ("Xd", ref[1], got[1])):
                    if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                        n = (a.view(torch.int32 if a.dtype == torch.float32 else torch.int16) !=
                             b.view(torch.int32 if b.dtype == torch.float32 else torch.int16)).sum().item()
                        return (f"{name} differs from the engine's path at {R} rows ({mode} slabs, seed {seed}): "
                                f"{n} values")
    if s.stream is not None and torch.any(s.stream):
        return "the kernel left its queue state set"
    torch.cuda.synchronize()
    return None
