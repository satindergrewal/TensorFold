"""TF_GLM_KDA_SPLIT (patch 0160): the chunked KDA prompt recurrence (TF_GLM_KDA_CHUNKED=1, kda_chunk.cu) on the whole
GPU, with kda_chunk.cu's bits (kda_split.cu).

kda_chunk.cu runs one block of 256 threads a head through a prompt chunk's 32-row sub-chunks in order, its 128 x 128
state in registers (255 a thread): 16 heads a rank keep 16 SMs busy. kda_split.cu computes the same values in three
kernels: every sub-chunk's state-independent work at once (block (sub-chunk, head): the conv, norms, decays, beta,
L, M, T, decayed keys and queries; kda_chunk.cu's code for it), then the sequential part split over the value
columns (block (head, value piece): read sums, V', read-outs and the state update for a piece's columns), then the
gated RMSNorm (a warp a row and head). Every element keeps kda_chunk.cu's operations, order and roundings, so the
setting changes speed only (tests/K2 checks the bits on the CPU, gpu_test.sh against kda_chunk.cu on the GPU).

TF_GLM_KDA_SPLIT: 0 (off, the default); ``auto`` (or 1): pieces of 16 value columns (8 a head, 256 threads) when the
heads' pieces fit the GPU's SMs (16 heads: 128 blocks on 188 SMs; the fastest shape measured on RTX PRO 6000 at
1,024-4,096 rows), else 32 (4 a head); or a shape ``COLS[:THREADS]`` from SHAPES (more threads: the same work a block
in more warps; 32:512 holds half the SMs of 16:256 for a similar time, for runs beside other streams). Every shape
gives the same bits.
TF_GLM_KDA_SPLIT_PREP: sub-chunks a prep block handles in turn (default 0: as few as fill the SMs in one wave).
TF_GLM_KDA_SPLIT_ROWS: rows a pass covers (default 2,048, a multiple of 32): the records of a pass (~66 KB a 32-row
sub-chunk and head: ~68 MB at 2,048 rows and 16 heads, allocated at the first prompt chunk) are written by the first
kernel and read by the second; longer chunks take several passes (the same bits).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

DV = 128
CHUNK = 32                      # rows of a sub-chunk (kda_chunk.cu's BT)
# value columns a piece -> threads a state block (the first: the default for those columns); kda_split.cu's dispatch
SHAPES = {12: (192, 384), 16: (256, 128, 512), 32: (256, 128, 512), 8: (128,)}


def setting(value: str | None = None) -> tuple[int, int] | None:
    """TF_GLM_KDA_SPLIT as (columns a piece, threads), (0, 0) for auto, or None (off)."""

    value = (os.environ.get("TF_GLM_KDA_SPLIT", "") if value is None else value).strip().lower() or "0"
    if value == "0":
        return None
    if value in ("1", "auto"):
        return (0, 0)
    cols, colon, nt = value.partition(":")
    if cols.isdecimal() and int(cols) in SHAPES and (not colon or (nt.isdecimal() and int(nt) in SHAPES[int(cols)])):
        return int(cols), int(nt) if colon else SHAPES[int(cols)][0]
    shapes = ", ".join(f"{c}:{t}" for c, ts in SHAPES.items() for t in ts)
    raise ValueError(f"TF_GLM_KDA_SPLIT: 0 (off), auto (1), or columns[:threads] from {shapes}, not {value!r}")


def prep_chunks(value: str | None = None) -> int:
    """TF_GLM_KDA_SPLIT_PREP: sub-chunks a prep block handles in turn (0, the default: as few as fill the SMs in one
    wave, e.g. 6 for 64 sub-chunks of 16 heads on 188 SMs; 1: a block a sub-chunk)."""

    value = (os.environ.get("TF_GLM_KDA_SPLIT_PREP", "") if value is None else value).strip() or "0"
    if not value.isdecimal() or int(value) > 1024:
        raise ValueError(f"TF_GLM_KDA_SPLIT_PREP: sub-chunks a prep block, 0 (one wave) to 1,024, not {value!r}")
    return int(value)


def window_rows(value: str | None = None) -> int:
    value = (os.environ.get("TF_GLM_KDA_SPLIT_ROWS", "") if value is None else value).strip() or "2048"
    if not value.isdecimal() or int(value) % CHUNK or not CHUNK <= int(value) <= 65536:
        raise ValueError(f"TF_GLM_KDA_SPLIT_ROWS: a multiple of 32 from 32 to 65,536, not {value!r}")
    return int(value)


SPLIT = setting()
WINDOW = window_rows() // CHUNK          # sub-chunks a pass
PREP = prep_chunks()


def pieces(cols: int) -> int:
    return -(-DV // cols)


def auto_shape(heads: int, device) -> tuple[int, int]:
    """The shape ``auto`` takes for ``heads`` heads on ``device``: 16 columns a piece (256 threads) when heads x 8
    state blocks fit the SMs (one block an SM), else 32 (256 threads)."""

    device = torch.device(device)
    sms = _sms(device.index if device.index is not None else torch.cuda.current_device())
    return (16, 256) if heads * pieces(16) <= sms else (32, 256)


def shape(heads: int, device) -> tuple[int, int]:
    """(columns a piece, threads) for ``heads`` heads on ``device``: the setting's, or ``auto_shape``."""

    if SPLIT is None:
        raise RuntimeError("TF_GLM_KDA_SPLIT is off")
    return SPLIT if SPLIT[0] else auto_shape(heads, device)


@lru_cache(maxsize=None)
def _sms(index: int) -> int:
    return torch.cuda.get_device_properties(index).multi_processor_count


def describe() -> str:
    how = "auto: pieces of 16 value columns when their state blocks fit the SMs, else 32" if SPLIT == (0, 0) else \
        f"pieces of {SPLIT[0]} value columns, {SPLIT[1]} threads a state block"
    prep = "one wave" if not PREP else f"{PREP} sub-chunks a block"
    return (f"TF_GLM_KDA_SPLIT: the KDA prompt recurrence on the whole GPU ({how}; prep {prep}; passes of "
            f"{WINDOW * CHUNK} rows), kda_chunk.cu's bits")


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_glm_kda_split", sources=[str(here / "kda_split.cpp"), str(here / "kda_split.cu")],
                extra_cuda_cflags=["-O3", "--fmad=false"], verbose=False)


def prepare() -> None:
    """Build the extension now (engine start) rather than at the first prompt chunk."""

    _ext()


_records: dict = {}


def records(heads: int, device) -> torch.Tensor:
    """The pass records for ``heads`` heads on ``device`` (allocated once; prompt chunks run one at a time a device,
    on one stream: the engine's main stream or, under TF_GLM_KDA_OVERLAP, the recurrence stream)."""

    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (heads, device)
    r = _records.get(key)
    if r is None:
        r = _records[key] = torch.empty(WINDOW * heads * int(_ext().record_bytes()), dtype=torch.uint8,
                                        device=device)
    return r


def chain(p: torch.Tensor, b_off: int, a: torch.Tensor, g: torch.Tensor, conv_state: torch.Tensor,
          conv_w: torch.Tensor, state_in: torch.Tensor, a_log: torch.Tensor, dt_bias: torch.Tensor,
          norm_w: torch.Tensor, eps: float, lower: float, rows: int, out: torch.Tensor, state_out: torch.Tensor,
          pos: int = 0, *, cols: int | None = None, nt: int | None = None, window: int | None = None,
          prep: int | None = None) -> torch.Tensor:
    """kda_chunked.chain's call and bits (kda_chunk.cu's), split: projection rows p, gate rows a and g, conv state
    [3, 3 H 128] (read), state_in -> state_out [H, 128, 128], ``pos`` the first row's absolute position. Writes
    out[:rows] (bf16 [rows, H * 128]). ``cols`` / ``nt`` / ``window`` (sub-chunks) / ``prep`` (sub-chunks a prep
    block) override the settings (tests)."""

    rows = int(rows)
    H = a_log.numel()
    if cols is None:
        cols, nt = shape(H, p.device)
    window = int(window or WINDOW)
    recs = records(H, p.device)
    if recs.numel() < window * H * int(_ext().record_bytes()):          # a test's larger window
        recs = torch.empty(window * H * int(_ext().record_bytes()), dtype=torch.uint8, device=p.device)
    _ext().chain(p, p.stride(0), int(b_off), a, a.stride(0), g, g.stride(0), conv_state, conv_w, state_in, a_log,
                 dt_bias, norm_w, float(eps), float(lower), rows, int(pos) % CHUNK, out, state_out, recs, int(cols),
                 int(nt or SHAPES[int(cols)][0]), window, int(PREP if prep is None else prep))
    return out[:rows]
