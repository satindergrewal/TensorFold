"""Keyed top-k draws and greedy picks on the GPU with the host rule's tokens (TENSORFOLD_GPU_SAMPLE).

The samplers all-gather every rank's top candidates (``top_k + MARGIN`` values and ids a row) and draw on the host
(``exact_sampling.choose_rows``, or the first of value descending, id ascending for greedy rows): a pageable copy of
the whole pack and numpy work every round. Here the GPU draws from the gathered pack where it lies (``gpu_sample.cu``)
and only the tokens come back, in one small pinned copy:

- exact on the GPU: the merge by (value descending, id ascending), the top-k cut, value / max(T, 1e-6), the keyed
  uniform (splitmix64 of seed, position and id), the min_p threshold and its comparisons;
- certified: the two decisions that go through log / exp (numpy's float64 functions on the host, CUDA's here, both
  within a few ulps but not always the same ulp): the argmax is taken only when the best score leads the next by more
  than 2^-39 x max(1, |scores|), the top_p cut only when no cumulative probability is within 2^-34 of top_p.
A row that misses a margin (in practice almost never: it needs a near tie at 1e-12) is drawn on the host by the host
rule from its own candidates, so every token equals today's for the same seed. Rows with top_k off (the nucleus rule)
and callers that need the draft probabilities keep the host path.

TENSORFOLD_GPU_SAMPLE unset or 1: on, after a check at the first call (random candidate packs, every certified token
compared with the host rule; any difference turns it off for the process, with a line); 0: off.

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors)."""

from __future__ import annotations

import math
import os
import struct
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

COLS = 9
MAXC = 1024                         # candidates a row the kernel takes (ranks x per-rank candidates)
_state: dict = {"decided": None}
_pinned: dict = {}


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name="tensorfold_gpu_sample_v2", sources=[str(here / "gpu_sample.cpp"), str(here / "gpu_sample.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def wanted() -> bool:
    v = (os.environ.get("TENSORFOLD_GPU_SAMPLE", "") or "1").strip().lower()
    if v not in ("0", "1", "on", "off"):
        raise ValueError(f"TENSORFOLD_GPU_SAMPLE: 1 (on) or 0 (off), not {v!r}")
    return v in ("1", "on")


def reset() -> None:
    """Forget the decision (tests)."""
    _state["decided"] = None


def _buf(name: str, shape: tuple, dtype, device=None) -> torch.Tensor:
    """A cached pinned host buffer (device None) or device buffer of at least ``shape``."""

    key = (name, dtype, str(device))
    t = _pinned.get(key)
    n = int(np.prod(shape))
    if t is None or t.numel() < n:
        t = torch.empty((max(n, 64),), dtype=dtype, device=device, pin_memory=device is None)
        _pinned[key] = t
    return t[:n].view(shape)


def table(specs: Sequence[tuple]) -> np.ndarray:
    """specs [(offset, n, sampling or None, position)] -> the kernel's int64 table [rows, COLS]."""

    rows = len(specs)
    t = np.zeros((rows, COLS), dtype=np.int64)
    d = np.zeros((rows, 3), dtype=np.float64)
    for i, (off, n, s, pos) in enumerate(specs):
        greedy = s is None or s.temperature <= 0
        t[i, 0], t[i, 1], t[i, 2] = int(off), int(n), 0 if greedy else 1
        if not greedy:
            t[i, 3] = int(s.top_k)
            t[i, 4] = np.uint64(int(s.seed) & 0xFFFFFFFFFFFFFFFF).view(np.int64)
            t[i, 5] = int(pos)
            d[i] = (max(float(s.temperature), 1e-6), float(s.top_p), float(s.min_log))
        else:
            d[i] = (1.0, 1.0, -math.inf)
    t[:, 6:9] = d.view(np.int64)
    return t


def _bits(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", float(x)))[0]


def table_parts(parts: Sequence[tuple]) -> np.ndarray:
    """``table`` for parts [(offset, rows, n, positions, sampling)] (row r of a part at offset + r x 2n), built a
    part at a time: the rows of a part share everything but their offsets and positions."""

    sizes = [int(R) for _, R, _, _, _ in parts]
    consts = np.zeros((len(parts), COLS), dtype=np.int64)
    offs, poss = [], []
    for j, (off, R, n, positions, s) in enumerate(parts):
        greedy = s is None or s.temperature <= 0
        if greedy:
            consts[j] = (0, n, 0, 0, 0, 0, _bits(1.0), _bits(1.0), _bits(-math.inf))
            poss.append(np.zeros(R, dtype=np.int64))
        else:
            seed = int(s.seed) & 0xFFFFFFFFFFFFFFFF
            consts[j] = (0, n, 1, int(s.top_k), seed - (1 << 64) if seed >= 1 << 63 else seed, 0,
                         _bits(max(float(s.temperature), 1e-6)), _bits(float(s.top_p)), _bits(float(s.min_log)))
            poss.append(np.asarray(list(positions)[:R], dtype=np.int64))
        offs.append(int(off) + 2 * int(n) * np.arange(R, dtype=np.int64))
    t = np.repeat(consts, sizes, axis=0)
    if len(t):
        t[:, 0] = np.concatenate(offs)
        t[:, 5] = np.concatenate(poss)
    return t


def run(got: torch.Tensor, rank_stride: int, world: int, specs: Sequence[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """The kernel on a gathered pack for rows ``specs`` [(offset, n, sampling, position)]: (tokens int64 [rows],
    certified bool [rows])."""

    return launch(got, rank_stride, world, table(specs))


def launch(got: torch.Tensor, rank_stride: int, world: int, tab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The kernel on a gathered pack for the rows of table ``tab``; one pinned upload of the table, one pinned
    readback, waited for on an event."""

    rows = int(tab.shape[0])
    dev = got.device
    with torch.cuda.device(dev):                    # the copies, the kernel and the event on got's device's stream
        host_t = _buf("table", (rows, COLS), torch.int64)
        host_t.numpy()[:] = tab
        dev_t = _buf("table", (rows, COLS), torch.int64, dev)
        dev_t.copy_(host_t, non_blocking=True)
        out = _buf("out", (rows, 2), torch.int32, dev)
        _ext().sample_rows(got.contiguous().view(-1), int(rank_stride), int(world), dev_t, out, rows)
        host_o = _buf("out", (rows, 2), torch.int32)
        host_o.copy_(out, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(dev))
        ev.synchronize()
    o = host_o.numpy()
    return o[:, 0].astype(np.int64), o[:, 1] != 0


def host_rows(got_rows: np.ndarray, n: int, world: int, positions: Sequence[int], s) -> list[int]:
    """The host rule on gathered packs [world, R, 2n] (fp32, ids as int32 bits): the samplers' own code."""

    from tensorfold.engine.exact_sampling import choose_rows

    g = got_rows
    values = np.concatenate([g[q, :, :n] for q in range(world)], axis=1).astype(np.float32)
    tokens = np.concatenate([np.ascontiguousarray(g[q, :, n:]).view(np.int32) for q in range(world)],
                            axis=1).astype(np.int64)
    if s is None or s.temperature <= 0:
        order = np.lexsort((tokens, -values), axis=-1)
        return [int(tokens[i, order[i, 0]]) for i in range(values.shape[0])]
    return choose_rows(values, tokens, positions, s)


def draw(got: torch.Tensor, rank_stride: int, world: int, parts: Sequence[tuple]) -> list[list[int]]:
    """Tokens of ``parts`` [(offset of the part's first row in a rank's pack, rows, n, positions, sampling)], each row
    at offset + row x 2n, as the host rule draws them: certified rows from the GPU, the others on the host."""

    bases, at = [], 0
    for _, R, _, _, _ in parts:
        bases.append(at)
        at += R
    tokens, certain = launch(got, rank_stride, world, table_parts(parts))
    out = [tokens[b:b + R].tolist() for b, (_, R, _, _, _) in zip(bases, parts)]
    if certain.all():
        return out
    flat = got.contiguous().view(-1)
    for p, (b, (off, R, n, positions, s)) in enumerate(zip(bases, parts)):
        idx = [r for r in range(R) if not certain[b + r]]
        if not idx:
            continue
        g = torch.stack([flat[q * rank_stride + off:q * rank_stride + off + R * 2 * n].view(R, 2 * n)
                         for q in range(world)]).cpu().numpy()
        sampled = s is not None and s.temperature > 0
        redo = host_rows(g[:, idx], n, world, [positions[r] for r in idx] if sampled else [0] * len(idx), s)
        for r, t in zip(idx, redo):
            out[p][r] = t
    return out


def applies(parts: Sequence[tuple]) -> bool:
    """Whether every part fits the kernel: keyed top-k (top_k on) or greedy rows, at most MAXC candidates a row."""
    for _, _, n, _, s in parts:
        if s is not None and s.temperature > 0 and not s.top_k:
            return False
    return True


def enabled(world: int, n_max: int, device) -> bool:
    """TENSORFOLD_GPU_SAMPLE and this GPU's check (run once, at the first call)."""

    if not wanted():
        return False
    if world * n_max > MAXC:
        return False
    if _state["decided"] is None:
        try:
            ok, why = check(device)
        except Exception as exc:  # noqa: BLE001 - no kernel here: the host draws
            ok, why = False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"
        _state["decided"] = ok
        print(f"[tensorfold] GPU sampling: {'on (checked against the host rule)' if ok else 'off: ' + why}",
              flush=True)
    return bool(_state["decided"])


def check(device, rows: int = 512, seed: int = 7) -> tuple[bool, str]:
    """Random gathered packs (4 ranks, 28 candidates each, many tied values) under many sampling settings: every
    certified GPU token must equal the host rule's."""

    from tensorfold.engine.exact_sampling import Sampling

    rng = np.random.default_rng(seed)
    world, n = 4, 28
    vals = rng.standard_normal((world, rows, n)).astype(np.float32) * 3
    vals[:, ::3, :] = np.round(vals[:, ::3, :])                      # ties: whole numbers
    vals = -np.sort(-vals, axis=2)                                   # each rank's top-k comes sorted
    ids = np.stack([rng.permutation(1000)[:n] + 1000 * q for q in range(world) for _ in range(rows)])
    ids = ids.reshape(world, rows, n).astype(np.int32)
    pack = np.concatenate([vals, ids.view(np.float32)], axis=2)      # [world, rows, 2n]
    got = torch.from_numpy(np.ascontiguousarray(pack)).to(device)
    settings = [None, Sampling(1, 0.7, 20, 0.95), Sampling(2, 1.0, 20, 0.95), Sampling(3, 0.3, 5, 1.0),
                Sampling(4, 1.5, 40, 0.5), Sampling(5, 1.0, 1, 0.9), Sampling(6, 1.0, 20, 0.95, 0.05),
                Sampling(7, 1e-7, 20, 0.95), Sampling(2 ** 62 + 11, 0.8, 112, 0.99)]
    per = rows // len(settings)
    parts = []
    for j, s in enumerate(settings):
        positions = list(rng.integers(0, 1 << 20, size=per))
        parts.append((j * per * 2 * n, per, n, positions, s))
    flat_got = got.view(world, -1)
    specs, expect = [], []
    for off, R, nn, positions, s in parts:
        g = pack.reshape(world, -1)[:, off:off + R * 2 * nn].reshape(world, R, 2 * nn)
        expect += host_rows(g, nn, world, positions, s)
        specs += [(off + r * 2 * nn, nn, s, positions[r]) for r in range(R)]
    tokens, certain = run(flat_got, flat_got.shape[1], world, specs)
    bad = [i for i in range(len(specs)) if certain[i] and int(tokens[i]) != expect[i]]
    if bad:
        return False, f"{len(bad)} of {len(specs)} certified tokens differ from the host rule"
    if certain.mean() < 0.99:
        return False, f"only {certain.mean():.3f} of the rows certified"
    return True, ""


__all__ = ["COLS", "MAXC", "applies", "check", "draw", "enabled", "host_rows", "launch", "run", "table", "table_parts",
           "wanted"]
