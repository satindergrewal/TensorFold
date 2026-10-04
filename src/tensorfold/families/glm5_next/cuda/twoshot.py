"""TF_GLM_TWOSHOT_ROWS: a decode window's rank partials exchanged in two shots instead of one all-gather, with the
same bits (X6a).

Today each of a decode forward's 90 sites all-gathers every rank's fp32 partial [R, D] (16 KiB a row a rank, so a
rank receives 48 KiB a row at four ranks), and hc_post sums the four partials rank 0 first and rounds the sum to
bf16: branch = bf16(((g0 + g1) + g2) + g3). The all-gather's bytes grow with the rows, so wide windows pay for them.

Two shots. Rank r owns the rows [r H, r H + H) of the window, H = ceil(R / ranks) (the row split of the prompt
chunks' exchanges, ``hcsplit``):

1. scatter: every rank sends each other rank its partial of that rank's rows (one NCCL group of sends and receives,
   on the all-gather communicator or on patch 0103's second one, TF_GLM_TWOSHOT_SCATTER);
2. the owner adds the ranks' partials of its rows rank 0 first in fp32, exactly as hc_post adds them (the same Triton
   add chain: add.f32, no flush to zero), and rounds the sum to bf16 (cvt.rn.bf16.f32, as hc_post does);
3. an all-gather of the owners' rows (in place, the engine's all-gather: NCCL, or CUDA IPC under TF_GLM_COMM=ipc;
   TF_GLM_TWOSHOT_GATHER) gives every rank every row's branch: bf16 (TF_GLM_TWOSHOT_KIND=bf16, then widened to fp32
   by one small kernel) or the same values as fp32 (f32);
4. the exchange hands hc_post a [ranks, R, D] fp32 view whose rank 0 holds the branch and whose other ranks hold -0.0
   (fixed buffers, written once). hc_post's sum is then ((b + -0) + -0) + -0 = b for every value b (IEEE: x + -0 is x,
   for +0, -0, infinities and NaN too; the owner's sum was rounded by the same flags, so b is never a value hc_post's
   add would treat otherwise), and bf16(b) is b: hc_post, its fused form (refused for this view, so it runs unfused)
   and the MTP block's residual_add compute exactly the bits they compute from the four partials. The consumers are
   unchanged; only their stride between ranks comes from the tensor (``glue``).

Bytes a rank receives a row at four ranks: 12 KiB (scatter) + 6 KiB (bf16) or 12 KiB (f32), against 48 KiB; latency:
two collectives and one or two small kernels instead of one collective. On the one-all-gather curve measured on the
target box (about 19 us + 0.027 us a KiB received) that breaks even near 26 rows; the GPU test measures it.

Exactness: bit-identical (the same bits as the one all-gather, for every row count; the GPU test proves it at every
row count from 1 to the window's rows, in CUDA graphs and eagerly, with special values: -0, denormals, infinities,
NaN). Settings (every rank the same; the engine compares them at start):

  TF_GLM_TWOSHOT_ROWS      0 (default): off; N: decode windows of N rows or more exchange in two shots
  TF_GLM_TWOSHOT_KIND      bf16 (default): the second shot moves bf16 rows; f32: their fp32 copies (no widening)
  TF_GLM_TWOSHOT_SCATTER   nccl (default): the first shot on the all-gather communicator (peer to peer under
                           NCCL_P2P_LEVEL); exchange: on the second communicator (TF_NCCL_EXCHANGE_ENV, patch 0103)
  TF_GLM_TWOSHOT_GATHER    comm (default): the engine's all-gather (NCCL, or CUDA IPC under TF_GLM_COMM=ipc);
                           nccl: NCCL's even under TF_GLM_COMM=ipc

Every buffer is allocated once a decode buffer (``forward.Buffers``), so CUDA graphs capture the exchange; each row
count's sizes are fixed (a graph a row count). Memory a decode buffer, H = ceil(window rows / ranks): ranks x (H + 2)
x D x 4 bytes of received partials, ranks x ranks x (H + 1) x D x 4 bytes for the view (all but one rank's place
-0.0) and ranks x (H + 1) x D x 2 bytes of bf16 rows: about 24 MiB for a 256-row window at four ranks, 6 MiB for 63.

TF_GLM_TWOSHOT_SPLIT (X6c, a prototype, default off; needs TF_GLM_TWOSHOT_ROWS): windows of that many rows or more
gather their second shot in two halves, rows [0, R1) (R1 a multiple of the ranks, about half) and [R1, R): one scatter,
then the first half's owner sum and gather; the second half's on a side stream after it, while the first half's
hyper-connection glue (hc_post, then the next site's hc_pre) runs (``forward``: ``Split``). Every row's arithmetic is
the unsplit exchange's and the glue kernels are row-independent, so the bits are the same; whether the overlap beats
the second gather's own latency is what the GPU test's bench measures (with and without a stand-in for the next
block's input projection, which a full X6c would overlap too)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

KINDS = ("bf16", "f32")
SCATTERS = ("nccl", "exchange")
GATHERS = ("comm", "nccl")
BLOCK = 1024                     # columns a program of the owner's sum (hc_post's block)


@dataclass(frozen=True)
class Settings:
    rows: int = 0                # 0: off; else the fewest rows a window exchanges in two shots
    kind: str = "bf16"
    scatter: str = "nccl"
    gather: str = "comm"
    split: int = 0               # X6c prototype: 0 off; else the fewest rows whose second shot runs in two halves

    @classmethod
    def from_env(cls, env=None) -> "Settings":
        env = os.environ if env is None else env

        def pick(name: str, options: tuple[str, ...]) -> str:
            v = (env.get(name, "") or options[0]).strip().lower()
            if v not in options:
                raise ValueError(f"{name}: {' or '.join(options)}, not {v!r}")
            return v

        raw = (env.get("TF_GLM_TWOSHOT_ROWS", "") or "0").strip()
        if not raw.isdecimal() or int(raw) > 65536:
            raise ValueError(f"TF_GLM_TWOSHOT_ROWS: 0 (off) or the fewest rows a decode window exchanges in two shots, "
                             f"not {raw!r}")
        split = (env.get("TF_GLM_TWOSHOT_SPLIT", "") or "0").strip()
        if not split.isdecimal() or int(split) > 65536:
            raise ValueError(f"TF_GLM_TWOSHOT_SPLIT: 0 (off) or the fewest rows whose second shot runs in two halves, "
                             f"not {split!r}")
        if int(split) and not int(raw):
            raise ValueError("TF_GLM_TWOSHOT_SPLIT splits the two-shot exchange's second shot: it needs "
                             "TF_GLM_TWOSHOT_ROWS")
        return cls(int(raw), pick("TF_GLM_TWOSHOT_KIND", KINDS), pick("TF_GLM_TWOSHOT_SCATTER", SCATTERS),
                   pick("TF_GLM_TWOSHOT_GATHER", GATHERS), int(split))

    def code(self) -> list[int]:
        """What every rank must agree on (the engine's start comparison)."""
        return [self.rows, KINDS.index(self.kind), SCATTERS.index(self.scatter), GATHERS.index(self.gather),
                self.split]

    def describe(self) -> str:
        if not self.rows:
            return "decode partials: one all-gather a site (TF_GLM_TWOSHOT_ROWS=0)"
        return (f"decode partials of windows of {self.rows}+ rows: two shots (each rank sums its rows' partials rank 0 "
                f"first, then a {self.kind} all-gather; scatter on the "
                f"{'all-gather' if self.scatter == 'nccl' else 'send/receive'} communicator, gather by "
                f"{'the engine' if self.gather == 'comm' else 'NCCL'}), the same bits (TF_GLM_TWOSHOT_ROWS)"
                + (f"; prototype: windows of {self.split}+ rows gather their second half while the first half's "
                   f"hyper-connection glue runs (TF_GLM_TWOSHOT_SPLIT)" if self.split else ""))


_SETTINGS: Settings | None = None


def settings() -> Settings:
    """TF_GLM_TWOSHOT_* (read once)."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = Settings.from_env()
    return _SETTINGS


def code() -> list[int]:
    return settings().code()


# -- kernels -----------------------------------------------------------------------------------------------------------
@triton.jit
def _owner_sum(PART, RECV, OUT, row0, RECV_RS, rrow, D: tl.constexpr, WORLD: tl.constexpr, RANK: tl.constexpr,
               BLOCK: tl.constexpr, BF16: tl.constexpr):
    """Row i of this rank's rows (window row row0 + i), columns of block cb: the ranks' partials added rank 0 first in
    fp32 (glue._hc_post's chain: the first load, then one add a rank), this rank's own from PART, the others' from
    row rrow + i of their RECV slots; rounded to bf16 and stored at OUT's row row0 + i as bf16 (BF16) or as its fp32
    copy."""

    i = tl.program_id(0)
    cb = tl.program_id(1)
    d = cb * BLOCK + tl.arange(0, BLOCK)
    if RANK == 0:
        acc = tl.load(PART + (row0 + i) * D + d)
    else:
        acc = tl.load(RECV + (rrow + i) * D + d)
    for k in tl.static_range(1, WORLD):
        if k == RANK:
            acc = acc + tl.load(PART + (row0 + i) * D + d)
        else:
            acc = acc + tl.load(RECV + k * RECV_RS + (rrow + i) * D + d)
    branch = acc.to(tl.bfloat16)
    if BF16:
        tl.store(OUT + (row0 + i) * D + d, branch)
    else:
        tl.store(OUT + (row0 + i) * D + d, branch.to(tl.float32))


@triton.jit
def _widen(SRC, DST, n, BLOCK: tl.constexpr):
    """DST[:n] = SRC[:n] (bf16 to fp32: exact)."""

    j = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = j < n
    tl.store(DST + j, tl.load(SRC + j, mask=m).to(tl.float32), mask=m)


# -- the exchange ------------------------------------------------------------------------------------------------------
def rows_of(owner: int, R: int, world: int, first: int = 0) -> tuple[int, int]:
    """(first row, rows) of the window rows ``owner`` sums: [owner H, owner H + H) cut at R, H = ceil(R / world);
    ``first``: of the rows first .. R instead (the split's second half)."""

    H = -(-(int(R) - first) // int(world))
    lo = first + owner * H
    return lo, max(0, min(H, R - lo))


def split_rows(R: int, world: int) -> int | None:
    """X6c: the split's first half, rows 0 .. R1 (R1 a multiple of the ranks, so its gather carries no pad row that
    could land on the second half's rows), about half the window; None when the window is too small to split."""

    R1 = world * -(-int(R) // (2 * world))
    return R1 if 0 < R1 < R else None


def nccl_of(comm):
    """The NCCL object under the engine's communicator (``comm.NCCL``, or ``ipc.IpcComm``'s), else None."""

    for c in (comm, getattr(comm, "nccl", None)):
        if c is not None and all(hasattr(c, a) for a in ("lib", "comm", "xlib", "xcomm", "all_gather")):
            return c
    return None


class NcclP2P:
    """Shot 1's transport: one NCCL group of sends and receives on the current stream, on the all-gather
    communicator (``which`` "nccl") or on the send/receive one (TF_NCCL_EXCHANGE_ENV; "exchange")."""

    def __init__(self, nccl, which: str = "nccl") -> None:
        self.nccl = nccl
        self.lib, self.comm = (nccl.lib, nccl.comm) if which == "nccl" else (nccl.xlib, nccl.xcomm)

    def sendrecv(self, sends: dict, recvs: dict) -> None:
        """``sends[p]`` (a list of contiguous tensors) to rank p and ``recvs[p]`` from it, in list order (NCCL matches
        a peer's sends and receives in order; empty ones are skipped, as the peer's matching call skips them), peers
        in rank order."""

        from tensorfold.cuda.comm import _DTYPES

        lib, xc, check = self.lib, self.comm, self.nccl._check
        stream = torch.cuda.current_stream().cuda_stream
        check(lib.ncclGroupStart(), lib)
        try:
            for p in sorted(set(sends) | set(recvs)):
                for s in sends.get(p, ()):
                    if s.numel():
                        check(lib.ncclSend(s.data_ptr(), s.numel(), _DTYPES[s.dtype], p, xc, stream), lib)
                for r in recvs.get(p, ()):
                    if r.numel():
                        check(lib.ncclRecv(r.data_ptr(), r.numel(), _DTYPES[r.dtype], p, xc, stream), lib)
        finally:
            check(lib.ncclGroupEnd(), lib)


class TritonKernels:
    """The exchange's two kernels (``_owner_sum``, ``_widen``); tests may stand in another implementation."""

    @staticmethod
    def owner_sum(part, recv, out, lo: int, n: int, recv_rs: int, rrow: int, D: int, world: int, rank: int) -> None:
        _owner_sum[(n, D // BLOCK)](part, recv, out, lo, recv_rs, rrow, D=D, WORLD=world, RANK=rank, BLOCK=BLOCK,
                                    BF16=out.dtype == torch.bfloat16, num_warps=4)

    @staticmethod
    def widen(src, dst, m: int) -> None:
        _widen[(triton.cdiv(m, 4096),)](src, dst, m, BLOCK=4096, num_warps=4)


class TwoShot:
    """One decode buffer's two-shot exchange (see the module docstring): windows of ``rows_from`` .. ``rows_max``
    rows; ``comm``: the engine's communicator (its NCCL object does the scatter); ``settings``: the transport and
    kind (default TF_GLM_TWOSHOT_*)."""

    def __init__(self, comm, rank: int, world: int, rows_max: int, width: int, settings_: Settings | None = None,
                 *, check: bool = True, device=None, p2p=None, kernels=None) -> None:
        """``p2p`` / ``kernels``: shot 1's transport and the two kernels (default ``NcclP2P`` on the communicator
        TF_GLM_TWOSHOT_SCATTER names, ``TritonKernels``); ``check``: the start check (``check``)."""

        s = settings_ if settings_ is not None else settings()
        self.s = s
        self.comm, self.rank, self.world = comm, int(rank), int(world)
        self.nccl = nccl_of(comm)
        if self.nccl is None and p2p is None:
            raise ValueError("TF_GLM_TWOSHOT_ROWS: the two-shot exchange needs the engine's NCCL communicator "
                             "(TF_GLM_COMM=nccl or ipc)")
        if self.world < 2 or width % BLOCK:
            raise ValueError(f"TF_GLM_TWOSHOT_ROWS: {self.world} ranks and a width of {width}: two or more ranks and "
                             f"a multiple of {BLOCK} columns")
        self.rows_from = max(1, int(s.rows))
        self.rows_max, self.D = int(rows_max), int(width)
        dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
        H = self.Hmax = -(-self.rows_max // self.world)
        # rows a received slot holds (the split's two halves need up to two more) and the elements between the ranks'
        # places in the view (the split's second gather may end up to ranks - 1 rows past the window)
        self.Hcap = H + 2
        self.RS = self.world * (H + 1) * self.D
        self.recv = torch.empty((self.world, self.Hcap, self.D), dtype=torch.float32, device=dev)
        # the view's buffer: rank 0's place holds the branch (and, f32, the second shot's rows); the others -0.0
        self.fbuf = torch.full((self.world * self.RS,), -0.0, dtype=torch.float32, device=dev)
        self.fbuf[:self.RS].zero_()
        self.gbuf = (torch.empty((self.RS,), dtype=torch.bfloat16, device=dev) if s.kind == "bf16" else None)
        self.p2p = p2p if p2p is not None else NcclP2P(self.nccl, s.scatter)
        self.k = kernels if kernels is not None else TritonKernels()
        self.calls = 0
        # X6c (TF_GLM_TWOSHOT_SPLIT): the second half's owner sum and gather run on a side stream (none on the CPU:
        # the tests run the split in order)
        self.side = self.ev_a = self.ev_b = None
        if s.split and dev.type == "cuda":
            self.side = torch.cuda.Stream(device=dev)
            self.ev_a, self.ev_b = torch.cuda.Event(), torch.cuda.Event()
        if check:
            self.check()                                 # (its exchanges also open the scatter's connections)
        elif dev.type == "cuda":
            self.warm()

    @torch.no_grad()
    def warm(self) -> None:
        """One exchange of the widest window, every rank together, outside any CUDA graph capture: NCCL opens the
        scatter's peer connections on their first use, which must not happen inside a capture."""

        self.exchange(torch.zeros((self.rows_max, self.D), dtype=torch.float32, device=self.fbuf.device),
                      self.rows_max)
        torch.cuda.synchronize()

    def applies(self, R: int) -> bool:
        return self.rows_from <= R <= self.rows_max

    # -- the two shots ---------------------------------------------------------------------------------------------
    def _scatter(self, part: torch.Tensor, halves: list[tuple[int, int]]) -> None:
        """Shot 1: for each half (first row, end) of the window (one, or two when split), this rank's partial of each
        other rank's rows to it and the others' partials of this rank's rows into their ``recv`` slots, the halves one
        after the other in each slot (one group of sends and receives on the current stream)."""

        sends: dict = {p: [] for p in range(self.world) if p != self.rank}
        recvs: dict = {p: [] for p in sends}
        at = 0
        for first, end in halves:
            _, n_me = rows_of(self.rank, end, self.world, first)
            for p in sends:
                lo, n = rows_of(p, end, self.world, first)
                sends[p].append(part[lo:lo + n].reshape(-1))
                recvs[p].append(self.recv[p, at:at + n_me].reshape(-1))
            at += -(-(end - first) // self.world)
        self.p2p.sendrecv(sends, recvs)

    def _gather(self, buf: torch.Tensor, H: int, first: int = 0) -> None:
        """Shot 2: every rank's H rows of ``buf`` from row ``first`` (its own at its place), in place."""

        n, base = H * self.D, first * self.D
        send, recv = buf[base + self.rank * n:base + (self.rank + 1) * n], buf[base:base + self.world * n]
        if self.s.gather == "nccl" and self.nccl is not None:
            self.nccl.all_gather(send, recv)
        else:
            self.comm.all_gather(send, recv)

    def exchange(self, part: torch.Tensor, R: int, split: bool = False):
        """Every rank's fp32 partial ``part[:R]`` ([rows, D], contiguous) to the [ranks, R, D] fp32 view hc_post reads:
        rank 0's place the rows' branch bf16(((g0 + g1) + g2) + g3) (exact in fp32), the others' -0.0. ``split``: the
        caller takes a ``Split`` (TF_GLM_TWOSHOT_SPLIT, for windows that wide)."""

        if (not self.applies(R) or part.dtype != torch.float32 or not part.is_contiguous() or part.dim() != 2
                or part.shape[1] != self.D or part.shape[0] < R):
            raise ValueError(f"two-shot exchange of {R} rows of {tuple(part.shape)} {part.dtype}: {self.rows_from} to "
                             f"{self.rows_max} rows of a contiguous fp32 [rows, {self.D}] partial")
        self.calls += 1
        R1 = split_rows(R, self.world) if split and self.s.split and R >= self.s.split else None
        if R1 is not None:
            return self._exchange_split(part, R, R1)
        H = -(-R // self.world)
        lo, n = rows_of(self.rank, R, self.world)
        self._scatter(part, [(0, R)])
        out = self.gbuf if self.gbuf is not None else self.fbuf
        if n:
            self.k.owner_sum(part, self.recv, out, lo, n, self.Hcap * self.D, 0, self.D, self.world, self.rank)
        self._gather(out, H)
        if self.gbuf is not None:
            self.k.widen(self.gbuf, self.fbuf, R * self.D)
        return self.view(R)

    def _exchange_split(self, part: torch.Tensor, R: int, R1: int) -> "Split":
        """X6c: one scatter for both halves; the first half's owner sum, gather (and widening) on the current stream;
        the second half's on the side stream once the first half's gather is done (so no two gathers of a rank ever
        run at once), its end marked by ``ev_b``. The rows' arithmetic is the unsplit exchange's: the same view."""

        D, w = self.D, self.world
        HA, HB = R1 // w, -(-(R - R1) // w)
        self._scatter(part, [(0, R1), (R1, R)])
        out = self.gbuf if self.gbuf is not None else self.fbuf
        lo, n = rows_of(self.rank, R1, w)
        self.k.owner_sum(part, self.recv, out, lo, n, self.Hcap * D, 0, D, w, self.rank)
        self._gather(out, HA)
        if self.gbuf is not None:
            self.k.widen(self.gbuf, self.fbuf, R1 * D)
        lo, n = rows_of(self.rank, R, w, R1)

        def second() -> None:
            if n:
                self.k.owner_sum(part, self.recv, out, lo, n, self.Hcap * D, HA, D, w, self.rank)
            self._gather(out, HB, R1)
            if self.gbuf is not None:
                self.k.widen(self.gbuf[R1 * D:], self.fbuf[R1 * D:], (R - R1) * D)

        if self.side is None:
            second()
            return Split(self.view(R), R1, None)
        self.ev_a.record()
        self.side.wait_event(self.ev_a)
        with torch.cuda.stream(self.side):
            second()
            self.ev_b.record()
        return Split(self.view(R), R1, self.ev_b)

    def view(self, R: int) -> torch.Tensor:
        return self.fbuf.as_strided((self.world, R, self.D), (self.RS, self.D, 1))

    # -- the start check ---------------------------------------------------------------------------------------------
    @torch.no_grad()
    def check(self, rows=None) -> None:
        """Every rank together: hc_post and residual_add on the two-shot view against the one all-gather's partials,
        bit for bit, at a few row counts, on partials spread over many magnitudes with -0.0, denormal, infinite and
        NaN entries; raises on every rank when any rank differs."""

        from . import glue

        dev = self.fbuf.device
        bad = 0
        S = 4
        rows = rows or sorted({self.rows_from, min(self.rows_max, self.rows_from + 1), self.rows_max,
                               max(self.rows_from, min(self.rows_max, self.world + 1)),
                               *([max(self.rows_from, min(self.rows_max, self.s.split))] if self.s.split else [])})
        gather = self.nccl.all_gather if self.nccl is not None else self.comm.all_gather
        gen = torch.Generator(device=dev)
        for R in rows:
            gen.manual_seed(7000 + 31 * R + self.rank)
            part = special(torch.randn((R, self.D), device=dev, generator=gen) *
                           torch.exp2(torch.randint(-20, 21, (R, self.D), device=dev, generator=gen).float()),
                           gen, self.rank)
            gathered = torch.empty((self.world * R * self.D,), dtype=torch.float32, device=dev)
            gather(part.reshape(-1), gathered)
            gathered = gathered.view(self.world, R, self.D)
            view = self.exchange(part, R)
            gen.manual_seed(9000 + R)                       # the same streams, post and comb on every rank
            x0 = torch.randn((R, S * self.D), device=dev, generator=gen).to(torch.bfloat16)
            post = torch.rand((R, S), device=dev, generator=gen) * 2.0
            comb = torch.rand((R, S * S), device=dev, generator=gen) * 0.5
            a, b = x0.clone(), x0.clone()
            glue.hc_post(a, a, gathered, post, comb)
            glue.hc_post(b, b, view, post, comb)
            bad += int(not torch.equal(a.view(torch.int16), b.view(torch.int16)))
            r0 = x0[:, :self.D].contiguous()
            ra, rb = r0.clone(), r0.clone()
            glue.residual_add(ra, ra, gathered)
            glue.residual_add(rb, rb, view)
            bad += int(not torch.equal(ra.view(torch.int16), rb.view(torch.int16)))
            if self.s.split and R >= self.s.split and split_rows(R, self.world):     # X6c: the split's view too
                whole = view.clone()
                sp = self.exchange(part, R, split=True)
                sp.wait()
                bad += int(not torch.equal(whole.view(torch.int32), sp.view.view(torch.int32)))
        verdict = torch.tensor([bad], dtype=torch.int32, device=dev)
        every = torch.empty((self.world,), dtype=torch.int32, device=dev)
        gather(verdict, every)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        if int(every.sum()):
            raise RuntimeError("TF_GLM_TWOSHOT_ROWS: the two-shot exchange's hc_post differs from the all-gather's on "
                               f"rank {[r for r, v in enumerate(every.tolist()) if v]} at {list(rows)} rows: "
                               "set TF_GLM_TWOSHOT_ROWS=0")


class Split:
    """X6c: a two-shot exchange whose rows R1 .. R are still being gathered on a side stream. ``view``: as
    ``TwoShot.exchange``'s; rows 0 .. R1 may be read now, the rest after ``wait()`` (on the reading stream)."""

    def __init__(self, view: torch.Tensor, R1: int, event) -> None:
        self.view, self.R1, self.event = view, int(R1), event

    def wait(self) -> None:
        if self.event is not None:
            torch.cuda.current_stream().wait_event(self.event)


def special(part: torch.Tensor, gen: torch.Generator, rank: int) -> torch.Tensor:
    """Partials with the values that would show another rounding or flush: -0.0 on every rank in some columns (their
    sum is -0.0), denormal sums, sums past the largest fp32, one rank's infinity and one rank's NaN in a few."""

    p = part.clone()
    tiny = torch.finfo(torch.float32).tiny
    p[:, 0:8] = -0.0
    p[:, 8:16] = tiny * 0.25 * (1 + rank)                    # denormal partials and sums
    p[:, 16:24] = -tiny * 0.5 if rank % 2 else tiny * 0.375
    p[:, 24:32] = torch.finfo(torch.float32).max * 0.6       # their sum overflows
    if rank == 1:
        p[:, 32:34] = float("inf")
    if rank == 2:
        p[:, 34:36] = float("nan")
    return p


_TOLD = False


def attach(w, rows: int) -> TwoShot | None:
    """A decode buffer's exchange (``forward.Buffers``): None unless TF_GLM_TWOSHOT_ROWS is on, the engine runs on
    two ranks or more and the buffer holds windows that wide. Every rank builds its decode buffers together, so the
    start check's collectives meet (the ranks' TF_GLM_TWOSHOT_* settings are compared before, with the engine's)."""

    global _TOLD
    s = settings()
    if not s.rows or w.comm is None or w.world < 2 or rows < s.rows:
        return None
    ts = TwoShot(w.comm, w.rank, w.world, rows, w.cfg.hidden, s)
    if w.rank == 0 and not _TOLD:
        _TOLD = True
        print(f"[tensorfold] {s.describe()}; checked bit for bit at start against the all-gather", flush=True)
    return ts
