"""One-shot all-gather over CUDA IPC for the GPUs of one host (PCIe or NVLink peer to peer).

A decode round's all-gathers are small (a verify window's partials: rows x 4,096 fp32, the samplers' candidates, the
round messages) and NCCL spends most of each in fixed latency. Here every rank pushes its shard straight into a slot
of every peer's exported region and each rank copies the peers' shards out of its own slots in rank order (ipc.cu),
by one of three protocols:
  flag  SM stores into the peers' slots; the blocks arrive on a local counter (a GPU-scope release) and the last one
        fences once at system scope and writes one flag a peer; the peers poll their own flags and acquire with a
        load (Mia's RoCE one-shot protocol without its host proxy). One system fence a launch: on PCIe such a fence
        costs about half a microsecond and fences do not overlap, so a fence a warp or a block is far too slow;
  ll    SM stores of 8-byte words that each carry 4 payload bytes beside a tag of the launch, so a word that shows
        the tag is complete and no fence or flag is needed (the idea of NCCL's LL protocol): least latency, twice
        the bytes, for the smallest gathers;
  ce    copy-engine peer copies (cudaMemcpyAsync, captured as graph memcpy nodes), then a flag a peer; one slot a peer
        and an acknowledgement after each read, since a memcpy node's addresses are fixed. Where SM-driven
        peer-to-peer stores are slow (an IOMMU in translated mode, for one) the copy engines may still run at full
        speed.
Flags are plain stores and loads, so GPUs without peer-to-peer atomics qualify. A device epoch numbers the launches
(the SM protocols alternate two slots with it), so CUDA graphs replay the exchange; a wait past TF_GLM_IPC_WAIT_S
poisons the runtime and the host raises (as the RoCE one-shot does, roce.py).

``IpcComm`` sends all-gathers of up to TF_GLM_IPC_MAX_KB a rank this way and the rest (the prefill exchanges) through
NCCL, the protocol chosen by size bands (TF_GLM_IPC_PROTOCOLS); every choice depends on the byte count alone (the
same on every rank). The output layout is NCCL's and the bytes are copied, never computed: an all-gather adds
nothing, so every reply keeps its bits.

Credits: the protocol is Mia's one-shot RoCE all-gather (MiaAI-Lab, roce.cu / roce.py, itself b12x's "RoCEnante"),
without its host proxy; the CUDA IPC buffers follow b12x's PCIe one-shot collectives (local-inference-lab/b12x,
b12x/comm/pcie, Apache License 2.0); the tagged words follow NCCL's LL protocol. No b12x or NCCL source is copied.

Settings (every rank must see the same values; the setup compares them):
  TF_GLM_COMM=ipc         turns it on (GlmEngine); without peer-to-peer access between the ranks' GPUs (each rank must
                          see the others' devices, e.g. CUDA_VISIBLE_DEVICES with its own GPU first) it says so and
                          every gather stays on NCCL
  TF_GLM_IPC_MAX_KB       largest gather a rank sent this way (default 512 = 32 rows of 4,096 fp32)
  TF_GLM_IPC_PROTOCOLS    size bands, smallest first: name:KiB items, the last one's KiB optional (up to the max),
                          names ll, flag, ce; default "ll:32,flag" (LL up to 32 KiB, SM stores with flags above).
                          Examples: "flag" (one protocol), "ll:16,flag:64,ce". The GPU test's bench prints the best
                          bands for a machine. Each GPU exports 2 x (ranks - 1) flag slots of the max, LL slots of
                          twice its band's top, and ranks - 1 copy-engine slots of its band's top (BAR1 space)
  TF_GLM_IPC_WAIT_S       seconds a wait for a peer may take before it poisons the runtime (default 300, 1 to 3600).
                          Generous on purpose: in a multi-stream round the other ranks wait inside the round
                          message's gather while rank 0 alone admits a request (an image encode, a grammar compile),
                          and a dead rank is still reported instead of hanging as NCCL would
  TF_GLM_IPC_BLOCK_KB     shard KiB a block (the grid: one block for each, at most half the device's SMs, so every
                          block of a launch is resident at once; default 8)
  TF_GLM_IPC_THREADS      threads a block (default 512)
"""

from __future__ import annotations

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch

ABI = 3                                  # ipc.h's IPC_ABI
SLOT_ALIGN = 256
MAX_KB = 512
WAIT_S = 300.0
BLOCK_KB = 8
THREADS = 512
PROTOCOLS = "ll:32,flag"
PROTO = {"flag": 0, "ll": 1, "ce": 2}     # ipc.h's PROTO_FLAG, PROTO_LL, PROTO_CE
NAMES = {v: k for k, v in PROTO.items()}
MAX_BANDS = 3


class IpcUnavailable(RuntimeError):
    """CUDA IPC cannot run between these ranks (said on every rank alike): the caller keeps NCCL."""


def _setting(name: str, default: float, lo: float, hi: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = lo - 1
    if not lo <= value <= hi:
        raise ValueError(f"{name}: {lo:g} to {hi:g}, not {raw!r}")
    return value


def parse_protocols(spec: str, max_bytes: int) -> list[tuple[int, int]]:
    """``TF_GLM_IPC_PROTOCOLS`` -> [(protocol, largest bytes)] in rising order, the last band reaching ``max_bytes``.
    A band whose top passes the max is cut there, and bands after it are dropped (they would take no gather)."""

    items = [x.strip() for x in spec.split(",") if x.strip()]
    if not items or len(items) > MAX_BANDS:
        raise ValueError(f"TF_GLM_IPC_PROTOCOLS: 1 to {MAX_BANDS} name:KiB items, not {spec!r}")
    bands: list[tuple[int, int]] = []
    prev = 0
    for k, item in enumerate(items):
        name, _, kb = item.partition(":")
        name = name.strip().lower()
        if name not in PROTO:
            raise ValueError(f"TF_GLM_IPC_PROTOCOLS: protocols are {', '.join(PROTO)}, not {name!r} (in {spec!r})")
        last = k == len(items) - 1
        if kb.strip():
            try:
                top = int(float(kb) * 1024)
            except ValueError:
                top = 0
            if top <= 0:
                raise ValueError(f"TF_GLM_IPC_PROTOCOLS: a positive KiB, not {kb!r} (in {spec!r})")
        elif last:
            top = max_bytes
        else:
            raise ValueError(f"TF_GLM_IPC_PROTOCOLS: only the last band may leave out its KiB (in {spec!r})")
        if bands and bands[-1][1] >= max_bytes:
            break                                     # the bands so far take every gather
        if top <= prev:
            raise ValueError(f"TF_GLM_IPC_PROTOCOLS: the bands' KiB must rise (in {spec!r})")
        prev = top
        if last:
            top = max(top, max_bytes)                 # the last band reaches the max
        bands.append((PROTO[name], min(top, max_bytes)))
    return bands


def describe(bands: list[tuple[int, int]]) -> str:
    """The bands as a setting string (KiB), e.g. "ll:32,flag:512"."""

    return ",".join(f"{NAMES[p]}:{b / 1024:g}" for p, b in bands)


def settings() -> dict:
    """The TF_GLM_IPC_* values (bytes, ns, threads and bands), checked."""

    threads = int(_setting("TF_GLM_IPC_THREADS", THREADS, 32, 1024))
    if threads % 32:
        raise ValueError(f"TF_GLM_IPC_THREADS: a multiple of 32, not {threads}")
    max_bytes = int(_setting("TF_GLM_IPC_MAX_KB", MAX_KB, 1, 4096) * 1024)
    return {"max_bytes": max_bytes,
            "timeout_ns": int(_setting("TF_GLM_IPC_WAIT_S", WAIT_S, 1, 3600) * 1e9),
            "block_bytes": int(_setting("TF_GLM_IPC_BLOCK_KB", BLOCK_KB, 1, 4096) * 1024),
            "threads": threads,
            "bands": parse_protocols(os.environ.get("TF_GLM_IPC_PROTOCOLS", "").strip() or PROTOCOLS, max_bytes)}


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    return load(name=f"tensorfold_ipc_v{ABI}", sources=[str(here / "ipc.cpp"), str(here / "ipc.cu")],
                extra_cuda_cflags=["-O3"], verbose=False)


def plan(nbytes: int, block_bytes: int, max_blocks: int = 128, pack: int = 16) -> tuple[int, int]:
    """(blocks, packs a block) of a gather of ``nbytes`` a rank: ipc.cpp's ``plan`` (the CPU tests check the two
    agree), from the byte count alone so every rank's launch matches. The C++ side caps ``max_blocks`` at half the
    device's SMs (every block of a launch must be resident at once)."""

    packs = -(-nbytes // pack)
    grid = max(1, min(-(-nbytes // block_bytes), max_blocks, packs))
    chunk = -(-packs // grid)
    return -(-packs // chunk), chunk


class IpcGather:
    """One rank's side of the exchange: its exported region, the peers' mapped regions and the launches."""

    def __init__(self, comm, rank: int, world: int, max_bytes: int | None = None, timeout_ns: int | None = None,
                 block_bytes: int | None = None, threads: int | None = None,
                 bands: list[tuple[int, int]] | None = None, every_protocol: bool = False) -> None:
        """``comm``: a communicator with ``all_gather`` (NCCL) for the setup. ``bands``: [(protocol, largest
        bytes)] (default: TF_GLM_IPC_PROTOCOLS); ``every_protocol``: slots for all three protocols up to the max
        whatever the bands (tests and benchmarks force each). Raises ``IpcUnavailable`` on every rank when any rank
        cannot take part (then nothing is left mapped)."""

        given = settings()
        self.rank, self.world = rank, world
        self.max_bytes = int(given["max_bytes"] if max_bytes is None else max_bytes)
        self.timeout_ns = int(given["timeout_ns"] if timeout_ns is None else timeout_ns)
        self.block_bytes = int(given["block_bytes"] if block_bytes is None else block_bytes)
        self.threads = int(given["threads"] if threads is None else threads)
        if bands is None:
            bands = (given["bands"] if max_bytes is None else
                     parse_protocols(os.environ.get("TF_GLM_IPC_PROTOCOLS", "").strip() or PROTOCOLS, self.max_bytes))
        self.bands = [(int(p), int(b)) for p, b in bands]
        self.slot_bytes = -(-self.max_bytes // SLOT_ALIGN) * SLOT_ALIGN
        tops = {p: max((b for q, b in self.bands if q == p), default=0) for p in PROTO.values()}
        self.ll_bytes = self.max_bytes if every_protocol else tops[PROTO["ll"]]
        self.ce_bytes = self.max_bytes if every_protocol else tops[PROTO["ce"]]
        self.g = None
        self.err = None
        why, handle_len = "", 64
        handle = None
        try:
            if not 2 <= world <= 8:
                raise ValueError(f"2 to 8 ranks, not {world}")
            ext = _ext()                 # its first build (tens of seconds) here, before the verdict all-gather below
            layout = list(ext.layout())
            if layout[0] != ABI:
                raise RuntimeError(f"extension ABI {layout[0]}, expected {ABI}")
            if int(layout[3]) < self.threads:
                raise ValueError(f"TF_GLM_IPC_THREADS {self.threads}: at most {layout[3]}")
            handle_len = int(layout[7])
            flat = [v for band in self.bands for v in band]
            self.g = ext.Gather(world, rank, self.slot_bytes, self.ll_bytes, self.ce_bytes, flat, self.timeout_ns,
                                self.block_bytes, self.threads)
            handle = self.g.handle()
        except Exception as exc:         # noqa: BLE001  (any rank's failure: every rank keeps NCCL)
            why = f"{type(exc).__name__}: {exc}"
        words = -(-handle_len // 4)
        raw = bytearray(words * 4)
        if handle is not None:
            raw[:handle_len] = bytes(handle.numpy().tobytes())
        padded = (self.bands + [(-1, 0)] * MAX_BANDS)[:MAX_BANDS]
        self.max_grid = int(self.g.max_grid()) if self.g is not None else 0
        geometry = [ABI, self.slot_bytes, self.ll_bytes, self.ce_bytes, self.block_bytes, self.threads, handle_len,
                    self.max_grid, *[v for band in padded for v in band]]
        head = [1 if why else 0, *geometry]
        mine = torch.tensor(head + list(torch.frombuffer(raw, dtype=torch.int32).tolist()), dtype=torch.int32)
        got = torch.empty((world * mine.numel(),), dtype=torch.int32, device="cuda")
        comm.all_gather(mine.cuda(), got)
        rows = got.view(world, -1).cpu()
        failed = [r for r in range(world) if int(rows[r, 0])]
        if failed or len({tuple(r[1:len(head)].tolist()) for r in rows}) != 1:
            self.close()
            reason = (f"rank {', '.join(map(str, failed))} could not set up ({why or 'see that rank'})" if failed
                      else "the ranks' TF_GLM_IPC_* settings differ")
            raise IpcUnavailable(f"CUDA IPC all-gather unavailable: {reason}")
        handles = [torch.frombuffer(bytearray(rows[r, len(head):].numpy().tobytes()[:handle_len]), dtype=torch.uint8)
                   for r in range(world)]
        try:
            self.g.connect(handles)
        except Exception as exc:         # noqa: BLE001
            why = f"{type(exc).__name__}: {exc}"
        verdict = torch.tensor([1 if why else 0], dtype=torch.int32, device="cuda")
        all_verdicts = torch.empty((world,), dtype=torch.int32, device="cuda")
        comm.all_gather(verdict, all_verdicts)
        failed = [r for r, v in enumerate(all_verdicts.tolist()) if v]
        if failed:
            self.close()
            raise IpcUnavailable(f"CUDA IPC all-gather unavailable: rank {', '.join(map(str, failed))} cannot map "
                                 f"its peers' memory ({why or 'see that rank'}); peer-to-peer access needs every "
                                 "GPU visible to every rank")
        self.err = (ctypes.c_int64 * 4).from_address(self.g.error_address())

    def fits_bytes(self, n: int) -> bool:
        """Whether a gather of ``n`` bytes a rank goes over IPC (the same answer on every rank)."""

        return 0 < n <= self.max_bytes

    def protocols(self) -> list[int]:
        """The protocols this instance has slots for."""

        return [PROTO["flag"]] + ([PROTO["ll"]] if self.ll_bytes else []) + ([PROTO["ce"]] if self.ce_bytes else [])

    def capacity(self, protocol: int) -> int:
        return {PROTO["flag"]: self.max_bytes, PROTO["ll"]: self.ll_bytes, PROTO["ce"]: self.ce_bytes}[protocol]

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor, protocol: int = -1) -> None:
        """recv [world * n] <- every rank's send [n] in rank order: contiguous CUDA tensors, n bytes fitting;
        ``protocol`` -1 chooses by the size (the bands)."""

        self.g.run(send, recv, -1, -1, protocol)
        if self.err[0] and not torch.cuda.is_current_stream_capturing():
            self.check()

    def check(self) -> None:
        """Raise if a wait timed out on the GPU (it poisoned the runtime: later gathers move nothing)."""

        if self.err is not None and self.err[0]:
            seq, peer, block, waited = (int(v) for v in self.err)
            where = f"block {block}" if block >= 0 else "its acknowledgement"
            raise RuntimeError(f"CUDA IPC all-gather failed: sequence {seq} waited {waited / 1e9:.1f} s for rank "
                               f"{peer} ({where}) and gave up (TF_GLM_IPC_WAIT_S {self.timeout_ns / 1e9:g} s); "
                               "the runtime is poisoned, restart every rank")

    def close(self) -> None:
        if self.g is not None:
            self.err = None
            self.g.close()
            self.g = None


def _overlap(a: torch.Tensor, b: torch.Tensor) -> bool:
    a0, b0 = a.data_ptr(), b.data_ptr()
    return a0 < b0 + b.numel() * b.element_size() and b0 < a0 + a.numel() * a.element_size()


class IpcComm:
    """NCCL with small all-gathers over CUDA IPC: the same call, output layout and bits."""

    SELF_TEST_WAIT_NS = 30_000_000_000           # the self-test's wait limit: a path that drops writes fails soon

    def __init__(self, nccl, rank: int, world: int, max_bytes: int | None = None) -> None:
        self.nccl, self.rank, self.world = nccl, rank, world
        self.small = self.large = 0
        self.settled = False
        nccl.barrier()
        self.ipc = IpcGather(nccl, rank, world, max_bytes)
        try:
            self._self_test()
        except Exception:
            self.ipc.close()
            raise

    def _self_test(self) -> None:
        """Gathers over IPC and NCCL under a short wait limit: a few sizes by the bands (a tail, a decode row, the
        largest) and each protocol in use at a size it takes, three times each. The same bits on every rank, or
        ``IpcUnavailable`` on every rank (peer writes that never arrive, or arrive out of the order the protocol
        needs, would show here)."""

        ipc = self.ipc
        ipc.g.set_timeout(min(ipc.timeout_ns, self.SELF_TEST_WAIT_NS))
        gen = torch.Generator(device="cuda")
        gen.manual_seed(1234 + self.rank)
        cases = [(n, -1) for n in sorted({4, 168, 16 << 10, ipc.max_bytes - ipc.max_bytes % 4 or 4})]
        cases += [(min(16 << 10, ipc.capacity(p)) // 4 * 4 or 4, p) for p in ipc.protocols()]
        bad, why = 0, ""
        try:
            for n, protocol in cases:
                send = torch.randint(-2 ** 31, 2 ** 31 - 1, (n // 4,), dtype=torch.int32, device="cuda",
                                     generator=gen)
                want = torch.empty((self.world * send.numel(),), dtype=torch.int32, device="cuda")
                got = torch.full_like(want, -1)
                self.nccl.all_gather(send, want)
                for _ in range(3):
                    ipc.all_gather(send, got, protocol)
                    torch.cuda.synchronize()
                    ipc.check()
                    bad += int(not torch.equal(want, got))
        except RuntimeError as exc:                   # a wait timed out: this rank is poisoned, so stop here
            bad, why = bad + 1, str(exc)
        verdict = torch.tensor([bad], dtype=torch.int32, device="cuda")
        both = torch.empty((self.world,), dtype=torch.int32, device="cuda")
        self.nccl.all_gather(verdict, both)
        if int(both.sum()):
            ranks = ", ".join(str(r) for r, v in enumerate(both.tolist()) if v)
            raise IpcUnavailable(f"CUDA IPC all-gather unavailable: its self-test against NCCL failed on rank {ranks}"
                                 + (f" ({why})" if why else " (other bytes than NCCL's)"))
        ipc.g.set_timeout(ipc.timeout_ns)

    def describe(self) -> str:
        ipc = self.ipc
        return (f"all-gathers up to {ipc.max_bytes >> 10} KiB over CUDA IPC (one-shot peer-to-peer push, protocols "
                f"{describe(ipc.bands)}, blocks of {ipc.block_bytes / 1024:g} KiB x {ipc.threads} threads, at most "
                f"{ipc.max_grid}, wait limit {ipc.timeout_ns / 1e9:g} s), larger ones over NCCL")

    def settle(self) -> None:
        """Startup is over on this rank (its engine is built). Until then each eager IPC gather first meets the other
        ranks in an NCCL barrier: ranks build their CUDA extensions on their own and can drift apart by tens of
        seconds on a first start, past what an IPC wait allows; NCCL waits without that limit. Captured gathers
        never do."""

        self.settled = True

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        """IPC for a gather whose byte count fits, NCCL otherwise: decided by the byte count alone, which every rank's
        matching call shares. A buffer that is not contiguous goes through a contiguous copy, and a send that
        overlaps recv anywhere but at this rank's place in it through a copy of its own (local choices that never
        change the transport)."""

        n = send.numel() * send.element_size()
        if not self.ipc.fits_bytes(n):
            self.large += 1
            self.nccl.all_gather(send, recv)
            return
        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        self.small += 1
        s = send if send.is_contiguous() else send.contiguous()
        r = recv if recv.is_contiguous() else torch.empty(recv.shape, dtype=recv.dtype, device=recv.device)
        if _overlap(s, r) and s.data_ptr() != r.data_ptr() + self.rank * n:
            s = s.clone()
        if not self.settled and not torch.cuda.is_current_stream_capturing():
            self.nccl.barrier()
        self.ipc.all_gather(s, r)
        if r is not recv:
            recv.copy_(r)

    def barrier(self) -> None:
        self.nccl.barrier()

    def check(self) -> None:
        self.ipc.check()

    def __getattr__(self, name):
        if name in ("nccl", "ipc"):                  # not set yet (a failed construction): no recursion
            raise AttributeError(name)
        return getattr(self.nccl, name)
