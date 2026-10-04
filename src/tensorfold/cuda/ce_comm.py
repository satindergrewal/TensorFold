"""Paired sends and receives (``exchange`` / ``exchange_all``, NCCL's API in ``comm``) by the GPUs' copy engines,
peer to peer over PCIe between the rank processes of one host, through CUDA IPC (TENSORFOLD_EXCHANGE=ce).

Why: NCCL moves send/receive data with a few SMs a peer (two channels a PCIe peer), and on PCIe hosts its
peer-to-peer send/receive path can be several times slower than its host shared-memory one, which is itself bounded
by the SMs it takes. A copy engine writes straight into a peer GPU's memory at the link's speed and takes no SMs from
the compute it runs beside. The row-split prompt chunks of GLM-5.3-Flash move about 36 MiB out of every rank at each
of a chunk's 90 exchange sites (2,048 rows, four ranks): the bulk of a prompt's communication.

How. Setup (every rank together, through the NCCL bootstrap's store): each rank cudaMallocs a receive arena, exports
it (cudaIpcGetMemHandle) and maps every peer's (cudaIpcOpenMemHandle); creates a ring of interprocess CUDA events and
exports them; rank 0 creates a small host shared-memory table of step numbers. Every phase is voted on, so a problem
on one rank stops every rank with its reason; then a probe exchange is checked on every rank.

The arena holds one slot a peer, each in two regions used by alternate steps (TENSORFOLD_CE_SLOT_MIB a region,
default 8: 48 MiB a GPU at four ranks, whatever the chunk size). A call moves each peer's send tensors (laid back to
back, 256-byte aligned) in steps of at most a region: a step is this rank's cudaMemcpyAsync copies into every peer's
region (peers in rotated order, so each GPU receives from one sender at a time), one interprocess event recorded, then
a stream wait on every peer's event of the same step, then this rank's local copies from its own regions into the
receive tensors. A step number orders it all; every rank runs the same calls with the same sizes (NCCL's own contract:
the peer's matching call names this rank with the same sizes, dtypes and order), and this transport requires every
call to name every other rank with the same byte count, so every rank cuts a call into the same steps.

Ordering. A stream wait binds an event's last record at the time of the call, so the host table makes a receiver wait
for the record of the right step (the sender publishes its step number after recording; the receiver waits for it
before waiting on the event), and keeps a sender from recording an event slot again before every receiver has waited
on its previous use. A region is never overwritten while it may still be read: a sender's copies of step e start
after its stream waited for every peer's record of step e - 1, and a receiver's local copies out of the region of
step e - 2 sit on its stream before its own record of step e - 1 (a rank's steps run in its stream order: one stream,
or streams joined with stream waits between them, as the callers' chunks are).

Exact: transport only: the same bytes land in the same receive tensors as with NCCL. A failed step (a peer that stopped,
a CUDA error, a timeout) poisons the transport: every later call raises, and the ranks must be restarted together.
GPU-side waits on a peer that died mid-step do not time out (as with NCCL); the host-side waits do
(TENSORFOLD_CE_TIMEOUT_S, default 300).

Needs every rank on one host, each process seeing every GPU (its own first, as TensorFold uses device 0) with peer
access, and BAR1 room for the arena next to NCCL's peer buffers (``nvidia-smi -q -d MEMORY`` shows BAR1)."""

from __future__ import annotations

import ctypes
import glob
import mmap
import os
import time
import uuid

import numpy as np
import torch

SLOTS = 16                # interprocess events a rank cycles through
ALIGN = 256               # bytes: each tensor's start in a peer's payload
_D2D = 3                  # cudaMemcpyDeviceToDevice
_LAZY_PEER = 1            # cudaIpcMemLazyEnablePeerAccess


class _IpcHandle(ctypes.Structure):
    """cudaIpcMemHandle_t: 64 opaque bytes (unsigned: a c_char field would read as a string, cut at a zero byte)."""

    _fields_ = [("reserved", ctypes.c_ubyte * 64)]


def _loaded(fragment: str) -> list[str]:
    """Paths of the shared libraries this process has mapped whose file name starts with ``fragment``."""

    found = set()
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.split()[-1] if line.strip() else ""
                if path.startswith("/") and os.path.basename(path).startswith(fragment):
                    found.add(path)
    except OSError:
        return []
    return sorted(found)


def _cudart():
    """The CUDA runtime torch itself loaded (the same library instance: one set of runtime state)."""

    torch.cuda.init()
    names = _loaded("libcudart.so") + ["libcudart.so", "libcudart.so.13", "libcudart.so.12"]
    names += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*"))
    names += glob.glob("/usr/local/cuda/lib64/libcudart.so*")
    err = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            err = exc
            continue
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(_IpcHandle), ctypes.c_void_p]
        lib.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), _IpcHandle, ctypes.c_uint]
        lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                        ctypes.c_void_p]
        return lib
    raise RuntimeError(f"the CUDA runtime library was not found ({err})")


class _Raw:
    """A device pointer as a CUDA array (torch.as_tensor's view of memory this module allocated)."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False),
                                         "version": 3, "strides": None}


def transport(value: str | None = None) -> str:
    """TENSORFOLD_EXCHANGE: the paired sends and receives' transport, ``nccl`` (default) or ``ce``."""

    value = (os.environ.get("TENSORFOLD_EXCHANGE", "") if value is None else value).strip().lower() or "nccl"
    if value not in ("nccl", "ce"):
        raise ValueError(f"TENSORFOLD_EXCHANGE: nccl or ce, not {value!r}")
    return value


def slot_bytes(value: str | None = None) -> int:
    value = (os.environ.get("TENSORFOLD_CE_SLOT_MIB", "") if value is None else value).strip() or "8"
    if not value.isdecimal() or not 1 <= int(value) <= 64:
        raise ValueError(f"TENSORFOLD_CE_SLOT_MIB: MiB from 1 to 64, not {value!r}")
    return int(value) << 20


def timeout_s(value: str | None = None) -> float:
    value = (os.environ.get("TENSORFOLD_CE_TIMEOUT_S", "") if value is None else value).strip() or "300"
    try:
        return max(1.0, float(value))
    except ValueError:
        raise ValueError(f"TENSORFOLD_CE_TIMEOUT_S: seconds, not {value!r}") from None


def layout(tensors: list[torch.Tensor]) -> tuple[list[int], int]:
    """Each tensor's start in a peer's payload (ALIGN-byte aligned, in order) and the payload's bytes."""

    starts, at = [], 0
    for t in tensors:
        starts.append(at)
        at += -(-t.numel() * t.element_size() // ALIGN) * ALIGN
    return starts, at


class CeTransport:
    """``exchange`` / ``exchange_all`` over copy engines (one rank's side; created by every rank together)."""

    def __init__(self, rank: int, world: int, store, slot: int | None = None, tag: str = "ce") -> None:
        self.rank, self.world, self.store = int(rank), int(world), store
        if self.world < 2:
            raise ValueError("the copy-engine exchange needs two ranks or more")
        # peers in rotated order (rank + 1, rank + 2, ..): at each moment every rank copies to a different peer
        self.peers = [(self.rank + i) % self.world for i in range(1, self.world)]
        self.slot = slot                                         # bytes a region (two a peer slot): set in
        self.region = self.arena_bytes = 0                       # ``_first`` with the timeout, so a refused
        self.timeout = 300.0                                     # setting reaches every rank through the vote
        self.step = 0
        self.poisoned: str | None = None
        self.key = f"tf_{tag}"
        self.device = "cuda"
        self.lib = None
        self._name = None
        # two phases with a vote after each: every key a rank reads in ``_connect`` was written before the first vote
        # passed, so a rank that fails early never leaves the others blocked on the store
        try:
            for label, phase in (("local", self._first), ("connect", self._connect)):
                problem = ""
                try:
                    phase()
                except Exception as exc:               # noqa: BLE001  (every rank learns it and stops)
                    problem = f"{type(exc).__name__}: {exc}"
                self._vote(label, problem)
        finally:
            self._unlink()                             # rank 0's table: every rank has mapped it by now, or never will
        self._probe()

    # -- setup ---------------------------------------------------------------------------------------------------------
    def _check(self, code: int, what: str) -> None:
        if code != 0:
            raise RuntimeError(f"{what}: CUDA error {code} ({self.lib.cudaGetErrorString(code).decode()})")

    def _vote(self, label: str, problem: str) -> None:
        """Every rank's outcome of ``label``; all stop when any failed (no rank is left waiting in an exchange)."""

        s = self.store
        s.set(f"{self.key}/{label}/{self.rank}", ("0:" + problem) if problem else "1")
        bad = []
        for r in range(self.world):
            v = s.get(f"{self.key}/{label}/{r}")
            v = v.decode() if isinstance(v, bytes) else str(v)
            if v != "1":
                bad.append(f"rank {r}: {v[2:]}")
        if bad:
            raise RuntimeError("the copy-engine exchange could not start (" + "; ".join(bad) + "). It needs every "
                               "rank on one host, each process seeing every GPU (its own first) with peer access, "
                               f"and BAR1 room for its {self.arena_bytes >> 20} MiB arena a GPU beside NCCL's "
                               "peer buffers (nvidia-smi -q -d MEMORY shows BAR1; TENSORFOLD_CE_SLOT_MIB sizes the "
                               "arena); "
                               "TENSORFOLD_EXCHANGE=nccl is NCCL's exchange.")

    def _entries(self) -> int:
        return self.world + self.world * self.world  # each sender's last step; each receiver's last wait a sender

    def _first(self) -> None:
        """The first phase: the settings (read here, so a value one rank refuses stops every rank through the vote),
        then ``_local``."""

        self.region = slot_bytes() if self.slot is None else int(self.slot)
        self.arena_bytes = (self.world - 1) * 2 * self.region
        self.timeout = timeout_s()
        self._local()

    def _local(self) -> None:
        """This rank's part: the arena (allocated and exported), its events (exported), its host identity; rank 0
        also creates the host table. Writes every key ``_connect`` reads."""

        torch.cuda.current_device()
        self.lib = lib = _cudart()
        s = self.store
        # one host (the table is host shared memory): the kernel's boot id, the same in every container on it
        with open("/proc/sys/kernel/random/boot_id") as f:
            self.host = f.read().strip()
        s.set(f"{self.key}/host/{self.rank}", f"{self.host} {self.region}")
        ptr = ctypes.c_void_p()
        self._check(lib.cudaMalloc(ctypes.byref(ptr), self.arena_bytes), "cudaMalloc (arena)")
        self.base = int(ptr.value)
        arena = torch.as_tensor(_Raw(self.base, self.arena_bytes), device="cuda")
        if arena.data_ptr() != self.base or arena.numel() != self.arena_bytes:
            raise RuntimeError("the arena's tensor view does not cover the allocation")
        self.arena = arena                                      # uint8 [(world - 1) x 2 x region]
        handle = _IpcHandle()
        self._check(lib.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(self.base)), "cudaIpcGetMemHandle")
        s.set(f"{self.key}/mem/{self.rank}", bytes(handle))
        self.mine = [torch.cuda.Event(enable_timing=False, blocking=False, interprocess=True) for _ in range(SLOTS)]
        for k, ev in enumerate(self.mine):
            ev.record()                                         # so that each exists
            s.set(f"{self.key}/ev/{self.rank}/{k}", bytes(ev.ipc_handle()))
        torch.cuda.synchronize()
        if self.rank == 0:                                      # the host table, zeroed
            name = f"/dev/shm/tf-{self.key}-{uuid.uuid4().hex}"
            with open(name, "wb") as f:
                f.write(b"\0" * (8 * self._entries()))
            self._name = name
            s.set(f"{self.key}/table", name)

    def _connect(self) -> None:
        """The peers' parts: the same host, the host table mapped, the peers' arenas and events opened."""

        s, lib = self.store, self.lib
        for r in self.peers:
            v = s.get(f"{self.key}/host/{r}")
            host, region = (v.decode() if isinstance(v, bytes) else str(v)).split()
            if host != self.host:
                raise RuntimeError(f"rank {r} runs on another host")
            if int(region) != self.region:
                raise RuntimeError(f"rank {r} uses regions of {int(region) >> 20} MiB, this rank "
                                   f"{self.region >> 20} MiB (TENSORFOLD_CE_SLOT_MIB): give every rank the same")
        name = s.get(f"{self.key}/table")
        name = name.decode() if isinstance(name, bytes) else str(name)
        fd = os.open(name, os.O_RDWR)
        try:
            self._map = mmap.mmap(fd, 8 * self._entries())
        finally:
            os.close(fd)                                        # the mapping keeps the file
        self.tab = np.ndarray((self._entries(),), dtype=np.int64, buffer=self._map)
        self.remote = {}
        self.theirs = {}
        dev = torch.cuda.current_device()
        for p in self.peers:
            raw = bytes(s.get(f"{self.key}/mem/{p}"))
            if len(raw) != ctypes.sizeof(_IpcHandle):
                raise RuntimeError(f"rank {p}'s arena handle has {len(raw)} bytes, not {ctypes.sizeof(_IpcHandle)}")
            h = _IpcHandle.from_buffer_copy(raw)
            rp = ctypes.c_void_p()
            self._check(lib.cudaIpcOpenMemHandle(ctypes.byref(rp), h, _LAZY_PEER), f"cudaIpcOpenMemHandle (rank {p})")
            self.remote[p] = int(rp.value)
            self.theirs[p] = [torch.cuda.Event.from_ipc_handle(dev, bytes(s.get(f"{self.key}/ev/{p}/{k}")))
                              for k in range(SLOTS)]

    def _unlink(self) -> None:
        if self._name:
            try:
                os.unlink(self._name)
            except OSError:
                pass
            self._name = None

    def _probe(self) -> None:
        """One exchange of rank-numbered data through every peer slot and both regions (two steps), checked on every
        rank; every rank reaches every vote whatever fails, and the probe's waits give up after 60 s."""

        problem = ""
        timeout, self.timeout = self.timeout, min(self.timeout, 60.0)
        try:
            for step in range(2):
                n = 3 + step
                sends = {p: [torch.full((n,), float(1000 * self.rank + 10 * p + step), dtype=torch.float32,
                                        device=self.device)] for p in self.peers}
                recvs = {p: [torch.full((n,), -1.0, dtype=torch.float32, device=self.device)] for p in self.peers}
                self.exchange_all(sends, recvs)
                for p in self.peers:
                    if not bool(torch.all(recvs[p][0] == float(1000 * p + 10 * self.rank + step))):
                        raise RuntimeError(f"the startup pattern from rank {p} did not arrive (step {step})")
            self._sync()
        except Exception as exc:                       # noqa: BLE001
            problem = f"{type(exc).__name__}: {exc}"
        finally:
            self.timeout = timeout
        self._vote("probe", problem)

    # -- the protocol ------------------------------------------------------------------------------------------------
    def _await(self, idx: int, target: int, what: str) -> None:
        tab = self.tab
        if tab[idx] >= target:
            return
        t0 = time.monotonic()
        spins = 0
        while tab[idx] < target:
            spins += 1
            if spins > 2000:
                time.sleep(0.00002)
                if spins % 4096 == 0 and time.monotonic() - t0 > self.timeout:
                    raise RuntimeError(f"the copy-engine exchange: rank {self.rank} waited {self.timeout:.0f} s for "
                                       f"{what} at step {target}; a rank stopped or the ranks are out of step")

    def _step(self, copies: list[tuple[int, int, int]]) -> int:
        """One step on the current stream: this rank's copies (dst, src, bytes) into the peers' arenas, its event,
        then a wait for every peer's event of the same step (their copies into this rank's arena are then done).
        Returns the step's number."""

        e = self.step = self.step + 1
        k = e % SLOTS
        W = self.world
        for q in self.peers:                     # every receiver waited on slot k's previous record
            self._await(W + q * W + self.rank, e - SLOTS, f"rank {q}'s wait")
        stream = self._stream()
        for dst, src, nbytes in copies:
            if nbytes:
                self._copy(dst, src, nbytes, stream)
        self._record(k, stream)
        self.tab[self.rank] = e
        for p in self.peers:
            self._await(p, e, f"rank {p}'s copies")
            self._wait(p, k, stream)
            self.tab[W + self.rank * W + p] = e
        return e

    # the device operations (a CPU test replaces these)
    def _sync(self) -> None:
        torch.cuda.synchronize()

    def _stream(self):
        return torch.cuda.current_stream()

    def _copy(self, dst: int, src: int, nbytes: int, stream) -> None:
        self._check(self.lib.cudaMemcpyAsync(ctypes.c_void_p(dst), ctypes.c_void_p(src), nbytes, _D2D,
                                             ctypes.c_void_p(stream.cuda_stream)), "cudaMemcpyAsync (to a peer)")

    def _record(self, k: int, stream) -> None:
        self.mine[k].record(stream)

    def _wait(self, p: int, k: int, stream) -> None:
        stream.wait_event(self.theirs[p][k])

    # -- the API (comm.NCCL's) -----------------------------------------------------------------------------------------
    def _region(self, receiver: int, sender: int, step: int) -> int:
        """Byte offset in ``receiver``'s arena of ``sender``'s region for ``step``: slot (sender - receiver - 1) mod
        N, region step mod 2."""

        return (((sender - receiver - 1) % self.world) * 2 + step % 2) * self.region

    def exchange_all(self, sends: dict[int, list[torch.Tensor]], recvs: dict[int, list[torch.Tensor]]) -> None:
        """Every ``sends[p][i]`` to peer p and ``recvs[p][i]`` from it, on the current stream (``comm.NCCL``'s
        contract; here every call names every other rank, with the same byte count to each)."""

        if self.poisoned is not None:
            raise RuntimeError(f"the copy-engine exchange stopped after an error ({self.poisoned}); restart every rank")
        if set(sends) != set(self.peers) or set(recvs) != set(self.peers):
            raise ValueError("ce exchange_all: every other rank, each with its sends and receives")
        payload = None
        for p in self.peers:
            if len(sends[p]) != len(recvs[p]):
                raise ValueError("ce exchange_all: one receive per send")
            for s, r in zip(sends[p], recvs[p]):
                if s.numel() != r.numel() or s.dtype != r.dtype or not (s.is_contiguous() and r.is_contiguous()):
                    raise ValueError("ce exchange_all: each send and receive must be contiguous and alike")
            size = layout(sends[p])[1]
            if payload is not None and size != payload:
                raise ValueError("ce exchange_all: the same bytes to every peer")
            payload = size
        try:
            self._exchange(sends, recvs, payload or 0)
        except Exception as exc:
            self.poisoned = f"{type(exc).__name__}: {exc}"
            raise

    def exchange(self, sends: list[torch.Tensor], recvs: list[torch.Tensor], peer: int) -> None:
        """``exchange_all`` with one peer: two ranks only (every call of this transport names every other rank)."""

        if self.world != 2 or peer != self.peers[0]:
            raise ValueError("ce exchange: one peer only at two ranks (use exchange_all)")
        self.exchange_all({peer: list(sends)}, {peer: list(recvs)})

    def _exchange(self, sends, recvs, payload: int) -> None:
        R = self.region
        steps = max(1, -(-payload // R))
        lay = {p: layout(sends[p])[0] for p in self.peers}
        for i in range(steps):
            lo, hi = i * R, min(payload, (i + 1) * R)
            e = self.step + 1                                  # the step _step will number
            copies = []
            for q in self.peers:
                dst = self.remote[q] + self._region(q, self.rank, e)
                for t, at in zip(sends[q], lay[q]):
                    a, z = max(lo, at), min(hi, at + t.numel() * t.element_size())
                    if a < z:
                        copies.append((dst + a - lo, t.data_ptr() + a - at, z - a))
            if self._step(copies) != e:
                raise RuntimeError("the copy-engine exchange's step count moved under a call")
            for p in self.peers:                               # this step's bytes from p into its receives
                src = self._region(self.rank, p, e)
                for t, at in zip(recvs[p], lay[p]):
                    a, z = max(lo, at), min(hi, at + t.numel() * t.element_size())
                    if a < z:
                        dst = t.view(-1).view(torch.uint8)
                        dst[a - at:z - at].copy_(self.arena[src + a - lo:src + z - lo])
