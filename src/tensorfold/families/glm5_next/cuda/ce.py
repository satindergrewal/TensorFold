"""TF_GLM_HC_EXCHANGE=ce (patch 0122): the row split's exchanges (``hcsplit``) by the GPUs' copy engines, peer to
peer over PCIe, between the rank processes through CUDA IPC, instead of NCCL send/receive.

Why: on a host whose GPUs sit on PCIe (no NVLink), NCCL moves send/receive data with a few SMs per peer, and over
host shared memory where its peer-to-peer path is slow; a prompt chunk's row split moves about 36 MiB out of every
rank at each of its 90 exchange sites (2,048 rows, four ranks). A copy engine writes straight into the peer's memory at
the link's speed and takes no SMs from the compute running beside it.

How: each rank allocates a receive arena with cudaMalloc, exports it (cudaIpcGetMemHandle) and maps every peer's
(cudaIpcOpenMemHandle). The arena holds the partials' staging blocks exactly as ``hcsplit`` lays them out in its
all-gather buffer ([N, n, D] fp32 a piece: the peers' partials of this rank's rows, its own copied in by hc_post's
caller) and a staging area for the swapped bf16 rows ([T, N - 1, H, D]: tensor, sender, row). An exchange step is: the
sender's copies into each peer's arena (cudaMemcpyAsync on the exchange stream), then one interprocess CUDA event
recorded; each receiver makes its stream wait for every peer's event of that step and (rows) copies the rows from its
staging into place. Every rank runs the same steps in the same order, so a step number names them. A small host
shared-memory table (one int64 per sender: the last step it recorded; one per receiver and sender: the last step the
receiver waited for) makes a receiver wait for the record of the right step (a stream wait refers to the event's last
record at the time of the call) and keeps a sender from re-recording one of its SLOTS events before every receiver has
waited on its previous use. A staging area is never overwritten while its reader may still need it: a rank issues all
its steps in its stream order (one stream, or streams joined by stream waits between kinds of chunk) and a step's copies
start after its stream waited for every peer's record of the step before; a reader's last read of a staging area (its
local row copies, or hc_post on a partials block, which its next rows step waits for through ev_glue) sits before one of
its own later records, and the next write into that area comes only after a step that waited for that record.

Exact: the same bytes land in the same places NCCL's exchange put them; the consumers are unchanged (transport only).
A failed step (a peer that stopped, a CUDA error, a timeout) poisons the exchange: every later step raises, and the
ranks must be restarted together. GPU-side waits on a peer that died mid-step do not time out (as with NCCL); the
host-side ones do (TF_GLM_CE_TIMEOUT_S). The same protocol, with its own staging, also runs behind comm.exchange /
exchange_all as a transport of the communicator (``tensorfold.cuda.ce_comm``, TENSORFOLD_EXCHANGE=ce).

Needs: every rank on one host, each process seeing every GPU (its own first: TensorFold uses device 0) with peer
access between them, and room in each GPU's BAR1 for the arena (about 7 x R x D bytes: 56 MiB at 2,048 rows, 112 MiB
at 4,096; TF_GLM_CE_ARENA_MIB caps it, default 192). The startup check exchanges a pattern and stops the server, on
every rank, if anything is missing or wrong."""

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
        lib.cudaMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t, ctypes.c_void_p]
        return lib
    raise RuntimeError(f"TF_GLM_HC_EXCHANGE=ce: the CUDA runtime library was not found ({err})")


class _Raw:
    """A device pointer as a CUDA array (torch.as_tensor's view of memory this module allocated)."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {"shape": (nbytes,), "typestr": "|u1", "data": (ptr, False),
                                         "version": 3, "strides": None}


def arena_bytes(rows: int, world: int, width: int, tensors: int, narrow: int = 0) -> tuple[int, int]:
    """(partials' staging bytes, rows' staging bytes) for prompt buffers of ``rows`` rows: H = ceil(rows / world)
    rows a rank; [N, H, D] fp32 (hc_post reads every rank's slot) and [T, N - 1, H, D] bf16 (one slot a peer), then
    ``narrow`` bytes a row for the narrow row buffers ``rows`` may carry beside them ([N - 1, H, narrow]: a sender's
    rows of each narrow buffer back to back; TF_GLM_MOE_GLUE rowsplit's picks and weights)."""

    H = -(-int(rows) // int(world))
    return world * H * width * 4, tensors * (world - 1) * H * width * 2 + (world - 1) * H * int(narrow)


def arena_cap() -> int:
    value = (os.environ.get("TF_GLM_CE_ARENA_MIB", "") or "192").strip()
    if not value.isdecimal() or int(value) < 1:
        raise ValueError(f"TF_GLM_CE_ARENA_MIB: MiB (a whole number), not {value!r}")
    return int(value) << 20


def timeout_s() -> float:
    value = (os.environ.get("TF_GLM_CE_TIMEOUT_S", "") or "300").strip()
    try:
        return max(1.0, float(value))
    except ValueError:
        raise ValueError(f"TF_GLM_CE_TIMEOUT_S: seconds, not {value!r}") from None


class CeExchange:
    """One rank's copy-engine exchange for a prompt buffer's row split (set up collectively by every rank)."""

    def __init__(self, rank: int, world: int, store, rows: int, width: int, tensors: int, tag: str = "hcce",
                 narrow: int = 0) -> None:
        self.rank, self.world, self.store = int(rank), int(world), store
        self.narrow = int(narrow)            # bytes a row of narrow row buffers ``rows`` carries (0: none)
        # peers in rotated order (rank + 1, rank + 2, ..): at each moment every rank copies to a different peer, so
        # no GPU takes three senders' traffic at once while another takes none
        self.peers = [(self.rank + i) % self.world for i in range(1, self.world)]
        self.width, self.tensors = int(width), int(tensors)
        self.H = -(-int(rows) // self.world)
        self.rs_bytes, self.ag_bytes = arena_bytes(rows, world, width, tensors, self.narrow)
        self.ag_main = self.tensors * (self.world - 1) * self.H * self.width * 2      # the bf16 rows' part
        self.timeout = timeout_s()
        self.step = 0
        self.key = f"tf_{tag}"
        self.device = "cuda"
        self.lib = None
        self.poisoned: str | None = None
        self._name = None
        # two phases with a vote after each: every key a rank reads in ``_connect`` was written before the first
        # vote passed, so a rank that fails early never leaves the others blocked on the store
        try:
            for label, phase in (("local", self._local), ("connect", self._connect)):
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
            raise RuntimeError("TF_GLM_HC_EXCHANGE=ce could not start (" + "; ".join(bad) + "). It needs every rank "
                               "on one host, each process seeing every GPU (its own first) with peer access, and "
                               f"BAR1 room for the arena ({(self.rs_bytes + self.ag_bytes) >> 20} MiB a GPU beside "
                               "NCCL's peer buffers: nvidia-smi -q -d MEMORY shows BAR1; shorter prompt chunks "
                               "shrink it); TF_GLM_HC_EXCHANGE=p2p is NCCL's exchange.")

    def _local(self) -> None:
        """This rank's part: the arena (allocated and exported), its events (exported), its host identity; rank 0
        also creates the host table. Writes every key ``_connect`` reads."""

        total = self.rs_bytes + self.ag_bytes
        if total > arena_cap():
            raise RuntimeError(f"the arena would take {total >> 20} MiB, past TF_GLM_CE_ARENA_MIB "
                               f"({arena_cap() >> 20}): shorter prompt chunks (TF_GLM_PREFILL_ROWS) or a larger cap")
        torch.cuda.current_device()
        self.lib = lib = _cudart()
        s = self.store
        # one host (the table is host shared memory): the kernel's boot id, the same in every container on it
        with open("/proc/sys/kernel/random/boot_id") as f:
            self.host = f.read().strip()
        s.set(f"{self.key}/host/{self.rank}", self.host)
        # the receive arena, exported
        ptr = ctypes.c_void_p()
        self._check(lib.cudaMalloc(ctypes.byref(ptr), total), "cudaMalloc (arena)")
        self.base = int(ptr.value)
        arena = torch.as_tensor(_Raw(self.base, total), device="cuda")
        if arena.data_ptr() != self.base or arena.numel() != total:
            raise RuntimeError("the arena's tensor view does not cover the allocation")
        self.rs = arena[:self.rs_bytes].view(torch.float32)
        self.ag = arena[self.rs_bytes:self.rs_bytes + self.ag_main].view(torch.bfloat16).view(
            self.tensors, self.world - 1, self.H, self.width)
        self.nw = arena[self.rs_bytes + self.ag_main:]       # the narrow rows' part (bytes), empty without it
        handle = _IpcHandle()
        self._check(lib.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(self.base)), "cudaIpcGetMemHandle")
        s.set(f"{self.key}/mem/{self.rank}", bytes(handle))
        # events, one ring a rank, exported (recorded once so they exist)
        self.mine = [torch.cuda.Event(enable_timing=False, blocking=False, interprocess=True) for _ in range(SLOTS)]
        for k, ev in enumerate(self.mine):
            ev.record()
            s.set(f"{self.key}/ev/{self.rank}/{k}", bytes(ev.ipc_handle()))
        torch.cuda.synchronize()
        if self.rank == 0:                            # the host table, zeroed
            name = f"/dev/shm/tf-{self.key}-{uuid.uuid4().hex}"
            with open(name, "wb") as f:
                f.write(b"\0" * (8 * self._entries()))
            self._name = name
            s.set(f"{self.key}/table", name)

    def _entries(self) -> int:
        return self.world + self.world * self.world  # each sender's last step; each receiver's last wait a sender

    def _connect(self) -> None:
        """The peers' parts: the same host, the host table mapped, the peers' arenas and events opened."""

        s, lib = self.store, self.lib
        for r in self.peers:
            v = s.get(f"{self.key}/host/{r}")
            if (v.decode() if isinstance(v, bytes) else str(v)) != self.host:
                raise RuntimeError(f"rank {r} runs on another host")
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
        """Every rank writes its rank number into its slot of each peer's partials staging, through one step; each
        checks what arrived. Every rank reaches every vote whatever fails (no rank is left waiting on another's key),
        and the probe's own waits give up after 60 s."""

        d, n = self.width, min(self.H, 2)
        problem = ""
        try:
            src = torch.full((n, d), float(self.rank + 1), dtype=torch.float32, device=self.device)
            self.rs[:self.world * n * d].zero_()
            self._sync()
        except Exception as exc:                       # noqa: BLE001
            problem = f"{type(exc).__name__}: {exc}"
        self._vote("zeroed", problem)
        timeout, self.timeout = self.timeout, min(self.timeout, 60.0)
        try:
            copies = [(self.remote[q] + (self.rank * n * d) * 4, src.data_ptr(), n * d * 4) for q in self.peers]
            self.exchange(copies)
            got = self.rs[:self.world * n * d].view(self.world, n, d)
            for p in self.peers:
                if not bool(torch.all(got[p] == float(p + 1))):
                    raise RuntimeError(f"the startup pattern from rank {p} did not arrive")
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
                    raise RuntimeError(f"TF_GLM_HC_EXCHANGE=ce: rank {self.rank} waited {self.timeout:.0f} s for "
                                       f"{what} at step {target}; a rank stopped or the ranks are out of step")

    def exchange(self, copies: list[tuple[int, int, int]]) -> None:
        """One step on the current stream: this rank's copies (dst, src, bytes) into the peers' arenas, its event,
        then a wait for every peer's event of the same step (its copies into this rank's arena are then done)."""

        if self.poisoned is not None:
            raise RuntimeError(f"TF_GLM_HC_EXCHANGE=ce stopped after an error ({self.poisoned}); restart every rank")
        try:
            self._exchange(copies)
        except Exception as exc:
            self.poisoned = f"{type(exc).__name__}: {exc}"
            raise

    def _exchange(self, copies: list[tuple[int, int, int]]) -> None:
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

    # -- the row split's two exchanges ---------------------------------------------------------------------------------
    def partials(self, part: torch.Tensor, H: int, off: int, n: int, at: int | None = None) -> None:
        """Piece (off, n): this rank's fp32 partial rows of each peer q (part[q H + off ..]) into q's staging block
        of the piece ([N, n, D] at N x ``at`` rows of the arena, ``at`` = off unless given; this rank's slot); the
        peers' into this rank's."""

        d = self.width
        if part.shape[1] != d or part.dtype != torch.float32 or not part.is_contiguous():
            raise ValueError("ce partials: contiguous fp32 rows of the arena's width")
        at = off if at is None else at
        if at + n > self.H:
            raise ValueError(f"ce partials: arena rows {at} .. {at + n} past its {self.H}")
        row = d * 4
        copies = [(self.remote[q] + (self.world * at + self.rank * n) * row, part[q * H + off].data_ptr(), n * row)
                  for q in self.peers]
        self.exchange(copies)

    def block(self, off: int, n: int) -> torch.Tensor:
        """The piece's staging block [N, n, D] in this rank's arena (``hcsplit``'s ``got``)."""

        d = self.width
        return self.rs[self.world * off * d:self.world * (off + n) * d].view(self.world, n, d)

    def rows(self, outs: list[torch.Tensor], H: int, off: int, n: int, at: int | None = None,
             narrow: tuple | list = ()) -> None:
        """Piece (off, n) of bf16 row buffers: this rank's rows (rank H + off ..) of each into every peer's staging
        (its rows ``at`` .., ``at`` = off unless given), then the peers' rows from this rank's staging into place
        (rows p H + off ..). ``narrow``: contiguous row buffers of any dtype whose rows together take at most the
        arena's ``narrow`` bytes, moved the same way in the same step (their staging: a sender's slot of [H, row]
        a buffer, back to back)."""

        if len(outs) > self.tensors:
            raise ValueError(f"ce rows: {len(outs)} buffers, the staging holds {self.tensors}")
        at = off if at is None else at
        if at + n > self.H:
            raise ValueError(f"ce rows: staging rows {at} .. {at + n} past its {self.H}")
        d = self.width
        row = d * 2
        mine = self.rank * H + off
        N = self.world
        copies = []
        for t, buf in enumerate(outs):
            if buf.shape[1] != d or buf.dtype != torch.bfloat16 or not buf.is_contiguous():
                raise ValueError("ce rows: contiguous bf16 rows of the arena's width")
            for q in self.peers:                 # sender r's slot in receiver q's staging: (r - q - 1) mod N
                pos = ((t * (N - 1) + (self.rank - q - 1) % N) * self.H + at) * row
                copies.append((self.remote[q] + self.rs_bytes + pos, buf[mine].data_ptr(), n * row))
        lay = self._narrow_layout(narrow)
        for buf, start, rb in lay:
            for q in self.peers:
                pos = ((self.rank - q - 1) % N) * self.H * self.narrow + start * self.H + at * rb
                copies.append((self.remote[q] + self.rs_bytes + self.ag_main + pos, buf[mine].data_ptr(), n * rb))
        self.exchange(copies)
        for t, buf in enumerate(outs):
            for p in self.peers:
                buf[p * H + off:p * H + off + n].copy_(self.ag[t, (p - self.rank - 1) % N, at:at + n])
        for buf, start, rb in lay:
            for p in self.peers:
                pos = ((p - self.rank - 1) % N) * self.H * self.narrow + start * self.H + at * rb
                got = self.nw[pos:pos + n * rb].view(buf.dtype).view(n, *buf.shape[1:])
                buf[p * H + off:p * H + off + n].copy_(got)

    def _narrow_layout(self, narrow) -> list[tuple[torch.Tensor, int, int]]:
        """(buffer, byte offset of its rows within a row's narrow bytes, row bytes) for each narrow buffer."""

        out, start = [], 0
        for buf in narrow:
            rb = buf[0].numel() * buf.element_size()
            if not buf.is_contiguous() or rb % 4:
                raise ValueError("ce rows: contiguous narrow rows of whole 4-byte words")
            out.append((buf, start, rb))
            start += rb
        if start > self.narrow:
            raise ValueError(f"ce rows: narrow rows of {start} bytes, the arena holds {self.narrow}")
        return out
