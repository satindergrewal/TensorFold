"""Launch tables: which of a kernel's same-bit launch configurations runs, per GPU model, shape and row count.

The GLM kernels were tuned on a GB10 (48 SMs, LPDDR5x) and nothing autotunes. Each kernel listed in ``CHOICES``
has several launch configurations that give the same bits by construction (only tiles, warps, loads, staging or fused
epilogues change; every output keeps its own chain of float operations). ``tools/glm_autotune.py`` runs them on a
GPU, keeps only those whose outputs equal the default configuration's bit for bit at every tested row count, times
those, and writes the winners to a JSON launch table for that GPU model.

Settings:
  TF_GLM_TUNE=<table.json>   read the table at start. Unset, empty, 0 or off: no table, every kernel runs exactly as
                             without this module (the off switch).
  TF_GLM_TUNE_RECORD=<file>  every tunable call adds its (kernel, shape, rows) to that file (rewritten when a new one
                             appears), so the harness can tune exactly the shapes a start and a few requests use.

A table file holds one table per GPU model, each keyed by the device's name, SM count and compute capability; the
engine takes the one matching the GPU it runs on, and on any other GPU (or with no match) every kernel keeps its
default launch. A table entry takes precedence over the kernel's own environment setting (TF_GLM_EXL3_LOADS,
TF_GLM_Q4_TILE, ...) for the shapes and row counts it lists; everything it does not list keeps the setting or default.

Every value is checked at load against the kernel's list of same-bit choices (``CHOICES``): a value outside it, or a
malformed file, refuses the start, so a table can only pick among configurations that keep the bits. The checksum of
the whole file's tables joins the settings every rank compares at start (``code``), so all ranks read one file.

Row buckets: a kernel's entry for a shape is one value for every row count, or a dict {"<rows>": value} whose keys
are the upper bounds of row buckets; a call of R rows takes the smallest bound >= R (none: the default).

Licensed under the Apache License, Version 2.0. Builds on TensorFold (Ash Hart and the TensorFold contributors) and
on the GLM-5.3-Flash recipe and patches 0001-0056 by MiaAI-Lab."""

from __future__ import annotations

import json
import os
import threading
import zlib
from typing import Any, Callable

FORMAT = "tensorfold-launch-table/1"
CALLS_FORMAT = "tensorfold-launch-calls/1"
OFF = ("", "0", "off", "none")


def _ints(*allowed: int) -> Callable[[Any], int]:
    def check(v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int) or v not in allowed:
            raise ValueError(f"one of {allowed}, not {v!r}")
        return int(v)
    return check


def _keys(**fields: Callable[[Any], Any]) -> Callable[[Any], dict]:
    def check(v: Any) -> dict:
        if not isinstance(v, dict) or set(v) - set(fields) or not set(v):
            raise ValueError(f"a dict of {sorted(fields)}, not {v!r}")
        return {k: fields[k](x) for k, x in v.items()}
    return check


def _int_range(lo: int, hi: int) -> Callable[[Any], int]:
    def check(v: Any) -> int:
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            raise ValueError(f"an integer {lo}..{hi}, not {v!r}")
        return int(v)
    return check


def _exl3(check: Callable[[Any], dict]) -> Callable[[Any], dict]:
    def both(v: Any) -> dict:
        out = check(v)
        if out.get("wn", 1) == 4 and out.get("ld", 0) != 0:
            raise ValueError("4 warps along N only with the 32-bit loads (ld 0)")
        return out
    return both


# Every tunable kernel's same-bit choices (the kernels' own notes give the reasons; the harness re-checks the bits).
CHOICES: dict[str, Callable[[Any], Any]] = {
    # decode EXL3 experts, gate/up (exl3.cu dec_kernel): loads (0 32-bit; 1, 2, 3: 16-byte, 1, 2 or 4 steps ahead),
    # warps along N a block (1 as before; the K warps and their order are fixed), the gate/up epilogue fused into the
    # last block (2) or its own kernel (0), one rotated input a row for a layer whose experts share one suh (1) or
    # one a pair (0)
    "exl3_dec_gu": _exl3(_keys(ld=_ints(0, 1, 2, 3), wn=_ints(1, 2, 4), fuse=_ints(0, 2), xrow=_ints(0, 1))),
    # decode EXL3 experts, down: loads, warps along N, epilogue fused (1) or its own kernel (0)
    "exl3_dec_dn": _exl3(_keys(ld=_ints(0, 1, 2, 3), wn=_ints(1, 2, 4), fuse=_ints(0, 1))),
    # prompt EXL3 experts: members a pass (exl3_mm.prompt_pass)
    "exl3_prompt": _keys(passm=_ints(64, 128)),
    # 4-bit dense matmul of decode rows: the lane matmul's tile config (qmm.cu dispatch_cfg 0..15), -1 the stock tile
    "q4_dec": _int_range(-1, 15),
    # 4-bit dense matmul of prompt chunks: the prefill tile (qmm_prefill.cu dispatch 0..11)
    "q4_prefill": _int_range(0, 11),
    # BF16 / FP8 dense matmuls of decode rows (qmm._bmm / _fmm): Triton warps and stages, the FP8 column tile
    "b16_dec": _keys(warps=_ints(2, 4, 8), stages=_int_range(1, 8)),
    "f8_dec": _keys(warps=_ints(2, 4, 8), stages=_int_range(1, 8), bn=_ints(16, 32, 64, 128)),
    # KDA decode chain's step kernel: warps a block (4 value rows each; 32 / warps blocks a head), rows staged a time
    "kda_step": _keys(warps=_ints(1, 2, 4, 8), tr=_ints(8, 16)),
    # L2 prefetch: programs one wave of a 4-bit decode matmul holds (l2pf.ONE_WAVE; GB10's 192 = 4 x 48 SMs)
    "l2pf": _keys(one_wave=_int_range(1, 1 << 16)),
}

_lock = threading.Lock()
_loaded: tuple[str, dict, int] | None = None        # (path, this GPU's kernels, crc of the file's tables)
_record: dict | None = None
_said = False
_PATH = os.environ.get("TF_GLM_TUNE", "")            # read once: lookups run on every eager prompt matmul
_RECORD = os.environ.get("TF_GLM_TUNE_RECORD", "")
_DEVICE: dict | None = None                         # a test's or the harness's device (None: the current GPU)


def reset(path: str | None = None, record_path: str | None = None, device: dict | None = None) -> None:
    """Forget the loaded table: the next lookup reads ``path`` (default: TF_GLM_TUNE again) for ``device`` (default:
    the current GPU, ``device_key``). For tests and the harness."""

    global _loaded, _record, _said, _PATH, _RECORD, _DEVICE
    with _lock:
        _loaded, _record, _said = None, None, False
        _PATH = os.environ.get("TF_GLM_TUNE", "") if path is None else path
        _RECORD = os.environ.get("TF_GLM_TUNE_RECORD", "") if record_path is None else record_path
        _DEVICE = device


def device_key() -> dict | None:
    """The current GPU as a table's key: {"name", "sms", "capability"}; None without a CUDA GPU."""

    if _DEVICE is not None:
        return dict(_DEVICE)
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        p = torch.cuda.get_device_properties(torch.cuda.current_device())
        return {"name": str(p.name), "sms": int(p.multi_processor_count), "capability": [int(p.major), int(p.minor)]}
    except Exception:  # noqa: BLE001 - no usable GPU: no table applies
        return None


def same_device(a: dict, b: dict | None) -> bool:
    return b is not None and a.get("name") == b.get("name") and a.get("sms") == b.get("sms") and \
        list(a.get("capability", ())) == list(b.get("capability", ()))


def _say(text: str) -> None:
    print(f"[tensorfold] {text}", flush=True)


def validate_kernels(kernels: Any, where: str) -> dict:
    """A table's kernels, every value checked against ``CHOICES`` (ValueError otherwise); row buckets as ints."""

    if not isinstance(kernels, dict):
        raise ValueError(f"{where}: 'kernels' must be an object")
    out: dict = {}
    for name, shapes in kernels.items():
        if name not in CHOICES:
            raise ValueError(f"{where}: unknown kernel {name!r} (known: {', '.join(sorted(CHOICES))})")
        if not isinstance(shapes, dict):
            raise ValueError(f"{where}: {name} must map shapes to values")
        check = CHOICES[name]
        table: dict = {}
        for key, val in shapes.items():
            try:
                if isinstance(val, dict) and val and all(isinstance(k, str) and k.isdigit() for k in val):
                    table[str(key)] = {int(b): check(v) for b, v in sorted(val.items(), key=lambda kv: int(kv[0]))}
                else:
                    table[str(key)] = check(val)
            except ValueError as exc:
                raise ValueError(f"{where}: {name}[{key!r}]: {exc}") from None
        out[name] = table
    return out


def validate_device(dev: Any, where: str) -> dict:
    if not (isinstance(dev, dict) and isinstance(dev.get("name"), str) and isinstance(dev.get("sms"), int)
            and isinstance(dev.get("capability"), list) and len(dev["capability"]) == 2
            and all(isinstance(c, int) for c in dev["capability"])):
        raise ValueError(f"{where}: 'device' must be {{name: str, sms: int, capability: [major, minor]}}")
    return {"name": dev["name"], "sms": dev["sms"], "capability": list(dev["capability"])}


def parse(doc: Any, where: str) -> list[tuple[dict, dict]]:
    """A launch table file's (device, kernels) pairs, all validated."""

    if not isinstance(doc, dict) or doc.get("format") != FORMAT or not isinstance(doc.get("tables"), list):
        raise ValueError(f"{where}: not a {FORMAT} file (format, tables)")
    out = []
    for i, t in enumerate(doc["tables"]):
        if not isinstance(t, dict):
            raise ValueError(f"{where}: tables[{i}] must be an object")
        dev = validate_device(t.get("device"), f"{where}: tables[{i}]")
        if any(same_device(dev, d) for d, _ in out):
            raise ValueError(f"{where}: two tables for {dev['name']} ({dev['sms']} SMs)")
        out.append((dev, validate_kernels(t.get("kernels", {}), f"{where}: tables[{i}]")))
    return out


def load(path: str | None = None) -> dict:
    """This GPU's kernels from the table file ``path`` (default TF_GLM_TUNE); {} when off or no table matches."""

    global _loaded, _said
    path = _PATH if path is None else path
    with _lock:
        if _loaded is not None and _loaded[0] == path:
            return _loaded[1]
        if path.strip().lower() in OFF:
            _loaded = (path, {}, 0)
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"TF_GLM_TUNE={path}: cannot read the launch table ({exc})") from None
        tables = parse(doc, f"TF_GLM_TUNE={path}")
        canon = json.dumps([[d, k] for d, k in tables], sort_keys=True, separators=(",", ":"), default=str)
        crc = (zlib.crc32(canon.encode()) & 0x7FFFFFFF) or 1
        dev = device_key()
        kernels = next((k for d, k in tables if same_device(d, dev)), {})
        _loaded = (path, kernels, crc)
        if not _said:
            _said = True
            what = "no GPU" if dev is None else f"{dev['name']} ({dev['sms']} SMs, sm_{dev['capability'][0]}" \
                                                f"{dev['capability'][1]})"
            if kernels:
                n = sum(len(v) for v in kernels.values())
                _say(f"launch table {path}: {n} entries over {len(kernels)} kernels for {what} (crc {crc:08x})")
            else:
                _say(f"launch table {path}: no table for {what} among {len(tables)}: default launches")
        return kernels


def code() -> int:
    """The checksum of the table file's tables (0: off), for the settings every rank compares at start."""

    load()
    return _loaded[2] if _loaded is not None else 0


def pick(kernel: str, key: str, rows: int | None = None) -> Any:
    """The table's value for ``kernel`` at shape ``key`` and ``rows`` rows, or None (the kernel's default)."""

    if _RECORD:
        record(kernel, key, rows)
    loaded = _loaded
    table = loaded[1] if loaded is not None and loaded[0] == _PATH else load()
    entry = table.get(kernel, {}).get(key) if table else None
    if isinstance(entry, dict) and entry and all(isinstance(b, int) for b in entry):
        if rows is None:
            return None
        for bound, value in entry.items():               # sorted ascending at load
            if rows <= bound:
                return value
        return None
    return entry


def record(kernel: str, key: str, rows: int | None) -> None:
    """TF_GLM_TUNE_RECORD: remember (kernel, key, rows); the file is rewritten when a new one appears."""

    global _record
    path = _RECORD
    if not path or path.strip().lower() in OFF:
        return
    item = (kernel, key, -1 if rows is None else int(rows))
    with _lock:
        if _record is None:
            _record = {}
            try:
                with open(path, encoding="utf-8") as f:
                    for k, v in (json.load(f).get("calls") or {}).items():
                        for s, rs in v.items():
                            for r in rs:
                                _record[(k, s, int(r))] = True
            except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
                pass
        if item in _record:
            return
        _record[item] = True
        calls: dict = {}
        for k, s, r in sorted(_record):
            calls.setdefault(k, {}).setdefault(s, []).append(r)
        tmp = f"{path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"format": CALLS_FORMAT, "calls": calls}, f, indent=1, sort_keys=True)
            os.replace(tmp, path)
        except OSError:
            pass


def shape(*dims: int) -> str:
    """The key of a shape: dims joined by 'x' (e.g. a weight's n x k)."""

    return "x".join(str(int(d)) for d in dims)


__all__ = ["CALLS_FORMAT", "CHOICES", "FORMAT", "code", "device_key", "load", "parse", "pick", "record", "reset",
           "same_device", "shape", "validate_device", "validate_kernels"]
