"""An ADMIT prompt field, with an optional reference to its resume snapshot.

Only the snapshot already required by ADMIT may be named. Placement and evictions
have finished before packing. The scheduler is the sole cache writer; its ordered
operation stream creates identical immutable ids/kids on every rank. Later EVICT
operations cannot overtake an ADMIT, even in the same message. No speculative or
server-side tokenization cache keys cross the wire. Full prompts remain the
fallback for cold, serial, image, displaced and non-prefix requests.
"""
from __future__ import annotations


def pack_prompt(prompt, hit=None) -> list[int]:
    if hit is not None:
        cut = len(hit.ids)
        if cut > 4 and cut <= len(prompt) and hit.ids == prompt[:cut] and hit.kid >= 0:
            return [-1, hit.kid, cut, len(prompt) - cut, *prompt[cut:]]
    return [len(prompt), *prompt]


def unpack_prompt(payload, start: int, kept_by_id, resume_kid: int):
    n = payload[start]
    if n >= 0:
        end = start + 1 + n
        if end > len(payload):
            raise RuntimeError("truncated ADMIT prompt")
        return payload[start + 1:end], end
    if n != -1 or start + 4 > len(payload):
        raise RuntimeError("invalid delta ADMIT header")
    kid, cut, count = payload[start + 1:start + 4]
    if kid != resume_kid or kid < 0 or cut < 1 or count < 0 or start + 4 + count > len(payload):
        raise RuntimeError("invalid delta ADMIT reference")
    hit = kept_by_id(kid)
    if len(hit.ids) != cut:
        raise RuntimeError("delta ADMIT snapshot length differs")
    # 0147 already retains these arrays. tolist copies native integers in C and
    # does not parse the shared tensor's full prompt into Python on every rank.
    arr = getattr(hit, "ids_np", None)
    prefix = arr.tolist() if arr is not None and len(arr) == cut else list(hit.ids)
    end = start + 4 + count
    prefix.extend(payload[start + 4:end])
    return prefix, end
