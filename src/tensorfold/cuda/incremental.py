"""Exact host prefix reuse, enabled only by TF_W1_INCREMENTAL=1.

The accepted tokenizer profile extracts literal added tokens before processing each
remaining piece independently (Split, ByteLevel, deterministic BPE). A cut at the
start of one such token therefore separates independent pieces. We check the new
literal barrier and reject matches crossing the cut, including longer added tokens.
Unknown profiles, normalization, padding, truncation and token-adding processors
use the original full encode. See tests/W1 for the real-tokenizer contract tests.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
import re
from typing import Any


def enabled() -> bool:
    value = os.environ.get("TF_W1_INCREMENTAL", "0")
    if value not in ("0", "1"):
        raise ValueError("TF_W1_INCREMENTAL must be 0 or 1")
    return value == "1"


def _extend(state, ids, before: int):
    state = state.copy()
    if ids:
        state.update((("," if before else "") + ",".join(str(int(t)) for t in ids)).encode())
    return state


@dataclass(frozen=True)
class Boundary:
    char: int
    token: int
    state: Any


@dataclass(frozen=True)
class Rendered:
    text: str
    ids: tuple[int, ...]
    boundaries: tuple[Boundary, ...]
    state: Any
    reused: int = 0

    def seed(self, salt: int | None = None) -> int:
        """engine.exact_sampling.seed_for(self.ids, salt) from the kept hash state (``salt``: TENSORFOLD_SEED_SALT by
        default, as seed_for's)."""

        if salt is None:
            from tensorfold.engine.exact_sampling import SEED_SALT as salt
        state = self.state.copy()
        state.update(f"|{salt}".encode())
        return int.from_bytes(state.digest()[:8], "little") & ((1 << 63) - 1)


class PromptIds(list):
    """A request owns its immutable rendering until its kept snapshots release it."""

    def __init__(self, record: Rendered):
        super().__init__(record.ids)
        self.record = record


def copy_prompt(prompt):
    return PromptIds(prompt.record) if isinstance(prompt, PromptIds) else list(prompt)


def keep_rendered(snap, prompt) -> None:
    if isinstance(prompt, PromptIds):
        # A grid snapshot may end inside a tokenized piece. Only earlier barriers
        # can be reused, while the immutable request keeps the character mapping.
        snap.w1_rendered = (prompt.record, len(snap.ids))


class IncrementalTokenizer:
    """Only engine-kept snapshots are eligible; active requests are never an LRU.

    ``kept`` returns a snapshot of the cache list. Eviction can race a reader safely:
    a local immutable rendering reference has no dependency on GPU cache lifetime.
    The server owns its tokenizer and does not mutate it after construction.
    """

    def __init__(self, tokenizer, kept=lambda: ()):
        self.tok, self.kept = tokenizer, kept
        cfg = json.loads(tokenizer.to_str())
        added = cfg.get("added_tokens", [])
        self.literals = {t["id"]: t["content"] for t in added}
        pp = cfg.get("post_processor")
        pre = cfg.get("pre_tokenizer") or {}
        parts = pre.get("pretokenizers", [])
        model = cfg.get("model") or {}
        self.supported = bool(added) and all(
            t["content"] and not any(t.get(k) for k in ("normalized", "single_word", "lstrip", "rstrip"))
            for t in added)
        self.supported &= (cfg.get("normalizer") is None and cfg.get("truncation") is None
                           and cfg.get("padding") is None and not tokenizer.encode_special_tokens
                           and model.get("type") == "BPE" and model.get("dropout") in (None, 0)
                           and pre.get("type") == "Sequence" and len(parts) == 2
                           and parts[0].get("type") == "Split" and parts[1].get("type") == "ByteLevel"
                           and not parts[1].get("add_prefix_space")
                           and (pp is None or (pp.get("type") == "ByteLevel" and not pp.get("trim_offsets"))))
        self.pattern = re.compile("|".join(re.escape(s) for s in sorted(set(self.literals.values()),
                                                                     key=lambda s: (-len(s), s))))
        self.longest = max(map(len, self.literals.values()), default=0)

    def _barrier(self, text: str, char: int) -> bool:
        if self.pattern.match(text, char) is None:
            return False
        # Search every possible start: finditer could hide a crossing match inside
        # an earlier overlapping match. False rejections merely encode in full.
        for start in range(max(0, char - self.longest + 1), char):
            match = self.pattern.match(text, start)
            if match is not None and match.end() > char:
                return False
        return True

    def encode(self, text: str, *, add_special_tokens: bool = False):
        if not self.supported:
            return self.tok.encode(text, add_special_tokens=add_special_tokens).ids
        best = None
        records = tuple(getattr(s, "w1_rendered", None) for s in tuple(self.kept()))
        for item in records:
            if item is None:
                continue
            record, limit = item
            if text == record.text and limit == len(record.ids):
                return PromptIds(Rendered(text, record.ids, record.boundaries, record.state, len(record.ids)))
            boundaries = record.boundaries
            lo, hi = 0, len(boundaries)
            # Prefix equality is monotone; no Python scan of the long history.
            while lo < hi:
                mid = (lo + hi) // 2
                b = boundaries[mid]
                if b.token <= limit and text.startswith(record.text[:b.char]):
                    lo = mid + 1
                else:
                    hi = mid
            for index in range(lo - 1, -1, -1):
                b = boundaries[index]
                if best is not None and b.token <= best[1].token:
                    break
                if self._barrier(text, b.char):
                    best = record, b, index
                    break
        if best is None:
            old, char, cut, boundaries, state = (), 0, 0, [], hashlib.sha256()
        else:
            record, boundary, index = best
            char, cut = boundary.char, boundary.token
            old, boundaries, state = record.ids[:cut], list(record.boundaries[:index]), boundary.state
        tail = self.tok.encode(text[char:], add_special_tokens=add_special_tokens)
        ids = tail.ids
        pos = 0
        # Offsets are character indices, not UTF-8 bytes. No decoded token text is
        # used for byte fallback, emoji, CRLF or combining characters.
        for i, (token, (start, end)) in enumerate(zip(ids, tail.offsets)):
            literal = self.literals.get(token)
            if literal is not None and text[char + start:char + end] == literal:
                state = _extend(state, ids[pos:i], cut + pos)
                pos = i
                boundaries.append(Boundary(char + start, cut + i, state))
        state = _extend(state, ids[pos:], cut + pos)
        return PromptIds(Rendered(text, (*old, *ids), tuple(boundaries), state, cut))
