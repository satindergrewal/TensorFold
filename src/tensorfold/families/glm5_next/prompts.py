"""GLM-5.3's prompts, the same on both servers: earlier turns keep their reasoning, and with thinking off no effort
line and no opened think block."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

EFFORT_LINE = "<|system|>Reasoning Effort: Max"
OPENED = "<|assistant|><think>"
CLEAR_THINKING = "TF_GLM_CLEAR_THINKING"


def clear_thinking() -> bool:
    """The template's ``clear_thinking`` when a request leaves it unset: False unless ``TF_GLM_CLEAR_THINKING=1``.

    zai-org's GLM-5.3-Flash template defaults it to false, keeping every assistant turn's reasoning in the prompt
    (its model card: pass ``clear_thinking=true`` for chat). Checkpoints that carry the template's first revision
    (``Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw`` among them) clear it unless told otherwise, so the reasoning before the
    last user message renders as ``<think></think>``: each new user message changes the earlier turns' tokens, and
    an agent's whole previous tool loop is prefilled again. Passing false renders those checkpoints as the current
    template does; on the current template it changes nothing."""

    value = os.environ.get(CLEAR_THINKING, "0").strip() or "0"
    if value not in ("0", "1"):
        raise ValueError(f"{CLEAR_THINKING}={value!r}: expected 0 or 1")
    return value == "1"


def _call_id(message: Any) -> Any:
    """The id GLM's template matches a tool result or call by (its ``id_of``)."""

    return message.get("tool_call_id") or message.get("id") if isinstance(message, dict) else None


def agent_history(messages: Any) -> Any:
    """A chat as GLM-5.3's template can render every part of it: a call's arguments sent as "" or null are no
    arguments (the template reads them as a mapping), and results whose ids match none of their calls' ids are
    rendered in the order sent (the template sorts results by id and silently drops one it cannot place).

    A call whose arguments are not a JSON object (a truncated or garbled string from a model, client or proxy) is
    left out with its result, logged, rather than refused: an agent replays its history on every later turn, so a
    refusal would end the conversation. Its message keeps its other calls, content and reasoning; its result is the
    one with the call's id, else (no ids) the one in the call's place."""

    if not isinstance(messages, list):
        return messages
    out = list(messages)
    for i, message in enumerate(out):
        calls = message.get("tool_calls") if isinstance(message, dict) and message.get("role") == "assistant" else None
        if not isinstance(calls, list) or not calls:
            continue
        fixed, bad = [], []
        for j, call in enumerate(calls):
            function = call.get("function") if isinstance(call, dict) else None
            arguments = function.get("arguments") if isinstance(function, dict) else {}
            if arguments is None or (isinstance(arguments, str) and not arguments.strip()):
                call = {**call, "function": {**function, "arguments": {}}}
            elif isinstance(arguments, str):
                try:
                    parsed = json.loads(arguments)
                except ValueError:
                    parsed = None
                if not isinstance(parsed, dict):
                    print(f"[tensorfold] messages[{i}].tool_calls[{j}] ({_call_id(call)!r}): arguments are not a JSON "
                          f"object ({arguments[:80]!r}); the call and its result are left out of the prompt",
                          file=sys.stderr, flush=True)
                    bad.append(j)
                    continue
            fixed.append(call)
        end = i + 1
        while end < len(out) and isinstance(out[end], dict) and out[end].get("role") == "tool":
            end += 1
        if bad:
            ids = {_call_id(calls[j]) for j in bad} - {None}
            by_place = not ids and end - i - 1 == len(calls)      # no ids: results in their calls' order
            out[i + 1:end] = [r for k, r in enumerate(out[i + 1:end])
                              if not (_call_id(r) in ids or (by_place and k in bad))]
            end = i + 1
            while end < len(out) and isinstance(out[end], dict) and out[end].get("role") == "tool":
                end += 1
        if fixed:
            out[i] = {**message, "tool_calls": fixed}
        else:
            out[i] = {k: v for k, v in message.items() if k != "tool_calls"}
            continue
        known = {_call_id(call) for call in fixed}
        if any(_call_id(result) and _call_id(result) not in known for result in out[i + 1:end]):
            out[i + 1:end] = [{k: v for k, v in result.items() if k not in ("tool_call_id", "id")}
                              for result in out[i + 1:end]]
    return out


def thinking_off(text: str) -> str:
    """The checkpoint text as the thinking-off template renders it: no effort line, an empty think block."""

    text = text.replace(EFFORT_LINE, "", 1)
    return text + "</think>" if text.endswith(OPENED) else text


class GlmTokenizer:
    """The checkpoint's tokenizer (the Mac's), its chat template read through ``thinking_off`` when thinking is off
    and given ``clear_thinking`` (``clear_thinking()``) when the call leaves it unset."""

    def __init__(self, inner: Any, clear: bool | None = None) -> None:
        self._inner = inner
        self._clear = clear_thinking() if clear is None else clear

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def apply_chat_template(self, messages: Any, *args: Any, tokenize: bool = True, **kwargs: Any) -> Any:
        messages = agent_history(messages)
        kwargs.setdefault("clear_thinking", self._clear)
        if kwargs.get("enable_thinking", True) is not False:
            return self._inner.apply_chat_template(messages, *args, tokenize=tokenize, **kwargs)
        text = thinking_off(self._inner.apply_chat_template(messages, *args, tokenize=False, **kwargs))
        return list(self._inner.encode(text, add_special_tokens=False)) if tokenize else text
