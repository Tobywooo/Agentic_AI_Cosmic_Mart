"""Prompt-based JSON tool-calling protocol (for servers without native function calling).

The model must reply with exactly one JSON object, either
    {"thought": "...", "action": "<tool>", "action_input": {...}}
or
    {"thought": "...", "final_answer": "<message to the customer>"}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass
class AgentStep:
    thought: str = ""
    action: str | None = None
    action_input: dict = field(default_factory=dict)
    final_answer: str | None = None


def _candidates(text: str):
    """Yield every JSON object that can be decoded from `text`, outermost first."""
    decoder = json.JSONDecoder()
    sources = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for source in sources:
        idx = source.find("{")
        while idx != -1:
            try:
                obj, _ = decoder.raw_decode(source, idx)
                if isinstance(obj, dict):
                    yield obj
            except json.JSONDecodeError:
                pass
            idx = source.find("{", idx + 1)


def parse_agent_output(text: str) -> AgentStep | None:
    for obj in _candidates(text or ""):
        thought = str(obj.get("thought", "") or "")
        if isinstance(obj.get("final_answer"), str) and obj["final_answer"].strip():
            return AgentStep(thought=thought, final_answer=obj["final_answer"].strip())
        action = obj.get("action")
        if isinstance(action, str) and action.strip():
            args = obj.get("action_input", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    args = {"value": args}
            return AgentStep(thought=thought, action=action.strip(), action_input=args or {})
    return None


_PARTIAL_FINAL = re.compile(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)', re.DOTALL)


def salvage_reply(text: str) -> str | None:
    """Best-effort customer reply from output that failed to parse (e.g. JSON cut off at the token limit).

    Plain prose is returned as-is. Anything that looks like protocol JSON is never shown raw: we recover
    the (possibly truncated) final_answer text if present, otherwise return None so a safe default is used.
    """
    text = (text or "").strip()
    if not text:
        return None
    if "{" not in text and '"action"' not in text:
        return text
    match = _PARTIAL_FINAL.search(text)
    if match and match.group(1).strip():
        raw = match.group(1).rstrip("\\")
        try:
            return json.loads(f'"{raw}"').strip()
        except json.JSONDecodeError:
            return raw.replace("\\n", "\n").replace('\\"', '"').strip()
    return None


def format_final(answer: str) -> str:
    """How earlier agent replies are replayed in history, so the model keeps to the protocol."""
    return json.dumps({"final_answer": answer}, ensure_ascii=False)
