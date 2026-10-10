from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .types import AgentMessage, ToolResultMessage

DEFAULT_TOKEN_BUDGET = 60_000
DEFAULT_RECENCY_KEEP = 3
TODO_TOOL_NAMES = {"todo_write"}

PROTECTED_TOOL_NAMES = TODO_TOOL_NAMES | {
    "t2i",
    "i2i",
    "search",
    "code_scene_draft",
    "code_text_draft",
    "format_prompt",
}


@dataclass
class CompactionConfig:
    token_budget: int = DEFAULT_TOKEN_BUDGET
    recency_keep: int = DEFAULT_RECENCY_KEEP


def estimate_tokens(messages: List[AgentMessage]) -> int:
    total_chars = 0
    for m in messages:
        if hasattr(m, "content"):
            c = getattr(m, "content")
            if isinstance(c, str):
                total_chars += len(c)
            elif isinstance(c, list):
                for block in c:
                    text = getattr(block, "text", None)
                    if text:
                        total_chars += len(text)
                    name = getattr(block, "name", None)
                    if name:
                        total_chars += len(name) + 20
    return total_chars // 4


def compact(
    messages: List[AgentMessage], config: CompactionConfig = None
) -> List[AgentMessage]:
    config = config or CompactionConfig()
    if estimate_tokens(messages) <= config.token_budget:
        return list(messages)

    n = len(messages)
    out = list(messages)

    tr_indices = [i for i, m in enumerate(out) if isinstance(m, ToolResultMessage)]

    if config.recency_keep > 0 and tr_indices:
        recent_protected = set(tr_indices[-config.recency_keep:])
    else:
        recent_protected = set()

    for idx in tr_indices:
        if idx in recent_protected:
            continue
        m = out[idx]
        if not isinstance(m, ToolResultMessage):
            continue
        if m.tool_name in PROTECTED_TOOL_NAMES:
            continue
        if m.content.startswith("[Old tool result content cleared"):
            continue
        stub = f"[Old tool result content cleared by compaction; tool={m.tool_name}]"
        out[idx] = ToolResultMessage(
            tool_call_id=m.tool_call_id,
            tool_name=m.tool_name,
            content=stub,
            is_error=m.is_error,
        )

        if estimate_tokens(out) <= config.token_budget:
            break

    return out

