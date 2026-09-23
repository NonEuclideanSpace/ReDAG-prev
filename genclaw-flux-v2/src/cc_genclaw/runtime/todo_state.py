from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal


@dataclass
class TodoItem:
    content: str
    activeForm: str
    status: Literal["pending", "in_progress", "completed"] = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TodoState:

    items: List[TodoItem] = field(default_factory=list)

    def replace(self, items: List[TodoItem]) -> None:
        self.items = list(items)

    def is_empty(self) -> bool:
        return not self.items

    def render_for_reminder(self) -> str:
        if not self.items:
            return ""
        lines = ["Your todo list is currently as follows. Please check carefully and use TodoWrite to update it.\n"]
        for i, item in enumerate(self.items, start=1):
            marker = {
                "pending": "[ ]",
                "in_progress": "[~]",
                "completed": "[x]",
            }[item.status]
            lines.append(f"  {i}. {marker} {item.content}  ({item.status})")
        lines.append("")
        lines.append("Reminder: only ONE task should be in_progress at a time.")
        lines.append("Mark each task completed as soon as it's done; do not batch.")
        return "\n".join(lines)

    def snapshot(self) -> List[Dict[str, Any]]:
        return [it.to_dict() for it in self.items]


_GLOBAL_TODO_STATE = TodoState()


def get_todo_state() -> TodoState:
    return _GLOBAL_TODO_STATE


def reset_todo_state() -> None:
    _GLOBAL_TODO_STATE.items = []

