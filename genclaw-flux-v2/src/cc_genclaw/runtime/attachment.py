from __future__ import annotations

from typing import List

from .perception_state import PerceptionState, get_perception_state
from .todo_state import TodoState, get_todo_state
from ..types import AgentMessage, UserMessage


def make_todo_reminder_hook(state: TodoState = None):
    state = state or get_todo_state()

    def hook(messages: List[AgentMessage]) -> List[AgentMessage]:
        if state.is_empty():
            return messages
        body = state.render_for_reminder()
        return list(messages) + [UserMessage(content=body, is_system_reminder=True)]

    return hook


def make_perception_reminder_hook(state: PerceptionState = None):
    state = state or get_perception_state()

    def hook(messages: List[AgentMessage]) -> List[AgentMessage]:
        if not state.is_initialized():
            return messages
        body = state.render_for_reminder()
        if not body:
            return messages
        return list(messages) + [UserMessage(content=body, is_system_reminder=True)]

    return hook

