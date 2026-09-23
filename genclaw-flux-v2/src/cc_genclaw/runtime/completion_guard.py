from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List

from ..types import AgentMessage, UserMessage
from .perception_state import get_perception_state

_FINAL_IMAGE_TOOLS = {"t2i", "i2i", "code_text_draft"}


@dataclass
class CompletionState:
    image_produced: bool = False
    latest_final_path: str = ""
    guard_fired: int = 0
    max_fires: int = 1

    def note_tool_result(self, tool_name: str, result_content: str) -> None:
        try:
            data = json.loads(result_content)
        except Exception:
            return
        if tool_name not in _FINAL_IMAGE_TOOLS:
            return
        if (
            tool_name == "code_text_draft"
            and isinstance(data, dict)
            and (
                data.get("requires_i2i")
                or data.get("output_role") == "text_structure_reference"
                or data.get("render_type") in {"embedded", "embedded_svg"}
            )
        ):
            return
        if isinstance(data, dict) and data.get("status") == "success" and data.get("final_path"):
            self.image_produced = True
            self.latest_final_path = str(data.get("final_path") or "")


_GLOBAL_COMPLETION_STATE = CompletionState()


def get_completion_state() -> CompletionState:
    return _GLOBAL_COMPLETION_STATE


def reset_completion_state() -> None:
    global _GLOBAL_COMPLETION_STATE
    _GLOBAL_COMPLETION_STATE = CompletionState()


_REMINDER_BODY = (
    "[completion-check] The user's original request is an image task, but "
    "this run has NOT produced any final image yet (no image tool returned "
    "a successful final_path). Do NOT stop here and do NOT hand the task "
    "back to the user as a follow-up step. Either:\n"
    "  (a) call the appropriate image tool now (t2i / i2i / code_text_draft, "
    "or code_scene_draft -> i2i) to actually produce the image and complete "
    "the request; or\n"
    "  (b) if the request genuinely cannot be fulfilled with an image, state "
    "clearly to the user WHY.\n"
    "Reasoning output alone is not the deliverable — the image is."
)

def make_completion_after_tool_hook(state: CompletionState = None):
    state = state or get_completion_state()

    from ..hooks import AfterToolCallContext

    def hook(ctx: "AfterToolCallContext"):
        if ctx.result is not None and not ctx.is_error:
            content = ctx.result.content
            state.note_tool_result(ctx.tool_name, content if isinstance(content, str) else str(content))
        return None

    return hook


def make_completion_followup_hook(state: CompletionState = None):
    state = state or get_completion_state()

    def hook() -> List[AgentMessage]:
        if state.image_produced:
            return []
        if not get_perception_state().is_initialized():
            return []
        if state.guard_fired >= state.max_fires:
            return []
        state.guard_fired += 1
        return [UserMessage(content=_REMINDER_BODY, is_system_reminder=True)]

    return hook

