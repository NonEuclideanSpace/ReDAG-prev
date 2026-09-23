from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .loop import LoopConfig, run_agent_loop
from .types import (
    AgentContext,
    AgentEvent,
    AgentMessage,
    AgentTool,
    UserMessage,
)


@dataclass
class Agent:

    system_prompt: str
    tools: List[AgentTool] = field(default_factory=list)
    messages: List[AgentMessage] = field(default_factory=list)

    config: Optional[LoopConfig] = None
    _subscribers: List[Callable[[AgentEvent], None]] = field(default_factory=list)

    def subscribe(self, fn: Callable[[AgentEvent], None]) -> Callable[[], None]:
        self._subscribers.append(fn)
        def _unsub() -> None:
            try:
                self._subscribers.remove(fn)
            except ValueError:
                pass
        return _unsub

    def _emit(self, event: AgentEvent) -> None:
        for s in self._subscribers:
            try:
                s(event)
            except Exception:
                pass

    def prompt(self, user_input: str, user_image_path: Optional[str] = None) -> List[AgentMessage]:
        if self.config is None:
            raise RuntimeError("Agent.config must be set before calling prompt()")

        try:
            from .runtime.perception_init import run_perception_init
            run_perception_init(user_input, user_image_path)
        except Exception as e:
            import sys as _sys
            print(
                f"[Agent.prompt] perception_init crashed unexpectedly: "
                f"{type(e).__name__}: {e}. Proceeding without painter notes.",
                file=_sys.stderr,
            )

        prompt_msg = UserMessage(content=user_input)
        ctx = AgentContext(
            system_prompt=self.system_prompt,
            messages=list(self.messages),
            tools=list(self.tools),
        )
        new_msgs = run_agent_loop([prompt_msg], ctx, self.config, self._emit)
        self.messages.extend(new_msgs)
        return new_msgs

    def continue_run(self) -> List[AgentMessage]:
        if self.config is None:
            raise RuntimeError("Agent.config must be set")
        if not self.messages:
            raise ValueError("Cannot continue: no messages")
        last = self.messages[-1]
        last_role = getattr(last, "role", None)
        if last_role not in ("user", "tool"):
            raise ValueError(
                f"Cannot continue: last message role must be 'user' or 'tool', "
                f"got {last_role!r}"
            )

        ctx = AgentContext(
            system_prompt=self.system_prompt,
            messages=list(self.messages),
            tools=list(self.tools),
        )
        new_msgs = run_agent_loop([], ctx, self.config, self._emit)
        self.messages.extend(new_msgs)
        return new_msgs

    def reset(self) -> None:
        self.messages = []

