from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Union,
    runtime_checkable,
)


@dataclass
class TextContent:

    text: str
    type: Literal["text"] = "text"


@dataclass
class ToolCallContent:

    id: str
    name: str
    arguments: Dict[str, Any]
    type: Literal["tool_call"] = "tool_call"


AssistantBlock = Union[TextContent, ToolCallContent]


@dataclass
class UserMessage:

    content: str
    role: Literal["user"] = "user"
    timestamp: float = field(default_factory=time.time)
    is_system_reminder: bool = False


@dataclass
class AssistantMessage:

    content: List[AssistantBlock]
    role: Literal["assistant"] = "assistant"
    timestamp: float = field(default_factory=time.time)
    stop_reason: str = "end_turn"
    error_message: Optional[str] = None


@dataclass
class ToolResultMessage:

    tool_call_id: str
    tool_name: str
    content: str
    role: Literal["tool"] = "tool"
    is_error: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class SystemMessage:

    content: str
    role: Literal["system"] = "system"
    timestamp: float = field(default_factory=time.time)


AgentMessage = Union[UserMessage, AssistantMessage, ToolResultMessage, SystemMessage]


@dataclass
class ToolResult:

    content: str
    details: Any = None
    is_error: bool = False
    terminate: bool = False


@runtime_checkable
class AgentTool(Protocol):

    name: str
    """Tool name as the model sees it (e.g., 'read_file', 'bash')."""

    description: str
    """Tool description (rendered from capability.yaml, fed into system prompt)."""

    parameters_schema: Dict[str, Any]
    """JSON-schema for the tool's parameters (OpenAI tools format)."""

    def execute(self, args: Dict[str, Any], signal: Optional[Any] = None) -> ToolResult:
        ...


@dataclass
class AgentEvent:

    type: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentStartEvent(AgentEvent):
    type: str = "agent_start"


@dataclass
class AgentEndEvent(AgentEvent):
    messages: List[AgentMessage] = field(default_factory=list)
    type: str = "agent_end"


@dataclass
class TurnStartEvent(AgentEvent):
    type: str = "turn_start"


@dataclass
class TurnEndEvent(AgentEvent):
    message: Optional[AssistantMessage] = None
    tool_results: List[ToolResultMessage] = field(default_factory=list)
    type: str = "turn_end"


@dataclass
class MessageStartEvent(AgentEvent):
    message: Optional[AgentMessage] = None
    type: str = "message_start"


@dataclass
class MessageEndEvent(AgentEvent):
    message: Optional[AgentMessage] = None
    type: str = "message_end"


@dataclass
class ToolExecutionStartEvent(AgentEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    args: Dict[str, Any] = field(default_factory=dict)
    type: str = "tool_execution_start"


@dataclass
class ToolExecutionEndEvent(AgentEvent):
    tool_call_id: str = ""
    tool_name: str = ""
    result: Optional[ToolResult] = None
    is_error: bool = False
    type: str = "tool_execution_end"


@dataclass
class AgentContext:

    system_prompt: str
    messages: List[AgentMessage] = field(default_factory=list)
    tools: List[AgentTool] = field(default_factory=list)


EventSink = Callable[[AgentEvent], None]


def make_tool_call_id() -> str:
    return f"call_{uuid.uuid4().hex[:24]}"


def msg_to_dict(msg: AgentMessage) -> Dict[str, Any]:
    d = asdict(msg)
    return d

