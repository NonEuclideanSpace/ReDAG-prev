from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import ToolResult


class BaseTool:

    name: str = ""
    description: str = ""
    parameters_schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    should_defer: bool = False
    """If True, this tool is hidden behind ToolSearch until discovered.

    When False (default), the full 7-section description is sent on every
    LLM call (eager loading; matches cc's default for most tools).

    When True, the LLM only sees `short_description` initially. To learn
    the full description, the model must call `tool_search`. Once
    discovered (tracked in runtime.tool_discovery._GLOBAL_DISCOVERED),
    subsequent LLM calls receive the full description for this tool.
    """

    short_description: Optional[str] = None
    """One-liner sent to the LLM when this tool is deferred and not yet
    discovered. Mirrors cc's `DESCRIPTION` constant (vs the long `PROMPT`).

    If `should_defer=True` but `short_description=None`, the loader falls
    back to the first sentence of `description`.
    """

    def execute(self, args: Dict[str, Any], signal: Optional[Any] = None) -> ToolResult:
        raise NotImplementedError


    @staticmethod
    def ok(content: str, **kwargs: Any) -> ToolResult:
        return ToolResult(content=content, **kwargs)

    @staticmethod
    def err(message: str, **kwargs: Any) -> ToolResult:
        return ToolResult(content=message, is_error=True, **kwargs)

