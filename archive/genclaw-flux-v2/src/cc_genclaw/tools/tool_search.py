from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseTool
from ..runtime.tool_discovery import get_discovered_tools_state
from ..types import ToolResult


_REGISTERED_TOOLS: List[BaseTool] = []


def register_tools(tools: List[BaseTool]) -> None:
    global _REGISTERED_TOOLS
    _REGISTERED_TOOLS = list(tools)


def get_registered_tools() -> List[BaseTool]:
    return list(_REGISTERED_TOOLS)


class ToolSearchTool(BaseTool):
    name = "tool_search"
    description = (
        "Search for and load deferred tools by keyword or exact name. "
        "Some tools are kept hidden behind a short summary to save tokens; "
        "use this to discover and learn their full description before calling them."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query. Use 'select:<ToolName>' for exact match (e.g. "
                    "'select:bash'), or a keyword like 'shell' or 'git' for "
                    "case-insensitive substring search."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Cap on number of matches returned (default 5).",
            },
        },
        "required": ["query"],
    }

    should_defer = False

    def execute(self, args: Dict[str, Any], signal: Optional[Any] = None) -> ToolResult:
        query = (args.get("query") or "").strip()
        if not query:
            return self.err("query is required")

        max_results = int(args.get("max_results") or 5)

        state = get_discovered_tools_state()
        candidates = [
            t for t in _REGISTERED_TOOLS
            if t.should_defer and not state.is_discovered(t.name)
        ]

        if not candidates:
            return self.ok(
                "[no deferred tools available — all are already discovered or "
                "no tool_search-eligible tools registered]"
            )

        matches: List[BaseTool] = []

        if query.lower().startswith("select:"):
            target = query[len("select:"):].strip()
            for t in candidates:
                if t.name == target:
                    matches.append(t)
                    break
            if not matches:
                return self.err(
                    f"No deferred tool named {target!r}. "
                    f"Available deferred tools: {[t.name for t in candidates]}"
                )
        else:
            q = query.lower()
            for t in candidates:
                hay = " ".join([
                    t.name.lower(),
                    (t.short_description or "").lower(),
                    (t.description or "").lower()[:500],
                ])
                if q in hay:
                    matches.append(t)
                if len(matches) >= max_results:
                    break

        if not matches:
            return self.ok(
                f"[no deferred tools matched query={query!r}; "
                f"available deferred tools: {[t.name for t in candidates]}]"
            )

        state.mark_discovered([t.name for t in matches])

        body_parts = [
            f"Discovered {len(matches)} tool(s). Their full descriptions are now "
            "available, and from the next turn onward the LLM will see the "
            "complete tool schema for each. Inline copy below for immediate use:\n"
        ]
        for t in matches:
            body_parts.append(f"=== Tool: {t.name} ===")
            body_parts.append(t.description)
            body_parts.append("")
        return self.ok("\n".join(body_parts))

