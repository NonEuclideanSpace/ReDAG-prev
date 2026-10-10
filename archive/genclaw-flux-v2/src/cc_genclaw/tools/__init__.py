from .base import BaseTool
from .code_scene_draft import CodeSceneDraftTool
from .code_text_draft import CodeTextDraftTool
from .format_prompt import FormatPromptTool
from .i2i import I2ITool
from .reason import ReasonTool
from .search import SearchTool
from .t2i import T2ITool
from .todo_write import TodoWriteTool
from .tool_search import ToolSearchTool, register_tools
from .vlm_review import VLMReviewTool

__all__ = [
    "BaseTool",
    "TodoWriteTool", "ToolSearchTool",
    "T2ITool", "I2ITool",
    "SearchTool", "ReasonTool", "FormatPromptTool",
    "CodeSceneDraftTool", "CodeTextDraftTool",
    "VLMReviewTool",
    "register_tools",
    "default_tools",
]


def default_tools() -> list:
    tools = [
        TodoWriteTool(), ToolSearchTool(),
        T2ITool(), I2ITool(),
        SearchTool(), ReasonTool(), FormatPromptTool(),
        CodeSceneDraftTool(), CodeTextDraftTool(), VLMReviewTool(),
    ]
    register_tools(tools)
    return tools

