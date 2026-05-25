from __future__ import annotations

from typing import Any


SEARCH_SUBAGENT_ALLOWED_TOOLS = {"web_search", "web_fetch"}


def filter_tools_for_search_subagent(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool for tool in tools if _tool_name(tool) in SEARCH_SUBAGENT_ALLOWED_TOOLS]


def _tool_name(tool: dict[str, Any]) -> str | None:
    function = tool.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        return name if isinstance(name, str) else None
    name = tool.get("name")
    return name if isinstance(name, str) else None
