from __future__ import annotations

from typing import Any


def shell_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Run a shell command in the workspace. Only available to the Main Agent.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    }
