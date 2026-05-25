from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def openai_tool_schema(name: str, description: str, args_model: type[ToolArgs]) -> dict[str, Any]:
    parameters = args_model.model_json_schema()
    parameters.setdefault("additionalProperties", False)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
