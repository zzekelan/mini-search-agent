from __future__ import annotations

import json
from typing import Any


def project_timeline_to_openai(
    entries: list[dict[str, Any]],
    *,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for entry in entries:
        role = entry["role"]
        parts = entry.get("parts", [])
        if role == "user":
            content = _join_text(parts)
            if content:
                messages.append({"role": "user", "content": content})
            continue

        assistant_text = _join_text(parts)
        tool_calls = [_project_tool_call(part) for part in parts if part.get("type") == "tool_call"]
        if assistant_text or tool_calls:
            message: dict[str, Any] = {
                "role": "assistant",
                "content": assistant_text or None,
            }
            if tool_calls:
                message["tool_calls"] = tool_calls
            messages.append(message)

        for part in parts:
            if part.get("type") == "tool_result":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": part["call_id"],
                        "content": part.get("content", ""),
                    }
                )

    return messages


def _join_text(parts: list[dict[str, Any]]) -> str:
    return "\n".join(
        part.get("text", "")
        for part in parts
        if part.get("type") == "text" and part.get("text")
    )


def _project_tool_call(part: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": part["call_id"],
        "type": "function",
        "function": {
            "name": part["tool_name"],
            "arguments": json.dumps(part.get("arguments", {}), ensure_ascii=False),
        },
    }
