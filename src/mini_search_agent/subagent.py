from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .llm import ChatClient
from .prompts import PromptRegistry
from .session import (
    Session,
    SessionStore,
    TelemetryLogger,
    TimelineWriter,
    text_part,
    tool_call_part,
    tool_result_part,
)
from .tool_filter import filter_tools_for_search_subagent
from .tools.base import ToolResult
from .tools.web_fetch import web_fetch_tool_schema
from .tools.web_search import web_search_tool_schema


@dataclass
class SubagentTool:
    workspace: Path | str
    client: ChatClient
    parent_session: Session
    parent_timeline: TimelineWriter
    parent_telemetry: TelemetryLogger
    parent_tools: list[dict[str, Any]] | None = None
    prompt_registry: PromptRegistry = PromptRegistry()

    @property
    def name(self) -> str:
        return "subagent"

    def run(
        self,
        *,
        description: str,
        prompt: str,
        run_id: str = "run-001",
        call_id: str | None = None,
    ) -> ToolResult:
        started = time.perf_counter()
        tool_call_id = call_id or _next_subagent_call_id(self.parent_timeline)
        allowed_tools = filter_tools_for_search_subagent(
            self.parent_tools or [web_search_tool_schema(), web_fetch_tool_schema()]
        )

        self.parent_telemetry.emit(
            "subagent.started",
            run_id=run_id,
            actor="main",
            metadata={
                "description": description,
                "allowed_tools": [_schema_name(tool) for tool in allowed_tools],
            },
        )

        sub_session = SessionStore(self.workspace).create_sub_session(self.parent_session)
        sub_timeline = TimelineWriter(sub_session)
        sub_telemetry = TelemetryLogger(sub_session)
        sub_run_id = "run-001"
        sub_telemetry.emit(
            "session.started",
            run_id=sub_run_id,
            actor="subagent",
            metadata={"kind": "sub", "parent_session_id": self.parent_session.session_id},
        )
        sub_timeline.append(role="user", parts=[text_part(prompt)], produced_by_run=sub_run_id)

        system_prompt = self.prompt_registry.load("search_subagent")
        messages = [
            {"role": "developer", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        sub_telemetry.emit(
            "llm.request.started",
            run_id=sub_run_id,
            actor="subagent",
            metadata={"message_count": len(messages), "allowed_tools": [_schema_name(tool) for tool in allowed_tools]},
        )
        response = self.client.complete(messages, tools=allowed_tools)
        sub_telemetry.emit(
            "llm.response.finished",
            run_id=sub_run_id,
            actor="subagent",
            latency_ms=_elapsed_ms(started),
            metadata={"content_length": len(response.content), "tool_call_count": len(response.tool_calls)},
        )
        sub_timeline.append(role="assistant", parts=[text_part(response.content)], produced_by_run=sub_run_id)

        metadata = {"sub_session_path": str(sub_session.path)}
        result = ToolResult(content=response.content, metadata=metadata)
        self.parent_timeline.append(
            role="assistant",
            parts=[
                tool_call_part(
                    tool_call_id,
                    "subagent",
                    {"description": description, "prompt": prompt},
                ),
                tool_result_part(
                    tool_call_id,
                    "subagent",
                    result.content,
                    metadata=metadata,
                ),
            ],
            produced_by_run=run_id,
        )
        self.parent_telemetry.emit(
            "subagent.completed",
            run_id=run_id,
            actor="main",
            latency_ms=_elapsed_ms(started),
            metadata={**metadata, "description": description, "status": "completed"},
        )
        return result


def subagent_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "subagent",
            "description": "Spawn a Search Subagent for one focused source-collection angle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Short query angle label."},
                    "prompt": {"type": "string", "description": "Focused source collection instructions."},
                },
                "required": ["description", "prompt"],
                "additionalProperties": False,
            },
        },
    }


def _next_subagent_call_id(parent_timeline: TimelineWriter) -> str:
    count = 0
    for entry in parent_timeline.read_entries():
        for part in entry.get("parts", []):
            if part.get("type") == "tool_call" and part.get("tool_name") == "subagent":
                count += 1
    return f"subagent-{count + 1:03d}"


def _schema_name(schema: dict[str, Any]) -> str:
    function = schema.get("function")
    if isinstance(function, dict) and isinstance(function.get("name"), str):
        return function["name"]
    name = schema.get("name")
    return name if isinstance(name, str) else ""


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
