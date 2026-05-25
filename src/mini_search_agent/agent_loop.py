from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from .llm import ChatClient
from .projection import project_timeline_to_openai
from .session import (
    TelemetryLogger,
    TimelineWriter,
    text_part,
    tool_call_part,
    tool_result_part,
)
from .tool_schema import ToolArgs
from .tools.base import ToolResult


ToolHandler = Callable[[dict[str, Any]], ToolResult]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    schema: dict[str, Any]
    handler: ToolHandler
    args_model: type[ToolArgs] | None = None


@dataclass(frozen=True)
class AgentLoopResult:
    content: str
    tool_results: list[ToolResult] = field(default_factory=list)


def run_agent_loop(
    *,
    client: ChatClient,
    system_prompt: str,
    initial_user_text: str,
    timeline: TimelineWriter,
    telemetry: TelemetryLogger,
    tools: list[ToolSpec],
    run_id: str,
    actor: str,
    max_turns: int = 12,
    response_format: dict[str, Any] | None = None,
) -> AgentLoopResult:
    if not timeline.read_entries():
        timeline.append(role="user", parts=[text_part(initial_user_text)], produced_by_run=run_id)

    tool_results: list[ToolResult] = []
    tool_map = {tool.name: tool for tool in tools}
    tool_schemas = [tool.schema for tool in tools]
    messages = project_timeline_to_openai(timeline.read_entries(), system_prompt=system_prompt)

    for _ in range(max_turns):
        telemetry.emit(
            "llm.request.started",
            run_id=run_id,
            actor=actor,
            metadata={
                "message_count": len(messages),
                "tool_names": list(tool_map),
                "response_format_type": response_format.get("type") if response_format else None,
            },
        )
        started = time.perf_counter()
        response = client.complete(
            messages,
            tools=tool_schemas if tool_schemas else None,
            response_format=response_format,
        )
        telemetry.emit(
            "llm.response.finished",
            run_id=run_id,
            actor=actor,
            latency_ms=_elapsed_ms(started),
            metadata={"content_length": len(response.content), "tool_call_count": len(response.tool_calls)},
        )

        if not response.tool_calls:
            content = response.content.strip()
            timeline.append(role="assistant", parts=[text_part(content)], produced_by_run=run_id)
            return AgentLoopResult(content=content, tool_results=tool_results)

        messages.append(_assistant_message_from_response(response))
        call_parts = []
        for call in response.tool_calls:
            call_parts.append(
                tool_call_part(
                    _call_id(call),
                    _call_name(call),
                    _call_arguments(call),
                )
            )
        parts = [text_part(response.content)] if response.content.strip() else []
        timeline.append(role="assistant", parts=[*parts, *call_parts], produced_by_run=run_id)

        result_parts = []
        for call in response.tool_calls:
            name = _call_name(call)
            call_id = _call_id(call)
            arguments = _call_arguments(call)
            spec = tool_map.get(name)
            if spec is None:
                result = ToolResult(
                    content=f"Tool {name!r} is not available to this agent.",
                    metadata={"tool_name": name},
                    is_error=True,
                )
            else:
                try:
                    validated_arguments = _validate_tool_arguments(spec, arguments)
                except ValidationError as exc:
                    result = ToolResult(
                        content=f"Tool {name!r} arguments failed validation: {exc}",
                        metadata={"tool_name": name},
                        is_error=True,
                    )
                else:
                    result = spec.handler(validated_arguments)
            tool_results.append(result)
            result_parts.append(
                tool_result_part(
                    call_id,
                    name,
                    result.content,
                    is_error=result.is_error,
                    metadata=result.metadata,
                )
            )
            messages.append({"role": "tool", "tool_call_id": call_id, "content": result.content})
        timeline.append(role="assistant", parts=result_parts, produced_by_run=run_id)

    message = f"Agent loop exceeded max_turns={max_turns}"
    telemetry.emit("run.failed", run_id=run_id, actor=actor, status="error", metadata={"error": message})
    raise RuntimeError(message)


def _call_id(call: dict[str, Any]) -> str:
    return str(call.get("id") or call.get("call_id") or "call-unknown")


def _call_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or call.get("tool_name") or "")


def _call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments", {})
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _assistant_message_from_response(response) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": response.content or None}
    if response.reasoning_content:
        message["reasoning_content"] = response.reasoning_content
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": _call_id(call),
                "type": "function",
                "function": {
                    "name": _call_name(call),
                    "arguments": _call_arguments_json(call),
                },
            }
            for call in response.tool_calls
        ]
    return message


def _call_arguments_json(call: dict[str, Any]) -> str:
    arguments = call.get("arguments", {})
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments, ensure_ascii=False)


def _validate_tool_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> dict[str, Any]:
    if spec.args_model is None:
        return arguments
    return spec.args_model.model_validate(arguments).model_dump()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
