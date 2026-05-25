from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from pydantic import Field

from ..session import TelemetryLogger
from ..tool_schema import ToolArgs, openai_tool_schema
from .base import ToolResult


EXA_MCP_ENDPOINT = "https://mcp.exa.ai/mcp"
EXA_PROVIDER_NAME = "Exa MCP"


class WebSearchArgs(ToolArgs):
    query: str = Field(description="Search query for one focused angle of the Research Question.")


@dataclass
class ExaWebSearchTool:
    client: httpx.Client | None = None
    endpoint: str = EXA_MCP_ENDPOINT

    @property
    def name(self) -> str:
        return "web_search"

    def run(
        self,
        *,
        query: str,
        telemetry: TelemetryLogger | None = None,
        run_id: str = "run-001",
        actor: str = "main",
    ) -> ToolResult:
        started = time.perf_counter()
        if telemetry:
            telemetry.emit(
                "tool.web_search.started",
                run_id=run_id,
                actor=actor,
                metadata={"query": query, "provider": EXA_PROVIDER_NAME},
            )

        close_client = self.client is None
        client = self.client or httpx.Client(timeout=30)
        try:
            response = client.post(
                self.endpoint,
                json=build_exa_request(query),
                headers={
                    "accept": "application/json, text/event-stream",
                    "user-agent": "mini-search-agent/0.1",
                },
            )
            response.raise_for_status()
            content = parse_exa_response(response.text)
            if not content:
                raise ValueError("Exa MCP response did not contain text content")

            metadata = {"query": query}
            if telemetry:
                telemetry.emit(
                    "tool.web_search.finished",
                    run_id=run_id,
                    actor=actor,
                    latency_ms=_elapsed_ms(started),
                    metadata={
                        "query": query,
                        "provider": EXA_PROVIDER_NAME,
                        "returned_text_length": len(content),
                    },
                )
            return ToolResult(content=content, metadata=metadata)
        except Exception as exc:
            message = f"web_search failed for query {query!r}: {exc}"
            if telemetry:
                telemetry.emit(
                    "tool.web_search.finished",
                    run_id=run_id,
                    actor=actor,
                    status="error",
                    latency_ms=_elapsed_ms(started),
                    metadata={
                        "query": query,
                        "provider": EXA_PROVIDER_NAME,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                    },
                )
            return ToolResult(content=message, metadata={"query": query}, is_error=True)
        finally:
            if close_client:
                client.close()


def web_search_tool_schema() -> dict[str, Any]:
    return openai_tool_schema(
        "web_search",
        "Search the public web with real Exa-backed search. Returns provider text with candidate URLs; use web_fetch to verify pages before citing them.",
        WebSearchArgs,
    )


def build_exa_request(query: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search_exa",
            "arguments": {
                "query": query,
                "type": "auto",
                "numResults": 8,
                "livecrawl": "fallback",
            },
        },
    }


def parse_exa_response(response_text: str) -> str | None:
    sse_output = _parse_exa_event_stream(response_text)
    if sse_output:
        return sse_output

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return None

    return _read_exa_response_text(payload)


def _parse_exa_event_stream(response_text: str) -> str | None:
    lines = response_text.splitlines()
    data_segments: list[str] = []
    for line in lines:
        if line == "":
            text = _parse_exa_event_stream_payload(data_segments)
            if text:
                return text
            data_segments = []
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if field != "data":
            continue
        if separator and value.startswith(" "):
            value = value[1:]
        data_segments.append(value)
    return _parse_exa_event_stream_payload(data_segments)


def _parse_exa_event_stream_payload(data_segments: list[str]) -> str | None:
    payload_text = "\n".join(data_segments).strip()
    if not payload_text:
        return None
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    return _read_exa_response_text(payload)


def _read_exa_response_text(payload: dict[str, Any]) -> str | None:
    if payload.get("error"):
        error = payload["error"]
        code = error.get("code")
        message = error.get("message") or "unknown Exa MCP error"
        suffix = f" ({code})" if isinstance(code, int) else ""
        raise ValueError(f"Public search backend failed{suffix}: {message}")

    content = payload.get("result", {}).get("content")
    if not isinstance(content, list):
        return None

    for item in content:
        if (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
            and item["text"].strip()
        ):
            return item["text"]
    return None


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
