from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import LLMConfig


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: str | None = None


@dataclass(frozen=True)
class ModelStreamEvent:
    type: str
    delta: str = ""
    response: ModelResponse | None = None


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        ...


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMConfig):
        self._config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> ModelResponse:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for LLM calls") from exc

        client = OpenAI(api_key=self._config.api_key, base_url=self._config.base_url)
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        if response_format:
            kwargs["response_format"] = response_format

        response = client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        model_extra = getattr(message, "model_extra", None) or {}
        reasoning_content = getattr(message, "reasoning_content", None) or model_extra.get("reasoning_content")
        tool_calls = [
            {
                "id": tool_call.id,
                "name": tool_call.function.name,
                "arguments": tool_call.function.arguments,
            }
            for tool_call in (message.tool_calls or [])
        ]
        return ModelResponse(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content if isinstance(reasoning_content, str) else None,
        )

    def stream_complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> Iterator[ModelStreamEvent]:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for LLM calls") from exc

        client = OpenAI(api_key=self._config.api_key, base_url=self._config.base_url)
        kwargs: dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["parallel_tool_calls"] = True
        if response_format:
            kwargs["response_format"] = response_format

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls: dict[int, dict[str, str]] = {}

        for chunk in client.chat.completions.create(**kwargs):
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            choice = choices[0]
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            content_delta = getattr(delta, "content", None)
            if content_delta:
                content_parts.append(content_delta)
                yield ModelStreamEvent(type="content_delta", delta=content_delta)

            reasoning_delta = _first_string_attr(delta, ["reasoning_content", "reasoning", "reasoning_text"])
            if reasoning_delta:
                reasoning_parts.append(reasoning_delta)

            for tool_call_delta in getattr(delta, "tool_calls", None) or []:
                index = int(getattr(tool_call_delta, "index", len(tool_calls)))
                current = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                call_id = getattr(tool_call_delta, "id", None)
                if call_id:
                    current["id"] = str(call_id)
                function = getattr(tool_call_delta, "function", None)
                if function is not None:
                    name = getattr(function, "name", None)
                    if name:
                        current["name"] = str(name)
                    arguments = getattr(function, "arguments", None)
                    if arguments:
                        current["arguments"] += str(arguments)
                        yield ModelStreamEvent(type="tool_call_delta", delta=str(arguments))

        yield ModelStreamEvent(
            type="done",
            response=ModelResponse(
                content="".join(content_parts),
                tool_calls=[
                    {
                        "id": call["id"] or f"call-{index}",
                        "name": call["name"],
                        "arguments": call["arguments"],
                    }
                    for index, call in sorted(tool_calls.items())
                ],
                reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
            ),
        )


def _first_string_attr(value: Any, names: list[str]) -> str | None:
    for name in names:
        attr = getattr(value, name, None)
        if isinstance(attr, str) and attr:
            return attr
    return None
