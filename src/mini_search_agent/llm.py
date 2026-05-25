from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import LLMConfig


@dataclass(frozen=True)
class ModelResponse:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reasoning_content: str | None = None


class ChatClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelResponse:
        ...


class OpenAICompatibleChatClient:
    def __init__(self, config: LLMConfig):
        self._config = config

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
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
