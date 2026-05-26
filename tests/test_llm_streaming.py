from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from mini_search_agent.config import LLMConfig
from mini_search_agent.llm import OpenAICompatibleChatClient


class FakeChatCompletions:
    def __init__(self, chunks):
        self.chunks = chunks
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.chunks, list):
            return iter(self.chunks)
        return self.chunks


class OpenAICompatibleStreamingTest(unittest.TestCase):
    def test_complete_enables_parallel_tool_calls_when_tools_are_present(self):
        completions = FakeChatCompletions(_completion_response())
        fake_openai = SimpleNamespace(OpenAI=lambda api_key, base_url: _client(completions))
        client = OpenAICompatibleChatClient(
            LLMConfig(
                provider="openai-compatible",
                api_key="test-key",
                model="test-model",
                base_url="https://llm.example/v1",
            )
        )

        with patch.dict(sys.modules, {"openai": fake_openai}):
            response = client.complete(
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function", "function": {"name": "web_search"}}],
            )

        self.assertEqual(response.content, "done")
        self.assertTrue(completions.kwargs["parallel_tool_calls"])

    def test_stream_complete_enables_parallel_tool_calls_when_tools_are_present(self):
        completions = FakeChatCompletions([_chunk(finish_reason="stop")])
        fake_openai = SimpleNamespace(OpenAI=lambda api_key, base_url: _client(completions))
        client = OpenAICompatibleChatClient(
            LLMConfig(
                provider="openai-compatible",
                api_key="test-key",
                model="test-model",
                base_url="https://llm.example/v1",
            )
        )

        with patch.dict(sys.modules, {"openai": fake_openai}):
            list(
                client.stream_complete(
                    [{"role": "user", "content": "hi"}],
                    tools=[{"type": "function", "function": {"name": "web_search"}}],
                )
            )

        self.assertTrue(completions.kwargs["parallel_tool_calls"])

    def test_stream_complete_translates_openai_chunks_to_model_events(self):
        completions = FakeChatCompletions(
            [
                _chunk(content="Hello "),
                _chunk(content="there"),
                _chunk(
                    tool_calls=[
                        _tool_delta(
                            index=0,
                            call_id="call-001",
                            name="web_search",
                            arguments='{"query":',
                        )
                    ]
                ),
                _chunk(tool_calls=[_tool_delta(index=0, arguments='"agentic search"}')]),
                _chunk(finish_reason="tool_calls"),
            ]
        )
        fake_openai = SimpleNamespace(OpenAI=lambda api_key, base_url: _client(completions))
        client = OpenAICompatibleChatClient(
            LLMConfig(
                provider="openai-compatible",
                api_key="test-key",
                model="test-model",
                base_url="https://llm.example/v1",
            )
        )

        with patch.dict(sys.modules, {"openai": fake_openai}):
            events = list(client.stream_complete([{"role": "user", "content": "hi"}]))

        self.assertTrue(completions.kwargs["stream"])
        self.assertEqual([event.type for event in events], ["content_delta", "content_delta", "tool_call_delta", "tool_call_delta", "done"])
        self.assertEqual(events[0].delta, "Hello ")
        self.assertEqual(events[1].delta, "there")
        response = events[-1].response
        self.assertIsNotNone(response)
        self.assertEqual(response.content, "Hello there")
        self.assertEqual(
            response.tool_calls,
            [
                {
                    "id": "call-001",
                    "name": "web_search",
                    "arguments": '{"query":"agentic search"}',
                }
            ],
        )


def _client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def _completion_response(content="done"):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=[],
                    model_extra={},
                )
            )
        ]
    )


def _chunk(*, content=None, tool_calls=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, tool_calls=tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


def _tool_delta(*, index, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


if __name__ == "__main__":
    unittest.main()
