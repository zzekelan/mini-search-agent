from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

from mini_search_agent.agent_loop import ToolSpec, run_agent_loop
from mini_search_agent.llm import ModelResponse
from mini_search_agent.session import SessionStore, TelemetryLogger, TimelineWriter
from mini_search_agent.tools.base import ToolResult
from mini_search_agent.tools.web_search import WebSearchArgs, web_search_tool_schema


class ReasoningClient:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools=None, response_format=None):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return ModelResponse(
                content="",
                reasoning_content="private chain summary",
                tool_calls=[
                    {
                        "id": "call-001",
                        "name": "web_search",
                        "arguments": {"query": "agentic search"},
                    }
                ],
            )
        return ModelResponse(content="done")


class AgentLoopTest(unittest.TestCase):
    def test_response_format_is_sent_to_client(self):
        class ResponseFormatClient:
            def __init__(self):
                self.response_format = None

            def complete(self, messages, tools=None, response_format=None):
                self.response_format = response_format
                return ModelResponse(content='{"ok": true}')

        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        response_format = {"type": "json_object"}
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            timeline = TimelineWriter(session, clock=lambda: now)
            telemetry = TelemetryLogger(session, clock=lambda: now)
            client = ResponseFormatClient()

            run_agent_loop(
                client=client,
                system_prompt="system prompt",
                initial_user_text="question",
                timeline=timeline,
                telemetry=telemetry,
                tools=[],
                run_id="run-001",
                actor="subagent",
                response_format=response_format,
            )

        self.assertEqual(client.response_format, response_format)

    def test_reasoning_content_is_preserved_between_tool_turns_without_timeline_storage(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            timeline = TimelineWriter(session, clock=lambda: now)
            telemetry = TelemetryLogger(session, clock=lambda: now)
            client = ReasoningClient()

            result = run_agent_loop(
                client=client,
                system_prompt="system prompt",
                initial_user_text="question",
                timeline=timeline,
                telemetry=telemetry,
                tools=[
                    ToolSpec(
                        name="web_search",
                        schema=web_search_tool_schema(),
                        args_model=WebSearchArgs,
                        handler=lambda arguments: ToolResult(content="search result"),
                    )
                ],
                run_id="run-001",
                actor="main",
            )
            entries = timeline.read_entries()

        self.assertEqual(result.content, "done")
        self.assertEqual(client.calls[1][1]["role"], "user")
        assistant_message = client.calls[1][2]
        self.assertEqual(assistant_message["role"], "assistant")
        self.assertEqual(assistant_message["reasoning_content"], "private chain summary")
        self.assertEqual(assistant_message["tool_calls"][0]["function"]["arguments"], '{"query": "agentic search"}')
        timeline_text = "\n".join(str(entry) for entry in entries)
        self.assertNotIn("private chain summary", timeline_text)

    def test_tool_arguments_are_validated_with_pydantic_before_handler_runs(self):
        class InvalidToolClient:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, tools=None, response_format=None):
                self.calls += 1
                if self.calls == 1:
                    return ModelResponse(
                        content="",
                        tool_calls=[
                            {
                                "id": "call-001",
                                "name": "web_search",
                                "arguments": {"unexpected": "value"},
                            }
                        ],
                    )
                return ModelResponse(content="done")

        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        handler_calls = []
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            timeline = TimelineWriter(session, clock=lambda: now)
            telemetry = TelemetryLogger(session, clock=lambda: now)

            run_agent_loop(
                client=InvalidToolClient(),
                system_prompt="system prompt",
                initial_user_text="question",
                timeline=timeline,
                telemetry=telemetry,
                tools=[
                    ToolSpec(
                        name="web_search",
                        schema=web_search_tool_schema(),
                        args_model=WebSearchArgs,
                        handler=lambda arguments: handler_calls.append(arguments) or ToolResult(content="search result"),
                    )
                ],
                run_id="run-001",
                actor="main",
            )
            entries = timeline.read_entries()

        self.assertEqual(handler_calls, [])
        result_parts = [part for entry in entries for part in entry["parts"] if part["type"] == "tool_result"]
        self.assertTrue(result_parts[0]["is_error"])
        self.assertIn("arguments failed validation", result_parts[0]["content"])


if __name__ == "__main__":
    unittest.main()
