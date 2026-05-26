from __future__ import annotations

import io
import time
import tempfile
import unittest
from datetime import datetime, timezone

from mini_search_agent.agent_loop import ToolSpec, run_agent_loop
from mini_search_agent.console import RunConsoleView
from mini_search_agent.llm import ModelResponse, ModelStreamEvent
from mini_search_agent.session import SessionStore, TelemetryLogger, TimelineWriter
from mini_search_agent.tools.base import ToolResult
from mini_search_agent.tools.web_search import WebSearchArgs, web_search_tool_schema


class TtyOutput(io.StringIO):
    def isatty(self):
        return True


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

    def test_tty_console_animates_model_planning_and_tool_execution_while_waiting(self):
        class DelayedStreamingClient:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, tools=None, response_format=None):
                raise AssertionError("main agent should use stream_complete")

            def stream_complete(self, messages, tools=None, response_format=None):
                self.calls += 1
                if self.calls == 1:
                    time.sleep(0.05)
                    yield ModelStreamEvent(
                        type="done",
                        response=ModelResponse(
                            content="",
                            tool_calls=[
                                {
                                    "id": "call-001",
                                    "name": "web_search",
                                    "arguments": {"query": "agentic rl 2025"},
                                }
                            ],
                        ),
                    )
                    return
                yield ModelStreamEvent(type="content_delta", delta="done")
                yield ModelStreamEvent(type="done", response=ModelResponse(content="done"))

        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        output = TtyOutput()
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            timeline = TimelineWriter(session, clock=lambda: now)
            telemetry = TelemetryLogger(session, clock=lambda: now)

            run_agent_loop(
                client=DelayedStreamingClient(),
                system_prompt="system prompt",
                initial_user_text="question",
                timeline=timeline,
                telemetry=telemetry,
                tools=[
                    ToolSpec(
                        name="web_search",
                        schema=web_search_tool_schema(),
                        args_model=WebSearchArgs,
                        handler=lambda arguments: time.sleep(0.05) or ToolResult(content="search result"),
                    )
                ],
                run_id="run-001",
                actor="main",
                run_console=RunConsoleView(output, spinner_interval_seconds=0.01),
            )

        text = output.getvalue()
        self.assertGreater(text.count("Main Agent planning"), 1)
        self.assertGreater(text.count("web_search: agentic rl 2025 running"), 1)

    def test_console_reports_tool_error_when_handler_raises(self):
        class ToolErrorClient:
            def __init__(self):
                self.calls = 0

            def complete(self, messages, tools=None, response_format=None):
                self.calls += 1
                return ModelResponse(
                    content="",
                    tool_calls=[
                        {
                            "id": "call-001",
                            "name": "web_search",
                            "arguments": {"query": "agentic rl"},
                        }
                    ],
                )

        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            timeline = TimelineWriter(session, clock=lambda: now)
            telemetry = TelemetryLogger(session, clock=lambda: now)

            with self.assertRaises(RuntimeError):
                run_agent_loop(
                    client=ToolErrorClient(),
                    system_prompt="system prompt",
                    initial_user_text="question",
                    timeline=timeline,
                    telemetry=telemetry,
                    tools=[
                        ToolSpec(
                            name="web_search",
                            schema=web_search_tool_schema(),
                            args_model=WebSearchArgs,
                            handler=lambda arguments: (_ for _ in ()).throw(RuntimeError("boom")),
                        )
                    ],
                    run_id="run-001",
                    actor="main",
                    run_console=RunConsoleView(output),
                )

        self.assertIn("[tool] web_search: agentic rl error", output.getvalue())

    def test_tty_streaming_text_appends_like_typewriter_after_spinner(self):
        class MultiDeltaClient:
            def complete(self, messages, tools=None, response_format=None):
                raise AssertionError("main agent should use stream_complete")

            def stream_complete(self, messages, tools=None, response_format=None):
                yield ModelStreamEvent(type="content_delta", delta="hello")
                yield ModelStreamEvent(type="content_delta", delta=" ")
                yield ModelStreamEvent(type="content_delta", delta="world")
                yield ModelStreamEvent(type="done", response=ModelResponse(content="hello world"))

        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        output = TtyOutput()
        console = RunConsoleView(output, spinner_interval_seconds=0.01)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            timeline = TimelineWriter(session, clock=lambda: now)
            telemetry = TelemetryLogger(session, clock=lambda: now)

            result = run_agent_loop(
                client=MultiDeltaClient(),
                system_prompt="system prompt",
                initial_user_text="question",
                timeline=timeline,
                telemetry=telemetry,
                tools=[],
                run_id="run-001",
                actor="main",
                run_console=console,
            )
            console.run_finished()

        self.assertEqual(result.content, "hello world")
        self.assertIn("hello world\n", output.getvalue())


if __name__ == "__main__":
    unittest.main()
