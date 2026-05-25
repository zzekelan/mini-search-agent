from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from mini_search_agent.llm import ModelResponse
from mini_search_agent.session import SessionStore, TelemetryLogger, TimelineWriter, read_jsonl
from mini_search_agent.subagent import SubagentTool, subagent_tool_schema
from mini_search_agent.tool_filter import filter_tools_for_search_subagent
from mini_search_agent.tools.base import ToolResult
from mini_search_agent.tools import shell_tool_schema, web_fetch_tool_schema, web_search_tool_schema


class RecordingClient:
    def __init__(self):
        self.messages = None
        self.tools = None

    def complete(self, messages, tools=None):
        self.messages = messages
        self.tools = tools
        return ModelResponse(content="child model output")


class ModelRequestedToolsClient:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                content="Using a model-requested tool.",
                tool_calls=[
                    {
                        "id": "fetch-001",
                        "name": "web_fetch",
                        "arguments": {"url": "https://example.com/source"},
                    }
                ],
            )
        if self.calls == 2:
            return ModelResponse(
                content="Using another model-requested tool.",
                tool_calls=[
                    {
                        "id": "search-001",
                        "name": "web_search",
                        "arguments": {"query": "hybrid retrieval reranking"},
                    }
                ],
            )
        return ModelResponse(content="child done")


class SubagentTest(unittest.TestCase):
    def test_subagent_tool_schema_accepts_description_and_prompt(self):
        schema = subagent_tool_schema()

        self.assertEqual(schema["function"]["name"], "subagent")
        self.assertEqual(schema["function"]["parameters"]["required"], ["description", "prompt"])

    def test_tool_filter_removes_shell_and_subagent_from_search_subagent(self):
        filtered = filter_tools_for_search_subagent(
            [
                web_search_tool_schema(),
                web_fetch_tool_schema(),
                shell_tool_schema(),
                subagent_tool_schema(),
            ]
        )

        names = [tool["function"]["name"] for tool in filtered]
        self.assertEqual(names, ["web_search", "web_fetch"])

    def test_create_sub_session_uses_child_folder_and_minimal_metadata(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SessionStore(temp_dir, clock=lambda: now)
            parent = store.create_main_session()
            child = store.create_sub_session(parent)

            metadata = json.loads((child.path / "session.json").read_text(encoding="utf-8"))

        self.assertEqual(child.session_id, "sub-001")
        self.assertEqual(child.timeline_path.name, "timeline.jsonl")
        self.assertEqual(metadata, {
            "session_id": "sub-001",
            "kind": "sub",
            "parent_session_id": "session-2026-05-26-001",
            "created_at": "2026-05-26T00:00:00Z",
        })

    def test_subagent_run_is_isolated_and_parent_records_tool_result(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            parent = SessionStore(workspace, clock=lambda: now).create_main_session()
            parent_timeline = TimelineWriter(parent, clock=lambda: now)
            parent_telemetry = TelemetryLogger(parent, clock=lambda: now)
            client = RecordingClient()

            result = SubagentTool(
                workspace=workspace,
                client=client,
                parent_session=parent,
                parent_timeline=parent_timeline,
                parent_telemetry=parent_telemetry,
                parent_tools=[web_search_tool_schema(), web_fetch_tool_schema(), shell_tool_schema(), subagent_tool_schema()],
            ).run(description="retrieval angle", prompt="Find sources about hybrid retrieval")

            child_path = Path(result.metadata["sub_session_path"])
            parent_entries = parent_timeline.read_entries()
            child_entries = read_jsonl(child_path / "timeline.jsonl")
            parent_events = parent_telemetry.read_events()
            child_events = read_jsonl(child_path / "telemetry.jsonl")

        self.assertEqual(result.content, "child model output")
        self.assertEqual([tool["function"]["name"] for tool in client.tools], ["web_search", "web_fetch"])
        self.assertIn("Search Subagent", client.messages[0]["content"])
        self.assertEqual(parent_entries[0]["parts"][0]["tool_name"], "subagent")
        self.assertEqual(parent_entries[0]["parts"][1]["metadata"]["sub_session_path"], str(child_path))
        self.assertEqual([event["event"] for event in parent_events], ["subagent.started", "subagent.completed"])
        self.assertEqual([entry["role"] for entry in child_entries], ["user", "assistant"])
        self.assertEqual(
            [event["event"] for event in child_events],
            ["session.started", "llm.request.started", "llm.response.finished"],
        )

    def test_subagent_executes_model_requested_tool_calls_in_child_session(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        calls = []

        def web_search(arguments):
            calls.append(("web_search", arguments))
            return ToolResult(content="URL: https://example.com/source", metadata={"query": arguments["query"]})

        def web_fetch(arguments):
            calls.append(("web_fetch", arguments))
            return ToolResult(
                content="fetched evidence",
                metadata={
                    "url": arguments["url"],
                    "status_code": 200,
                    "content_type": "text/html",
                    "truncated": False,
                },
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            parent = SessionStore(workspace, clock=lambda: now).create_main_session()
            parent_timeline = TimelineWriter(parent, clock=lambda: now)
            parent_telemetry = TelemetryLogger(parent, clock=lambda: now)

            result = SubagentTool(
                workspace=workspace,
                client=ModelRequestedToolsClient(),
                parent_session=parent,
                parent_timeline=parent_timeline,
                parent_telemetry=parent_telemetry,
                parent_tools=[web_search_tool_schema(), web_fetch_tool_schema(), shell_tool_schema()],
                tool_handlers={"web_search": web_search, "web_fetch": web_fetch},
            ).run(description="retrieval angle", prompt="Find sources about hybrid retrieval")

            child_path = Path(result.metadata["sub_session_path"])
            child_entries = read_jsonl(child_path / "timeline.jsonl")

        self.assertEqual(
            calls,
            [
                ("web_fetch", {"url": "https://example.com/source"}),
                ("web_search", {"query": "hybrid retrieval reranking"}),
            ],
        )
        flattened_parts = [part for entry in child_entries for part in entry["parts"]]
        self.assertIn("web_search", [part.get("tool_name") for part in flattened_parts])
        self.assertIn("web_fetch", [part.get("tool_name") for part in flattened_parts])
        self.assertEqual(result.content, "child done")


if __name__ == "__main__":
    unittest.main()
