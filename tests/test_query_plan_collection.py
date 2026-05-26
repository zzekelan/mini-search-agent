from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_search_agent.llm import ModelResponse, ModelStreamEvent
from mini_search_agent.runner import run_research
from mini_search_agent.session import read_jsonl


class ScriptedResearchClient:
    def __init__(self):
        self.main_calls = 0
        self.subagent_calls = 0

    def complete(self, messages, tools=None, response_format=None):
        system_prompt = messages[0]["content"]
        if system_prompt.startswith("You are a Search Subagent"):
            self.subagent_calls += 1
            return ModelResponse(content=self._subagent_result(self.subagent_calls))

        self.main_calls += 1
        if self.main_calls == 1:
            return ModelResponse(
                content="I will request two focused source collectors.",
                tool_calls=[
                    {
                        "id": "call-001",
                        "name": "subagent",
                        "arguments": {"description": "products", "prompt": "Find product sources"},
                    },
                    {
                        "id": "call-002",
                        "name": "subagent",
                        "arguments": {"description": "papers", "prompt": "Find paper sources"},
                    },
                ],
            )
        return ModelResponse(
            content="\n".join(
                [
                    "Final answer with citations [W001] [W002].",
                    "",
                    "## Sources",
                    "[W001] Source 1 - https://example.com/source-1",
                    "[W002] Source 2 - https://example.com/source-2",
                ]
            )
        )

    def _subagent_result(self, number):
        return json.dumps(
            {
                "query": f"query {number}",
                "candidate_urls": [
                    {
                        "url": f"https://example.com/source-{number}",
                        "reason": "relevant",
                    }
                ],
                "fetched_sources": [
                    {
                        "title": f"Source {number}",
                        "url": f"https://example.com/source-{number}",
                        "fetch_status": "success",
                        "reliability": "high",
                        "evidence": f"evidence {number}",
                        "notes": "verified",
                    }
                ],
            }
        )


class StreamingToolStatusClient:
    def __init__(self):
        self.main_calls = 0
        self.subagent_calls = 0

    def stream_complete(self, messages, tools=None, response_format=None):
        self.main_calls += 1
        if self.main_calls == 1:
            yield ModelStreamEvent(type="content_delta", delta="Collecting sources.\n")
            yield ModelStreamEvent(
                type="done",
                response=ModelResponse(
                    content="Collecting sources.",
                    tool_calls=[
                        {
                            "id": "call-001",
                            "name": "subagent",
                            "arguments": {"description": "source", "prompt": "Find one source"},
                        }
                    ],
                ),
            )
            return
        yield ModelStreamEvent(type="content_delta", delta="Final answer [W001].\n\n## Sources\n[W001] Source 1")
        yield ModelStreamEvent(
            type="done",
            response=ModelResponse(content="Final answer [W001].\n\n## Sources\n[W001] Source 1"),
        )

    def complete(self, messages, tools=None, response_format=None):
        system_prompt = messages[0]["content"]
        if system_prompt.startswith("You are a Search Subagent"):
            self.subagent_calls += 1
            return ModelResponse(content=ScriptedResearchClient()._subagent_result(1))
        raise AssertionError("main agent should use stream_complete")


class TtyOutput(io.StringIO):
    def isatty(self):
        return True


class QueryPlanCollectionTest(unittest.TestCase):
    def test_model_requested_subagents_write_source_notes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai-compatible",
                        "LLM_API_KEY=test-key",
                        "LLM_MODEL=test-model",
                        "LLM_BASE_URL=https://llm.example/v1",
                    ]
                ),
                encoding="utf-8",
            )
            client = ScriptedResearchClient()
            output = io.StringIO()
            with patch.dict(os.environ, {}, clear=True):
                answer = run_research("Compare search agents", workspace=workspace, client=client, output=output)

            session_path = next((workspace / ".msa" / "sessions").iterdir())
            telemetry = read_jsonl(session_path / "telemetry.jsonl")
            source_index = next((workspace / ".msa" / "research").glob("*/sources/index.md"))
            source_count = len(list((source_index.parent / "web").glob("W*.md")))
            index_text = source_index.read_text(encoding="utf-8")

        self.assertIn("[W001]", answer)
        self.assertEqual(client.subagent_calls, 2)
        self.assertEqual(source_count, 2)
        self.assertIn("[W002]", index_text)
        self.assertEqual(
            [event["event"] for event in telemetry].count("subagent.started"),
            2,
        )
        self.assertEqual(
            [event["event"] for event in telemetry].count("source_note.created"),
            2,
        )

    def test_interactive_run_reports_tool_call_status_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai-compatible",
                        "LLM_API_KEY=test-key",
                        "LLM_MODEL=test-model",
                        "LLM_BASE_URL=https://llm.example/v1",
                    ]
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            client = StreamingToolStatusClient()
            with patch.dict(os.environ, {}, clear=True):
                run_research("Compare search agents", workspace=workspace, client=client, output=output, interactive=True)

        text = output.getvalue()
        self.assertNotIn("[tool] subagent: source pending", text)
        self.assertIn("[tool] subagent: source running", text)
        self.assertIn("[tool] subagent: source done", text)

    def test_tty_interactive_run_updates_tool_call_status_in_place(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai-compatible",
                        "LLM_API_KEY=test-key",
                        "LLM_MODEL=test-model",
                        "LLM_BASE_URL=https://llm.example/v1",
                    ]
                ),
                encoding="utf-8",
            )
            output = TtyOutput()
            with patch.dict(os.environ, {}, clear=True):
                run_research(
                    "Compare search agents",
                    workspace=workspace,
                    client=StreamingToolStatusClient(),
                    output=output,
                    interactive=True,
                )

        text = output.getvalue()
        self.assertIn("\r", text)
        self.assertIn("subagent: source running", text)
        self.assertIn("[done] subagent: source done", text)
        self.assertNotIn("[tool] subagent: source running", text)


if __name__ == "__main__":
    unittest.main()
