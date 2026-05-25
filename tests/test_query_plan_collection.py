from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_search_agent.llm import ModelResponse
from mini_search_agent.runner import run_research
from mini_search_agent.session import read_jsonl


class ScriptedResearchClient:
    def __init__(self):
        self.main_calls = 0
        self.subagent_calls = 0

    def complete(self, messages, tools=None):
        developer = messages[0]["content"]
        if developer.startswith("You are a Search Subagent"):
            self.subagent_calls += 1
            return ModelResponse(content=self._subagent_result(self.subagent_calls))

        self.main_calls += 1
        if self.main_calls == 1:
            return ModelResponse(
                content="I will use three search angles.",
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
                    {
                        "id": "call-003",
                        "name": "subagent",
                        "arguments": {"description": "limitations", "prompt": "Find limitation sources"},
                    },
                ],
            )
        return ModelResponse(
            content="\n".join(
                [
                    "Final answer with citations [W001] [W002] [W003].",
                    "",
                    "## Sources",
                    "[W001] Source 1 - https://example.com/source-1",
                    "[W002] Source 2 - https://example.com/source-2",
                    "[W003] Source 3 - https://example.com/source-3",
                ]
            )
        )

    def _subagent_result(self, number):
        return "\n".join(
            [
                "## Search Subagent Result",
                "",
                "### Query",
                f"query {number}",
                "",
                "### Candidate URLs",
                f"- https://example.com/source-{number} - relevant",
                "",
                "### Fetched Sources",
                f"#### Source {number}",
                f"- URL: https://example.com/source-{number}",
                "- Fetch status: success",
                "- Reliability: high",
                f"- Evidence: evidence {number}",
                "- Notes: verified",
            ]
        )


class QueryPlanCollectionTest(unittest.TestCase):
    def test_main_agent_collects_three_subagent_angles_and_writes_source_notes(self):
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
        self.assertEqual(client.subagent_calls, 3)
        self.assertEqual(source_count, 3)
        self.assertIn("[W003]", index_text)
        self.assertEqual(
            [event["event"] for event in telemetry].count("subagent.started"),
            3,
        )
        self.assertEqual(
            [event["event"] for event in telemetry].count("source_note.created"),
            3,
        )


if __name__ == "__main__":
    unittest.main()
