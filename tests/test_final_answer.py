from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_search_agent.citations import extract_cited_source_ids
from mini_search_agent.llm import ModelResponse
from mini_search_agent.runner import run_research
from mini_search_agent.session import read_jsonl


class FinalAnswerClient:
    def __init__(self):
        self.main_calls = 0
        self.subagent_calls = 0

    def complete(self, messages, tools=None):
        system_prompt = messages[0]["content"]
        if system_prompt.startswith("You are a Search Subagent"):
            self.subagent_calls += 1
            return ModelResponse(content=self._subagent_result(self.subagent_calls))

        self.main_calls += 1
        if self.main_calls == 1:
            return ModelResponse(
                content="Collecting three angles.",
                tool_calls=[
                    {"id": "call-001", "name": "subagent", "arguments": {"description": "a", "prompt": "A"}},
                    {"id": "call-002", "name": "subagent", "arguments": {"description": "b", "prompt": "B"}},
                    {"id": "call-003", "name": "subagent", "arguments": {"description": "c", "prompt": "C"}},
                ],
            )
        return ModelResponse(
            content="\n".join(
                [
                    "The answer cites all available notes [W001] [W002] [W003].",
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


class FinalAnswerTest(unittest.TestCase):
    def test_extract_cited_source_ids_preserves_first_seen_order(self):
        self.assertEqual(extract_cited_source_ids("Use [W002], [W001], [W002]."), ["W002", "W001"])

    def test_final_answer_is_stdout_and_timeline_only_with_citation_telemetry(self):
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
            with patch.dict(os.environ, {}, clear=True):
                answer = run_research("Need cited answer", workspace=workspace, client=FinalAnswerClient(), output=output)

            session_path = next((workspace / ".msa" / "sessions").iterdir())
            timeline = read_jsonl(session_path / "main.jsonl")
            telemetry = read_jsonl(session_path / "telemetry.jsonl")
            research_files = list((workspace / ".msa" / "research").glob("**/*"))

        final_event = [event for event in telemetry if event["event"] == "final_answer.completed"][0]
        self.assertEqual(output.getvalue(), answer + "\n")
        self.assertIn("## Sources", answer)
        self.assertEqual(final_event["metadata"]["cited_source_ids"], ["W001", "W002", "W003"])
        self.assertEqual(final_event["metadata"]["available_source_ids"], ["W001", "W002", "W003"])
        self.assertEqual(final_event["metadata"]["unknown_cited_source_ids"], [])
        self.assertTrue(final_event["metadata"]["has_sources_section"])
        self.assertIn("The answer cites all available notes", timeline[-1]["parts"][0]["text"])
        self.assertFalse(any(path.name.startswith("final") for path in research_files))


if __name__ == "__main__":
    unittest.main()
