from __future__ import annotations

import io
import json
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

    def complete(self, messages, tools=None, response_format=None):
        system_prompt = messages[0]["content"]
        if system_prompt.startswith("You are a Search Subagent"):
            self.subagent_calls += 1
            return ModelResponse(content=self._subagent_result(self.subagent_calls))

        self.main_calls += 1
        if self.main_calls == 1:
            return ModelResponse(
                content="Collecting a model-requested source note.",
                tool_calls=[
                    {"id": "call-001", "name": "subagent", "arguments": {"description": "a", "prompt": "A"}},
                ],
            )
        return ModelResponse(
            content="\n".join(
                [
                    "The answer cites the recorded note [W001].",
                    "",
                    "## Sources",
                    "[W001] Source 1 - https://example.com/source-1",
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
        self.assertEqual(final_event["metadata"]["cited_source_ids"], ["W001"])
        self.assertEqual(final_event["metadata"]["available_source_ids"], ["W001"])
        self.assertEqual(final_event["metadata"]["unknown_cited_source_ids"], [])
        self.assertTrue(final_event["metadata"]["has_sources_section"])
        self.assertIn("The answer cites the recorded note", timeline[-1]["parts"][0]["text"])
        self.assertFalse(any(path.name.startswith("final") for path in research_files))


if __name__ == "__main__":
    unittest.main()
