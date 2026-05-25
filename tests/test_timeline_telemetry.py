from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from mini_search_agent.llm import ModelResponse
from mini_search_agent.projection import project_timeline_to_openai
from mini_search_agent.runner import run_research
from mini_search_agent.session import (
    SessionStore,
    TelemetryLogger,
    TimelineWriter,
    read_jsonl,
    text_part,
    tool_call_part,
    tool_result_part,
)


class RecordingClient:
    def complete(self, messages, tools=None):
        return ModelResponse(content="telemetry answer")


class TimelineTelemetryTest(unittest.TestCase):
    def test_main_session_creates_metadata_timeline_and_telemetry_files(self):
        now = datetime(2026, 5, 26, 1, 2, 3, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()

            self.assertEqual(session.session_id, "session-2026-05-26-001")
            self.assertTrue((session.path / "session.json").exists())
            self.assertTrue(session.timeline_path.exists())
            self.assertTrue(session.telemetry_path.exists())
            metadata = json.loads((session.path / "session.json").read_text(encoding="utf-8"))

        self.assertEqual(metadata["kind"], "main")
        self.assertEqual(metadata["created_at"], "2026-05-26T01:02:03Z")

    def test_timeline_entries_project_to_openai_messages(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            writer = TimelineWriter(session, clock=lambda: now)
            writer.append(role="user", parts=[text_part("question")], produced_by_run="run-001")
            writer.append(
                role="assistant",
                parts=[
                    text_part("looking"),
                    tool_call_part("call-001", "web_search", {"query": "q"}),
                    tool_result_part("call-001", "web_search", "result", metadata={"ignored": True}),
                ],
                produced_by_run="run-001",
            )

            messages = project_timeline_to_openai(
                writer.read_entries(),
                system_prompt="system instructions",
            )

        self.assertEqual(messages[0], {"role": "system", "content": "system instructions"})
        self.assertEqual(messages[1], {"role": "user", "content": "question"})
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call-001")
        self.assertEqual(messages[3], {"role": "tool", "tool_call_id": "call-001", "content": "result"})

    def test_run_research_writes_timeline_and_telemetry_for_real_user_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai-compatible",
                        "LLM_API_KEY=super-secret-key",
                        "LLM_MODEL=test-model",
                        "LLM_BASE_URL=https://llm.example/v1",
                    ]
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                output = io.StringIO()
                run_research("Research this", workspace=workspace, client=RecordingClient(), output=output)

            session_path = next((workspace / ".msa" / "sessions").iterdir())
            timeline = read_jsonl(session_path / "main.jsonl")
            telemetry = read_jsonl(session_path / "telemetry.jsonl")

        self.assertEqual(output.getvalue(), "telemetry answer\n")
        self.assertEqual([entry["role"] for entry in timeline], ["user", "assistant"])
        self.assertEqual(timeline[0]["parts"][0]["text"], "Research this")
        self.assertEqual(timeline[1]["parts"][0]["text"], "telemetry answer")
        self.assertEqual(
            [event["event"] for event in telemetry],
            [
                "session.started",
                "llm.request.started",
                "llm.response.finished",
                "final_answer.completed",
                "stdout.finalized",
            ],
        )
        telemetry_text = "\n".join(json.dumps(event) for event in telemetry)
        self.assertNotIn("super-secret-key", telemetry_text)

    def test_telemetry_redacts_sensitive_metadata_fields(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            logger = TelemetryLogger(session, clock=lambda: now)

            logger.emit(
                "llm.request.started",
                run_id="run-001",
                metadata={"api_key": "secret", "token": "secret", "model": "safe"},
            )
            event = logger.read_events()[0]

        self.assertEqual(event["metadata"]["api_key"], "[redacted]")
        self.assertEqual(event["metadata"]["token"], "[redacted]")
        self.assertEqual(event["metadata"]["model"], "safe")


if __name__ == "__main__":
    unittest.main()
