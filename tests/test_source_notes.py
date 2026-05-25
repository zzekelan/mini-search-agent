from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone

from mini_search_agent.session import SessionStore, TelemetryLogger
from mini_search_agent.sources import (
    SourceStore,
    record_sources_from_subagent_result,
    subagent_result_json_schema,
    subagent_result_response_format,
)


class SourceNotesTest(unittest.TestCase):
    def test_subagent_result_schema_exposes_json_fields(self):
        schema = subagent_result_json_schema()

        self.assertEqual(subagent_result_response_format(), {"type": "json_object"})
        self.assertIn("query", schema["properties"])
        self.assertIn("candidate_urls", schema["properties"])
        self.assertIn("fetched_sources", schema["properties"])

    def test_add_source_writes_note_and_index_under_msa_research(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SourceStore(temp_dir, topic_slug="Hybrid Retrieval")

            note = store.add_source(
                title="Hybrid Retrieval Survey",
                url="https://example.com/survey",
                retrieved_at="2026-05-26T00:00:00Z",
                fetch_status="success",
                reliability="high",
                queries=["hybrid retrieval reranking"],
                evidence="Hybrid retrieval combines sparse and dense signals.",
                notes="Survey article.",
            )

            note_text = note.path.read_text(encoding="utf-8")
            index_text = (store.sources_root / "index.md").read_text(encoding="utf-8")

        self.assertEqual(note.source_id, "W001")
        self.assertIn("# W001 - Hybrid Retrieval Survey", note_text)
        self.assertIn("- Reliability: high", note_text)
        self.assertIn("[W001](web/W001-hybrid-retrieval-survey.md)", index_text)
        self.assertIn("hybrid retrieval reranking", index_text)

    def test_add_source_deduplicates_by_url_and_merges_queries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SourceStore(temp_dir, topic_slug="topic")
            first = store.add_source(
                title="Same Source",
                url="https://example.com/source",
                retrieved_at="2026-05-26T00:00:00Z",
                fetch_status="success",
                reliability="medium",
                queries=["query A"],
                evidence="Evidence A",
            )
            second = store.add_source(
                title="Different Title",
                url="https://example.com/source",
                retrieved_at="2026-05-26T01:00:00Z",
                fetch_status="success",
                reliability="high",
                queries=["query B", "query A"],
                evidence="Evidence B",
            )

            notes = store.list_sources()
            index_text = (store.sources_root / "index.md").read_text(encoding="utf-8")

        self.assertEqual(first.source_id, second.source_id)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].queries, ("query A", "query B"))
        self.assertIn("query A; query B", index_text)

    def test_add_source_continues_existing_numbering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = SourceStore(temp_dir, topic_slug="topic")
            store.add_source(
                title="First",
                url="https://example.com/first",
                retrieved_at="2026-05-26T00:00:00Z",
                fetch_status="success",
                reliability="high",
                queries=["first"],
                evidence="First evidence.",
            )

            second = store.add_source(
                title="Second",
                url="https://example.com/second",
                retrieved_at="2026-05-26T00:01:00Z",
                fetch_status="partial",
                reliability="low",
                queries=["second"],
                evidence="Second evidence.",
            )

        self.assertEqual(second.source_id, "W002")

    def test_source_writes_are_recorded_in_telemetry(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            telemetry = TelemetryLogger(session, clock=lambda: now)
            store = SourceStore(temp_dir, topic_slug="topic")

            store.add_source(
                title="Telemetry Source",
                url="https://example.com/source",
                retrieved_at="2026-05-26T00:00:00Z",
                fetch_status="success",
                reliability="high",
                queries=["query A"],
                evidence="Evidence.",
                telemetry=telemetry,
            )
            store.add_source(
                title="Telemetry Source",
                url="https://example.com/source",
                retrieved_at="2026-05-26T00:01:00Z",
                fetch_status="success",
                reliability="high",
                queries=["query B"],
                evidence="Evidence.",
                telemetry=telemetry,
            )
            events = telemetry.read_events()

        self.assertEqual(
            [event["event"] for event in events],
            [
                "source_note.created",
                "source_index.updated",
                "source_note.deduplicated",
                "source_index.updated",
            ],
        )
        self.assertEqual(events[0]["metadata"]["source_id"], "W001")
        self.assertEqual(events[2]["metadata"]["merged_queries"], ["query A", "query B"])

    def test_record_sources_from_subagent_json_result_uses_pydantic_schema(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        result_json = json.dumps(
            {
                "query": "hybrid retrieval reranking",
                "candidate_urls": [
                    {
                        "url": "https://example.com/source",
                        "reason": "Relevant benchmark source.",
                    }
                ],
                "fetched_sources": [
                    {
                        "title": "Hybrid Retrieval Benchmark",
                        "url": "https://example.com/source",
                        "fetch_status": "success",
                        "reliability": "high",
                        "evidence": "Hybrid retrieval improves recall.",
                        "notes": "Fetched and verified.",
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            telemetry = TelemetryLogger(session, clock=lambda: now)
            store = SourceStore(temp_dir, topic_slug="topic")

            notes = record_sources_from_subagent_result(
                result_json,
                store=store,
                telemetry=telemetry,
                run_id="run-001",
            )
            events = telemetry.read_events()

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].source_id, "W001")
        self.assertEqual(notes[0].queries, ("hybrid retrieval reranking",))
        self.assertEqual(events[0]["event"], "source_note.created")

    def test_record_sources_from_invalid_subagent_json_records_parse_failure(self):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            telemetry = TelemetryLogger(session, clock=lambda: now)
            store = SourceStore(temp_dir, topic_slug="topic")

            notes = record_sources_from_subagent_result(
                "not json",
                store=store,
                telemetry=telemetry,
                run_id="run-001",
            )
            events = telemetry.read_events()

        self.assertEqual(notes, [])
        self.assertEqual(events[0]["event"], "source_note.parse_failed")


if __name__ == "__main__":
    unittest.main()
