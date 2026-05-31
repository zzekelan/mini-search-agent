from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval import CodeEvalResult, LLMJudgeResult, run_checks, save_eval_results
from eval.checks import check_citation_source_ids, check_subagent_fetch
from eval.data import EvalData, SubagentData, build_eval_data


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")


def _make_session(
    session_dir: Path,
    session_id: str,
    kind: str = "main",
    parent_session_id: str | None = None,
) -> Path:
    session_path = session_dir / session_id
    session_path.mkdir(parents=True, exist_ok=True)
    metadata: dict = {
        "session_id": session_id,
        "kind": kind,
        "created_at": "2026-05-26T00:00:00Z",
    }
    if parent_session_id:
        metadata["parent_session_id"] = parent_session_id
    (session_path / "session.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return session_path


class BuildEvalDataTest(unittest.TestCase):
    def test_reads_main_session_and_persists_eval_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_path = _make_session(tmp_path, "session-2026-05-26-001")
            _write_jsonl(session_path / "telemetry.jsonl", [
                {
                    "event": "session.started",
                    "run_id": "run-001",
                    "session_id": "session-2026-05-26-001",
                    "actor": "main",
                    "status": "ok",
                },
                {
                    "event": "llm.request.started",
                    "run_id": "run-001",
                    "actor": "main",
                },
                {
                    "event": "llm.response.finished",
                    "run_id": "run-001",
                    "latency_ms": 500.0,
                    "metadata": {"content_length": 100, "tool_call_count": 0},
                },
                {
                    "event": "final_answer.completed",
                    "run_id": "run-001",
                    "metadata": {
                        "cited_source_ids": ["W001"],
                        "available_source_ids": ["W001", "W002"],
                        "unknown_cited_source_ids": [],
                        "uncited_available_source_ids": ["W002"],
                        "has_sources_section": True,
                    },
                },
                {
                    "event": "stdout.finalized",
                    "run_id": "run-001",
                    "metadata": {"content_length": 200},
                },
            ])
            eval_dir = tmp_path / "evals" / "session-2026-05-26-001"

            data = build_eval_data(session_path, eval_dir)

            # In-memory
            self.assertEqual(data.session_id, "session-2026-05-26-001")
            self.assertEqual(data.turn_id, "run-001")
            self.assertEqual(data.final_answer_metadata["cited_source_ids"], ["W001"])
            self.assertEqual(data.final_answer_metadata["unknown_cited_source_ids"], [])
            self.assertEqual(data.subagents, [])

            # Persisted
            saved = eval_dir / "eval_data.json"
            self.assertTrue(saved.exists())
            loaded = EvalData.load(saved)
            self.assertEqual(loaded.session_id, data.session_id)
            self.assertEqual(loaded.turn_id, data.turn_id)

    def test_reads_sub_sessions_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_path = _make_session(tmp_path, "session-2026-05-26-001")
            _write_jsonl(session_path / "telemetry.jsonl", [
                {
                    "event": "session.started",
                    "run_id": "run-001",
                    "session_id": "session-2026-05-26-001",
                },
                {
                    "event": "final_answer.completed",
                    "run_id": "run-001",
                    "metadata": {
                        "cited_source_ids": [],
                        "unknown_cited_source_ids": [],
                    },
                },
                {"event": "stdout.finalized", "run_id": "run-001", "metadata": {}},
            ])

            sub_path = _make_session(
                session_path / "sub",
                "sub-001",
                kind="sub",
                parent_session_id="session-2026-05-26-001",
            )
            _write_jsonl(sub_path / "timeline.jsonl", [
                {"entry_id": "entry-001", "role": "user", "parts": [
                    {"type": "text", "text": "Find sources"}
                ]},
                {"entry_id": "entry-002", "role": "assistant", "parts": [
                    {"type": "text", "text": "done"},
                ]},
            ])
            _write_jsonl(sub_path / "telemetry.jsonl", [
                {
                    "event": "session.started",
                    "run_id": "run-001",
                    "actor": "subagent",
                },
                {
                    "event": "tool.web_search.started",
                    "run_id": "run-001",
                    "actor": "subagent",
                    "metadata": {"query": "test search"},
                },
                {
                    "event": "tool.web_search.finished",
                    "run_id": "run-001",
                    "actor": "subagent",
                    "latency_ms": 100.0,
                },
            ])
            eval_dir = tmp_path / "evals" / "session-2026-05-26-001"

            data = build_eval_data(session_path, eval_dir)

        self.assertEqual(len(data.subagents), 1)
        sub = data.subagents[0]
        self.assertEqual(sub.sub_session_id, "sub-001")
        self.assertEqual(len(sub.timeline), 2)
        self.assertEqual(len(sub.telemetry), 3)


class CitationSourceIdsTest(unittest.TestCase):
    def test_passes_when_no_unknown_cited_ids(self):
        data = EvalData(
            session_id="s1",
            turn_id="t1",
            final_answer_metadata={
                "cited_source_ids": ["W001", "W002"],
                "available_source_ids": ["W001", "W002", "W003"],
                "unknown_cited_source_ids": [],
            },
        )

        result = check_citation_source_ids(data)

        self.assertIsInstance(result, CodeEvalResult)
        self.assertEqual(result.check_id, "citation_source_ids")
        self.assertEqual(result.score, 1)
        self.assertEqual(result.label, "pass")

    def test_fails_when_cited_ids_are_unknown(self):
        data = EvalData(
            session_id="s1",
            turn_id="t1",
            final_answer_metadata={
                "unknown_cited_source_ids": ["W099", "W100"],
                "cited_source_ids": ["W001", "W099", "W100"],
                "available_source_ids": ["W001"],
            },
        )

        result = check_citation_source_ids(data)

        self.assertEqual(result.score, 0)
        self.assertIn("W099", result.label)
        self.assertIn("W100", result.label)

    def test_passes_when_metadata_is_empty(self):
        data = EvalData(session_id="s1", turn_id="t1")

        result = check_citation_source_ids(data)

        self.assertEqual(result.score, 1)
        self.assertEqual(result.label, "pass")


class SubagentFetchTest(unittest.TestCase):
    def test_passes_when_subagent_has_search_and_fetch(self):
        sub = SubagentData(
            sub_session_id="sub-001",
            telemetry=[
                {"event": "tool.web_search.started"},
                {"event": "tool.web_search.finished"},
                {"event": "tool.web_fetch.started"},
                {"event": "tool.web_fetch.finished"},
            ],
        )
        data = EvalData(session_id="s1", turn_id="t1", subagents=[sub])

        result = check_subagent_fetch(data)

        self.assertEqual(result.score, 1)
        self.assertEqual(result.label, "pass")

    def test_passes_when_subagent_has_no_tool_calls(self):
        sub = SubagentData(sub_session_id="sub-002", telemetry=[
            {"event": "session.started"},
        ])
        data = EvalData(session_id="s1", turn_id="t1", subagents=[sub])

        result = check_subagent_fetch(data)

        self.assertEqual(result.score, 1)
        self.assertEqual(result.label, "pass")

    def test_passes_when_no_subagents_exist(self):
        data = EvalData(session_id="s1", turn_id="t1", subagents=[])

        result = check_subagent_fetch(data)

        self.assertEqual(result.score, 1)
        self.assertEqual(result.label, "pass")

    def test_fails_when_subagent_searches_but_never_fetches(self):
        sub = SubagentData(
            sub_session_id="sub-003",
            telemetry=[
                {"event": "session.started"},
                {"event": "tool.web_search.started"},
                {"event": "tool.web_search.finished"},
                {"event": "tool.web_search.started"},
                {"event": "tool.web_search.finished"},
                {"event": "tool.web_search.started"},
                {"event": "tool.web_search.finished"},
                {"event": "tool.web_search.started"},
                {"event": "tool.web_search.finished"},
                {"event": "tool.web_search.started"},
                {"event": "tool.web_search.finished"},
            ],
        )
        data = EvalData(session_id="s1", turn_id="t1", subagents=[sub])

        result = check_subagent_fetch(data)

        self.assertEqual(result.score, 0)
        self.assertIn("sub-003", result.label)
        self.assertIn("searches=5", result.label)
        self.assertIn("fetches=0", result.label)

    def test_handles_multiple_subagents_with_mixed_results(self):
        sub1 = SubagentData(
            sub_session_id="sub-001",
            telemetry=[
                {"event": "tool.web_search.started"},
                {"event": "tool.web_fetch.started"},
            ],
        )
        sub2 = SubagentData(
            sub_session_id="sub-002",
            telemetry=[
                {"event": "tool.web_search.started"},
                {"event": "tool.web_search.started"},
            ],
        )
        data = EvalData(
            session_id="s1", turn_id="t1", subagents=[sub1, sub2],
        )

        result = check_subagent_fetch(data)

        self.assertEqual(result.score, 0)
        self.assertIn("sub-002", result.label)
        self.assertNotIn("sub-001", result.label)


class RunChecksTest(unittest.TestCase):
    def test_returns_results_for_all_registered_checks_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_path = _make_session(tmp_path, "session-2026-05-26-001")
            _write_jsonl(session_path / "telemetry.jsonl", [
                {
                    "event": "session.started",
                    "run_id": "run-001",
                    "session_id": "session-2026-05-26-001",
                },
                {
                    "event": "llm.request.started",
                    "run_id": "run-001",
                },
                {
                    "event": "llm.response.finished",
                    "run_id": "run-001",
                    "metadata": {"content_length": 100, "tool_call_count": 0},
                },
                {
                    "event": "final_answer.completed",
                    "run_id": "run-001",
                    "metadata": {
                        "cited_source_ids": ["W001"],
                        "available_source_ids": ["W001"],
                        "unknown_cited_source_ids": [],
                    },
                },
                {"event": "stdout.finalized", "run_id": "run-001", "metadata": {}},
            ])
            evals_root = tmp_path / "evals"

            results = run_checks(session_path, evals_root=evals_root)

            # Results in memory
            check_ids = {r.check_id for r in results}
            self.assertEqual(len(results), 2)
            self.assertIn("citation_source_ids", check_ids)
            self.assertIn("subagent_fetch", check_ids)
            for r in results:
                self.assertIsInstance(r, CodeEvalResult)

            # EvalData persisted
            eval_data_path = evals_root / "session-2026-05-26-001" / "eval_data.json"
            self.assertTrue(eval_data_path.exists())


class SaveEvalResultsTest(unittest.TestCase):
    def test_writes_results_json_with_code_and_llm_entries(self):
        results = [
            CodeEvalResult(check_id="ck1", score=1, label="pass"),
            CodeEvalResult(check_id="ck2", score=0, label="fail: broken"),
            LLMJudgeResult(
                check_id="source_precision", score=0.7,
                label="PARTIAL_MATCH", explanation="W001 matches, W002 does not.",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "evals" / "session-2026-05-26-001"

            path = save_eval_results("session-2026-05-26-001", results, eval_dir)

            self.assertEqual(path.name, "results.json")
            report = json.loads(path.read_text("utf-8"))
            self.assertEqual(report["session_id"], "session-2026-05-26-001")
            self.assertEqual(len(report["results"]), 3)

            # Code result
            self.assertEqual(report["results"][0]["kind"], "code")
            self.assertEqual(report["results"][0]["check_id"], "ck1")
            self.assertEqual(report["results"][0]["score"], 1)

            # LLM judge result
            self.assertEqual(report["results"][2]["kind"], "llm")
            self.assertEqual(report["results"][2]["check_id"], "source_precision")
            self.assertEqual(report["results"][2]["score"], 0.7)
            self.assertEqual(report["results"][2]["explanation"],
                             "W001 matches, W002 does not.")

    def test_creates_eval_dir_if_missing(self):
        results = [CodeEvalResult(check_id="ck1", score=1, label="pass")]
        with tempfile.TemporaryDirectory() as tmp:
            eval_dir = Path(tmp) / "nested" / "evals" / "s1"

            path = save_eval_results("s1", results, eval_dir)

            self.assertTrue(path.exists())
            self.assertEqual(path.name, "results.json")


class EvalSessionTest(unittest.TestCase):
    def test_persists_eval_data_and_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            session_path = _make_session(tmp_path, "session-2026-05-26-001")
            _write_jsonl(session_path / "telemetry.jsonl", [
                {
                    "event": "session.started",
                    "run_id": "run-001",
                },
                {
                    "event": "final_answer.completed",
                    "run_id": "run-001",
                    "metadata": {
                        "cited_source_ids": [],
                        "unknown_cited_source_ids": [],
                    },
                },
                {"event": "stdout.finalized", "run_id": "run-001", "metadata": {}},
            ])
            evals_root = tmp_path / "evals"
            from eval import eval_session

            results = eval_session(session_path, evals_root=evals_root, judges=[])

            self.assertEqual(len(results), 2)

            eval_dir = evals_root / "session-2026-05-26-001"
            self.assertTrue((eval_dir / "eval_data.json").exists())
            self.assertTrue((eval_dir / "results.json").exists())

            report = json.loads((eval_dir / "results.json").read_text("utf-8"))
            self.assertEqual(report["session_id"], "session-2026-05-26-001")

    def test_defaults_to_dot_msa_evals(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Simulate the default .msa/evals path
            msa_dir = tmp_path / ".msa"
            evals_default = msa_dir / "evals"
            session_path = _make_session(msa_dir / "sessions", "session-2026-05-26-001")
            _write_jsonl(session_path / "telemetry.jsonl", [
                {
                    "event": "session.started",
                    "run_id": "run-001",
                },
                {
                    "event": "final_answer.completed",
                    "run_id": "run-001",
                    "metadata": {
                        "cited_source_ids": [],
                        "unknown_cited_source_ids": [],
                    },
                },
                {"event": "stdout.finalized", "run_id": "run-001", "metadata": {}},
            ])

            # Patch DEFAULT_EVALS_ROOT temporarily
            import eval as eval_mod
            old_default = eval_mod.DEFAULT_EVALS_ROOT
            eval_mod.DEFAULT_EVALS_ROOT = evals_default
            try:
                from eval import eval_session
                results = eval_session(session_path, judges=[])
                self.assertEqual(len(results), 2)
                self.assertTrue((evals_default / "session-2026-05-26-001" / "results.json").exists())
            finally:
                eval_mod.DEFAULT_EVALS_ROOT = old_default


if __name__ == "__main__":
    unittest.main()
