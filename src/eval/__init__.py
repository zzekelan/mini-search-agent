from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EVALS_ROOT = Path(".msa/evals")


@dataclass
class EvalResult:
    """Base class for eval check results."""
    check_id: str


@dataclass
class CodeEvalResult(EvalResult):
    """Result from a deterministic code eval."""
    score: int
    label: str


@dataclass
class LLMJudgeResult(EvalResult):
    """Result from an LLM-as-a-judge eval."""
    score: int
    label: str
    explanation: str


def _serialize_result(r: EvalResult) -> dict:
    entry: dict = {"check_id": r.check_id, "score": r.score, "label": r.label}
    if isinstance(r, LLMJudgeResult):
        entry["kind"] = "llm"
        entry["explanation"] = r.explanation
    else:
        entry["kind"] = "code"
    return entry


def run_checks(
    session_path: Path,
    evals_root: Path | None = None,
) -> list[EvalResult]:
    """Run all registered code checks on a session.

    Builds and persists EvalData to the eval session directory.
    """
    from .data import build_eval_data
    from .checks import ALL_CHECKS

    root = evals_root or DEFAULT_EVALS_ROOT
    eval_dir = root / session_path.name
    data = build_eval_data(session_path, eval_dir)
    return [check(data) for check in ALL_CHECKS]


def save_eval_results(
    session_id: str,
    results: list[EvalResult],
    eval_dir: Path,
) -> Path:
    """Write eval results to eval_dir/results.json."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_path = eval_dir / "results.json"
    report = {
        "session_id": session_id,
        "results": [_serialize_result(r) for r in results],
    }
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output_path


def eval_session(
    session_path: Path,
    evals_root: Path | None = None,
) -> list[EvalResult]:
    """Run all checks on a session, persist EvalData and results."""
    root = evals_root or DEFAULT_EVALS_ROOT
    eval_dir = root / session_path.name

    results = run_checks(session_path, evals_root)
    # Future: results += run_llm_judges(...)

    save_eval_results(session_path.name, results, eval_dir)
    return results
