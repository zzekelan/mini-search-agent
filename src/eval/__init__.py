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

    Delegates to eval_session() without LLM judges.
    """
    return eval_session(session_path, evals_root=evals_root, judges=[])


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
    judges: list | None = None,
    workspace: Path | None = None,
) -> list[EvalResult]:
    """Run all checks on a session, persist EvalData and results.

    If *judges* is None, defaults to all registered LLM_JUDGES.
    Pass an empty list to skip LLM judges.
    """
    from .data import build_eval_data
    from .checks import ALL_CHECKS

    if judges is None:
        from .judge_prompts import LLM_JUDGES as default_judges
        judges = default_judges

    root = evals_root or DEFAULT_EVALS_ROOT
    eval_dir = root / session_path.name
    ws = workspace or Path.cwd()

    data = build_eval_data(session_path, eval_dir)
    code_results = [check(data) for check in ALL_CHECKS]

    llm_results: list[EvalResult] = []
    if judges:
        from .judge import run_llm_judges
        llm_results = run_llm_judges(session_path, eval_dir, ws, judges)

    results = list(code_results) + llm_results
    save_eval_results(session_path.name, results, eval_dir)
    return results
