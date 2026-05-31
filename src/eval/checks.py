from __future__ import annotations

from dataclasses import dataclass, field

from . import CodeEvalResult
from .data import EvalData


@dataclass
class LLMJudgeSpec:
    """Specification for one LLM Judge check."""
    check_id: str
    system_prompt_template: str
    max_turns: int = 60


def check_citation_source_ids(data: EvalData) -> CodeEvalResult:
    """Fail if any cited source IDs do not exist in available sources."""
    meta = data.final_answer_metadata
    unknown = meta.get("unknown_cited_source_ids", [])

    if not unknown:
        return CodeEvalResult(
            check_id="citation_source_ids",
            score=1,
            label="pass",
        )

    return CodeEvalResult(
        check_id="citation_source_ids",
        score=0,
        label=f"fail: unknown cited IDs: {unknown}",
    )


def check_subagent_fetch(data: EvalData) -> CodeEvalResult:
    """Fail if any Search Subagent searched but never fetched."""
    failures: list[str] = []
    for sub in data.subagents:
        searches = sum(
            1 for e in sub.telemetry if e["event"] == "tool.web_search.started"
        )
        fetches = sum(
            1 for e in sub.telemetry if e["event"] == "tool.web_fetch.started"
        )
        if searches > 0 and fetches == 0:
            failures.append(
                f"{sub.sub_session_id}(searches={searches}, fetches={fetches})"
            )

    if not failures:
        return CodeEvalResult(
            check_id="subagent_fetch",
            score=1,
            label="pass",
        )

    return CodeEvalResult(
        check_id="subagent_fetch",
        score=0,
        label=f"fail: {', '.join(failures)}",
    )


ALL_CHECKS = [check_citation_source_ids, check_subagent_fetch]
