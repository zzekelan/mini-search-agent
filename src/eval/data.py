from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mini_search_agent.session import read_jsonl


@dataclass
class SubagentData:
    """Timeline + Telemetry for one Search Subagent Sub-session."""
    sub_session_id: str
    timeline: list[dict[str, Any]] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class EvalData:
    """Merged Session Timeline + Telemetry for eval consumption."""
    session_id: str
    turn_id: str
    timeline: list[dict[str, Any]] = field(default_factory=list)
    telemetry: list[dict[str, Any]] = field(default_factory=list)
    final_answer_metadata: dict[str, Any] = field(default_factory=dict)
    subagents: list[SubagentData] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Persist EvalData as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dataclasses.asdict(self), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> EvalData:
        """Load EvalData from a JSON file."""
        raw = json.loads(path.read_text("utf-8"))
        return cls(
            session_id=raw["session_id"],
            turn_id=raw["turn_id"],
            final_answer_metadata=raw.get("final_answer_metadata", {}),
            subagents=[SubagentData(**s) for s in raw.get("subagents", [])],
        )


def _build_raw(session_path: Path) -> EvalData:
    """Build EvalData from session files (no I/O beyond reading)."""

    session_json = json.loads((session_path / "session.json").read_text("utf-8"))
    session_id: str = session_json["session_id"]

    timeline = read_jsonl(session_path / "main.jsonl")
    telemetry = read_jsonl(session_path / "telemetry.jsonl")

    turn_id = "unknown"
    for event in telemetry:
        if "run_id" in event:
            turn_id = str(event["run_id"])
            break

    final_answer_metadata: dict[str, Any] = {}
    for event in telemetry:
        if event["event"] == "final_answer.completed":
            final_answer_metadata = event.get("metadata", {})
            break

    subagents: list[SubagentData] = []
    sub_dir = session_path / "sub"
    if sub_dir.exists() and sub_dir.is_dir():
        for sub_path in sorted(sub_dir.iterdir()):
            if not sub_path.is_dir():
                continue
            sub_session = json.loads(
                (sub_path / "session.json").read_text("utf-8")
            )
            sub_timeline = read_jsonl(sub_path / "timeline.jsonl")
            sub_telemetry = read_jsonl(sub_path / "telemetry.jsonl")
            subagents.append(SubagentData(
                sub_session_id=sub_session["session_id"],
                timeline=sub_timeline,
                telemetry=sub_telemetry,
            ))

    return EvalData(
        session_id=session_id,
        turn_id=turn_id,
        timeline=timeline,
        telemetry=telemetry,
        final_answer_metadata=final_answer_metadata,
        subagents=subagents,
    )


def build_eval_data(session_path: Path, eval_dir: Path) -> EvalData:
    """Build EvalData from a session and persist as eval_data.json."""
    data = _build_raw(session_path)
    data.save(eval_dir / "eval_data.json")
    return data
