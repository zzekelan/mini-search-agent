from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Clock = Callable[[], datetime]

_SESSION_CREATION_LOCK = threading.Lock()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Session:
    session_id: str
    kind: str
    path: Path
    timeline_path: Path
    telemetry_path: Path


class SessionStore:
    def __init__(self, workspace: Path | str = ".", clock: Clock = utc_now):
        self.workspace = Path(workspace)
        self.clock = clock

    def create_main_session(self) -> Session:
        now = self.clock()
        with _SESSION_CREATION_LOCK:
            sessions_root = self.workspace / ".msa" / "sessions"
            sessions_root.mkdir(parents=True, exist_ok=True)
            date_prefix = now.date().isoformat()
            next_number = _next_session_number(sessions_root, date_prefix)
            session_id = f"session-{date_prefix}-{next_number:03d}"
            session_path = sessions_root / session_id
            session_path.mkdir(parents=True, exist_ok=False)

            metadata = {
                "session_id": session_id,
                "kind": "main",
                "created_at": _format_time(now),
            }
            (session_path / "session.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            timeline_path = session_path / "main.jsonl"
            telemetry_path = session_path / "telemetry.jsonl"
            timeline_path.touch()
            telemetry_path.touch()
            return Session(
                session_id=session_id,
                kind="main",
                path=session_path,
                timeline_path=timeline_path,
                telemetry_path=telemetry_path,
            )

    def create_sub_session(self, parent: Session) -> Session:
        now = self.clock()
        with _SESSION_CREATION_LOCK:
            sub_root = parent.path / "sub"
            sub_root.mkdir(parents=True, exist_ok=True)
            next_number = _next_sub_session_number(sub_root)
            session_id = f"sub-{next_number:03d}"
            session_path = sub_root / session_id
            session_path.mkdir(parents=True, exist_ok=False)
            metadata = {
                "session_id": session_id,
                "kind": "sub",
                "parent_session_id": parent.session_id,
                "created_at": _format_time(now),
            }
            (session_path / "session.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            timeline_path = session_path / "timeline.jsonl"
            telemetry_path = session_path / "telemetry.jsonl"
            timeline_path.touch()
            telemetry_path.touch()
            return Session(
                session_id=session_id,
                kind="sub",
                path=session_path,
                timeline_path=timeline_path,
                telemetry_path=telemetry_path,
            )


class TimelineWriter:
    def __init__(self, session: Session, clock: Clock = utc_now):
        self.session = session
        self.clock = clock

    def append(self, *, role: str, parts: list[dict[str, Any]], produced_by_run: str) -> dict[str, Any]:
        if role not in {"user", "assistant"}:
            raise ValueError("Timeline role must be user or assistant")
        sequence = _jsonl_count(self.session.timeline_path) + 1
        entry = {
            "entry_id": f"entry-{sequence:03d}",
            "produced_by_run": produced_by_run,
            "role": role,
            "sequence": sequence,
            "created_at": _format_time(self.clock()),
            "parts": parts,
        }
        _append_jsonl(self.session.timeline_path, entry)
        return entry

    def read_entries(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.session.timeline_path)


class TelemetryLogger:
    def __init__(self, session: Session, clock: Clock = utc_now):
        self.session = session
        self.clock = clock
        self._lock = threading.Lock()

    def emit(
        self,
        event: str,
        *,
        run_id: str,
        actor: str = "main",
        status: str = "ok",
        latency_ms: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "event": event,
            "timestamp": _format_time(self.clock()),
            "session_id": self.session.session_id,
            "run_id": run_id,
            "actor": actor,
            "status": status,
            "latency_ms": latency_ms,
            "metadata": _sanitize_metadata(metadata or {}),
        }
        with self._lock:
            _append_jsonl(self.session.telemetry_path, record)
        return record

    def read_events(self) -> list[dict[str, Any]]:
        with self._lock:
            return _read_jsonl(self.session.telemetry_path)


def text_part(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def tool_call_part(call_id: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "call_id": call_id,
        "tool_name": tool_name,
        "arguments": arguments,
    }


def tool_result_part(
    call_id: str,
    tool_name: str,
    content: str,
    *,
    is_error: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    part = {
        "type": "tool_result",
        "call_id": call_id,
        "tool_name": tool_name,
        "content": content,
        "is_error": is_error,
    }
    if metadata:
        part["metadata"] = metadata
    return part


def error_part(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def _next_session_number(sessions_root: Path, date_prefix: str) -> int:
    prefix = f"session-{date_prefix}-"
    numbers = []
    for path in sessions_root.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        suffix = path.name.removeprefix(prefix)
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def _next_sub_session_number(sub_root: Path) -> int:
    numbers = []
    for path in sub_root.iterdir():
        if not path.is_dir() or not path.name.startswith("sub-"):
            continue
        suffix = path.name.removeprefix("sub-")
        if suffix.isdigit():
            numbers.append(int(suffix))
    return max(numbers, default=0) + 1


def _format_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in metadata.items():
        lowered = key.lower()
        if "key" in lowered or "secret" in lowered or "token" in lowered:
            redacted[key] = "[redacted]"
        else:
            redacted[key] = value
    return redacted
