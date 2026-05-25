from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .session import TelemetryLogger


@dataclass(frozen=True)
class SourceNote:
    source_id: str
    title: str
    url: str
    retrieved_at: str
    fetch_status: str
    reliability: str
    queries: tuple[str, ...]
    evidence: str
    notes: str
    path: Path


class SourceStore:
    def __init__(self, workspace: Path | str = ".", topic_slug: str = "default"):
        self.workspace = Path(workspace)
        self.topic_slug = slugify(topic_slug) or "default"
        self.sources_root = self.workspace / ".msa" / "research" / self.topic_slug / "sources"
        self.web_root = self.sources_root / "web"

    def add_source(
        self,
        *,
        title: str,
        url: str,
        fetch_status: str,
        reliability: str,
        queries: list[str],
        evidence: str,
        notes: str = "",
        retrieved_at: str | None = None,
        telemetry: TelemetryLogger | None = None,
        run_id: str = "run-001",
        actor: str = "main",
    ) -> SourceNote:
        self.web_root.mkdir(parents=True, exist_ok=True)
        self.sources_root.mkdir(parents=True, exist_ok=True)
        retrieved = retrieved_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        existing = self._find_by_url(url)

        if existing:
            merged_queries = _merge_queries(list(existing.queries), queries)
            note = SourceNote(
                source_id=existing.source_id,
                title=existing.title,
                url=existing.url,
                retrieved_at=existing.retrieved_at,
                fetch_status=fetch_status or existing.fetch_status,
                reliability=reliability or existing.reliability,
                queries=tuple(merged_queries),
                evidence=evidence or existing.evidence,
                notes=notes or existing.notes,
                path=existing.path,
            )
            self._write_note(note)
            self._write_index()
            if telemetry:
                telemetry.emit(
                    "source_note.deduplicated",
                    run_id=run_id,
                    actor=actor,
                    metadata={
                        "source_id": note.source_id,
                        "url": note.url,
                        "merged_queries": list(note.queries),
                    },
                )
                telemetry.emit(
                    "source_index.updated",
                    run_id=run_id,
                    actor=actor,
                    metadata={"topic_slug": self.topic_slug, "source_count": len(self.list_sources())},
                )
            return note

        source_id = self._next_source_id()
        path = self.web_root / f"{source_id}-{slugify(title) or 'source'}.md"
        note = SourceNote(
            source_id=source_id,
            title=title.strip(),
            url=url.strip(),
            retrieved_at=retrieved,
            fetch_status=fetch_status.strip(),
            reliability=reliability.strip(),
            queries=tuple(_merge_queries([], queries)),
            evidence=evidence.strip(),
            notes=notes.strip(),
            path=path,
        )
        self._write_note(note)
        self._write_index()
        if telemetry:
            telemetry.emit(
                "source_note.created",
                run_id=run_id,
                actor=actor,
                metadata={"source_id": note.source_id, "url": note.url, "queries": list(note.queries)},
            )
            telemetry.emit(
                "source_index.updated",
                run_id=run_id,
                actor=actor,
                metadata={"topic_slug": self.topic_slug, "source_count": len(self.list_sources())},
            )
        return note

    def list_sources(self) -> list[SourceNote]:
        if not self.web_root.exists():
            return []
        notes = [_parse_note(path) for path in sorted(self.web_root.glob("W*.md"))]
        return [note for note in notes if note is not None]

    def _find_by_url(self, url: str) -> SourceNote | None:
        normalized = url.strip()
        for note in self.list_sources():
            if note.url == normalized:
                return note
        return None

    def _next_source_id(self) -> str:
        highest = 0
        for note in self.list_sources():
            match = re.fullmatch(r"W(\d{3})", note.source_id)
            if match:
                highest = max(highest, int(match.group(1)))
        return f"W{highest + 1:03d}"

    def _write_note(self, note: SourceNote) -> None:
        note.path.write_text(render_source_note(note), encoding="utf-8")

    def _write_index(self) -> None:
        self.sources_root.mkdir(parents=True, exist_ok=True)
        self.web_root.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Sources",
            "",
            "| ID | Title | URL | Reliability | Queries |",
            "| --- | --- | --- | --- | --- |",
        ]
        for note in self.list_sources():
            relative = note.path.relative_to(self.sources_root)
            queries = "; ".join(note.queries)
            lines.append(
                f"| [{note.source_id}]({relative.as_posix()}) | {_escape_table(note.title)} | {note.url} | {note.reliability} | {_escape_table(queries)} |"
            )
        (self.sources_root / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_source_note(note: SourceNote) -> str:
    queries = "; ".join(note.queries)
    return "\n".join(
        [
            f"# {note.source_id} - {note.title}",
            "",
            f"- ID: {note.source_id}",
            f"- Title: {note.title}",
            f"- URL: {note.url}",
            f"- Retrieved at: {note.retrieved_at}",
            f"- Fetch status: {note.fetch_status}",
            f"- Reliability: {note.reliability}",
            f"- Queries: {queries}",
            f"- Evidence: {note.evidence}",
            f"- Notes: {note.notes}",
            "",
        ]
    )


def record_sources_from_subagent_result(
    markdown: str,
    *,
    store: SourceStore,
    telemetry: TelemetryLogger | None = None,
    run_id: str = "run-001",
    actor: str = "main",
) -> list[SourceNote]:
    query = _extract_section_text(markdown, "### Query").strip()
    sources: list[SourceNote] = []
    for block in _extract_fetched_source_blocks(markdown):
        title = block["title"]
        fields = block["fields"]
        url = fields.get("URL", "").strip()
        if not url:
            continue
        sources.append(
            store.add_source(
                title=title,
                url=url,
                fetch_status=fields.get("Fetch status", "partial"),
                reliability=fields.get("Reliability", "medium"),
                queries=[query] if query else [],
                evidence=fields.get("Evidence", ""),
                notes=fields.get("Notes", ""),
                telemetry=telemetry,
                run_id=run_id,
                actor=actor,
            )
        )
    return sources


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def _extract_section_text(markdown: str, heading: str) -> str:
    start = markdown.find(heading)
    if start == -1:
        return ""
    start += len(heading)
    next_heading = re.search(r"\n#{2,3} ", markdown[start:])
    end = start + next_heading.start() if next_heading else len(markdown)
    return markdown[start:end].strip()


def _extract_fetched_source_blocks(markdown: str) -> list[dict[str, Any]]:
    section = _extract_section_text(markdown, "### Fetched Sources")
    if not section:
        return []
    matches = list(re.finditer(r"^####\s+(.+)$", section, flags=re.MULTILINE))
    blocks: list[dict[str, Any]] = []
    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(section)
        body = section[start:end]
        fields: dict[str, str] = {}
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped.startswith("- ") or ": " not in stripped:
                continue
            key, value = stripped[2:].split(": ", 1)
            fields[key.strip()] = value.strip()
        blocks.append({"title": title, "fields": fields})
    return blocks


def _parse_note(path: Path) -> SourceNote | None:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- ") or ": " not in line:
            continue
        key, value = line[2:].split(": ", 1)
        fields[key] = value
    required = ["ID", "Title", "URL", "Retrieved at", "Fetch status", "Reliability", "Queries", "Evidence", "Notes"]
    if any(key not in fields for key in required):
        return None
    queries = tuple(query.strip() for query in fields["Queries"].split(";") if query.strip())
    return SourceNote(
        source_id=fields["ID"],
        title=fields["Title"],
        url=fields["URL"],
        retrieved_at=fields["Retrieved at"],
        fetch_status=fields["Fetch status"],
        reliability=fields["Reliability"],
        queries=queries,
        evidence=fields["Evidence"],
        notes=fields["Notes"],
        path=path,
    )


def _merge_queries(existing: list[str], incoming: list[str]) -> list[str]:
    merged: list[str] = []
    for query in [*existing, *incoming]:
        stripped = query.strip()
        if stripped and stripped not in merged:
            merged.append(stripped)
    return merged


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|")
