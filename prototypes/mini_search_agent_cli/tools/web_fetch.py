from __future__ import annotations

import re
import time
from dataclasses import dataclass
from html import unescape
from typing import Any

import httpx

from ..session import TelemetryLogger
from .base import ToolResult


BINARY_CONTENT_TYPE_PREFIXES = (
    "image/",
    "audio/",
    "video/",
    "application/octet-stream",
    "application/zip",
    "application/x-zip",
    "application/gzip",
    "application/x-gzip",
    "application/pdf",
    "application/wasm",
    "font/",
)


@dataclass
class WebFetchTool:
    client: httpx.Client | None = None
    max_chars: int = 50_000

    @property
    def name(self) -> str:
        return "web_fetch"

    def run(
        self,
        *,
        url: str,
        telemetry: TelemetryLogger | None = None,
        run_id: str = "run-001",
        actor: str = "main",
    ) -> ToolResult:
        started = time.perf_counter()
        if telemetry:
            telemetry.emit(
                "tool.web_fetch.started",
                run_id=run_id,
                actor=actor,
                metadata={"url": url},
            )

        close_client = self.client is None
        client = self.client or httpx.Client(timeout=30, follow_redirects=True)
        status_code: int | None = None
        content_type = ""
        try:
            response = client.get(url, headers={"user-agent": "mini-search-agent/0.1"})
            status_code = response.status_code
            content_type = response.headers.get("content-type", "")
            response.raise_for_status()

            content, strategy = _content_to_text(response, content_type)
            truncated = len(content) > self.max_chars
            if truncated:
                content = content[: self.max_chars].rstrip() + "\n\n[truncated]"

            metadata = {
                "url": url,
                "status_code": status_code,
                "content_type": content_type,
                "truncated": truncated,
                "extraction_strategy": strategy,
            }
            if telemetry:
                telemetry.emit(
                    "tool.web_fetch.finished",
                    run_id=run_id,
                    actor=actor,
                    latency_ms=_elapsed_ms(started),
                    metadata=metadata,
                )
            return ToolResult(content=content, metadata=metadata)
        except Exception as exc:
            metadata = {
                "url": url,
                "status_code": status_code,
                "content_type": content_type,
                "truncated": False,
                "failure_reason": str(exc),
            }
            if telemetry:
                telemetry.emit(
                    "tool.web_fetch.finished",
                    run_id=run_id,
                    actor=actor,
                    status="error",
                    latency_ms=_elapsed_ms(started),
                    metadata=metadata,
                )
            return ToolResult(content=f"web_fetch failed for URL {url!r}: {exc}", metadata=metadata, is_error=True)
        finally:
            if close_client:
                client.close()


def web_fetch_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": "Fetch a specific URL and return readable text plus metadata. Use this to verify candidate URLs before citing them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute HTTP or HTTPS URL to fetch.",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    }


def _content_to_text(response: httpx.Response, content_type: str) -> tuple[str, str]:
    if _is_binary_content_type(content_type):
        return (
            f"Binary or unsupported content was not returned as text. Content-Type: {content_type or 'unknown'}.",
            "binary-description",
        )

    text = response.text
    if _is_html_content_type(content_type) or _looks_like_html(text):
        extracted, strategy = _extract_html(text)
        return extracted, strategy

    return _normalize_text(text), "plain-text"


def _extract_html(html: str) -> tuple[str, str]:
    try:
        import trafilatura

        extracted = trafilatura.extract(html)
        if extracted and extracted.strip():
            return _normalize_text(extracted), "trafilatura"
    except Exception:
        pass

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
        return _normalize_text(text), "beautifulsoup"
    except Exception:
        return _regex_html_to_text(html), "regex-html"


def _is_binary_content_type(content_type: str) -> bool:
    lowered = content_type.lower()
    return any(lowered.startswith(prefix) for prefix in BINARY_CONTENT_TYPE_PREFIXES)


def _is_html_content_type(content_type: str) -> bool:
    lowered = content_type.lower()
    return lowered.startswith("text/html") or lowered.startswith("application/xhtml+xml")


def _looks_like_html(text: str) -> bool:
    return bool(re.search(r"<html\b|<body\b|<article\b|<p\b", text, flags=re.IGNORECASE))


def _regex_html_to_text(html: str) -> str:
    without_scripts = re.sub(r"<(script|style)\b[^>]*>[\s\S]*?</\1>", "", html, flags=re.IGNORECASE)
    without_tags = re.sub(r"<[^>]+>", "\n", without_scripts)
    return _normalize_text(unescape(without_tags))


def _normalize_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+\n", "\n", text.replace("\r\n", "\n"))).strip()


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
