from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone

import httpx

from mini_search_agent.session import SessionStore, TelemetryLogger
from mini_search_agent.tools.web_fetch import WebFetchTool, web_fetch_tool_schema


class WebFetchToolTest(unittest.TestCase):
    def test_tool_schema_exposes_web_fetch_name(self):
        schema = web_fetch_tool_schema()

        self.assertEqual(schema["function"]["name"], "web_fetch")
        self.assertEqual(schema["function"]["parameters"]["required"], ["url"])

    def test_fetch_html_extracts_readable_text_and_records_metadata(self):
        html = """
        <html><head><script>hidden()</script></head>
        <body><h1>Hybrid Retrieval</h1><p>Dense plus sparse retrieval.</p></body></html>
        """

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"}, text=html)

        telemetry, result = self._run_fetch(handler, "https://example.com/article")

        self.assertFalse(result.is_error)
        self.assertIn("Hybrid Retrieval", result.content)
        self.assertIn("Dense plus sparse retrieval", result.content)
        self.assertNotIn("hidden()", result.content)
        self.assertEqual(result.metadata["status_code"], 200)
        self.assertEqual(result.metadata["content_type"], "text/html; charset=utf-8")
        self.assertFalse(result.metadata["truncated"])
        self.assertIn(result.metadata["extraction_strategy"], {"trafilatura", "beautifulsoup", "regex-html"})
        self.assertEqual(telemetry[-1]["metadata"]["url"], "https://example.com/article")

    def test_fetch_plain_text_passes_text_through(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="plain evidence")

        _, result = self._run_fetch(handler, "https://example.com/plain.txt")

        self.assertEqual(result.content, "plain evidence")
        self.assertEqual(result.metadata["extraction_strategy"], "plain-text")

    def test_fetch_binary_content_returns_explanation(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"%PDF")

        _, result = self._run_fetch(handler, "https://example.com/file.pdf")

        self.assertIn("Binary or unsupported content", result.content)
        self.assertEqual(result.metadata["extraction_strategy"], "binary-description")

    def test_fetch_truncates_over_limit_content(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, headers={"content-type": "text/plain"}, text="abcdef")

        client = httpx.Client(transport=httpx.MockTransport(handler))
        tool = WebFetchTool(client=client, max_chars=3)

        result = tool.run(url="https://example.com/long")

        self.assertEqual(result.content, "abc\n\n[truncated]")
        self.assertTrue(result.metadata["truncated"])

    def test_fetch_failure_is_tool_error_with_telemetry(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="missing")

        telemetry, result = self._run_fetch(handler, "https://example.com/missing")

        self.assertTrue(result.is_error)
        self.assertIn("web_fetch failed", result.content)
        self.assertEqual(telemetry[-1]["status"], "error")
        self.assertEqual(telemetry[-1]["metadata"]["status_code"], 404)
        self.assertIn("failure_reason", telemetry[-1]["metadata"])

    def _run_fetch(self, handler, url):
        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            logger = TelemetryLogger(session, clock=lambda: now)
            client = httpx.Client(transport=httpx.MockTransport(handler))

            result = WebFetchTool(client=client).run(url=url, telemetry=logger, run_id="run-001")
            events = logger.read_events()

        return events, result


if __name__ == "__main__":
    unittest.main()
