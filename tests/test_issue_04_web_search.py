from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone

import httpx

from prototypes.mini_search_agent_cli.session import SessionStore, TelemetryLogger
from prototypes.mini_search_agent_cli.tools.web_search import (
    ExaWebSearchTool,
    build_exa_request,
    parse_exa_response,
    web_search_tool_schema,
)


class WebSearchToolTest(unittest.TestCase):
    def test_tool_schema_exposes_provider_neutral_web_search_name(self):
        schema = web_search_tool_schema()

        self.assertEqual(schema["function"]["name"], "web_search")
        self.assertIn("query", schema["function"]["parameters"]["required"])

    def test_build_exa_request_uses_agreed_mcp_shape(self):
        request = build_exa_request("agentic search")

        self.assertEqual(request["method"], "tools/call")
        self.assertEqual(request["params"]["name"], "web_search_exa")
        self.assertEqual(
            request["params"]["arguments"],
            {
                "query": "agentic search",
                "type": "auto",
                "numResults": 8,
                "livecrawl": "fallback",
            },
        )

    def test_parse_exa_plain_json_text_response(self):
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"content": [{"type": "text", "text": "Plain JSON response from Exa MCP."}]},
        }

        self.assertEqual(parse_exa_response(json.dumps(payload)), "Plain JSON response from Exa MCP.")

    def test_parse_exa_sse_text_response(self):
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": "Top web results for \"Alan Turing\"\n1. Alan Turing\nURL: https://en.wikipedia.org/wiki/Alan_Turing",
                    }
                ]
            },
        }
        response_text = f"event: message\ndata: {json.dumps(payload)}\n\n"

        self.assertIn("URL: https://en.wikipedia.org/wiki/Alan_Turing", parse_exa_response(response_text))

    def test_web_search_returns_exa_text_and_records_telemetry(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content.decode("utf-8"))
            self.assertEqual(body["params"]["arguments"]["query"], "hybrid retrieval")
            return httpx.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "URL: https://example.com/paper"}]},
                },
            )

        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            telemetry = TelemetryLogger(session, clock=lambda: now)
            client = httpx.Client(transport=httpx.MockTransport(handler))

            result = ExaWebSearchTool(client=client).run(
                query="hybrid retrieval",
                telemetry=telemetry,
                run_id="run-001",
            )

            events = telemetry.read_events()

        self.assertFalse(result.is_error)
        self.assertEqual(result.content, "URL: https://example.com/paper")
        self.assertEqual(result.metadata, {"query": "hybrid retrieval"})
        self.assertEqual([event["event"] for event in events], ["tool.web_search.started", "tool.web_search.finished"])
        self.assertEqual(events[-1]["metadata"]["provider"], "Exa MCP")
        self.assertEqual(events[-1]["metadata"]["returned_text_length"], len(result.content))

    def test_web_search_failure_is_reported_without_fake_results(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, text="provider down")

        now = datetime(2026, 5, 26, tzinfo=timezone.utc)
        with tempfile.TemporaryDirectory() as temp_dir:
            session = SessionStore(temp_dir, clock=lambda: now).create_main_session()
            telemetry = TelemetryLogger(session, clock=lambda: now)
            client = httpx.Client(transport=httpx.MockTransport(handler))

            result = ExaWebSearchTool(client=client).run(
                query="current research",
                telemetry=telemetry,
                run_id="run-001",
            )
            events = telemetry.read_events()

        self.assertTrue(result.is_error)
        self.assertIn("web_search failed", result.content)
        self.assertEqual(result.metadata, {"query": "current research"})
        self.assertEqual(events[-1]["status"], "error")
        self.assertIn("provider", events[-1]["metadata"])


if __name__ == "__main__":
    unittest.main()
