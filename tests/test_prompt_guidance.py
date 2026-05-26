from __future__ import annotations

import unittest

from mini_search_agent.prompts import PromptRegistry


class PromptGuidanceTest(unittest.TestCase):
    def test_main_prompt_guides_recorded_source_citations(self):
        prompt = PromptRegistry().load("main_agent")

        self.assertIn("web_fetch", prompt)
        self.assertIn("Recorded Source Notes", prompt)
        self.assertIn("If a subagent result has no Recorded Source Notes", prompt)
        self.assertIn("Do not copy source IDs from a Search Subagent summary", prompt)
        self.assertIn("Do not ask Search Subagents to create `W001`", prompt)
        self.assertIn("Use direct `web_search` or `web_fetch` only for narrow follow-up checks", prompt)
        self.assertIn("prefer four to six Search Subagents", prompt)
        self.assertIn("multiple independent `web_search` queries", prompt)
        self.assertIn("If coverage is thin", prompt)

    def test_search_subagent_prompt_guides_fetch_first_source_notes(self):
        prompt = PromptRegistry().load("search_subagent")

        self.assertIn("Do not invent `W001`", prompt)
        self.assertIn("search snippets are not fetched evidence", prompt)
        self.assertIn("Use `web_fetch` before adding a URL to `fetched_sources`", prompt)
        self.assertIn("Prefer primary or official sources", prompt)
        self.assertIn("Return only a valid json object", prompt)
        self.assertIn('"fetched_sources"', prompt)
        self.assertIn("Search comprehensively within your assigned angle", prompt)
        self.assertIn("Do not stop early", prompt)


if __name__ == "__main__":
    unittest.main()
