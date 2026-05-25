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

    def test_search_subagent_prompt_guides_fetch_first_source_notes(self):
        prompt = PromptRegistry().load("search_subagent")

        self.assertIn("Do not invent `W001`", prompt)
        self.assertIn("search snippets are not fetched evidence", prompt)
        self.assertIn("Use `web_fetch` before listing a URL under `### Fetched Sources`", prompt)
        self.assertIn("Prefer primary or official sources", prompt)


if __name__ == "__main__":
    unittest.main()
