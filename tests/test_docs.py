from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationTest(unittest.TestCase):
    def test_readme_covers_required_demo_topics_without_examples_or_eval_report(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for expected in ["## Install", "## Run The Demo", "## Environment", "## System Design", "## Telemetry"]:
            self.assertIn(expected, readme)
        for env_name in ["LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"]:
            self.assertIn(env_name, readme)
        self.assertIn("https://mcp.exa.ai/mcp", readme)
        self.assertNotIn("2025-2026 年搜索智能体", readme)
        self.assertNotIn("eval report", readme.lower())

    def test_examples_contain_only_the_three_agreed_questions(self):
        examples = (ROOT / "examples" / "questions.md").read_text(encoding="utf-8")
        question_lines = [line for line in examples.splitlines() if line[:3] in {"1. ", "2. ", "3. "}]

        self.assertEqual(
            question_lines,
            [
                "1. 2025-2026 年搜索智能体或 Deep Research 产品/论文有哪些代表性进展？",
                "2. RAG 中 hybrid retrieval + reranking 相比单纯 dense retrieval 的收益和局限是什么？",
                "3. 比较 Exa、Tavily、SerpAPI、DuckDuckGo 在 agentic search 场景中的适用性。",
            ],
        )
        self.assertEqual(len(question_lines), 3)


if __name__ == "__main__":
    unittest.main()
