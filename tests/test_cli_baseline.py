from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mini_search_agent.config import ConfigError, load_llm_config
from mini_search_agent.llm import ModelResponse
from mini_search_agent.prompts import PromptRegistry
from mini_search_agent.runner import run_research


class RecordingClient:
    def __init__(self):
        self.messages = None

    def complete(self, messages, tools=None):
        self.messages = messages
        return ModelResponse(content="baseline answer")


class CliBaselineTest(unittest.TestCase):
    def test_prompt_registry_loads_main_and_search_subagent_prompts(self):
        registry = PromptRegistry()

        self.assertIn("Main Agent", registry.load("main_agent"))
        self.assertIn("Search Subagent", registry.load("search_subagent"))

    def test_shell_environment_overrides_dotenv_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai-compatible",
                        "LLM_API_KEY=from-dotenv",
                        "LLM_MODEL=dotenv-model",
                        "LLM_BASE_URL=https://dotenv.example/v1",
                    ]
                ),
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "LLM_API_KEY": "from-shell",
                    "LLM_MODEL": "shell-model",
                    "LLM_BASE_URL": "https://shell.example/v1",
                },
                clear=False,
            ):
                config = load_llm_config(workspace)

        self.assertEqual(config.api_key, "from-shell")
        self.assertEqual(config.model, "shell-model")
        self.assertEqual(config.base_url, "https://shell.example/v1")

    def test_run_research_sends_prompt_and_question_to_client_and_prints_answer(self):
        client = RecordingClient()
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / ".env").write_text(
                "\n".join(
                    [
                        "LLM_PROVIDER=openai-compatible",
                        "LLM_API_KEY=test-key",
                        "LLM_MODEL=test-model",
                        "LLM_BASE_URL=https://llm.example/v1",
                    ]
                ),
                encoding="utf-8",
            )

            answer = run_research("What changed?", workspace=workspace, client=client, output=output)

        self.assertEqual(answer, "baseline answer")
        self.assertEqual(output.getvalue(), "baseline answer\n")
        self.assertEqual(client.messages[0]["role"], "system")
        self.assertIn("Main Agent", client.messages[0]["content"])
        self.assertEqual(client.messages[1], {"role": "user", "content": "What changed?"})

    def test_missing_llm_configuration_is_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ConfigError) as raised:
                    load_llm_config(Path(temp_dir))

        self.assertIn("LLM_PROVIDER", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
