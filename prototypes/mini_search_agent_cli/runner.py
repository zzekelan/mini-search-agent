from __future__ import annotations

from pathlib import Path
from typing import TextIO

from .config import load_llm_config
from .llm import ChatClient, ModelResponse, OpenAICompatibleChatClient
from .prompts import PromptRegistry


def run_research(
    question: str,
    *,
    workspace: Path | str = ".",
    client: ChatClient | None = None,
    output: TextIO | None = None,
) -> str:
    config = load_llm_config(workspace)
    prompt = PromptRegistry().load("main_agent")
    chat_client = client or OpenAICompatibleChatClient(config)
    response: ModelResponse = chat_client.complete(
        [
            {"role": "developer", "content": prompt},
            {"role": "user", "content": question},
        ]
    )
    answer = response.content.strip()
    if output is not None:
        output.write(answer)
        output.write("\n")
    return answer
