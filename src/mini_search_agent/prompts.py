from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class PromptNotFoundError(FileNotFoundError):
    """Raised when an agent system prompt is missing from the prompt registry."""


@dataclass(frozen=True)
class PromptRegistry:
    root: Path | None = None

    def load(self, name: str) -> str:
        prompt_path = self._root() / f"{name}.md"
        if not prompt_path.exists():
            raise PromptNotFoundError(f"Prompt not found: {name}")
        return prompt_path.read_text(encoding="utf-8").strip()

    def _root(self) -> Path:
        if self.root is not None:
            return self.root
        return Path(__file__).with_name("prompts")
