from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class ConfigError(Exception):
    """Raised when the CLI cannot build a valid runtime configuration."""


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    api_key: str
    model: str
    base_url: str


def load_llm_config(workspace: Path | str = ".") -> LLMConfig:
    workspace_path = Path(workspace)
    values = _read_dotenv(workspace_path / ".env")
    merged = {**values, **{key: value for key, value in os.environ.items() if value is not None}}

    provider = merged.get("LLM_PROVIDER", "").strip()
    api_key = merged.get("LLM_API_KEY", "").strip()
    model = merged.get("LLM_MODEL", "").strip()
    base_url = merged.get("LLM_BASE_URL", "").strip()

    missing = [
        name
        for name, value in {
            "LLM_PROVIDER": provider,
            "LLM_API_KEY": api_key,
            "LLM_MODEL": model,
            "LLM_BASE_URL": base_url,
        }.items()
        if not value
    ]
    if missing:
        raise ConfigError(f"Missing required environment variable(s): {', '.join(missing)}")

    if provider != "openai-compatible":
        raise ConfigError("Only LLM_PROVIDER=openai-compatible is supported")

    return LLMConfig(provider=provider, api_key=api_key, model=model, base_url=base_url)


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    try:
        from dotenv import dotenv_values
    except ImportError:
        return _read_dotenv_without_dependency(path)

    return {key: str(value) for key, value in dotenv_values(path).items() if value is not None}


def _read_dotenv_without_dependency(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
