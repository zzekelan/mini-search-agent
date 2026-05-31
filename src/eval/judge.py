"""LLM Judge execution: run agent-based evaluations.

Each judge creates an isolated Session under the eval directory
so its Timeline + Telemetry are auditable.
"""

from __future__ import annotations

import json
from pathlib import Path

from mini_search_agent.agent_loop import ToolSpec, run_agent_loop
from mini_search_agent.config import load_llm_config
from mini_search_agent.llm import OpenAICompatibleChatClient
from mini_search_agent.session import SessionStore, TelemetryLogger, TimelineWriter
from mini_search_agent.tools import (
    ExaWebSearchTool,
    ShellTool,
    WebFetchTool,
    shell_tool_schema,
    web_fetch_tool_schema,
    web_search_tool_schema,
)

from . import LLMJudgeResult
from .checks import LLMJudgeSpec


def _judge_tools(workspace: Path) -> list[ToolSpec]:
    shell = ShellTool(workspace=workspace)
    return [
        ToolSpec(
            name="shell", schema=shell_tool_schema(),
            handler=lambda args: shell.run(**args), args_model=None,
        ),
        ToolSpec(
            name="web_search", schema=web_search_tool_schema(),
            handler=lambda args: ExaWebSearchTool().run(**args), args_model=None,
        ),
        ToolSpec(
            name="web_fetch", schema=web_fetch_tool_schema(),
            handler=lambda args: WebFetchTool().run(**args), args_model=None,
        ),
    ]


def _find_topic_slug(session_path: Path) -> str:
    """Find topic_slug from a session's telemetry or sibling sessions."""
    from mini_search_agent.session import read_jsonl

    def _scan(sub_path: Path) -> str:
        tel = sub_path / "telemetry.jsonl"
        if not tel.exists():
            return ""
        for event in read_jsonl(tel):
            if event.get("event") == "source_index.updated":
                slug = event.get("metadata", {}).get("topic_slug", "")
                if slug:
                    return slug
        return ""

    # Own telemetry first
    slug = _scan(session_path)
    if slug:
        return slug

    # Sibling sessions
    parent = session_path.parent
    if parent.exists():
        for sibling in sorted(parent.iterdir()):
            if not sibling.is_dir() or sibling.name == session_path.name:
                continue
            slug = _scan(sibling)
            if slug:
                return slug

    return "unknown"


def run_llm_judge(
    session_name: str,
    topic_slug: str,
    eval_dir: Path,
    workspace: Path,
    spec: LLMJudgeSpec,
) -> LLMJudgeResult:
    """Run one LLM judge check.

    Creates a judge Session under eval_dir/llm_judge_session/<check_id>/.
    """
    prompt = spec.system_prompt_template
    prompt = prompt.replace("{session_name}", session_name)
    prompt = prompt.replace("{topic_slug}", topic_slug)

    judge_root = eval_dir / "llm_judge_session" / spec.check_id
    store = SessionStore(workspace)
    session = store.create_main_session(parent_dir=judge_root)
    timeline = TimelineWriter(session)
    telemetry = TelemetryLogger(session)

    config = load_llm_config(workspace)
    client = OpenAICompatibleChatClient(config)

    result = run_agent_loop(
        client=client,
        system_prompt=prompt,
        initial_user_text=(
            f"Evaluate session {session_name} for {spec.check_id}. "
            f"topic={topic_slug}. "
            f"Start by reading eval_data.json, then main.jsonl, then check each source."
        ),
        timeline=timeline,
        telemetry=telemetry,
        tools=_judge_tools(workspace),
        run_id="judge-001",
        actor="judge",
        max_turns=spec.max_turns,
        response_format={"type": "json_object"},
    )

    # Parse JSON from output (may be in markdown code block)
    content = result.content
    data: dict = {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        if "```json" in content:
            start = content.index("```json") + 7
            end = content.index("```", start)
            try:
                data = json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

    return LLMJudgeResult(
        check_id=spec.check_id,
        score=data.get("score", 0),
        label=data.get("label", "ERROR"),
        explanation=data.get("explanation", content[:500]),
    )


def run_llm_judges(
    session_path: Path,
    eval_dir: Path,
    workspace: Path,
    specs: list[LLMJudgeSpec],
) -> list[LLMJudgeResult]:
    """Run all specified LLM judge checks on a session."""
    session_name = session_path.name
    topic_slug = _find_topic_slug(session_path)

    results: list[LLMJudgeResult] = []
    for spec in specs:
        r = run_llm_judge(session_name, topic_slug, eval_dir, workspace, spec)
        results.append(r)

    return results
