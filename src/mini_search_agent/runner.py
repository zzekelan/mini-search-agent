from __future__ import annotations

import time
from pathlib import Path
from typing import TextIO

from .agent_loop import ToolSpec, run_agent_loop
from .citations import extract_cited_source_ids
from .config import load_llm_config
from .llm import ChatClient, OpenAICompatibleChatClient
from .prompts import PromptRegistry
from .session import SessionStore, TelemetryLogger, TimelineWriter
from .sources import SourceStore, record_sources_from_subagent_result, slugify
from .subagent import SubagentTool, subagent_tool_schema
from .tools import (
    ExaWebSearchTool,
    ShellTool,
    WebFetchTool,
    shell_tool_schema,
    web_fetch_tool_schema,
    web_search_tool_schema,
)


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
    session = SessionStore(workspace).create_main_session()
    timeline = TimelineWriter(session)
    telemetry = TelemetryLogger(session)
    run_id = "run-001"

    telemetry.emit("session.started", run_id=run_id, metadata={"kind": session.kind})
    topic_slug = slugify(question)[:80] or "research"
    source_store = SourceStore(workspace, topic_slug=topic_slug)
    web_search = ExaWebSearchTool()
    web_fetch = WebFetchTool()
    shell = ShellTool(workspace=workspace)

    parent_tool_schemas = [
        web_search_tool_schema(),
        web_fetch_tool_schema(),
        shell_tool_schema(),
        subagent_tool_schema(),
    ]
    subagent_tool = SubagentTool(
        workspace=workspace,
        client=chat_client,
        parent_session=session,
        parent_timeline=timeline,
        parent_telemetry=telemetry,
        parent_tools=parent_tool_schemas,
    )

    def run_subagent(arguments):
        result = subagent_tool.run(
            description=str(arguments.get("description", "")),
            prompt=str(arguments.get("prompt", "")),
            run_id=run_id,
            record_parent_timeline=False,
        )
        notes = record_sources_from_subagent_result(
            result.content,
            store=source_store,
            telemetry=telemetry,
            run_id=run_id,
        )
        if notes:
            recorded = "\n\n### Recorded Source Notes\n" + "\n".join(
                f"- [{note.source_id}] {note.title} - {note.url}" for note in notes
            )
            return type(result)(content=result.content + recorded, metadata=result.metadata)
        return result

    tools = [
        ToolSpec(
            name="web_search",
            schema=web_search_tool_schema(),
            handler=lambda arguments: web_search.run(
                query=str(arguments.get("query", "")),
                telemetry=telemetry,
                run_id=run_id,
            ),
        ),
        ToolSpec(
            name="web_fetch",
            schema=web_fetch_tool_schema(),
            handler=lambda arguments: web_fetch.run(
                url=str(arguments.get("url", "")),
                telemetry=telemetry,
                run_id=run_id,
            ),
        ),
        ToolSpec(
            name="shell",
            schema=shell_tool_schema(),
            handler=lambda arguments: shell.run(
                command=str(arguments.get("command", "")),
                telemetry=telemetry,
                run_id=run_id,
            ),
        ),
        ToolSpec(name="subagent", schema=subagent_tool_schema(), handler=run_subagent),
    ]
    started = time.perf_counter()
    try:
        result = run_agent_loop(
            client=chat_client,
            system_prompt=prompt,
            initial_user_text=question,
            timeline=timeline,
            telemetry=telemetry,
            tools=tools,
            run_id=run_id,
            actor="main",
        )
    except Exception as exc:
        telemetry.emit(
            "run.failed",
            run_id=run_id,
            status="error",
            latency_ms=_elapsed_ms(started),
            metadata={"error": str(exc), "error_type": type(exc).__name__},
        )
        raise

    answer = result.content.strip()
    available_source_ids = [note.source_id for note in source_store.list_sources()]
    cited_source_ids = extract_cited_source_ids(answer)
    telemetry.emit(
        "final_answer.completed",
        run_id=run_id,
        metadata={
            "cited_source_ids": cited_source_ids,
            "available_source_ids": available_source_ids,
            "uncited_available_source_ids": [
                source_id for source_id in available_source_ids if source_id not in cited_source_ids
            ],
            "unknown_cited_source_ids": [
                source_id for source_id in cited_source_ids if source_id not in available_source_ids
            ],
            "has_sources_section": "## Sources" in answer,
        },
    )
    if output is not None:
        output.write(answer)
        output.write("\n")
    telemetry.emit(
        "stdout.finalized",
        run_id=run_id,
        metadata={"content_length": len(answer)},
    )
    return answer


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
