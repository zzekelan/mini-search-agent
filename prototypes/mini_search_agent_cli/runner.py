from __future__ import annotations

import time
from pathlib import Path
from typing import TextIO

from .config import load_llm_config
from .llm import ChatClient, ModelResponse, OpenAICompatibleChatClient
from .prompts import PromptRegistry
from .session import SessionStore, TelemetryLogger, TimelineWriter, text_part


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
    timeline.append(role="user", parts=[text_part(question)], produced_by_run=run_id)

    messages = [
        {"role": "developer", "content": prompt},
        {"role": "user", "content": question},
    ]
    telemetry.emit(
        "llm.request.started",
        run_id=run_id,
        metadata={"provider": config.provider, "model": config.model, "message_count": len(messages)},
    )
    started = time.perf_counter()
    try:
        response: ModelResponse = chat_client.complete(messages)
    except Exception as exc:
        telemetry.emit(
            "run.failed",
            run_id=run_id,
            status="error",
            latency_ms=_elapsed_ms(started),
            metadata={"error": str(exc), "error_type": type(exc).__name__},
        )
        raise

    telemetry.emit(
        "llm.response.finished",
        run_id=run_id,
        latency_ms=_elapsed_ms(started),
        metadata={"content_length": len(response.content), "tool_call_count": len(response.tool_calls)},
    )
    answer = response.content.strip()
    timeline.append(role="assistant", parts=[text_part(answer)], produced_by_run=run_id)
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
