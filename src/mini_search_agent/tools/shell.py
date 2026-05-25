from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from ..session import TelemetryLogger
from ..tool_schema import ToolArgs, openai_tool_schema
from .base import ToolResult


class ShellArgs(ToolArgs):
    command: str = Field(description="Shell command to run in the workspace.")


@dataclass
class ShellTool:
    workspace: Path | str = "."
    timeout_seconds: int = 30

    @property
    def name(self) -> str:
        return "shell"

    def run(
        self,
        *,
        command: str,
        telemetry: TelemetryLogger | None = None,
        run_id: str = "run-001",
        actor: str = "main",
    ) -> ToolResult:
        started = time.perf_counter()
        if telemetry:
            telemetry.emit(
                "tool.shell.started",
                run_id=run_id,
                actor=actor,
                metadata={"command": command},
            )
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=self.workspace,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            content = _format_shell_output(completed.stdout, completed.stderr, completed.returncode)
            metadata = {"command": command, "return_code": completed.returncode}
            if telemetry:
                telemetry.emit(
                    "tool.shell.finished",
                    run_id=run_id,
                    actor=actor,
                    status="ok" if completed.returncode == 0 else "error",
                    latency_ms=_elapsed_ms(started),
                    metadata=metadata,
                )
            return ToolResult(content=content, metadata=metadata, is_error=completed.returncode != 0)
        except Exception as exc:
            metadata = {"command": command, "failure_reason": str(exc)}
            if telemetry:
                telemetry.emit(
                    "tool.shell.finished",
                    run_id=run_id,
                    actor=actor,
                    status="error",
                    latency_ms=_elapsed_ms(started),
                    metadata=metadata,
                )
            return ToolResult(content=f"shell failed: {exc}", metadata=metadata, is_error=True)


def shell_tool_schema() -> dict[str, Any]:
    return openai_tool_schema(
        "shell",
        "Run a shell command in the workspace. Only available to the Main Agent.",
        ShellArgs,
    )


def _format_shell_output(stdout: str, stderr: str, return_code: int) -> str:
    sections = [f"return_code: {return_code}"]
    if stdout:
        sections.append(f"stdout:\n{stdout.rstrip()}")
    if stderr:
        sections.append(f"stderr:\n{stderr.rstrip()}")
    return "\n\n".join(sections)


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
