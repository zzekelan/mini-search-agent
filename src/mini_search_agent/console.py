from __future__ import annotations

from typing import TextIO


class RunConsoleView:
    def __init__(self, output: TextIO):
        self._output = output
        self._wrote_text = False
        self._ends_with_newline = True

    def llm_content_delta(self, text: str) -> None:
        if not text:
            return
        self._output.write(text)
        self._output.flush()
        self._wrote_text = True
        self._ends_with_newline = text.endswith("\n")

    def run_finished(self) -> None:
        if self._wrote_text and not self._ends_with_newline:
            self._output.write("\n")
            self._output.flush()

    def tool_call_pending(self, *, tool_name: str) -> None:
        self._write_status_line(f"[tool] {tool_name} pending")

    def tool_call_started(self, *, tool_name: str) -> None:
        self._write_status_line(f"[tool] {tool_name} running")

    def tool_call_finished(self, *, tool_name: str, is_error: bool) -> None:
        status = "error" if is_error else "done"
        self._write_status_line(f"[tool] {tool_name} {status}")

    def _write_status_line(self, line: str) -> None:
        if self._wrote_text and not self._ends_with_newline:
            self._output.write("\n")
        self._output.write(line)
        self._output.write("\n")
        self._output.flush()
        self._wrote_text = True
        self._ends_with_newline = True
