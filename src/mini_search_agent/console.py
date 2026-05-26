from __future__ import annotations

import shutil
import threading
from dataclasses import dataclass
from typing import TextIO


@dataclass
class _ToolStatus:
    label: str
    status: str


class RunConsoleView:
    def __init__(
        self,
        output: TextIO,
        *,
        spinner_interval_seconds: float = 0.12,
        terminal_width: int | None = None,
    ):
        self._output = output
        self._is_tty = bool(getattr(output, "isatty", lambda: False)())
        self._spinner_interval_seconds = spinner_interval_seconds
        self._terminal_width = terminal_width
        self._wrote_text = False
        self._ends_with_newline = True
        self._lock = threading.Lock()
        self._spinner_lock = threading.RLock()
        self._spinner_stop: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None
        self._tool_status_lock = threading.RLock()
        self._tool_statuses: dict[str, _ToolStatus] = {}
        self._tool_rendered_lines = 0
        self._tool_frame_index = 0
        self._tool_renderer_stop: threading.Event | None = None
        self._tool_renderer_thread: threading.Thread | None = None

    def llm_content_delta(self, text: str) -> None:
        if not text:
            return
        self._stop_spinner(clear=True)
        with self._lock:
            self._output.write(text)
            self._output.flush()
            self._wrote_text = True
            self._ends_with_newline = text.endswith("\n")

    def run_finished(self) -> None:
        self._stop_spinner(clear=False)
        self._stop_tool_renderer()
        with self._lock:
            if self._wrote_text and not self._ends_with_newline:
                self._output.write("\n")
                self._output.flush()
                self._ends_with_newline = True

    def model_response_started(self) -> None:
        if self._is_tty:
            self._start_spinner("Main Agent planning")

    def model_response_finished(self) -> None:
        self._stop_spinner(clear=True)

    def tool_call_pending(self, *, label: str) -> None:
        if self._is_tty:
            self._write_status_line(f"[wait] {label} pending")
            return
        self._write_status_line(f"[tool] {label} pending")

    def tool_call_started(self, *, label: str, call_id: str | None = None) -> None:
        if self._is_tty:
            self._start_tool_status(label=label, call_id=call_id)
            return
        self._write_status_line(f"[tool] {label} running")

    def tool_call_finished(self, *, label: str, is_error: bool, call_id: str | None = None) -> None:
        status = "error" if is_error else "done"
        if self._is_tty:
            self._finish_tool_status(label=label, call_id=call_id, status=status)
            return
        self._write_status_line(f"[tool] {label} {status}")

    def _start_spinner(self, label: str) -> None:
        with self._spinner_lock:
            self._stop_spinner(clear=True)
            stop = threading.Event()
            self._spinner_stop = stop
            self._spinner_thread = threading.Thread(
                target=self._spin,
                args=(label, stop),
                daemon=True,
            )
            self._spinner_thread.start()

    def _start_tool_status(self, *, label: str, call_id: str | None) -> None:
        self._stop_spinner(clear=True)
        with self._tool_status_lock:
            key = self._tool_status_key(label=label, call_id=call_id)
            self._tool_statuses[key] = _ToolStatus(label=label, status="running")
            self._ensure_tool_renderer_locked()
            self._redraw_tool_statuses_locked()

    def _finish_tool_status(self, *, label: str, call_id: str | None, status: str) -> None:
        key = self._tool_status_key(label=label, call_id=call_id)
        thread_to_join: threading.Thread | None = None
        with self._tool_status_lock:
            self._tool_statuses[key] = _ToolStatus(label=label, status=status)
            self._redraw_tool_statuses_locked()
            if self._tool_statuses and all(
                tool_status.status in {"done", "error"}
                for tool_status in self._tool_statuses.values()
            ):
                stop = self._tool_renderer_stop
                thread_to_join = self._tool_renderer_thread
                self._tool_renderer_stop = None
                self._tool_renderer_thread = None
                if stop is not None:
                    stop.set()
                self._tool_statuses.clear()
                self._tool_rendered_lines = 0

        if thread_to_join is not None and thread_to_join is not threading.current_thread():
            thread_to_join.join()

    def _tool_status_key(self, *, label: str, call_id: str | None) -> str:
        return call_id or label

    def _ensure_tool_renderer_locked(self) -> None:
        if self._tool_renderer_thread is not None and self._tool_renderer_thread.is_alive():
            return
        stop = threading.Event()
        self._tool_renderer_stop = stop
        self._tool_renderer_thread = threading.Thread(
            target=self._render_tool_statuses,
            args=(stop,),
            daemon=True,
        )
        self._tool_renderer_thread.start()

    def _render_tool_statuses(self, stop: threading.Event) -> None:
        while not stop.wait(self._spinner_interval_seconds):
            with self._tool_status_lock:
                if not self._tool_statuses:
                    return
                self._tool_frame_index += 1
                self._redraw_tool_statuses_locked()

    def _redraw_tool_statuses_locked(self) -> None:
        lines = self._tool_status_lines_locked()
        if not lines:
            return

        with self._lock:
            if self._tool_rendered_lines > 0:
                self._output.write(f"\033[{self._tool_rendered_lines}A")
            elif self._wrote_text and not self._ends_with_newline:
                self._output.write("\n")

            for line in lines:
                self._output.write("\r\033[2K")
                self._output.write(line)
                self._output.write("\n")

            self._output.flush()
            self._wrote_text = True
            self._ends_with_newline = True
            self._tool_rendered_lines = len(lines)

    def _tool_status_lines_locked(self) -> list[str]:
        frames = ["|", "/", "-", "\\"]
        lines: list[str] = []
        terminal_width = self._terminal_columns()
        for index, tool_status in enumerate(self._tool_statuses.values()):
            if tool_status.status == "running":
                marker = f"[{frames[(self._tool_frame_index + index) % len(frames)]}]"
            elif tool_status.status == "error":
                marker = "[error]"
            else:
                marker = "[✓]"
            lines.append(
                _format_status_line(
                    marker=marker,
                    label=tool_status.label,
                    status=tool_status.status,
                    max_width=terminal_width,
                )
            )
        return lines

    def _terminal_columns(self) -> int:
        output_columns = getattr(self._output, "columns", None)
        if isinstance(output_columns, int) and output_columns > 0:
            return output_columns
        if self._terminal_width is not None:
            return max(20, self._terminal_width)
        return max(20, shutil.get_terminal_size(fallback=(80, 24)).columns)

    def _stop_tool_renderer(self) -> None:
        thread_to_join: threading.Thread | None = None
        with self._tool_status_lock:
            stop = self._tool_renderer_stop
            thread_to_join = self._tool_renderer_thread
            self._tool_renderer_stop = None
            self._tool_renderer_thread = None
            if stop is not None:
                stop.set()
            self._tool_statuses.clear()
            self._tool_rendered_lines = 0

        if thread_to_join is not None and thread_to_join is not threading.current_thread():
            thread_to_join.join()

    def _spin(self, label: str, stop: threading.Event) -> None:
        frames = ["|", "/", "-", "\\"]
        index = 0
        first_frame = True
        while not stop.is_set():
            with self._lock:
                if first_frame and self._wrote_text and not self._ends_with_newline:
                    self._output.write("\n")
                self._output.write("\r\033[2K")
                self._output.write(f"[{frames[index % len(frames)]}] {label}")
                self._output.flush()
                self._wrote_text = True
                self._ends_with_newline = False
            index += 1
            first_frame = False
            stop.wait(self._spinner_interval_seconds)

    def _stop_spinner(self, *, clear: bool) -> None:
        with self._spinner_lock:
            stop = self._spinner_stop
            thread = self._spinner_thread
            self._spinner_stop = None
            self._spinner_thread = None
            stopped_spinner = stop is not None or thread is not None
            if stop is not None:
                stop.set()
            if thread is not None:
                thread.join()
            if clear and self._is_tty and stopped_spinner:
                with self._lock:
                    self._output.write("\r\033[2K")
                    self._output.flush()
                    self._ends_with_newline = True

    def _write_status_line(self, line: str) -> None:
        with self._lock:
            if self._wrote_text and not self._ends_with_newline:
                self._output.write("\n")
            self._output.write(line)
            self._output.write("\n")
            self._output.flush()
            self._wrote_text = True
            self._ends_with_newline = True


def _format_status_line(*, marker: str, label: str, status: str, max_width: int) -> str:
    prefix = f"{marker} "
    available_label_width = max_width - len(prefix)
    if available_label_width <= 0:
        return prefix[:max_width]
    return f"{prefix}{_truncate_label(label, available_label_width)}"


def _truncate_label(label: str, max_width: int) -> str:
    if len(label) <= max_width:
        return label
    if max_width <= 3:
        return label[:max_width]
    return label[: max_width - 3].rstrip() + "..."
