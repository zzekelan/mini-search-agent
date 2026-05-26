from __future__ import annotations

import threading
from typing import TextIO


class RunConsoleView:
    def __init__(self, output: TextIO, *, spinner_interval_seconds: float = 0.12):
        self._output = output
        self._is_tty = bool(getattr(output, "isatty", lambda: False)())
        self._spinner_interval_seconds = spinner_interval_seconds
        self._wrote_text = False
        self._ends_with_newline = True
        self._lock = threading.Lock()
        self._spinner_lock = threading.RLock()
        self._spinner_stop: threading.Event | None = None
        self._spinner_thread: threading.Thread | None = None

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

    def tool_call_started(self, *, label: str) -> None:
        if self._is_tty:
            self._start_spinner(f"{label} running")
            return
        self._write_status_line(f"[tool] {label} running")

    def tool_call_finished(self, *, label: str, is_error: bool) -> None:
        status = "error" if is_error else "done"
        if self._is_tty:
            marker = "[error]" if is_error else "[done]"
            self._stop_spinner(clear=True)
            self._write_status_line(f"{marker} {label} {status}")
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
