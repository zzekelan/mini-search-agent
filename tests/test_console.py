from __future__ import annotations

import io
import time
import unittest

from mini_search_agent.console import RunConsoleView


class TtyOutput(io.StringIO):
    def isatty(self):
        return True


class RunConsoleViewTest(unittest.TestCase):
    def test_tty_renders_parallel_tool_status_rows_and_updates_finished_row(self):
        output = TtyOutput()
        console = RunConsoleView(output, spinner_interval_seconds=0.01)

        console.tool_call_started(label="subagent: OpenAI", call_id="call-openai")
        console.tool_call_started(label="subagent: Google", call_id="call-google")
        time.sleep(0.03)
        console.tool_call_finished(label="subagent: OpenAI", call_id="call-openai", is_error=False)
        console.tool_call_finished(label="subagent: Google", call_id="call-google", is_error=True)
        console.run_finished()

        text = output.getvalue()
        self.assertIn("subagent: OpenAI running", text)
        self.assertIn("subagent: Google running", text)
        self.assertIn("[done] subagent: OpenAI done", text)
        self.assertIn("[error] subagent: Google error", text)
        self.assertIn("\033[2A", text)
        self.assertNotIn("[tool] subagent: OpenAI running", text)

    def test_tty_truncates_long_tool_status_rows_to_terminal_width(self):
        output = TtyOutput()
        console = RunConsoleView(output, spinner_interval_seconds=0.01, terminal_width=48)
        long_label = "web_search: Anthropic demystifying evals for AI agents evaluation methodology"

        console.tool_call_started(label=long_label, call_id="call-long")
        time.sleep(0.02)
        console.tool_call_finished(label=long_label, call_id="call-long", is_error=False)
        console.run_finished()

        drawn_lines = [
            segment.split("\n", 1)[0]
            for segment in output.getvalue().split("\r\033[2K")
            if segment and not segment.startswith("\033[")
        ]

        self.assertTrue(drawn_lines)
        self.assertTrue(all(len(line) <= 48 for line in drawn_lines))
        self.assertIn("...", output.getvalue())
        self.assertNotIn("evaluation methodology running", output.getvalue())


if __name__ == "__main__":
    unittest.main()
