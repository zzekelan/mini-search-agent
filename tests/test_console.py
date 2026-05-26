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


if __name__ == "__main__":
    unittest.main()
