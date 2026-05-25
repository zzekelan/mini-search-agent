from __future__ import annotations

import unittest

from pydantic import ValidationError

from mini_search_agent.subagent import SubagentArgs, subagent_tool_schema
from mini_search_agent.tools import (
    ShellArgs,
    WebFetchArgs,
    WebSearchArgs,
    shell_tool_schema,
    web_fetch_tool_schema,
    web_search_tool_schema,
)


class ToolSchemaTest(unittest.TestCase):
    def test_tool_schemas_are_generated_from_pydantic_json_schema(self):
        for schema, model, required in [
            (web_search_tool_schema(), WebSearchArgs, ["query"]),
            (web_fetch_tool_schema(), WebFetchArgs, ["url"]),
            (shell_tool_schema(), ShellArgs, ["command"]),
            (subagent_tool_schema(), SubagentArgs, ["description", "prompt"]),
        ]:
            parameters = schema["function"]["parameters"]
            self.assertEqual(parameters["additionalProperties"], False)
            self.assertEqual(parameters["required"], required)
            self.assertEqual(parameters["properties"], model.model_json_schema()["properties"])

    def test_tool_arg_models_reject_unknown_fields(self):
        with self.assertRaises(ValidationError):
            WebSearchArgs.model_validate({"query": "ok", "extra": "nope"})


if __name__ == "__main__":
    unittest.main()
