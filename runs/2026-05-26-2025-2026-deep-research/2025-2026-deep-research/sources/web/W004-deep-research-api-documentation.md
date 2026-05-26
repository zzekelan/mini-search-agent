# W004 - Deep research API documentation

- ID: W004
- Title: Deep research API documentation
- URL: https://developers.openai.com/api/docs/guides/deep-research
- Retrieved at: 2026-05-26T08:23:48.287190Z
- Fetch status: success
- Reliability: high
- Queries: OpenAI Deep Research 2025-2026 product details, capabilities, benchmarks, comparisons, and technical reports
- Evidence: API supports o3-deep-research and o4-mini-deep-research models. Requires at least one data source: web search, remote MCP servers, or file search with vector stores. Supports code interpreter tool. Background mode recommended for long tasks. max_tool_calls parameter to control cost/latency. Supports web search (search, open_page, find_in_page actions), file search (max 2 vector stores), remote MCP servers (search+fetch interface only), code interpreter. Does NOT support function calling. Prompt injection risks documented with CRM data exfiltration example. MCP require_approval must be 'never' for deep research. Safety mitigations: trusted MCP only, log review, phased calling, LLM monitor in loop.
- Notes: Fetched successfully. Detailed technical documentation.
