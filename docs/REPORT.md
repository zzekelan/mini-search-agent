# Mini Search Agent Report

Report time: 2026-05-26 08:45 Asia/Shanghai

## Method

Mini Search Agent is a CLI research agent for open-ended research questions. Its current design uses a Main Agent plus Search Subagents:

1. The Main Agent receives the Research Question and decomposes it into a broad Query Plan.
2. The Main Agent dispatches focused Search Subagents with the `subagent` tool.
3. Each Search Subagent is limited to `web_search` and `web_fetch`.
4. `web_search` is backed by real Exa MCP search.
5. `web_fetch` directly fetches candidate URLs and extracts readable text.
6. Each Search Subagent returns structured JSON.
7. The Main Agent writes the final answer to stdout with `[W001]`-style citations.
8. Session timelines and telemetry are written under `.msa/sessions/<session>/`.

The project intentionally keeps research behavior prompt-led. Runtime tool filtering constrains available tools, and telemetry records what happened, but the runtime does not currently reject an answer for missing research steps.

## External Library Dependencies

Runtime dependencies declared in `pyproject.toml`:

| Library | Usage |
| --- | --- |
| `openai` | OpenAI-compatible chat completion client. |
| `httpx` | HTTP requests for web search backend calls, direct URL fetches, and Jina API calls. |
| `pydantic` | Tool argument validation and structured Subagent result parsing. |
| `python-dotenv` | Loading local `.env` configuration. |
| `rank-bm25` | BM25 keyword retrieval in `local_search`. |
| `trafilatura` | Primary HTML-to-readable-text extraction. |

## External Service Dependencies

| Service | Usage |
| --- | --- |
| Exa MCP | Public web search backend for the `web_search` tool. Endpoint: `https://mcp.exa.ai/mcp`. |
| Jina AI | Embedding API (`jina-embeddings-v3`) and Reranker API (`jina-reranker-v2-base-multilingual`) for `local_search`. Requires `JINA_API_KEY` in `.env`. |

## Complete Run Result

Test time: 2026-05-26 Asia/Shanghai

Research question:

```text
2025-2026 年搜索智能体或 Deep Research 产品/论文有哪些代表性进展？
```

Session:

```text
runs/2026-05-26-2025-2026-deep-research/session-2026-05-26-001
```

Research artifacts:

```text
runs/2026-05-26-2025-2026-deep-research/2025-2026-deep-research
```

The Main Agent dispatched five Search Subagents:

| Subagent | Topic | Search calls | Fetch calls | Notes |
| --- | --- | ---: | ---: | --- |
| sub-001 | OpenAI Deep Research 产品进展 | 4 | 8 | Included OpenAI launch, system card, API docs, media coverage, papers. |
| sub-002 | Google Deep Research / Gemini 进展 | 4 | 18 | Included Gemini Deep Research, Project Mariner, Search updates, DeepMind work, API docs. |
| sub-003 | Perplexity 及其他 AI 搜索产品进展 | 5 | 8 | Included Perplexity, DRACO, You.com ARI, Grep AI, related comparisons. |
| sub-004 | 学术论文：搜索智能体与 Deep Research | 5 | 0 | Failure case: returned `fetched_sources` without calling `web_fetch`. |
| sub-005 | 中国厂商 Deep Research 产品进展 | 6 | 5 | Included 百度千帆、通义、Kimi、智谱、腾讯等来源. |

The Main Agent also performed two direct follow-up `web_fetch` calls for arXiv URLs after the subagents completed.

Final telemetry summary:

```text
subagent.started=5
subagent.completed=5
source_note.created=78
final_answer.completed=1
stdout.finalized=1
unknown_cited_source_ids=[]
```

Final answer:

- Printed to stdout.
- Length: 13,341 characters.
- Included a `Sources` section.
- Cited only Source Note IDs known to the session.

Research artifact summary:

```text
.msa/research/2025-2026-deep-research/sources/index.md
```

The source index contains 78 Source Notes, covering:

- OpenAI Deep Research.
- Google Gemini Deep Research and Project Mariner.
- Perplexity Deep Research and DRACO.
- You.com ARI and Research API.
- Academic papers and benchmarks for Deep Research Agents and search agents.
- Chinese vendor products including 百度千帆、通义 DeepResearch、Kimi、智谱 AutoGLM、腾讯元宝.

This run shows that the strengthened prompts increased search breadth and source coverage compared with earlier minimal runs.

## Failure Case

Failure time: 2026-05-26 Asia/Shanghai

Affected sub-session:

```text
runs/2026-05-26-2025-2026-deep-research/session-2026-05-26-001/sub/sub-004
```

Failure summary:

`sub-004` searched for academic papers and returned 26 entries in `fetched_sources`, but it never called `web_fetch`.

Telemetry:

```text
sub-004 search=5 fetch=0 errors=0
```

Timeline shape:

1. `entry-001`: the Main Agent assigned the academic-paper task.
2. `entry-002`: the Subagent called `web_search` five times.
3. `entry-003`: the Subagent received five `web_search` results.
4. `entry-004`: the Subagent returned final JSON.

There were no `web_fetch` tool calls.

The search queries were:

```text
search agent deep research paper 2025 arXiv
LLM agent web search research 2025 paper
autonomous information seeking agent 2025
multi-step reasoning search agent paper 2025
deep research benchmark agent 2025
```

Despite the lack of fetch calls, the final JSON included entries such as:

```text
https://arxiv.org/html/2508.05668v2
https://arxiv.org/pdf/2509.24107
https://arxiv.org/html/2509.25189v1
https://arxiv.org/pdf/2504.21776
https://aclanthology.org/2025.findings-emnlp.130.pdf
https://github.com/Ayanami0730/deep_research_bench
```

These were recorded as Source Notes `W041` through `W066`.

Cause:

- Exa search results include rich highlights, especially for papers.The Subagent treated search highlights as if they were fetched content.

- The prompt said to use `web_fetch`, but it did not explicitly state that a URL may appear in `fetched_sources` only after a successful `web_fetch` call for that exact URL.

Impact:

- Source Notes `W041` through `W066` are not reliable as fetched evidence.
- Final citations were internally valid, but some cited Source Notes may not satisfy the fetch-verification requirement.
- The failure is visible in telemetry, but not obvious from the source index alone.

Fix:

The Main Agent prompt should require generated Subagent tasks to name `web_fetch` explicitly when describing fetch requirements.

## Still Working On

- retrieval metrics
- eval-based agent tuning
