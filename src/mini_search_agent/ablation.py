"""Ablation script: compare local_search with and without reranker.

Usage:
    uv run ablation "your search query"
"""

from __future__ import annotations

import sys
from pathlib import Path

from mini_search_agent.config import load_llm_config
from mini_search_agent.llm import OpenAICompatibleChatClient
from mini_search_agent.tools.local_search import local_search, SearchResult


def _print_results(label: str, result: SearchResult, top_k: int = 5) -> str:
    """Print results and return a text summary for the judge."""
    print(f"\n{'='*70}")
    print(f"  {label}  (latency: {result.latency_ms:.0f}ms)")
    print(f"{'='*70}")
    lines: list[str] = []
    for i, (doc, score) in enumerate(result.documents[:top_k]):
        snippet = doc.text[:120].replace("\n", " ")
        line = f"[{i+1}] {doc.source_id} | {doc.title[:70]} | score={score:.4f}"
        lines.append(line)
        print(f"  {line}")
        print(f"      {snippet}...")
    return "\n".join(lines)


def _judge_comparison(
    query: str,
    no_rerank_text: str,
    rerank_text: str,
    client: OpenAICompatibleChatClient,
) -> str:
    """Ask LLM to compare the two result sets."""
    prompt = f"""You are evaluating a search ablation study comparing two retrieval methods for the same query.

Query: "{query}"

## Results WITHOUT reranker (BM25 + Embedding + RRF fusion only):
{no_rerank_text}

## Results WITH reranker (BM25 + Embedding + RRF fusion + Jina Reranker):
{rerank_text}

Compare the two result sets. Answer:
1. Which ranking is better for this query? Why?
2. Does the reranker improve relevance ordering? Give specific examples of documents that moved up or down.
3. Overall verdict: "reranker wins" / "no-reranker wins" / "tie" / "inconclusive"

Keep your response concise (under 300 words)."""

    response = client.complete(
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content.strip()


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run ablation <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    workspace = Path.cwd()

    print(f"\n🔍 Query: {query}")
    print(f"📁 Workspace: {workspace}")

    # Run without reranker
    print("\n--- Running WITHOUT reranker ---")
    result_no_rerank = local_search(
        query, workspace=workspace, rerank_enabled=False, top_k=5
    )
    no_rerank_text = _print_results("WITHOUT reranker (BM25 + Embedding + RRF)", result_no_rerank)

    # Run with reranker
    print("\n--- Running WITH reranker ---")
    result_rerank = local_search(
        query, workspace=workspace, rerank_enabled=True, top_k=5
    )
    rerank_text = _print_results("WITH reranker (BM25 + Embedding + RRF + Reranker)", result_rerank)

    # LLM judge comparison
    print(f"\n{'='*70}")
    print("  LLM JUDGE COMPARISON")
    print(f"{'='*70}")
    try:
        config = load_llm_config(workspace)
        client = OpenAICompatibleChatClient(config)
        verdict = _judge_comparison(query, no_rerank_text, rerank_text, client)
        print(f"\n{verdict}")
    except Exception as exc:
        print(f"\n  (LLM judge unavailable: {exc})")

    print()


if __name__ == "__main__":
    main()
