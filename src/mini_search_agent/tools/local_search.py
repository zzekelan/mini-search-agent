"""Local hybrid retrieval: BM25 + Jina embedding + optional Jina reranker.

Corpus is assembled from all past Sessions (web_fetch results in timeline)
and all Source Notes (metadata: title, url, source_id).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from rank_bm25 import BM25Okapi


# ── Jina API ───────────────────────────────────────────────────────

JINA_EMBED_URL = "https://api.jina.ai/v1/embeddings"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
EMBED_MODEL = "jina-embeddings-v3"
RERANK_MODEL = "jina-reranker-v2-base-multilingual"


def _load_jina_key(workspace: Path) -> str:
    """Load JINA_API_KEY from .env or environment."""
    import os
    key = os.environ.get("JINA_API_KEY", "")
    if key:
        return key
    env_file = workspace / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.strip().startswith("JINA_API_KEY="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("JINA_API_KEY not found in .env or environment")


def _jina_headers(api_key: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }


# ── Document model ──────────────────────────────────────────────────

@dataclass(frozen=True)
class Document:
    source_id: str
    title: str
    url: str
    text: str

    @property
    def uid(self) -> str:
        return _hash_url(self.url)


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


# ── Corpus loader ───────────────────────────────────────────────────

def load_corpus(workspace: Path) -> list[Document]:
    """Assemble corpus from all sessions and source notes."""

    # Step 1: collect full texts from all timeline web_fetch results
    text_by_url: dict[str, str] = {}
    sessions_root = workspace / ".msa" / "sessions"
    if sessions_root.exists():
        for tl_path in sorted(sessions_root.rglob("timeline.jsonl")):
            try:
                for entry in _read_jsonl(tl_path):
                    for part in entry.get("parts", []):
                        if (
                            part.get("type") == "tool_result"
                            and part.get("tool_name") == "web_fetch"
                            and not part.get("is_error", False)
                        ):
                            content = part.get("content", "")
                            if content and len(content) > 200:
                                url = _extract_url_from_result(part)
                                if url and url not in text_by_url:
                                    text_by_url[url] = content
            except Exception:
                continue

    # Step 2: collect metadata from all Source Notes
    documents: list[Document] = []
    research_root = workspace / ".msa" / "research"
    if research_root.exists():
        for src_path in sorted(research_root.rglob("W*.md")):
            try:
                note = _parse_source_note_fields(src_path)
                url = note.get("URL", "")
                if not url:
                    continue
                text = text_by_url.get(url, "")
                if not text:
                    # Fallback: use evidence + notes if no full text
                    text = f"{note.get('Evidence', '')}\n{note.get('Notes', '')}"
                if not text.strip():
                    continue
                documents.append(Document(
                    source_id=note.get("ID", ""),
                    title=note.get("Title", ""),
                    url=url,
                    text=text,
                ))
            except Exception:
                continue

    return documents


def _extract_url_from_result(part: dict) -> str:
    meta = part.get("metadata", {})
    return str(meta.get("url", "")).strip()


def _parse_source_note_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("- ") or ": " not in line:
            continue
        key, value = line[2:].split(": ", 1)
        fields[key] = value
    return fields


# ── Simple tokenizer for BM25 ───────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    return re.findall(r"\w+", text.lower())


# ── BM25 index ──────────────────────────────────────────────────────

class BM25Index:
    def __init__(self, documents: list[Document]):
        self.documents = documents
        self._tokenized = [_tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        if not self._bm25:
            return []
        tokenized_query = _tokenize(query)
        if not tokenized_query:
            return []
        scores = self._bm25.get_scores(tokenized_query)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.documents[i], float(scores[i])) for i, _ in indexed[:top_k] if scores[i] > 0]


# ── Jina Embedding index ────────────────────────────────────────────

class EmbeddingIndex:
    def __init__(self, documents: list[Document], api_key: str, cache_dir: Path):
        self.documents = documents
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._embeddings: list[list[float]] | None = None

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Call Jina embeddings API with batching and retry."""
        embeddings: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for idx, doc in enumerate(self.documents):
            cache_path = self._cache_path(doc)
            if cache_path.exists():
                try:
                    data = json.loads(cache_path.read_text())
                    if data.get("model") == EMBED_MODEL:
                        embeddings[idx] = data["embedding"]
                        continue
                except Exception:
                    pass
            uncached_indices.append(idx)

        if not uncached_indices:
            return [e for e in embeddings if e is not None]

        cached_count = len(texts) - len(uncached_indices)
        if cached_count:
            print(f"  Embedding cache hit: {cached_count}/{len(texts)}, need {len(uncached_indices)} API calls")

        BATCH_SIZE = 10
        for batch_start in range(0, len(uncached_indices), BATCH_SIZE):
            batch_indices = uncached_indices[batch_start:batch_start + BATCH_SIZE]
            batch_texts = [texts[i][:8000] for i in batch_indices]
            batch_n = batch_start // BATCH_SIZE + 1
            total_batches = (len(uncached_indices) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"  Embedding batch {batch_n}/{total_batches}: {len(batch_texts)} texts ...")

            result = _api_post_with_retry(
                JINA_EMBED_URL,
                headers=_jina_headers(self.api_key),
                json_payload={
                    "model": EMBED_MODEL,
                    "input": batch_texts,
                    "task": "retrieval.passage",
                },
            )
            data_list = result.get("data", [])
            for i, idx in enumerate(batch_indices):
                if i < len(data_list):
                    emb = data_list[i].get("embedding", [])
                    embeddings[idx] = emb
                    cache_path = self._cache_path(self.documents[idx])
                    cache_path.write_text(json.dumps({
                        "model": EMBED_MODEL,
                        "embedding": emb,
                    }))

            if batch_start + BATCH_SIZE < len(uncached_indices):
                time.sleep(2.0)

        return [e for e in embeddings if e is not None]

    def build(self) -> None:
        texts = [doc.text for doc in self.documents]
        self._embeddings = self._embed_texts(texts)

    def search(self, query: str, top_k: int = 20) -> list[tuple[Document, float]]:
        if not self._embeddings:
            return []
        result = _api_post_with_retry(
            JINA_EMBED_URL,
            headers=_jina_headers(self.api_key),
            json_payload={
                "model": EMBED_MODEL,
                "input": [query],
                "task": "retrieval.query",
            },
        )
        query_emb = result["data"][0]["embedding"]

        scores = [_cosine_sim(query_emb, doc_emb) for doc_emb in self._embeddings]
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.documents[i], scores[i]) for i, _ in indexed[:top_k] if scores[i] > 0]

    def _cache_path(self, doc: Document) -> Path:
        return self.cache_dir / f"{doc.uid}.json"


def _api_post_with_retry(
    url: str,
    headers: dict[str, str],
    json_payload: dict[str, Any],
    max_retries: int = 6,
    timeout: int = 60,
) -> dict[str, Any]:
    """POST with exponential backoff on 429 rate limits."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = httpx.post(url, headers=headers, json=json_payload, timeout=timeout)
            if response.status_code == 429:
                if attempt == max_retries - 1:
                    raise RuntimeError("Jina API rate limit exhausted after all retries")
                wait = 2 ** attempt
                print(f"  Rate limited, retrying in {wait}s ...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, RuntimeError) as exc:
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
                if attempt == max_retries - 1:
                    raise RuntimeError("Jina API rate limit exhausted after all retries") from exc
                wait = 2 ** attempt
                print(f"  Rate limited, retrying in {wait}s ...")
                time.sleep(wait)
                continue
            last_exc = exc
            break
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                break
    raise RuntimeError(f"API call failed after {max_retries} retries: {last_exc}") from last_exc


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = (sum(x * x for x in a)) ** 0.5
    norm_b = (sum(x * x for x in b)) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── RRF fusion ──────────────────────────────────────────────────────

def _rrf_fusion(
    bm25_results: list[tuple[Document, float]],
    embed_results: list[tuple[Document, float]],
    k: int = 60,
    top_k: int = 20,
) -> list[tuple[Document, float]]:
    """Reciprocal Rank Fusion: combine BM25 and embedding rankings."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(bm25_results):
        scores[doc.uid] = scores.get(doc.uid, 0) + 1 / (k + rank + 1)
        doc_map[doc.uid] = doc

    for rank, (doc, _) in enumerate(embed_results):
        scores[doc.uid] = scores.get(doc.uid, 0) + 1 / (k + rank + 1)
        doc_map[doc.uid] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[uid], score) for uid, score in ranked[:top_k]]


# ── Jina Reranker ───────────────────────────────────────────────────

def rerank(
    query: str,
    documents: list[tuple[Document, float]],
    api_key: str,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """Re-rank documents using Jina Reranker API."""
    if not documents:
        return []
    doc_texts = [doc.text[:2000] for doc, _ in documents]
    result = _api_post_with_retry(
        JINA_RERANK_URL,
        headers=_jina_headers(api_key),
        json_payload={
            "model": RERANK_MODEL,
            "query": query,
            "documents": doc_texts,
            "top_n": min(top_k, len(documents)),
        },
    )
    ranked: list[tuple[Document, float]] = []
    for item in result.get("results", []):
        idx = item.get("index", -1)
        score = item.get("relevance_score", 0.0)
        if 0 <= idx < len(documents):
            ranked.append((documents[idx][0], float(score)))
    return ranked


# ── High-level API ──────────────────────────────────────────────────

@dataclass
class SearchResult:
    documents: list[tuple[Document, float]]
    query: str
    reranked: bool
    latency_ms: float


def local_search(
    query: str,
    *,
    workspace: Path | None = None,
    rerank_enabled: bool = True,
    top_k: int = 5,
    fusion_top_k: int = 20,
) -> SearchResult:
    """Run hybrid retrieval with optional reranking.

    Args:
        query: Search query string.
        workspace: Project root (default: cwd).
        rerank_enabled: Apply Jina reranker after RRF fusion.
        top_k: Final number of results to return.
        fusion_top_k: Number of candidates from RRF fusion before reranking.
    """
    started = time.perf_counter()
    ws = workspace or Path.cwd()
    api_key = _load_jina_key(ws)

    print(f"Loading corpus from {ws / '.msa'} ...")
    documents = load_corpus(ws)
    print(f"  {len(documents)} documents loaded.")

    if not documents:
        return SearchResult(documents=[], query=query, reranked=rerank_enabled, latency_ms=0.0)

    # BM25
    print("Building BM25 index ...")
    bm25 = BM25Index(documents)
    bm25_results = bm25.search(query, top_k=fusion_top_k)
    print(f"  BM25 returned {len(bm25_results)} results.")

    # Embedding
    print("Building embedding index (this may call Jina API) ...")
    cache_dir = ws / ".msa" / "cache" / "embeddings"
    embed = EmbeddingIndex(documents, api_key, cache_dir)
    embed.build()
    embed_results = embed.search(query, top_k=fusion_top_k)
    print(f"  Embedding returned {len(embed_results)} results.")

    # RRF fusion
    fused = _rrf_fusion(bm25_results, embed_results, top_k=fusion_top_k)
    print(f"  RRF fused: {len(fused)} candidates.")

    # Optional reranker
    if rerank_enabled and fused:
        print("Applying Jina reranker ...")
        fused = rerank(query, fused, api_key, top_k=top_k)
        print(f"  Reranked to top-{len(fused)}.")
    else:
        fused = fused[:top_k]

    elapsed = (time.perf_counter() - started) * 1000
    return SearchResult(
        documents=fused,
        query=query,
        reranked=rerank_enabled,
        latency_ms=round(elapsed, 1),
    )


# ── Helpers ─────────────────────────────────────────────────────────

def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    result: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return result
