"""LLM Judge system prompts and spec registry.

Each judge checks one quality dimension. All four share the same
file-reading tools (shell, web_search, web_fetch).
"""

from __future__ import annotations

from .checks import LLMJudgeSpec

# ── Shared prelude (tools + constraints) ────────────────────────────

_PRELUDE = """\
You are a rigorous evaluator. Use shell (read-only cat/find/grep), web_search, and web_fetch to inspect data.

## Data locations
- `.msa/evals/{session_name}/eval_data.json` — merged timeline + telemetry, includes cited_source_ids
- `.msa/sessions/{session_name}/main.jsonl` — answer text
- `.msa/research/{topic_slug}/sources/web/W*-*.md` — source notes (title/url/evidence)

## Output (strict JSON)
{"score": <0~1>, "label": "<brief label>", "explanation": "<per-item breakdown>"}"""

# ── Individual judge prompts ────────────────────────────────────────

SOURCE_PRECISION_PROMPT = _PRELUDE + """

## Task: Source Precision
Check whether each cited source file genuinely supports the claim made about it in the answer.

Steps:
1. cat eval_data.json → get cited_source_ids
2. cat main.jsonl → extract the final answer
3. For each cited_source_id, cat the corresponding W*-*.md
4. Compare: claim in answer vs evidence field in source note
5. If source file does not exist, score=0, actual="source file not found"
6. Output JSON with per-citation breakdown

Additional output field "citations": [{"source_id": "W001", "score": <0~1>, "claimed": "...", "actual": "..."}]"""

CITATION_COVERAGE_PROMPT = _PRELUDE + """

## Task: Citation Coverage
Check whether every factual claim in the answer is backed by a source citation.

Steps:
1. cat main.jsonl → extract the final answer
2. Identify all factual claims (exclude opinions, summary statements)
3. Check each claim for a [Wxxx] citation marker
4. Compute: claims with citations / total claims
5. List claims that lack citations

Additional output field "claims": [{"text": "...", "has_citation": true/false, "missing_detail": "..."}]"""

FAITHFULNESS_PROMPT = _PRELUDE + """

## Task: Answer Faithfulness
Check whether the answer is faithful to the actual content of cited sources, without fabricating claims.

Steps:
1. cat eval_data.json → cited_source_ids
2. cat main.jsonl → final answer
3. For each cited_source_id, cat W*-*.md, read evidence
4. Compare: are all claims attributed to this source actually found in its evidence?
5. Classify each: hallucination / over-extrapolation / faithful

Additional output field "hallucinations": [{"source_id": "W001", "claim": "...", "issue": "hallucination/over-extrapolation/faithful"}]"""

FRESHNESS_PROMPT = _PRELUDE + """

## Task: Freshness
Check whether cited sources are appropriately recent for the research topic.

Steps:
1. cat eval_data.json → cited_source_ids
2. cat main.jsonl → answer + research question
3. For each cited_source_id, cat W*-*.md, check for publication date
4. Judge freshness by content type:
   - paper/technical report: <2 years is good
   - product/API docs: <6 months is good
   - news/events: <1 month is good
5. Classify: outdated / acceptable / fresh / no_date

Additional output field "freshness": [{"source_id": "W001", "date": "2024-03", "status": "outdated/acceptable/fresh/no_date", "reason": "..."}]"""

# ── Registry ────────────────────────────────────────────────────────

LLM_JUDGES: list[LLMJudgeSpec] = [
    LLMJudgeSpec(
        check_id="source_precision",
        system_prompt_template=SOURCE_PRECISION_PROMPT,
        max_turns=80,
    ),
    LLMJudgeSpec(
        check_id="citation_coverage",
        system_prompt_template=CITATION_COVERAGE_PROMPT,
        max_turns=60,
    ),
    LLMJudgeSpec(
        check_id="faithfulness",
        system_prompt_template=FAITHFULNESS_PROMPT,
        max_turns=80,
    ),
    LLMJudgeSpec(
        check_id="freshness",
        system_prompt_template=FRESHNESS_PROMPT,
        max_turns=60,
    ),
]
