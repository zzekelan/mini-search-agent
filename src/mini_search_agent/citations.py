from __future__ import annotations

import re


def extract_cited_source_ids(answer: str) -> list[str]:
    seen: list[str] = []
    for match in re.finditer(r"\[(W\d{3})\]", answer):
        source_id = match.group(1)
        if source_id not in seen:
            seen.append(source_id)
    return seen
