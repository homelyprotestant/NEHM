"""Small overlapping text windows used by PLM_Agent RCS."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

import tiktoken

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class SplitParagraph:
    page: int
    text: str
    local_index: int
    section: str | None = None


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(os.getenv("RAG_CHUNK_ENCODING") or "cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    tokens = _encoding().encode(text)
    return [
        _encoding().decode(tokens[i : i + max_tokens])
        for i in range(0, len(tokens), max_tokens)
    ]


def atomize_with_overlap(
    paragraphs: list[SplitParagraph],
    *,
    chunk_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[SplitParagraph]:
    target = chunk_tokens or int(os.getenv("RAG_ATOMIC_TOKENS", "180"))
    overlap = overlap_tokens or int(os.getenv("RAG_ATOMIC_OVERLAP", "60"))
    overlap = min(overlap, max(1, target // 3))

    spans: list[tuple[str, int, str | None]] = []
    for item in paragraphs:
        for sentence in _SENTENCE_SPLIT.split(item.text.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            parts = (
                _split_long_text(sentence, target)
                if count_tokens(sentence) > target
                else [sentence]
            )
            spans.extend((part, item.page, item.section) for part in parts)

    chunks: list[SplitParagraph] = []
    start = 0
    while start < len(spans):
        end = start
        while end < len(spans):
            text = " ".join(item[0] for item in spans[start : end + 1])
            if count_tokens(text) > target and end > start:
                break
            end += 1
            if count_tokens(text) >= target:
                break
        end = max(end, start + 1)
        window = spans[start:end]
        chunks.append(
            SplitParagraph(
                page=window[0][1],
                text=" ".join(item[0] for item in window).strip(),
                local_index=len(chunks) + 1,
                section=window[0][2],
            )
        )
        if end >= len(spans):
            break
        next_start = end
        while next_start > start + 1:
            tail = " ".join(item[0] for item in spans[next_start - 1 : end])
            if count_tokens(tail) >= overlap:
                break
            next_start -= 1
        start = max(start + 1, next_start - 1)
    return chunks
