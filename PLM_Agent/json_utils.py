"""Robust extraction of a JSON object from an LLM response."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", text.strip())
    if not match:
        match = re.search(r"\{[\s\S]*", text.strip())
    if not match:
        return {}
    snippet = match.group()
    for candidate in (snippet, _repair_json_closers(snippet)):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _repair_json_closers(text: str) -> str:
    snippet = text.strip()
    while snippet.endswith(","):
        snippet = snippet[:-1].rstrip()
    snippet += "]" * max(0, snippet.count("[") - snippet.count("]"))
    snippet += "}" * max(0, snippet.count("{") - snippet.count("}"))
    return snippet
