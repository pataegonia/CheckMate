from __future__ import annotations

import json
import re
from typing import Any


class ResponseParseError(ValueError):
    pass


def parse_json_response(raw: str) -> dict[str, Any]:
    """Parse Claude JSON, with one repair attempt when the response is not pure JSON."""
    text = (raw or "").strip()
    if not text:
        raise ResponseParseError("empty Claude response")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        repaired = repair_json_once(text)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError as exc:
            snippet = repaired[:200].replace("\n", " ")
            raise ResponseParseError(f"Claude response was not valid JSON after repair: {snippet}") from exc

    if not isinstance(parsed, dict):
        raise ResponseParseError("Claude response JSON must be an object")
    return parsed


def repair_json_once(text: str) -> str:
    repaired = _strip_code_fence(text)
    repaired = _extract_json_object(repaired)
    repaired = _normalize_json_quotes(repaired)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    return repaired.strip()


def _strip_code_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
        value = re.sub(r"```$", "", value).strip()
    return value


def _extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        snippet = text[:200].replace("\n", " ")
        raise ResponseParseError(f"no JSON object found in Claude response: {snippet}")
    return text[start : end + 1]


def _normalize_json_quotes(text: str) -> str:
    return (
        text.replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
