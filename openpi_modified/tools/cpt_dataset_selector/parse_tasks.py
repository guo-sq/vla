"""Parse meta/tasks.jsonl lines into normalized records."""

from __future__ import annotations

import json
import re
from typing import Any

# LeRobot-style task text keys (order = preference)
_TASK_TEXT_KEYS = (
    "task",
    "instruction",
    "language_instruction",
    "text",
    "description",
    "caption",
)


def _flatten_strings(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _flatten_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _flatten_strings(v, out)


def extract_task_text(row: dict[str, Any]) -> str:
    for k in _TASK_TEXT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # Fallback: any string values
    strings: list[str] = []
    _flatten_strings(row, strings)
    return " ".join(s for s in strings if s).strip()


def extract_task_index(row: dict[str, Any]) -> int | None:
    for key in ("task_index", "task_idx", "index"):
        v = row.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return _WS_RE.sub(" ", s.strip().lower())


def parse_jsonl_line(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def parse_tasks_jsonl_content(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, line in enumerate(content.splitlines()):
        obj = parse_jsonl_line(line)
        if obj is None:
            continue
        text = extract_task_text(obj)
        rows.append(
            {
                "line_no": i,
                "task_index": extract_task_index(obj),
                "raw": obj,
                "normalized_text": normalize_text(text),
                "task_text": text,
            }
        )
    return rows
