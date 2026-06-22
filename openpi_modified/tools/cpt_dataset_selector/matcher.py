"""Filter repos by taxonomy (structured fields + text synonyms)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
import sqlite3
from typing import Any

from .indexer import ensure_repos_robot_type_column
from .taxonomy_loader import Taxonomy
from .taxonomy_loader import TaxonomyOption
from .taxonomy_loader import load_taxonomy


@dataclass(frozen=True)
class QueryFilter:
    atomic_actions: tuple[str, ...] = ()
    object_categories: tuple[str, ...] = ()
    scenes: tuple[str, ...] = ()
    include_incomplete_meta: bool = False
    """When set with taxonomy filters, drop repos where matched_tasks/task_count < this (0-1)."""
    min_match_ratio: float | None = None
    dataset_filters: tuple[str, ...] = ()
    """When non-empty, only repos whose indexed `family` is in this set."""


def _option_by_id(opts: tuple[TaxonomyOption, ...], oid: str) -> TaxonomyOption | None:
    for o in opts:
        if o.id == oid:
            return o
    return None


def _collect_synonyms(opts: tuple[TaxonomyOption, ...], selected_ids: tuple[str, ...]) -> list[str]:
    """All synonyms for selected option ids (lowercased)."""
    syns: list[str] = []
    for oid in selected_ids:
        opt = _option_by_id(opts, oid)
        if opt is None:
            continue
        syns.append(opt.id.lower())
        syns.extend(s.lower() for s in opt.synonyms)
    return syns


def _text_matches_any(term: str, synonyms: list[str]) -> bool:
    if not term:
        return False
    return any(s and s in term for s in synonyms)


def _structured_value(raw: dict[str, Any], dimension: str, taxonomy: Taxonomy) -> str | None:
    keys = taxonomy.structured_field_aliases.get(dimension, ())
    for k in keys:
        if k in raw:
            v = raw[k]
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
            if isinstance(v, int | float):
                return str(v).lower()
    return None


def _structured_matches(
    raw: dict[str, Any],
    dimension: str,
    selected_ids: tuple[str, ...],
    taxonomy: Taxonomy,
) -> bool:
    if not selected_ids:
        return True
    opts = taxonomy.by_dimension[dimension]
    synonyms = _collect_synonyms(opts, selected_ids)
    val = _structured_value(raw, dimension, taxonomy)
    if val is None:
        return False
    return any(s and s in val for s in synonyms)


def _text_matches_dimension(
    normalized_text: str,
    dimension: str,
    selected_ids: tuple[str, ...],
    taxonomy: Taxonomy,
) -> bool:
    if not selected_ids:
        return True
    opts = taxonomy.by_dimension[dimension]
    synonyms = _collect_synonyms(opts, selected_ids)
    return _text_matches_any(normalized_text, synonyms)


def _dim_satisfied(
    dim: str,
    selected: tuple[str, ...],
    normalized_text: str,
    raw: dict[str, Any],
    taxonomy: Taxonomy,
) -> bool:
    if not selected:
        return True
    struct_ok = _structured_matches(raw, dim, selected, taxonomy)
    text_ok = _text_matches_dimension(normalized_text, dim, selected, taxonomy)
    # Mixed: structured field match OR normalized task text (substring) match
    return struct_ok or text_ok


def task_matches_filter(
    normalized_text: str,
    raw_json: str,
    q: QueryFilter,
    taxonomy: Taxonomy,
) -> bool:
    """AND across dimensions with selections; OR within dimension (via synonym lists)."""
    raw = json.loads(raw_json)
    dims: list[tuple[str, tuple[str, ...]]] = [
        ("atomic_actions", q.atomic_actions),
        ("object_categories", q.object_categories),
        ("scenes", q.scenes),
    ]
    return all(_dim_satisfied(dim, selected, normalized_text, raw, taxonomy) for dim, selected in dims)


def repo_matches_any_task(
    rows: list[tuple[str, str]],
    q: QueryFilter,
    taxonomy: Taxonomy,
) -> bool:
    """rows: list of (normalized_text, raw_json)"""
    if not rows:
        return False
    return any(task_matches_filter(norm, raw, q, taxonomy) for norm, raw in rows)


def _repo_row_public(r: dict[str, Any]) -> dict[str, Any]:
    """Expose DB `family` as `dataset` for API clarity."""
    row = {**r}
    row["dataset"] = row.pop("family", None)
    row["repo_duration"] = row.get("duration_hours")
    return row


# SQLite bind parameter limit is often 999; stay below for IN (...).
_TASKS_IN_CHUNK = 450


def _tasks_by_repo_for_ids(
    conn: sqlite3.Connection,
    repo_ids: list[str],
) -> dict[str, list[tuple[str, str]]]:
    """Batch-load tasks for many repos (avoids one connection + query per repo)."""
    by_repo: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if not repo_ids:
        return {}
    for i in range(0, len(repo_ids), _TASKS_IN_CHUNK):
        chunk = repo_ids[i : i + _TASKS_IN_CHUNK]
        ph = ",".join("?" * len(chunk))
        cur = conn.execute(
            f"SELECT repo_id, normalized_text, raw_json FROM tasks WHERE repo_id IN ({ph})",
            chunk,
        )
        for row in cur:
            by_repo[row["repo_id"]].append((row["normalized_text"], row["raw_json"]))
    return dict(by_repo)


def query_repos(
    db_path: str,
    q: QueryFilter,
    taxonomy: Taxonomy | None = None,
) -> list[dict[str, Any]]:
    """Return repo rows that match filter (any task matches)."""
    tax = taxonomy or load_taxonomy()
    has_taxonomy = bool(q.atomic_actions or q.object_categories or q.scenes)
    conn = sqlite3.connect(db_path)
    try:
        ensure_repos_robot_type_column(conn)
        conn.commit()
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            """
            SELECT r.repo_id, r.family, r.robot_type, r.total_episodes, r.duration_hours,
                   r.has_tasks_jsonl, r.error, r.task_count
            FROM repos r
            ORDER BY r.repo_id
            """
        )
        repos = [dict(row) for row in cur.fetchall()]

        candidates: list[dict[str, Any]] = []
        for r in repos:
            if not q.include_incomplete_meta and (not r["has_tasks_jsonl"] or r.get("error")):
                continue
            if q.dataset_filters and r["family"] not in q.dataset_filters:
                continue
            candidates.append(r)

        if not candidates:
            return []

        if not has_taxonomy:
            out: list[dict[str, Any]] = []
            for r in candidates:
                tc = int(r["task_count"] or 0)
                out.append(
                    {
                        **_repo_row_public(r),
                        "matched_tasks": tc,
                        "match_ratio": 1.0 if tc else None,
                    }
                )
            return out

        candidate_ids = [c["repo_id"] for c in candidates]
        tasks_by_repo = _tasks_by_repo_for_ids(conn, candidate_ids)

        out = []
        for r in candidates:
            rid = r["repo_id"]
            rows = tasks_by_repo.get(rid, [])
            if not repo_matches_any_task(rows, q, tax):
                continue
            matched = sum(1 for norm, raw in rows if task_matches_filter(norm, raw, q, tax))
            tc = int(r["task_count"] or 0)
            ratio = (matched / tc) if tc > 0 else 0.0
            if q.min_match_ratio is not None and tc > 0 and ratio < q.min_match_ratio - 1e-15:
                continue
            if q.min_match_ratio is not None and tc == 0:
                continue
            out.append(
                {
                    **_repo_row_public(r),
                    "matched_tasks": matched,
                    "match_ratio": ratio,
                }
            )
        return out
    finally:
        conn.close()


def fts_search(
    db_path: str,
    q: str,
    *,
    limit: int = 500,
) -> list[dict[str, Any]]:
    """Simple substring search on normalized_text (optional utility)."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT repo_id, line_no, task_index, raw_json, normalized_text
        FROM tasks
        WHERE normalized_text LIKE ?
        LIMIT ?
        """,
        (f"%{q.lower()}%", limit),
    )
    rows = [dict(x) for x in cur.fetchall()]
    conn.close()
    return rows
