"""Matcher unit tests."""

import json
from pathlib import Path

from tools.cpt_dataset_selector.indexer import full_rebuild
from tools.cpt_dataset_selector.matcher import QueryFilter
from tools.cpt_dataset_selector.matcher import query_repos
from tools.cpt_dataset_selector.matcher import task_matches_filter
from tools.cpt_dataset_selector.taxonomy_loader import load_taxonomy


def _make_dataset(root: Path, repo_id: str, lines: list[str]) -> None:
    p = root / repo_id / "meta"
    p.mkdir(parents=True)
    (p / "tasks.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_task_matches_pick_text():
    tax = load_taxonomy()
    raw = json.dumps({"task": "Pick up the apple from the table"})
    assert task_matches_filter(
        "pick up the apple from the table",
        raw,
        QueryFilter(atomic_actions=("pick",)),
        tax,
    )


def test_query_repos_filter(tmp_path: Path):
    root = tmp_path / "mnt"
    _make_dataset(root, "ds/a", ['{"task": "Pick the box"}'])
    _make_dataset(root, "ds/b", ['{"task": "Only rotate the knob"}'])
    db = tmp_path / "idx.sqlite3"
    full_rebuild(root, ["ds/a", "ds/b"], db, workers=1)
    tax = load_taxonomy()
    q = QueryFilter(atomic_actions=("pick",))
    rows = query_repos(str(db), q, taxonomy=tax)
    assert len(rows) == 1
    assert rows[0]["repo_id"] == "ds/a"


def test_structured_field_match(tmp_path: Path):
    root = tmp_path / "mnt"
    line = json.dumps({"task": "x", "action": "pick"})
    _make_dataset(root, "ds/c", [line])
    db = tmp_path / "idx2.sqlite3"
    full_rebuild(root, ["ds/c"], db, workers=1)
    tax = load_taxonomy()
    q = QueryFilter(atomic_actions=("pick",))
    rows = query_repos(str(db), q, taxonomy=tax)
    assert len(rows) == 1


def test_query_repos_dataset_filters_multiple(tmp_path: Path):
    root = tmp_path / "mnt"
    line = '{"task": "Pick the box"}'
    # Indexed `family` is the second path segment (see indexer._repo_family).
    _make_dataset(root, "root/foo/a", [line])
    _make_dataset(root, "root/bar/b", [line])
    _make_dataset(root, "root/baz/c", [line])
    db = tmp_path / "idx_ds.sqlite3"
    full_rebuild(root, ["root/foo/a", "root/bar/b", "root/baz/c"], db, workers=1)
    tax = load_taxonomy()
    q = QueryFilter(atomic_actions=("pick",), dataset_filters=("foo", "baz"))
    rows = query_repos(str(db), q, taxonomy=tax)
    ids = {r["repo_id"] for r in rows}
    assert ids == {"root/foo/a", "root/baz/c"}
