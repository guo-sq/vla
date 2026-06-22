"""manifest merge and load helpers."""

from pathlib import Path

from tools.cpt_dataset_selector.manifest import load_repo_ids_merged
from tools.cpt_dataset_selector.manifest import parse_config_paths_csv
from tools.cpt_dataset_selector.manifest import write_manifest


def test_parse_config_paths_csv(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text("REPO_ID = ()\n", encoding="utf-8")
    b.write_text("REPO_ID = ()\n", encoding="utf-8")
    raw = f" {a} , {b} "
    got = parse_config_paths_csv(raw)
    assert got is not None
    assert len(got) == 2
    assert got[0] == Path(a)
    assert got[1] == Path(b)
    assert parse_config_paths_csv(None) is None
    assert parse_config_paths_csv("  ") is None


def test_load_repo_ids_merged_dedup_order(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text('REPO_ID = ("x/1", "x/2")\n', encoding="utf-8")
    b.write_text('REPO_ID = ("x/2", "y/1")\n', encoding="utf-8")
    merged = load_repo_ids_merged([a, b])
    assert merged == ("x/1", "x/2", "y/1")


def test_write_manifest_multi_config_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.py"
    b = tmp_path / "b.py"
    a.write_text('REPO_ID = ("p/a",)\n', encoding="utf-8")
    b.write_text('REPO_ID = ("p/b",)\n', encoding="utf-8")
    out = tmp_path / "manifest.json"
    payload = write_manifest(out, config_paths=[a, b])
    assert payload["count"] == 2
    assert len(payload["config_paths"]) == 2
    assert payload["repo_ids"] == ["p/a", "p/b"]
