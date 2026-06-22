"""Unit tests for merge_batches — combine per-batch episode_details.json files
produced by ``wait_gpu_then_run.sh`` when it runs the production sparse
benchmark in 60-repo chunks.

Each batch writes to ``{output_dir}/batch_NNN/metrics/episode_details.json``;
after all batches finish (or crash) this script concatenates them into a
single ``{output_dir}/metrics/episode_details.json`` so the downstream
``fill_head_pred_ranges.py`` has one canonical input.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark.merge_batches import merge_batches
from scripts.benchmark.merge_batches import merge_skipped_repos


def _write_episode_details(batch_dir: Path, episodes: list[dict]) -> None:
    metrics_dir = batch_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "episode_details.json").write_text(json.dumps(episodes))


def test_merges_episodes_from_multiple_batches(tmp_path: Path):
    _write_episode_details(
        tmp_path / "batch_000",
        [
            {"episode_key": "a:0", "head_pred": -0.1},
            {"episode_key": "b:0", "head_pred": -0.2},
        ],
    )
    _write_episode_details(
        tmp_path / "batch_001",
        [
            {"episode_key": "c:0", "head_pred": -0.3},
        ],
    )

    out_path = tmp_path / "merged.json"
    count = merge_batches(tmp_path, out_path)

    assert count == 3
    merged = json.loads(out_path.read_text())
    keys = {e["episode_key"] for e in merged}
    assert keys == {"a:0", "b:0", "c:0"}


def test_dedups_on_episode_key_across_batches(tmp_path: Path):
    """If the same repo accidentally appears in two batches, keep the first occurrence."""
    _write_episode_details(
        tmp_path / "batch_000",
        [
            {"episode_key": "dup:0", "head_pred": -0.11},
        ],
    )
    _write_episode_details(
        tmp_path / "batch_001",
        [
            {"episode_key": "dup:0", "head_pred": -0.99},
        ],
    )

    out_path = tmp_path / "merged.json"
    count = merge_batches(tmp_path, out_path)

    assert count == 1
    merged = json.loads(out_path.read_text())
    # First batch wins (sorted batch dir iteration order).
    assert merged[0]["head_pred"] == pytest.approx(-0.11)


def test_skips_missing_batch_directories(tmp_path: Path):
    """If a batch crashed without producing a metrics file, skip it silently."""
    _write_episode_details(
        tmp_path / "batch_000",
        [
            {"episode_key": "ok:0", "head_pred": -0.5},
        ],
    )
    # batch_001 was created but its run_benchmark crashed before writing metrics
    (tmp_path / "batch_001").mkdir()

    out_path = tmp_path / "merged.json"
    count = merge_batches(tmp_path, out_path)

    assert count == 1


def test_ignores_non_batch_directories(tmp_path: Path):
    """Only directories matching the batch_NNN pattern are merged."""
    _write_episode_details(
        tmp_path / "batch_000",
        [
            {"episode_key": "a:0", "head_pred": -0.5},
        ],
    )
    (tmp_path / "visualization").mkdir()  # Unrelated subdir from a single-batch run
    _write_episode_details(
        tmp_path / "misc",
        [
            {"episode_key": "ignored:0", "head_pred": -0.9},
        ],
    )

    out_path = tmp_path / "merged.json"
    count = merge_batches(tmp_path, out_path)
    assert count == 1


def test_no_batches_produces_empty_output(tmp_path: Path):
    out_path = tmp_path / "merged.json"
    count = merge_batches(tmp_path, out_path)
    assert count == 0
    assert json.loads(out_path.read_text()) == []


# ---------------------------------------------------------------------------
# merge_skipped_repos — aggregate per-repo failures across batches
# ---------------------------------------------------------------------------


def _write_skipped(batch_dir: Path, entries: list[dict]) -> None:
    batch_dir.mkdir(exist_ok=True)
    (batch_dir / "skipped_repos.json").write_text(json.dumps(entries))


def test_merges_skipped_repos_from_multiple_batches(tmp_path: Path):
    """Per-repo errors logged by run_benchmark.py get aggregated into one file."""
    _write_skipped(
        tmp_path / "batch_000",
        [
            {"repo_id": "record.bad.1", "error": "RuntimeError: corrupt video"},
        ],
    )
    _write_skipped(
        tmp_path / "batch_002",
        [
            {"repo_id": "record.bad.2", "error": "RuntimeError: missing file"},
            {"repo_id": "record.bad.3", "error": "ValueError: shape mismatch"},
        ],
    )

    out_path = tmp_path / "skipped_repos.json"
    count = merge_skipped_repos(tmp_path, out_path)

    assert count == 3
    merged = json.loads(out_path.read_text())
    repo_ids = {e["repo_id"] for e in merged}
    assert repo_ids == {"record.bad.1", "record.bad.2", "record.bad.3"}


def test_no_skipped_repos_produces_empty_list(tmp_path: Path):
    """All batches succeeded → empty merged list."""
    # Create batch dirs without skipped files (simulating fully successful batches)
    (tmp_path / "batch_000").mkdir()
    (tmp_path / "batch_001").mkdir()

    out_path = tmp_path / "skipped_repos.json"
    count = merge_skipped_repos(tmp_path, out_path)

    assert count == 0
    assert json.loads(out_path.read_text()) == []


def test_skipped_repos_dedups_on_repo_id(tmp_path: Path):
    """Same repo failing in two batches (shouldn't happen, but defensive) → kept once."""
    _write_skipped(
        tmp_path / "batch_000",
        [
            {"repo_id": "record.bad", "error": "first error"},
        ],
    )
    _write_skipped(
        tmp_path / "batch_001",
        [
            {"repo_id": "record.bad", "error": "second error"},
        ],
    )

    count = merge_skipped_repos(tmp_path, tmp_path / "skipped.json")
    assert count == 1


def test_skipped_repos_ignores_non_batch_dirs(tmp_path: Path):
    _write_skipped(tmp_path / "batch_000", [{"repo_id": "ok_bad", "error": "e"}])
    _write_skipped(tmp_path / "random_dir", [{"repo_id": "ignored", "error": "e"}])

    count = merge_skipped_repos(tmp_path, tmp_path / "skipped.json")
    assert count == 1


# ---------------------------------------------------------------------------
# Regression: iteration order
# ---------------------------------------------------------------------------


def test_batches_iterated_in_sorted_order(tmp_path: Path):
    """batch_005 must not appear before batch_010 numerically, and batch_010
    must not appear before batch_002 just because of lex ordering."""
    _write_episode_details(tmp_path / "batch_002", [{"episode_key": "ep:2", "head_pred": 0.2}])
    _write_episode_details(tmp_path / "batch_005", [{"episode_key": "ep:5", "head_pred": 0.5}])
    _write_episode_details(tmp_path / "batch_010", [{"episode_key": "ep:10", "head_pred": 1.0}])

    out_path = tmp_path / "merged.json"
    merge_batches(tmp_path, out_path)

    merged = json.loads(out_path.read_text())
    keys = [e["episode_key"] for e in merged]
    assert keys == ["ep:2", "ep:5", "ep:10"]
