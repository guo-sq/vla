"""RED-phase tests for compute_separation_scores.py.

Separation Score = Cohen's d on tail_pred between two quadrant groups.
Used to compare 4 value models on the v0409 fold/flatten benchmarks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.compute_separation_scores import cohen_d
from scripts.benchmark.compute_separation_scores import compute_task_separation_scores
from scripts.benchmark.compute_separation_scores import load_episode_scores
from scripts.benchmark.compute_separation_scores import load_manifest_quadrants

# ---------------------------------------------------------------------------
# cohen_d primitive
# ---------------------------------------------------------------------------


class TestCohenD:
    def test_zero_separation_identical_distributions(self):
        a = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        b = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert cohen_d(a, b) == pytest.approx(0.0, abs=1e-9)

    def test_perfect_separation_large_d(self):
        # Non-overlapping, well separated → |d| much greater than 2
        a = np.array([0.0, 0.0, 0.0, 0.0])
        b = np.array([1.0, 1.0001, 0.9999, 1.0])
        d = cohen_d(a, b)
        assert abs(d) > 10.0

    def test_known_value(self):
        # mean_a=0, mean_b=1, sd_pooled=1 → d = (1 - 0) / 1 = 1
        a = np.array([-1.0, 0.0, 1.0])  # sd = 1
        b = np.array([0.0, 1.0, 2.0])  # sd = 1
        assert cohen_d(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_sign_direction(self):
        # cohen_d(a, b) should be (mean_b - mean_a)/sd_pooled
        low = np.array([-1.0, -0.9, -1.1])
        high = np.array([0.0, 0.1, -0.1])
        assert cohen_d(low, high) > 0  # high - low > 0

    def test_handles_single_element_gracefully(self):
        a = np.array([0.0])
        b = np.array([1.0])
        # Zero variance on both sides → pooled sd is 0 → should return NaN, not raise
        d = cohen_d(a, b)
        assert math.isnan(d)

    def test_empty_array_returns_nan(self):
        assert math.isnan(cohen_d(np.array([]), np.array([1.0, 2.0])))
        assert math.isnan(cohen_d(np.array([1.0, 2.0]), np.array([])))


# ---------------------------------------------------------------------------
# load_episode_scores
# ---------------------------------------------------------------------------


class TestLoadEpisodeScores:
    def test_builds_episode_key_to_tail_pred_map(self, tmp_path: Path):
        data = [
            {"episode_key": "repoA:0", "tail_pred": -0.01, "head_pred": -0.90},
            {"episode_key": "repoA:1", "tail_pred": -0.95, "head_pred": -0.01},
            {"episode_key": "repoB:0", "tail_pred": -0.50, "head_pred": -0.50},
        ]
        path = tmp_path / "episode_details.json"
        path.write_text(json.dumps(data))

        scores = load_episode_scores(path)

        assert scores["repoA:0"]["tail_pred"] == pytest.approx(-0.01)
        assert scores["repoA:1"]["head_pred"] == pytest.approx(-0.01)
        assert scores["repoB:0"]["tail_pred"] == pytest.approx(-0.50)
        assert len(scores) == 3

    def test_skips_entries_without_tail_pred(self, tmp_path: Path):
        data = [
            {"episode_key": "x:0", "tail_pred": -0.1, "head_pred": -0.2},
            {"episode_key": "y:0", "tail_pred": None, "head_pred": None},  # skipped
        ]
        path = tmp_path / "ep.json"
        path.write_text(json.dumps(data))
        scores = load_episode_scores(path)
        assert "x:0" in scores
        assert "y:0" not in scores


# ---------------------------------------------------------------------------
# load_manifest_quadrants
# ---------------------------------------------------------------------------


class TestLoadManifestQuadrants:
    def test_returns_quadrant_to_episode_keys(self, tmp_path: Path):
        manifest = {
            "task_name": "fold",
            "quotas": {"TP": 2, "TN": 1},
            "actual_counts": {"TP": 2, "TN": 1},
            "episodes": {
                "TP": ["repoA:0", "repoA:1"],
                "TN": ["repoB:0"],
                "FP": [],
                "edge": [],
            },
            "total_episodes": 3,
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest))

        quadrants = load_manifest_quadrants(path)

        assert quadrants["TP"] == ["repoA:0", "repoA:1"]
        assert quadrants["TN"] == ["repoB:0"]


# ---------------------------------------------------------------------------
# compute_task_separation_scores (end-to-end unit)
# ---------------------------------------------------------------------------


class TestComputeTaskSeparationScores:
    def _make_model_data(self, tmp_path: Path, model_name: str, episodes: list[dict]) -> Path:
        out = tmp_path / model_name / "metrics"
        out.mkdir(parents=True)
        (out / "episode_details.json").write_text(json.dumps(episodes))
        return tmp_path / model_name

    def _make_manifest(
        self,
        tmp_path: Path,
        task_name: str,
        episodes_by_q: dict[str, list[str]],
    ) -> Path:
        task_dir = tmp_path / task_name
        task_dir.mkdir(parents=True)
        manifest = {
            "task_name": task_name,
            "quotas": {q: len(eps) for q, eps in episodes_by_q.items()},
            "actual_counts": {q: len(eps) for q, eps in episodes_by_q.items()},
            "episodes": episodes_by_q,
            "total_episodes": sum(len(e) for e in episodes_by_q.values()),
        }
        (task_dir / "manifest.json").write_text(json.dumps(manifest))
        return task_dir / "manifest.json"

    def test_fold_task_tp_vs_tn_separation(self, tmp_path: Path):
        # fold task: TP (fold_success) tail≈0, TN (shuffle_success) tail≈-1
        # Perfect separation → large positive Cohen's d for (TN → TP direction)
        episodes = [
            {"episode_key": "tp1:0", "tail_pred": -0.01, "head_pred": -0.9},
            {"episode_key": "tp2:0", "tail_pred": 0.02, "head_pred": -0.85},
            {"episode_key": "tp3:0", "tail_pred": -0.05, "head_pred": -0.88},
            {"episode_key": "tn1:0", "tail_pred": -0.98, "head_pred": -0.02},
            {"episode_key": "tn2:0", "tail_pred": -0.95, "head_pred": 0.01},
            {"episode_key": "fp1:0", "tail_pred": -0.50, "head_pred": -0.50},
        ]
        model_dir = self._make_model_data(tmp_path, "modelA", episodes)
        manifest_path = self._make_manifest(
            tmp_path,
            "fold",
            {
                "TP": ["tp1:0", "tp2:0", "tp3:0"],
                "TN": ["tn1:0", "tn2:0"],
                "FP": ["fp1:0"],
                "edge": [],
            },
        )

        result = compute_task_separation_scores(
            model_dir=model_dir,
            manifest_path=manifest_path,
        )

        # TP tail≈0, TN tail≈-1 → TP vs TN is perfect separation
        assert "TP_vs_TN" in result["contrasts"]
        d_tp_tn = result["contrasts"]["TP_vs_TN"]["cohen_d"]
        # Sign convention: cohen_d(TN, TP) = (mean_TP - mean_TN) / sd_pooled → positive (TP > TN)
        assert d_tp_tn > 5.0

        # Sample sizes preserved
        assert result["contrasts"]["TP_vs_TN"]["n_a"] == 2  # TN
        assert result["contrasts"]["TP_vs_TN"]["n_b"] == 3  # TP

    def test_missing_model_episodes_are_dropped(self, tmp_path: Path):
        # Manifest references tp3:0 but model never scored it → skip, still report TP n=2
        episodes = [
            {"episode_key": "tp1:0", "tail_pred": -0.01, "head_pred": -0.9},
            {"episode_key": "tp2:0", "tail_pred": 0.02, "head_pred": -0.85},
            {"episode_key": "tn1:0", "tail_pred": -0.98, "head_pred": -0.02},
            {"episode_key": "tn2:0", "tail_pred": -0.95, "head_pred": 0.01},
        ]
        model_dir = self._make_model_data(tmp_path, "modelB", episodes)
        manifest_path = self._make_manifest(
            tmp_path,
            "fold",
            {
                "TP": ["tp1:0", "tp2:0", "tp3:0"],
                "TN": ["tn1:0", "tn2:0"],
                "FP": [],
                "edge": [],
            },
        )
        result = compute_task_separation_scores(
            model_dir=model_dir,
            manifest_path=manifest_path,
        )
        # n_b (TP) should be 2, not 3; missing key tracked
        assert result["contrasts"]["TP_vs_TN"]["n_b"] == 2
        assert "tp3:0" in result["missing_episodes"]
