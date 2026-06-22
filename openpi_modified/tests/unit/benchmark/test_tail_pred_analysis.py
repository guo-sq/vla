"""Tests for tail_pred_analysis — 1D tail_pred distribution analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.tail_pred_analysis import CategoryStats
from scripts.benchmark.tail_pred_analysis import PairSeparation
from scripts.benchmark.tail_pred_analysis import assign_semantic_label
from scripts.benchmark.tail_pred_analysis import compute_category_stats
from scripts.benchmark.tail_pred_analysis import compute_overlap_pct
from scripts.benchmark.tail_pred_analysis import compute_separation
from scripts.benchmark.tail_pred_analysis import load_benchmark_episodes
from scripts.benchmark.tail_pred_analysis import load_qc_data
from scripts.benchmark.tail_pred_analysis import merge_and_label

# ---------------------------------------------------------------------------
# Fixtures — reusable test data
# ---------------------------------------------------------------------------

MODELS = ["fast_mode_max3600", "1215_0227_max3600", "per_task_p90", "stage2_all_0322"]


def _ep(
    key: str,
    quadrant: str,
    tail_pred: float,
    role: str = "builder",
    *,
    success: bool = True,
    tail_gt: float = 0.0,
) -> dict:
    """Minimal episode detail dict."""
    return {
        "episode_key": key,
        "quadrant": quadrant,
        "role": role,
        "success": success,
        "tail_pred": tail_pred,
        "tail_gt": tail_gt,
        "n_frames": 100,
        "head_mse": 0.01,
        "tail_mse": 0.02,
        "mae": 0.03,
        "rmse": 0.04,
        "pearson": 0.9,
        "r_squared": 0.8,
    }


def _qc(
    key: str,
    status: str = "ok",
    intervention_count: int = 0,
    tail_pred: float | None = -0.5,
    *,
    label_success: bool = True,
    label_role: str = "builder",
) -> dict:
    """Minimal QC entry."""
    return {
        "episode_key": key,
        "status": status,
        "label_success": label_success,
        "label_role": label_role,
        "end_reason": "complete",
        "intervention_count": intervention_count,
        "tail_pred": tail_pred,
        "tail_mse": 0.01,
        "pearson": 0.9,
    }


@pytest.fixture
def expanded_dir(tmp_path: Path) -> Path:
    """Create a minimal expanded benchmark directory structure."""
    for model in MODELS:
        metrics_dir = tmp_path / "expanded" / model / "metrics"
        metrics_dir.mkdir(parents=True)
        episodes = [
            _ep("ep_tp_1", "true_positive", -0.1),
            _ep("ep_tp_2", "true_positive", -0.2),
            _ep("ep_tn_1", "true_negative", -0.8, success=False),
            _ep("ep_tn_2", "true_negative", -0.9, success=False),
            _ep("ep_fp_1", "false_positive", -0.6, success=False),
            _ep("ep_fn_1", "false_negative", -0.4),
        ]
        (metrics_dir / "episode_details.json").write_text(json.dumps(episodes))
    return tmp_path / "expanded"


@pytest.fixture
def error_dir(tmp_path: Path) -> Path:
    """Create a minimal error benchmark directory structure."""
    for model in MODELS:
        metrics_dir = tmp_path / "error" / model / "metrics"
        metrics_dir.mkdir(parents=True)
        episodes = [
            _ep("ep_err_1", "false_positive", -0.5, success=False),
            _ep("ep_err_2", "false_positive", -0.7, success=False),
        ]
        (metrics_dir / "episode_details.json").write_text(json.dumps(episodes))
    return tmp_path / "error"


@pytest.fixture
def qc_path(tmp_path: Path) -> Path:
    """Create a minimal QC JSON file."""
    qc_data = [
        _qc("ep_tp_1", intervention_count=0),
        _qc("ep_tp_2", intervention_count=2),  # has intervention
        _qc("ep_tn_1", intervention_count=0),
        _qc("ep_fn_1", intervention_count=0),
    ]
    path = tmp_path / "self_play_label_qc.json"
    path.write_text(json.dumps(qc_data))
    return path


# ---------------------------------------------------------------------------
# assign_semantic_label
# ---------------------------------------------------------------------------


class TestAssignSemanticLabel:
    def test_intervention_recovery_takes_priority(self):
        """Episode with intervention_count > 0 is 'intervention_recovery' regardless of quadrant."""
        label = assign_semantic_label(quadrant="true_positive", intervention_count=2)
        assert label == "intervention_recovery"

    def test_true_positive_is_fold_success(self):
        label = assign_semantic_label(quadrant="true_positive", intervention_count=0)
        assert label == "fold_success"

    def test_true_negative_is_shuffle_success(self):
        label = assign_semantic_label(quadrant="true_negative", intervention_count=0)
        assert label == "shuffle_success"

    def test_false_positive_is_fold_failure(self):
        label = assign_semantic_label(quadrant="false_positive", intervention_count=0)
        assert label == "fold_failure"

    def test_false_negative_is_other(self):
        label = assign_semantic_label(quadrant="false_negative", intervention_count=0)
        assert label == "other"

    def test_unknown_quadrant_is_other(self):
        label = assign_semantic_label(quadrant="something_else", intervention_count=0)
        assert label == "other"


# ---------------------------------------------------------------------------
# load_benchmark_episodes
# ---------------------------------------------------------------------------


class TestLoadBenchmarkEpisodes:
    def test_loads_all_models(self, expanded_dir: Path):
        result = load_benchmark_episodes(expanded_dir, MODELS)
        assert set(result.keys()) == set(MODELS)

    def test_episode_count_per_model(self, expanded_dir: Path):
        result = load_benchmark_episodes(expanded_dir, MODELS)
        for model in MODELS:
            assert len(result[model]) == 6

    def test_missing_model_raises(self, expanded_dir: Path):
        with pytest.raises(FileNotFoundError):
            load_benchmark_episodes(expanded_dir, ["nonexistent_model"])


# ---------------------------------------------------------------------------
# load_qc_data
# ---------------------------------------------------------------------------


class TestLoadQcData:
    def test_returns_dict_keyed_by_episode(self, qc_path: Path):
        qc = load_qc_data(qc_path)
        assert isinstance(qc, dict)
        assert "ep_tp_1" in qc
        assert "ep_tp_2" in qc

    def test_preserves_intervention_count(self, qc_path: Path):
        qc = load_qc_data(qc_path)
        assert qc["ep_tp_2"]["intervention_count"] == 2

    def test_only_entries_with_tail_pred(self, qc_path: Path):
        """All returned entries should have a non-null tail_pred."""
        qc = load_qc_data(qc_path)
        for entry in qc.values():
            assert entry.get("tail_pred") is not None


# ---------------------------------------------------------------------------
# merge_and_label
# ---------------------------------------------------------------------------


class TestMergeAndLabel:
    def test_deduplicates_episodes_across_expanded_and_error(self, expanded_dir: Path, error_dir: Path, qc_path: Path):
        expanded = load_benchmark_episodes(expanded_dir, MODELS)
        error = load_benchmark_episodes(error_dir, MODELS)
        qc = load_qc_data(qc_path)
        labeled = merge_and_label(expanded["fast_mode_max3600"], error["fast_mode_max3600"], qc)
        keys = [ep["episode_key"] for ep in labeled]
        assert len(keys) == len(set(keys)), "Duplicate episode keys found"

    def test_all_episodes_have_semantic_label(self, expanded_dir: Path, error_dir: Path, qc_path: Path):
        expanded = load_benchmark_episodes(expanded_dir, MODELS)
        error = load_benchmark_episodes(error_dir, MODELS)
        qc = load_qc_data(qc_path)
        labeled = merge_and_label(expanded["fast_mode_max3600"], error["fast_mode_max3600"], qc)
        for ep in labeled:
            assert "semantic_label" in ep
            assert ep["semantic_label"] in {
                "fold_success",
                "shuffle_success",
                "fold_failure",
                "intervention_recovery",
                "other",
            }

    def test_intervention_detected_from_qc(self, expanded_dir: Path, error_dir: Path, qc_path: Path):
        expanded = load_benchmark_episodes(expanded_dir, MODELS)
        error = load_benchmark_episodes(error_dir, MODELS)
        qc = load_qc_data(qc_path)
        labeled = merge_and_label(expanded["fast_mode_max3600"], error["fast_mode_max3600"], qc)
        ep_tp2 = next(ep for ep in labeled if ep["episode_key"] == "ep_tp_2")
        assert ep_tp2["semantic_label"] == "intervention_recovery"

    def test_total_count_matches(self, expanded_dir: Path, error_dir: Path, qc_path: Path):
        expanded = load_benchmark_episodes(expanded_dir, MODELS)
        error = load_benchmark_episodes(error_dir, MODELS)
        qc = load_qc_data(qc_path)
        labeled = merge_and_label(expanded["fast_mode_max3600"], error["fast_mode_max3600"], qc)
        # 6 expanded + 2 error = 8 total (no overlap in our test data)
        assert len(labeled) == 8


# ---------------------------------------------------------------------------
# compute_category_stats
# ---------------------------------------------------------------------------


class TestComputeCategoryStats:
    def test_basic_stats(self):
        values = [-0.1, -0.2, -0.3, -0.4, -0.5]
        stats = compute_category_stats(values)
        assert isinstance(stats, CategoryStats)
        assert stats.n == 5
        assert np.isclose(stats.mean, np.mean(values))
        assert np.isclose(stats.std, np.std(values, ddof=1))
        assert stats.min == -0.5
        assert stats.max == -0.1

    def test_percentiles(self):
        values = list(np.linspace(-1.0, 0.0, 101))
        stats = compute_category_stats(values)
        assert np.isclose(stats.p50, -0.5, atol=0.02)
        assert stats.p5 < stats.p25 < stats.p50 < stats.p75 < stats.p95

    def test_small_sample_warning(self):
        values = [-0.1, -0.2]
        stats = compute_category_stats(values)
        assert stats.warning == "small_sample"

    def test_no_warning_for_large_sample(self):
        values = list(np.linspace(-1.0, 0.0, 20))
        stats = compute_category_stats(values)
        assert stats.warning is None

    def test_single_value(self):
        stats = compute_category_stats([-0.5])
        assert stats.n == 1
        assert stats.mean == -0.5
        assert stats.warning == "small_sample"

    def test_nan_handling(self):
        values = [-0.1, float("nan"), -0.3]
        stats = compute_category_stats(values)
        assert stats.n == 3
        assert not np.isnan(stats.mean)


# ---------------------------------------------------------------------------
# compute_overlap_pct
# ---------------------------------------------------------------------------


class TestComputeOverlapPct:
    def test_no_overlap(self):
        a = [-0.9, -0.8, -0.7]
        b = [-0.3, -0.2, -0.1]
        pct = compute_overlap_pct(a, b)
        assert pct == 0.0

    def test_full_overlap(self):
        a = [-0.5, -0.4, -0.3]
        b = [-0.5, -0.4, -0.3]
        pct = compute_overlap_pct(a, b)
        assert pct == 1.0

    def test_partial_overlap(self):
        a = [-0.7, -0.6, -0.5, -0.4]
        b = [-0.5, -0.4, -0.3, -0.2]
        pct = compute_overlap_pct(a, b)
        assert 0.0 < pct < 1.0

    def test_empty_arrays(self):
        pct = compute_overlap_pct([], [])
        assert pct == 0.0


# ---------------------------------------------------------------------------
# compute_separation
# ---------------------------------------------------------------------------


class TestComputeSeparation:
    def test_separation_fields(self):
        a = [-0.1, -0.2, -0.3]
        b = [-0.7, -0.8, -0.9]
        sep = compute_separation(a, b)
        assert isinstance(sep, PairSeparation)
        assert hasattr(sep, "mean_distance")
        assert hasattr(sep, "overlap_pct")
        assert hasattr(sep, "midpoint_threshold")

    def test_mean_distance_positive(self):
        a = [-0.1, -0.2]
        b = [-0.8, -0.9]
        sep = compute_separation(a, b)
        assert sep.mean_distance > 0

    def test_midpoint_threshold(self):
        a = [-0.1, -0.2]
        b = [-0.8, -0.9]
        sep = compute_separation(a, b)
        expected_midpoint = (np.nanmean(a) + np.nanmean(b)) / 2
        assert np.isclose(sep.midpoint_threshold, expected_midpoint)


# ---------------------------------------------------------------------------
# End-to-end: JSON output structure
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_output_json_structure(self, expanded_dir: Path, error_dir: Path, qc_path: Path, tmp_path: Path):
        """Run the full pipeline on test data and verify JSON output."""
        from scripts.benchmark.tail_pred_analysis import run_analysis

        output_dir = tmp_path / "output"
        run_analysis(
            expanded_dir=expanded_dir,
            error_dir=error_dir,
            qc_path=qc_path,
            output_dir=output_dir,
            primary_model="fast_mode_max3600",
            models=MODELS,
        )

        json_path = output_dir / "tail_pred_1d_analysis.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())

        # metadata
        assert "metadata" in data
        assert data["metadata"]["primary_model"] == "fast_mode_max3600"
        assert data["metadata"]["n_episodes_total"] > 0

        # per_category
        assert "per_category" in data
        for cat_stats in data["per_category"].values():
            assert "n" in cat_stats
            assert "mean" in cat_stats
            assert "p50" in cat_stats

        # separation
        assert "separation" in data
        assert "pairwise" in data["separation"]
        assert "key_thresholds" in data["separation"]

        # model_comparison
        assert "model_comparison" in data

    def test_plots_created(self, expanded_dir: Path, error_dir: Path, qc_path: Path, tmp_path: Path):
        """Verify plot files are written."""
        from scripts.benchmark.tail_pred_analysis import run_analysis

        output_dir = tmp_path / "output"
        run_analysis(
            expanded_dir=expanded_dir,
            error_dir=error_dir,
            qc_path=qc_path,
            output_dir=output_dir,
            primary_model="fast_mode_max3600",
            models=MODELS,
        )

        assert (output_dir / "plots" / "tail_pred_distributions.png").exists()
        assert (output_dir / "plots" / "tail_pred_by_model.png").exists()
