"""Golden-value tests for benchmark classification_metrics.

These metrics drive the AUC / F1 / separation numbers reported per role in
benchmark v4 — the same numbers that decide which value-model variant looks
best on cleaned data. Hand-computed expectations cover the perfect-separation,
fully-overlapping, single-class, and tiny-input cases.
"""

import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.classification_metrics import compute_classification_report
from scripts.benchmark.classification_metrics import compute_optimal_threshold
from scripts.benchmark.classification_metrics import compute_separation_score
from scripts.benchmark.classification_metrics import compute_tail_auc


def _episode(success: bool, tail_pred: float) -> dict:
    return {"success": success, "tail_pred": tail_pred}


class TestComputeTailAuc:
    def test_perfect_separation_builder(self):
        # builder: high pred = success
        preds = [-0.9, -0.8, -0.2, -0.1]
        labels = [False, False, True, True]
        auc = compute_tail_auc(preds, labels, role="builder")
        assert auc == pytest.approx(1.0)

    def test_perfect_separation_destroyer(self):
        # destroyer: low pred = success → score = -pred → high after flip
        preds = [-0.9, -0.8, -0.2, -0.1]
        labels = [True, True, False, False]
        auc = compute_tail_auc(preds, labels, role="destroyer")
        assert auc == pytest.approx(1.0)

    def test_inverted_separation_yields_zero(self):
        preds = [-0.1, -0.2, -0.8, -0.9]
        labels = [False, False, True, True]
        auc = compute_tail_auc(preds, labels, role="builder")
        assert auc == pytest.approx(0.0)

    def test_random_overlap_is_half(self):
        # Two interleaved classes with identical mean → AUC = 0.5
        preds = [-0.5, -0.5, -0.5, -0.5]
        labels = [True, False, True, False]
        auc = compute_tail_auc(preds, labels, role="builder")
        assert auc == pytest.approx(0.5)

    def test_degenerate_single_class_returns_nan(self):
        preds = [-0.5, -0.6, -0.7]
        labels = [True, True, True]
        auc = compute_tail_auc(preds, labels, role="builder")
        assert auc != auc  # nan != nan

    def test_degenerate_too_few_samples_returns_nan(self):
        auc = compute_tail_auc([-0.5], [True], role="builder")
        assert auc != auc


class TestComputeOptimalThreshold:
    def test_perfect_separation_picks_correct_split(self):
        preds = [-0.9, -0.8, -0.2, -0.1]
        labels = [False, False, True, True]
        result = compute_optimal_threshold(preds, labels, role="builder")
        assert result["f1"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        # Threshold should land somewhere in the gap (-0.8, -0.2)
        assert -0.8 < result["threshold"] <= -0.2

    def test_destroyer_threshold_uses_le_comparison(self):
        # Destroyer: pred <= threshold means success
        preds = [-0.9, -0.8, -0.2, -0.1]
        labels = [True, True, False, False]
        result = compute_optimal_threshold(preds, labels, role="destroyer")
        assert result["f1"] == pytest.approx(1.0)
        assert -0.8 <= result["threshold"] < -0.2

    def test_completely_overlapping_classes_yield_low_f1(self):
        # Identical preds for both classes → no threshold can separate
        preds = [-0.5, -0.5, -0.5, -0.5]
        labels = [True, False, True, False]
        result = compute_optimal_threshold(preds, labels, role="builder")
        # Best achievable is predicting all positive → P=0.5, R=1, F1=2/3
        assert result["f1"] == pytest.approx(2 / 3)


class TestComputeSeparationScore:
    def test_positive_score_when_success_high(self):
        preds = [-0.9, -0.8, -0.2, -0.1]
        labels = [False, False, True, True]
        sep = compute_separation_score(preds, labels)
        # mean(success) = -0.15, mean(failure) = -0.85 → diff = 0.7
        assert sep == pytest.approx(0.7)

    def test_negative_score_when_inverted(self):
        preds = [-0.1, -0.2, -0.8, -0.9]
        labels = [False, False, True, True]
        sep = compute_separation_score(preds, labels)
        # mean(success) = -0.85, mean(failure) = -0.15 → diff = -0.7
        assert sep == pytest.approx(-0.7)

    def test_empty_class_returns_nan(self):
        preds = [-0.5, -0.6]
        labels = [True, True]
        sep = compute_separation_score(preds, labels)
        assert sep != sep


class TestComputeClassificationReport:
    def test_perfect_separation_full_report(self):
        episodes = [
            _episode(success=False, tail_pred=-0.9),
            _episode(success=False, tail_pred=-0.85),
            _episode(success=True, tail_pred=-0.15),
            _episode(success=True, tail_pred=-0.1),
        ]
        report = compute_classification_report(episodes, role="builder")
        assert report is not None
        assert report["auc"] == pytest.approx(1.0)
        assert report["optimal_threshold"]["f1"] == pytest.approx(1.0)
        assert report["separation_score"] == pytest.approx(0.75)
        assert report["n_success"] == 2
        assert report["n_failure"] == 2

    def test_returns_none_for_single_class(self):
        episodes = [
            _episode(success=True, tail_pred=-0.1),
            _episode(success=True, tail_pred=-0.2),
        ]
        assert compute_classification_report(episodes, role="builder") is None

    def test_returns_none_for_empty_input(self):
        assert compute_classification_report([], role="builder") is None

    def test_returns_none_for_too_few_samples(self):
        episodes = [_episode(success=True, tail_pred=-0.1)]
        assert compute_classification_report(episodes, role="builder") is None

    def test_falls_back_to_pred_array_when_tail_pred_missing(self):
        # Some callers haven't set 'tail_pred' yet, only 'pred'
        episodes = [
            {"success": False, "pred": [-0.9, -0.85, -0.8]},
            {"success": False, "pred": [-0.85, -0.8]},
            {"success": True, "pred": [-0.3, -0.15]},
            {"success": True, "pred": [-0.2, -0.1]},
        ]
        report = compute_classification_report(episodes, role="builder")
        assert report is not None
        assert report["auc"] == pytest.approx(1.0)
