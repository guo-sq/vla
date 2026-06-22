"""Tests for the dual-prompt aggregator.

The aggregator is the scriptable version of the feishu benchmark v4 section
5.2 table. It takes two ``run_benchmark.py`` output dirs (one per prompt) and
produces a comparison report that classifies each (model, quadrant) into one
of five qualitative buckets (aligned / reversed / collapsed / partial /
degenerate). Priority: degenerate > reversed > aligned > collapsed > partial.
"""

from __future__ import annotations

import json

import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.aggregate_dual_prompt import classify_dual_prompt_status
from scripts.benchmark.aggregate_dual_prompt import diff_hang_vs_takeoff
from scripts.benchmark.aggregate_dual_prompt import load_run_outputs
from scripts.benchmark.aggregate_dual_prompt import render_report_markdown

# ---------------------------------------------------------------------------
# Step 1–6: classify_dual_prompt_status
# ---------------------------------------------------------------------------


class TestClassifyDualPromptStatus:
    def test_classify_aligned_strict(self):
        assert classify_dual_prompt_status(0.97, 0.97) == "aligned"

    def test_classify_reversed(self):
        assert classify_dual_prompt_status(0.97, -0.95) == "reversed"

    def test_classify_collapsed(self):
        assert classify_dual_prompt_status(0.97, 0.07) == "collapsed"

    def test_classify_collapsed_mirror(self):
        """``collapsed`` is symmetric in (hang, takeoff): a takeoff-dominant model
        with hang near zero must classify as ``collapsed`` (not ``partial``).
        Untested before; will silently misclassify clothes multitask models that
        train primarily on flatten prompts.
        """
        assert classify_dual_prompt_status(0.07, 0.97) == "collapsed"

    def test_classify_partial(self):
        assert classify_dual_prompt_status(0.9712, 0.3136) == "partial"

    def test_classify_edge_weak_positive(self):
        """Boundary regression: both > 0.5 should be aligned, not partial."""
        assert classify_dual_prompt_status(0.6, 0.6) == "aligned"

    def test_classify_both_nan_is_degenerate(self):
        assert classify_dual_prompt_status(float("nan"), float("nan")) == "degenerate"

    def test_classify_one_nan_is_degenerate(self):
        assert classify_dual_prompt_status(0.9, float("nan")) == "degenerate"
        assert classify_dual_prompt_status(float("nan"), 0.9) == "degenerate"

    def test_classify_both_negative_is_degenerate(self):
        """Model that fails in both directions — no signal."""
        assert classify_dual_prompt_status(-0.5, -0.5) == "degenerate"

    def test_priority_reversed_beats_aligned(self):
        """h=0.6, t=-0.6 — both satisfy "> 0.5" loosely but reversed should win."""
        assert classify_dual_prompt_status(0.6, -0.6) == "reversed"


# ---------------------------------------------------------------------------
# Step 7: load_run_outputs
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_run_dir(tmp_path):
    """Build a minimal cleaned_test_split/ structure with 2 models."""
    base = tmp_path / "cleaned_test_split"
    for model, tp_pearson, tp_mse in [
        ("model_a", 0.9709, 0.0005),
        ("model_b", 0.9694, 0.0008),
    ]:
        mdir = base / model / "metrics"
        mdir.mkdir(parents=True)
        summary = {
            "true_positive": {
                "n_episodes": 293,
                "median_pearson": tp_pearson,
                "median_tail_mse": tp_mse,
            },
            "true_negative": {
                "n_episodes": 361,
                "median_pearson": 0.968,
                "median_tail_mse": 0.0045,
            },
            "false_positive": {
                "n_episodes": 70,
                "median_pearson": 0.969,
                "median_tail_mse": 0.0003,
            },
            "false_negative": {
                "n_episodes": 2,
                "median_pearson": float("nan"),
                "median_tail_mse": float("nan"),
            },
        }
        with open(mdir / "quadrant_summaries.json", "w") as f:
            json.dump(summary, f)
    return base


class TestLoadRunOutputs:
    def test_load_run_outputs(self, fake_run_dir):
        data = load_run_outputs(fake_run_dir)
        assert set(data.keys()) == {"model_a", "model_b"}
        assert data["model_a"]["true_positive"]["median_pearson"] == pytest.approx(0.9709)
        assert data["model_a"]["true_positive"]["median_tail_mse"] == pytest.approx(0.0005)
        assert data["model_b"]["true_positive"]["median_pearson"] == pytest.approx(0.9694)

    def test_load_run_outputs_skips_model_without_summary(self, tmp_path):
        base = tmp_path / "cleaned_test_split"
        # Model has a metrics/ dir but no summary file
        (base / "incomplete_model" / "metrics").mkdir(parents=True)
        # Full model
        mdir = base / "full_model" / "metrics"
        mdir.mkdir(parents=True)
        with open(mdir / "quadrant_summaries.json", "w") as f:
            json.dump({"true_positive": {"n_episodes": 1, "median_pearson": 0.9, "median_tail_mse": 0.01}}, f)
        data = load_run_outputs(base)
        assert "incomplete_model" not in data
        assert "full_model" in data

    def test_load_run_outputs_missing_dir_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            load_run_outputs(missing)


# ---------------------------------------------------------------------------
# Step 8: diff_hang_vs_takeoff + render_report_markdown
# ---------------------------------------------------------------------------


@pytest.fixture
def hang_data():
    return {
        "cross_neg_10k": {
            "true_positive": {"n_episodes": 293, "median_pearson": 0.970905, "median_tail_mse": 0.000453},
            "true_negative": {"n_episodes": 361, "median_pearson": 0.968050, "median_tail_mse": 0.004525},
        },
        "negatives_10k": {
            "true_positive": {"n_episodes": 293, "median_pearson": 0.969400, "median_tail_mse": 0.000800},
            "true_negative": {"n_episodes": 361, "median_pearson": 0.960400, "median_tail_mse": 0.002400},
        },
    }


@pytest.fixture
def takeoff_data():
    return {
        "cross_neg_10k": {
            "true_positive": {"n_episodes": 293, "median_pearson": 0.970800, "median_tail_mse": 0.000500},
            "true_negative": {"n_episodes": 361, "median_pearson": 0.967500, "median_tail_mse": 0.004300},
        },
        "negatives_10k": {
            "true_positive": {"n_episodes": 293, "median_pearson": -0.943200, "median_tail_mse": 0.915800},
            "true_negative": {"n_episodes": 361, "median_pearson": -0.968400, "median_tail_mse": 0.812000},
        },
    }


class TestDiffAndRender:
    def test_diff_contains_expected_rows(self, hang_data, takeoff_data):
        rows = diff_hang_vs_takeoff(hang_data, takeoff_data)
        # 2 models × 3 non-FN quadrants = 6 rows
        assert len(rows) == 6
        cross_tp = [r for r in rows if r["model"] == "cross_neg_10k" and r["quadrant"] == "true_positive"][0]
        assert cross_tp["hang_pearson"] == pytest.approx(0.970905)
        assert cross_tp["takeoff_pearson"] == pytest.approx(0.970800)
        assert cross_tp["status"] == "aligned"
        neg_tp = [r for r in rows if r["model"] == "negatives_10k" and r["quadrant"] == "true_positive"][0]
        assert neg_tp["status"] == "reversed"

    def test_render_markdown_has_header_and_row(self, hang_data, takeoff_data):
        rows = diff_hang_vs_takeoff(hang_data, takeoff_data)
        md = render_report_markdown(rows)
        # header presence
        assert "Hang" in md and "Takeoff" in md and "Status" in md
        # cross_neg TP row with 4-decimal precision and aligned status
        assert "cross_neg_10k" in md
        assert "0.9709" in md  # hang
        assert "0.9708" in md  # takeoff
        assert "aligned" in md
        # negatives TP row should be reversed
        assert "negatives_10k" in md
        assert "-0.9432" in md or "−0.9432" in md
        assert "reversed" in md


# ---------------------------------------------------------------------------
# Step 9: golden match against 飞书 5.2 table
# ---------------------------------------------------------------------------


@pytest.fixture
def feishu_5_2_hang_fixture(tmp_path):
    """Hand-rebuilt fixture with 飞书 5.2 numbers as ground truth."""
    base = tmp_path / "hang"
    models = {
        "negatives_10k": {"tp": (0.9694, 0.0008), "tn": (0.9604, 0.0024), "fp": (0.9763, None)},
        "fp_v2_fixed_5k": {"tp": (0.9694, None), "tn": (0.9684, None), "fp": (0.9678, None)},
        "fp_v2_fixed_10k": {"tp": (0.9712, 0.0003), "tn": (0.9696, 0.0046), "fp": (0.9699, None)},
        "multitask_10k": {"tp": (0.9719, 0.0003), "tn": (0.8670, 0.0003), "fp": (0.9712, None)},
        "cross_neg_10k": {"tp": (0.9709, 0.0005), "tn": (0.9681, 0.0045), "fp": (0.9693, None)},
    }
    for name, q in models.items():
        mdir = base / name / "metrics"
        mdir.mkdir(parents=True)
        payload = {}
        for qkey, (pearson, mse) in [
            ("true_positive", q["tp"]),
            ("true_negative", q["tn"]),
            ("false_positive", q["fp"]),
        ]:
            payload[qkey] = {
                "n_episodes": 1,
                "median_pearson": pearson,
                "median_tail_mse": mse if mse is not None else float("nan"),
            }
        with open(mdir / "quadrant_summaries.json", "w") as f:
            json.dump(payload, f)
    return base


@pytest.fixture
def feishu_5_2_takeoff_fixture(tmp_path):
    base = tmp_path / "takeoff"
    models = {
        "negatives_10k": {"tp": (-0.9432, 0.9158), "tn": (-0.9684, 0.8120), "fp": (-0.9393, None)},
        "fp_v2_fixed_10k": {"tp": (-0.9711, 0.9667), "tn": (-0.9696, 0.8697), "fp": (-0.9699, None)},
        "multitask_10k": {"tp": (0.0715, 0.0017), "tn": (0.9691, 0.0099), "fp": (0.3136, None)},
        "cross_neg_10k": {"tp": (0.9708, 0.0005), "tn": (0.9675, 0.0043), "fp": (0.9671, None)},
    }
    for name, q in models.items():
        mdir = base / name / "metrics"
        mdir.mkdir(parents=True)
        payload = {}
        for qkey, (pearson, mse) in [
            ("true_positive", q["tp"]),
            ("true_negative", q["tn"]),
            ("false_positive", q["fp"]),
        ]:
            payload[qkey] = {
                "n_episodes": 1,
                "median_pearson": pearson,
                "median_tail_mse": mse if mse is not None else float("nan"),
            }
        with open(mdir / "quadrant_summaries.json", "w") as f:
            json.dump(payload, f)
    return base


class TestAggregatorMatchesFeishuSection5_2:
    """Golden test: 4-decimal rounding of fixture data must reproduce 飞书 5.2 exactly."""

    def test_cross_neg_is_aligned_in_both_tp_and_tn(self, feishu_5_2_hang_fixture, feishu_5_2_takeoff_fixture):
        hang = load_run_outputs(feishu_5_2_hang_fixture)
        takeoff = load_run_outputs(feishu_5_2_takeoff_fixture)
        rows = diff_hang_vs_takeoff(hang, takeoff)
        cross_tp = [r for r in rows if r["model"] == "cross_neg_10k" and r["quadrant"] == "true_positive"][0]
        assert cross_tp["status"] == "aligned"
        assert round(cross_tp["hang_pearson"], 4) == 0.9709
        assert round(cross_tp["takeoff_pearson"], 4) == 0.9708
        cross_tn = [r for r in rows if r["model"] == "cross_neg_10k" and r["quadrant"] == "true_negative"][0]
        assert cross_tn["status"] == "aligned"

    def test_negatives_and_fp_v2_are_reversed_in_tp(self, feishu_5_2_hang_fixture, feishu_5_2_takeoff_fixture):
        hang = load_run_outputs(feishu_5_2_hang_fixture)
        takeoff = load_run_outputs(feishu_5_2_takeoff_fixture)
        rows = diff_hang_vs_takeoff(hang, takeoff)
        neg_tp = [r for r in rows if r["model"] == "negatives_10k" and r["quadrant"] == "true_positive"][0]
        fp_tp = [r for r in rows if r["model"] == "fp_v2_fixed_10k" and r["quadrant"] == "true_positive"][0]
        assert neg_tp["status"] == "reversed"
        assert fp_tp["status"] == "reversed"

    def test_multitask_tp_is_collapsed(self, feishu_5_2_hang_fixture, feishu_5_2_takeoff_fixture):
        hang = load_run_outputs(feishu_5_2_hang_fixture)
        takeoff = load_run_outputs(feishu_5_2_takeoff_fixture)
        rows = diff_hang_vs_takeoff(hang, takeoff)
        multitask_tp = [r for r in rows if r["model"] == "multitask_10k" and r["quadrant"] == "true_positive"][0]
        # hang=0.9719, takeoff=0.0715 → |t|<0.3 and h>0.5 → collapsed
        assert multitask_tp["status"] == "collapsed"

    def test_multitask_fp_is_partial(self, feishu_5_2_hang_fixture, feishu_5_2_takeoff_fixture):
        hang = load_run_outputs(feishu_5_2_hang_fixture)
        takeoff = load_run_outputs(feishu_5_2_takeoff_fixture)
        rows = diff_hang_vs_takeoff(hang, takeoff)
        multitask_fp = [r for r in rows if r["model"] == "multitask_10k" and r["quadrant"] == "false_positive"][0]
        # hang=0.9712, takeoff=0.3136 → not aligned (0.3136 < 0.5), not collapsed (0.3136 > 0.3) → partial
        assert multitask_fp["status"] == "partial"

    def test_rendered_markdown_contains_feishu_numbers(self, feishu_5_2_hang_fixture, feishu_5_2_takeoff_fixture):
        hang = load_run_outputs(feishu_5_2_hang_fixture)
        takeoff = load_run_outputs(feishu_5_2_takeoff_fixture)
        rows = diff_hang_vs_takeoff(hang, takeoff)
        md = render_report_markdown(rows)
        # TP cross_neg: +0.9709, +0.9708 aligned
        assert "0.9709" in md
        assert "0.9708" in md
        # TP negatives: +0.9694, -0.9432 reversed
        assert "0.9694" in md
        # hyphen-minus U+002D or unicode minus U+2212 — accept either
        assert ("-0.9432" in md) or ("−0.9432" in md)
        # TP multitask takeoff collapsed
        assert "0.0715" in md
        # TN multitask: hang 0.8670 / takeoff 0.9691
        assert "0.8670" in md
        assert "0.9691" in md


# ---------------------------------------------------------------------------
# Step 10: missing takeoff data → TBD
# ---------------------------------------------------------------------------


class TestHandlesMissingTakeoffForSomeModels:
    def test_model_only_in_hang_renders_tbd(self, tmp_path):
        hang_dir = tmp_path / "hang"
        takeoff_dir = tmp_path / "takeoff"
        # Model with hang data
        for base, models in [
            (hang_dir, {"selfplay_fixed_10k": 0.96, "shared": 0.97}),
            (takeoff_dir, {"shared": 0.95}),
        ]:
            for name, p in models.items():
                mdir = base / name / "metrics"
                mdir.mkdir(parents=True)
                with open(mdir / "quadrant_summaries.json", "w") as f:
                    json.dump(
                        {"true_positive": {"n_episodes": 1, "median_pearson": p, "median_tail_mse": 0.001}},
                        f,
                    )
        hang = load_run_outputs(hang_dir)
        takeoff = load_run_outputs(takeoff_dir)
        rows = diff_hang_vs_takeoff(hang, takeoff)
        sfixed = [r for r in rows if r["model"] == "selfplay_fixed_10k"]
        assert len(sfixed) >= 1
        # Missing takeoff → status = "TBD"
        assert all(r["status"] == "TBD" for r in sfixed)
        assert all(r["takeoff_pearson"] is None for r in sfixed)

    def test_render_tbd_in_markdown(self, tmp_path):
        hang_dir = tmp_path / "hang"
        takeoff_dir = tmp_path / "takeoff"
        mdir = hang_dir / "selfplay_fixed_10k" / "metrics"
        mdir.mkdir(parents=True)
        with open(mdir / "quadrant_summaries.json", "w") as f:
            json.dump({"true_positive": {"n_episodes": 1, "median_pearson": 0.96, "median_tail_mse": 0.001}}, f)
        takeoff_dir.mkdir()  # empty
        hang = load_run_outputs(hang_dir)
        takeoff = load_run_outputs(takeoff_dir)
        rows = diff_hang_vs_takeoff(hang, takeoff)
        md = render_report_markdown(rows)
        assert "TBD" in md
        assert "selfplay_fixed_10k" in md
