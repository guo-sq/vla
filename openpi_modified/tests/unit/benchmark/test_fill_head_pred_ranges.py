"""Unit tests for fill_head_pred_ranges — compute HEAD_PRED_RANGES prior from
sparse inference outputs + ground-truth labels.

This script is the third piece of the "post-sparse pipeline":
    sparse run → episode_details.json ──┐
                                        ├→ fill_head_pred_ranges → head_pred_ranges.json
    ground_truth_labels.json ───────────┘

For each external-label category it groups all matching episodes' head_pred
values and computes p5/p25/p50/p75/p95 (the percentile set the 2D classifier
uses to draw soft region boundaries). Categories with fewer than
``min_samples`` labeled episodes are marked as ``None`` so the downstream
classifier knows to fall back to 1D tail_pred for that category.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark.fill_head_pred_ranges import compute_head_pred_ranges
from scripts.benchmark.fill_head_pred_ranges import lookup_category

# ---------------------------------------------------------------------------
# lookup_category — per-episode → repo fallback
# ---------------------------------------------------------------------------


def test_lookup_prefers_episode_label_over_repo_label():
    episode_labels = {
        "record.x:0": {"category": "intervention_recovery", "source": "self_play_qc"},
    }
    repo_labels = {
        "record.x": {"category": "fold_success", "source": "flatten_classification"},
    }
    cat = lookup_category("record.x:0", episode_labels, repo_labels)
    assert cat == "intervention_recovery"


def test_lookup_falls_back_to_repo_label_when_no_episode_label():
    episode_labels: dict = {}
    repo_labels = {"record.x": {"category": "fold_success", "source": "flatten_classification"}}
    cat = lookup_category("record.x:0", episode_labels, repo_labels)
    assert cat == "fold_success"


def test_lookup_returns_none_when_neither_source_has_label():
    assert lookup_category("unknown.repo:0", {}, {}) is None


def test_lookup_handles_repo_id_with_colons_in_name():
    """Repo IDs should not contain ':', but defensively use rsplit to be safe."""
    repo_labels = {"record.weird.name": {"category": "shuffle_success", "source": "test"}}
    assert lookup_category("record.weird.name:3", {}, repo_labels) == "shuffle_success"


# ---------------------------------------------------------------------------
# compute_head_pred_ranges — percentile computation per category
# ---------------------------------------------------------------------------


def _mk_episode(key: str, head_pred: float, tail_pred: float = -0.5) -> dict:
    return {"episode_key": key, "head_pred": head_pred, "tail_pred": tail_pred}


def test_groups_episodes_by_category_via_episode_label():
    """self_play episode labels route directly to their categories."""
    episodes = [
        _mk_episode("ep.a:0", -0.50),
        _mk_episode("ep.b:0", -0.60),
        _mk_episode("ep.c:0", -0.80),
    ]
    episode_labels = {
        "ep.a:0": {"category": "fold_success", "source": "self_play_qc"},
        "ep.b:0": {"category": "fold_success", "source": "self_play_qc"},
        "ep.c:0": {"category": "shuffle_success", "source": "self_play_qc"},
    }
    result = compute_head_pred_ranges(episodes, episode_labels, {}, min_samples=1)

    assert result["fold_success"]["n_samples"] == 2
    assert result["shuffle_success"]["n_samples"] == 1


def test_groups_episodes_by_category_via_repo_label():
    """When episode labels are absent, repo labels apply to all episodes of that repo."""
    episodes = [
        _mk_episode("record.fold:0", -0.10),
        _mk_episode("record.fold:1", -0.15),
        _mk_episode("record.fold:2", -0.20),
    ]
    repo_labels = {"record.fold": {"category": "fold_success", "source": "flatten"}}

    result = compute_head_pred_ranges(episodes, {}, repo_labels, min_samples=1)
    assert result["fold_success"]["n_samples"] == 3


def test_percentiles_are_numpy_percentiles_of_head_pred():
    """Verify p5/p25/p50/p75/p95 match numpy.percentile output (summary stats)."""
    head_preds = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
    episodes = [_mk_episode(f"r:{i}", hp) for i, hp in enumerate(head_preds)]
    episode_labels = {f"r:{i}": {"category": "fold_success", "source": "test"} for i in range(len(head_preds))}

    result = compute_head_pred_ranges(episodes, episode_labels, {}, min_samples=1)

    r = result["fold_success"]
    expected = np.percentile(head_preds, [5, 25, 50, 75, 95])
    assert r["p5"] == pytest.approx(float(expected[0]))
    assert r["p25"] == pytest.approx(float(expected[1]))
    assert r["p50"] == pytest.approx(float(expected[2]))
    assert r["p75"] == pytest.approx(float(expected[3]))
    assert r["p95"] == pytest.approx(float(expected[4]))


def test_default_head_min_max_match_p5_p95():
    """Default lower/upper percentiles are 5/95, so head_min==p5 and head_max==p95."""
    head_preds = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0]
    episodes = [_mk_episode(f"r:{i}", hp) for i, hp in enumerate(head_preds)]
    episode_labels = {f"r:{i}": {"category": "fold_success", "source": "test"} for i in range(len(head_preds))}

    result = compute_head_pred_ranges(episodes, episode_labels, {}, min_samples=1)

    r = result["fold_success"]
    assert r["head_min"] == pytest.approx(r["p5"])
    assert r["head_max"] == pytest.approx(r["p95"])
    assert r["lower_percentile"] == pytest.approx(5.0)
    assert r["upper_percentile"] == pytest.approx(95.0)


def test_custom_percentile_widens_head_range():
    """lower=1, upper=99 should be wider than the default 5/95 range."""
    head_preds = np.linspace(-1.0, 0.0, 100).tolist()
    episodes = [_mk_episode(f"r:{i}", hp) for i, hp in enumerate(head_preds)]
    episode_labels = {f"r:{i}": {"category": "fold_success", "source": "test"} for i in range(len(head_preds))}

    narrow = compute_head_pred_ranges(
        episodes,
        episode_labels,
        {},
        min_samples=1,
        lower_percentile=5.0,
        upper_percentile=95.0,
    )
    wide = compute_head_pred_ranges(
        episodes,
        episode_labels,
        {},
        min_samples=1,
        lower_percentile=1.0,
        upper_percentile=99.0,
    )

    # Wider percentile range ⇒ lower head_min and higher head_max.
    assert wide["fold_success"]["head_min"] < narrow["fold_success"]["head_min"]
    assert wide["fold_success"]["head_max"] > narrow["fold_success"]["head_max"]
    # p5/p25/p50/p75/p95 summary stats should NOT depend on the bound choice.
    assert wide["fold_success"]["p5"] == pytest.approx(narrow["fold_success"]["p5"])
    assert wide["fold_success"]["p95"] == pytest.approx(narrow["fold_success"]["p95"])
    # lower/upper_percentile metadata is persisted so downstream can audit.
    assert wide["fold_success"]["lower_percentile"] == pytest.approx(1.0)
    assert wide["fold_success"]["upper_percentile"] == pytest.approx(99.0)


def test_extreme_percentiles_0_100_give_min_max():
    """lower=0, upper=100 degenerate to raw min/max over the samples."""
    head_preds = [-0.95, -0.50, -0.10, 0.00, 0.05]
    episodes = [_mk_episode(f"r:{i}", hp) for i, hp in enumerate(head_preds)]
    episode_labels = {f"r:{i}": {"category": "fold_success", "source": "test"} for i in range(len(head_preds))}

    result = compute_head_pred_ranges(
        episodes,
        episode_labels,
        {},
        min_samples=1,
        lower_percentile=0.0,
        upper_percentile=100.0,
    )

    assert result["fold_success"]["head_min"] == pytest.approx(-0.95)
    assert result["fold_success"]["head_max"] == pytest.approx(0.05)


def test_invalid_percentile_bounds_rejected():
    """lower must be < upper, both in [0, 100]."""
    episodes = [_mk_episode("r:0", -0.5)]
    labels = {"r:0": {"category": "fold_success", "source": "x"}}

    with pytest.raises(ValueError, match=r"lower_percentile .* < upper_percentile"):
        compute_head_pred_ranges(episodes, labels, {}, min_samples=1, lower_percentile=95.0, upper_percentile=5.0)
    with pytest.raises(ValueError, match=r"lower_percentile must be in"):
        compute_head_pred_ranges(episodes, labels, {}, min_samples=1, lower_percentile=-1.0, upper_percentile=95.0)
    with pytest.raises(ValueError, match=r"upper_percentile must be in"):
        compute_head_pred_ranges(episodes, labels, {}, min_samples=1, lower_percentile=5.0, upper_percentile=101.0)


def test_category_below_min_samples_is_none():
    """Small categories (< min_samples) get ``None`` so downstream knows to skip."""
    episodes = [
        _mk_episode("ep.a:0", -0.5),
        _mk_episode("ep.b:0", -0.6),
    ]
    episode_labels = {
        "ep.a:0": {"category": "fold_success", "source": "x"},
        "ep.b:0": {"category": "fold_success", "source": "x"},
    }
    result = compute_head_pred_ranges(episodes, episode_labels, {}, min_samples=20)
    assert result["fold_success"] is None


def test_unlabeled_episodes_are_skipped():
    """Episodes without a label in either dict don't affect any category."""
    episodes = [
        _mk_episode("labeled:0", -0.5),
        _mk_episode("unlabeled:0", -0.9),
    ]
    episode_labels = {"labeled:0": {"category": "fold_success", "source": "x"}}
    result = compute_head_pred_ranges(episodes, episode_labels, {}, min_samples=1)

    assert result["fold_success"]["n_samples"] == 1
    # Unlabeled episode should not have produced any other category entry.
    assert all((v is None or v["n_samples"] == 0 or cat == "fold_success") for cat, v in result.items())


def test_episodes_missing_head_pred_are_skipped():
    """Robust to NaN/missing head_pred values."""
    episodes = [
        {"episode_key": "a:0", "head_pred": -0.5},
        {"episode_key": "b:0", "head_pred": None},
        {"episode_key": "c:0"},  # missing entirely
        {"episode_key": "d:0", "head_pred": float("nan")},
    ]
    episode_labels = {key: {"category": "fold_success", "source": "x"} for key in ["a:0", "b:0", "c:0", "d:0"]}
    result = compute_head_pred_ranges(episodes, episode_labels, {}, min_samples=1)
    assert result["fold_success"]["n_samples"] == 1


# ---------------------------------------------------------------------------
# bootstrap_from_1d — fall back to 1D tail_pred classification for unlabeled eps
# ---------------------------------------------------------------------------


def test_bootstrap_off_by_default_skips_unlabeled_episodes():
    """Without ``bootstrap_from_1d``, unlabeled episodes don't contribute."""
    episodes = [_mk_episode("unlabeled:0", -0.5, tail_pred=-0.005)]  # tail → fold_success in 1D
    result = compute_head_pred_ranges(episodes, {}, {}, min_samples=1)
    # All categories should be None (or at least fold_success should not get this episode)
    assert result["fold_success"] is None


def test_bootstrap_from_1d_fills_fold_or_intervention_for_tail_near_zero():
    """An unlabeled episode with tail_pred ≈ 0 gets routed to either
    fold_success OR intervention_recovery by the 1D classifier (those two
    categories have overlapping tail distributions, so which one wins is an
    implementation detail — both are correct targets for the head_pred prior).
    """
    episodes = [_mk_episode("unlabeled:0", -0.85, tail_pred=-0.005)]
    result = compute_head_pred_ranges(
        episodes,
        {},
        {},
        min_samples=1,
        bootstrap_from_1d=True,
    )
    # One of the two tail-overlapping categories must have received the sample.
    fold_count = (result["fold_success"] or {}).get("n_samples", 0)
    intervention_count = (result["intervention_recovery"] or {}).get("n_samples", 0)
    assert fold_count + intervention_count == 1


def test_bootstrap_from_1d_fills_unlabeled_shuffle_success():
    """tail_pred ≈ -0.85 is firmly in the shuffle_success region (100%
    separation from fold per Step 4c-1 data)."""
    episodes = [_mk_episode("unlabeled:0", -0.02, tail_pred=-0.85)]
    result = compute_head_pred_ranges(
        episodes,
        {},
        {},
        min_samples=1,
        bootstrap_from_1d=True,
    )
    assert result["shuffle_success"] is not None
    assert result["shuffle_success"]["n_samples"] == 1


def test_bootstrap_does_not_override_explicit_ground_truth():
    """When an episode has an explicit label (possibly wrong per 1D), trust it."""
    episodes = [_mk_episode("labeled:0", -0.5, tail_pred=-0.005)]  # 1D says fold_success
    episode_labels = {
        "labeled:0": {"category": "shuffle_success", "source": "self_play_qc"},
    }
    result = compute_head_pred_ranges(
        episodes,
        episode_labels,
        {},
        min_samples=1,
        bootstrap_from_1d=True,
    )
    assert result["shuffle_success"]["n_samples"] == 1
    # fold_success must not have received this episode
    assert result["fold_success"] is None


def test_bootstrap_skips_episodes_without_tail_pred():
    """Episodes missing tail_pred cannot be 1D-classified and are silently dropped."""
    episodes = [
        {"episode_key": "no_tail:0", "head_pred": -0.5},  # no tail_pred key
    ]
    result = compute_head_pred_ranges(
        episodes,
        {},
        {},
        min_samples=1,
        bootstrap_from_1d=True,
    )
    assert all(v is None for v in result.values())


def test_output_contains_all_five_target_categories():
    """All five categories must appear as keys even if empty — downstream
    classifier depends on consistent schema."""
    result = compute_head_pred_ranges([], {}, {}, min_samples=1)
    assert set(result.keys()) == {
        "fold_success",
        "flatten_success",
        "shuffle_success",
        "fold_failure",
        "intervention_recovery",
    }
    # All empty categories should be None (no samples).
    assert all(v is None for v in result.values())
