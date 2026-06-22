"""Unit tests for merge_ground_truth_labels — fuse external label sources for
the 2D classifier's HEAD_PRED_RANGES prior.

Only **high-confidence external sources** feed HEAD_PRED_RANGES. Self-classified
tail_pred labels would be circular (we're trying to *improve* that classifier),
so they are deliberately excluded.

Label sources, from highest to lowest precedence:
  1. self_play_label_qc.json — per-episode QC'd labels (93.4% consistency)
     (role=builder + success/intervention combinations; role=destroyer for shuffle)
  2. flatten_classification.json — per-repo rule-based labels
     (only flatten high/medium and disarrange high are accepted)

Per-episode labels win over per-repo labels when both exist, because self_play
metadata reflects ground-truth annotation while flatten_classification is only a
rule-based pattern match on task descriptions.
"""

from __future__ import annotations

from scripts.benchmark.merge_ground_truth_labels import merge_labels

# ---------------------------------------------------------------------------
# self_play_label_qc.json → episode-level labels
# ---------------------------------------------------------------------------


def test_self_play_builder_success_no_intervention_is_fold_success():
    self_play = [{"episode_key": "self_play.x.0:0", "role": "builder", "success": True, "intervention_count": 0}]
    result = merge_labels(self_play, [])
    assert result["episode_labels"]["self_play.x.0:0"]["category"] == "fold_success"
    assert result["episode_labels"]["self_play.x.0:0"]["source"] == "self_play_qc"


def test_self_play_builder_success_with_intervention_is_recovery():
    self_play = [{"episode_key": "self_play.x.1:0", "role": "builder", "success": True, "intervention_count": 3}]
    result = merge_labels(self_play, [])
    assert result["episode_labels"]["self_play.x.1:0"]["category"] == "intervention_recovery"


def test_self_play_destroyer_success_is_shuffle_success():
    self_play = [{"episode_key": "self_play.x.2:0", "role": "destroyer", "success": True, "intervention_count": 0}]
    result = merge_labels(self_play, [])
    assert result["episode_labels"]["self_play.x.2:0"]["category"] == "shuffle_success"


def test_self_play_builder_failure_is_fold_failure():
    self_play = [{"episode_key": "self_play.x.3:0", "role": "builder", "success": False, "intervention_count": 0}]
    result = merge_labels(self_play, [])
    assert result["episode_labels"]["self_play.x.3:0"]["category"] == "fold_failure"


def test_self_play_unknown_role_is_skipped():
    """``role='unknown'`` carries no category signal and is dropped in strict mode."""
    self_play = [{"episode_key": "self_play.x.4:0", "role": "unknown", "success": True, "intervention_count": 0}]
    result = merge_labels(self_play, [])
    assert "self_play.x.4:0" not in result["episode_labels"]


def test_self_play_missing_role_field_is_skipped():
    """Older v0401 QC entries lack the ``role`` key entirely — drop them."""
    self_play = [{"episode_key": "self_play.x.5:0", "success": True, "intervention_count": 0}]
    result = merge_labels(self_play, [])
    assert "self_play.x.5:0" not in result["episode_labels"]


# ---------------------------------------------------------------------------
# flatten_classification.json → repo-level labels
# ---------------------------------------------------------------------------


def test_flatten_high_confidence_is_flatten_success():
    flatten = [{"repo_id": "record.flat.1", "final_label": "flatten", "confidence": "high"}]
    result = merge_labels([], flatten)
    assert result["repo_labels"]["record.flat.1"]["category"] == "flatten_success"
    assert result["repo_labels"]["record.flat.1"]["source"] == "flatten_classification"


def test_flatten_medium_confidence_also_accepted():
    """27 high + 10 medium = 37 flatten repos; we accept both tiers."""
    flatten = [{"repo_id": "record.flat.2", "final_label": "flatten", "confidence": "medium"}]
    result = merge_labels([], flatten)
    assert result["repo_labels"]["record.flat.2"]["category"] == "flatten_success"


def test_flatten_low_confidence_is_skipped():
    flatten = [{"repo_id": "record.flat.3", "final_label": "flatten", "confidence": "low"}]
    result = merge_labels([], flatten)
    assert "record.flat.3" not in result["repo_labels"]


def test_disarrange_high_confidence_is_shuffle_success():
    flatten = [{"repo_id": "r_policy.dis.1", "final_label": "disarrange", "confidence": "high"}]
    result = merge_labels([], flatten)
    assert result["repo_labels"]["r_policy.dis.1"]["category"] == "shuffle_success"


def test_fold_high_confidence_is_fold_success():
    """``final_label=fold + confidence=high`` comes from flatten_classifier's
    string-rule match on ``task_text`` (not a value-model prediction), so it is
    safe to treat as fold_success for HEAD_PRED_RANGES sampling. This broadens
    coverage from ~15 self_play episodes to hundreds of record.* episodes.
    """
    flatten = [{"repo_id": "record.fold.hi", "final_label": "fold", "confidence": "high"}]
    result = merge_labels([], flatten)
    assert result["repo_labels"]["record.fold.hi"]["category"] == "fold_success"


def test_fold_medium_confidence_is_skipped():
    """``fold + medium`` still has lower rule precision; skip to avoid polluting
    the fold_success head_pred distribution."""
    flatten = [{"repo_id": "record.fold.med", "final_label": "fold", "confidence": "medium"}]
    result = merge_labels([], flatten)
    assert "record.fold.med" not in result["repo_labels"]


def test_fold_low_confidence_is_skipped():
    flatten = [{"repo_id": "record.fold.lo", "final_label": "fold", "confidence": "low"}]
    result = merge_labels([], flatten)
    assert "record.fold.lo" not in result["repo_labels"]


def test_bimodal_label_is_skipped():
    """``bimodal`` explicitly means 'we don't know' — skip for HEAD_PRED_RANGES."""
    flatten = [{"repo_id": "record.bi.1", "final_label": "bimodal", "confidence": "low"}]
    result = merge_labels([], flatten)
    assert "record.bi.1" not in result["repo_labels"]


def test_non_task_label_is_skipped():
    """Non-task repos (reset_arm etc.) have no valid category."""
    flatten = [{"repo_id": "record.nt.1", "final_label": "non_task", "confidence": "high"}]
    result = merge_labels([], flatten)
    assert "record.nt.1" not in result["repo_labels"]


# ---------------------------------------------------------------------------
# Cross-source and degenerate cases
# ---------------------------------------------------------------------------


def test_episode_label_does_not_overwrite_repo_label_on_different_repo():
    """Episode labels and repo labels live in different buckets; no interference."""
    self_play = [{"episode_key": "self_play.x.6:0", "role": "builder", "success": True, "intervention_count": 0}]
    flatten = [{"repo_id": "record.flat.6", "final_label": "flatten", "confidence": "high"}]
    result = merge_labels(self_play, flatten)
    assert result["episode_labels"]["self_play.x.6:0"]["category"] == "fold_success"
    assert result["repo_labels"]["record.flat.6"]["category"] == "flatten_success"


def test_empty_inputs_produce_empty_output():
    result = merge_labels([], [])
    assert result == {"episode_labels": {}, "repo_labels": {}}
