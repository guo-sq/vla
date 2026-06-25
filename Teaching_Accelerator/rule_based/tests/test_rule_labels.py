import json
from pathlib import Path

import numpy as np

from teaching_accelerator.rule_labels import RuleConfig
from teaching_accelerator.rule_labels import compute_rule_labels
from teaching_accelerator.rule_labels import dilate_1d
from teaching_accelerator.rule_labels import merge_boolean_spans
from teaching_accelerator.sidecar import validate_record


def test_phase_dispersion_is_casualness_not_precision():
    base = np.zeros((10, 14), dtype=np.float32)
    divergent_a = np.vstack([base[:5], np.ones((5, 14), dtype=np.float32) * 3.0])
    divergent_b = np.vstack([base[:5], np.ones((5, 14), dtype=np.float32) * -3.0])
    actions = {
        ("repo", 0): divergent_a,
        ("repo", 1): divergent_b,
    }

    result = compute_rule_labels(actions, fps_by_key={("repo", 0): 30, ("repo", 1): 30}, config=RuleConfig(phase_bins=2))
    early_casual = result.scores_by_episode[("repo", 0)]["casualness_score"][:5].mean()
    late_casual = result.scores_by_episode[("repo", 0)]["casualness_score"][5:].mean()
    early_consistency = result.scores_by_episode[("repo", 0)]["phase_consistency_score"][:5].mean()
    late_consistency = result.scores_by_episode[("repo", 0)]["phase_consistency_score"][5:].mean()

    assert late_casual > early_casual
    assert early_consistency > late_consistency


def test_gripper_event_dilation_covers_neighbors():
    values = np.zeros(9, dtype=np.float32)
    values[4] = 1.0

    out = dilate_1d(values, radius=2)

    assert out.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]


def test_span_merge_drop_and_padding():
    mask = np.zeros(60, dtype=bool)
    mask[10:18] = True
    mask[24:32] = True
    mask[50:53] = True

    spans = merge_boolean_spans(mask, fps=30, min_span_frames=15, merge_gap_frames=10, padding_frames=2)

    assert len(spans) == 1
    assert spans[0]["start_frame"] == 8
    assert spans[0]["end_frame"] == 34


def test_sidecar_record_length_validation():
    length = 3
    record = {
        "repo_id": "repo",
        "episode_index": 0,
        "task": ["task"],
        "length": length,
        "fps": 30,
        "hard_score": [0.1] * length,
        "casualness_score": [0.2] * length,
        "phase_consistency_score": [0.3] * length,
        "gripper_event_score": [0.4] * length,
        "turn_score": [0.5] * length,
        "jerk_score": [0.6] * length,
        "coordination_score": [0.7] * length,
        "label": ["neutral"] * length,
        "acceleration_stride": [2] * length,
        "hard_spans": [{"start_frame": 0, "end_frame": 2, "start_s": 0.0, "end_s": 0.0667, "duration_s": 0.0667}],
    }

    validate_record(json.loads(json.dumps(record)))
