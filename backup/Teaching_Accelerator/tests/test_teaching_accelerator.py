import numpy as np

from teaching_accelerator.action_targets import build_accelerated_action_indices
from teaching_accelerator.labels import compute_scores_from_actions
from teaching_accelerator.labels import direction_change
from teaching_accelerator.labels import labels_and_strides_from_scores


def test_direction_change_ignores_straight_speed_and_flags_turns():
    velocity = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [0.0, 3.0],
            [-3.0, 0.0],
        ],
        dtype=np.float32,
    )

    turn = direction_change(velocity)

    assert turn[1] == 0.0
    assert turn[2] == 0.0
    assert turn[3] > 0.49
    assert turn[4] > 0.49


def test_high_phase_dispersion_is_casualness_not_precision():
    # First half is consistent across episodes, second half diverges.
    actions_by_episode = {
        0: np.array([[0.0], [0.0], [0.0], [4.0], [4.0], [4.0]], dtype=np.float32),
        1: np.array([[0.0], [0.0], [0.0], [-4.0], [-4.0], [-4.0]], dtype=np.float32),
    }

    scores, _, _ = compute_scores_from_actions(
        actions_by_episode,
        phase_bin_count=2,
        smoothing_half_window=0,
    )

    early_precision = np.mean([scores[0]["precision_score"][:3].mean(), scores[1]["precision_score"][:3].mean()])
    late_precision = np.mean([scores[0]["precision_score"][3:].mean(), scores[1]["precision_score"][3:].mean()])
    early_casual = np.mean([scores[0]["casualness_score"][:3].mean(), scores[1]["casualness_score"][:3].mean()])
    late_casual = np.mean([scores[0]["casualness_score"][3:].mean(), scores[1]["casualness_score"][3:].mean()])

    assert late_casual > early_casual
    assert early_precision > late_precision


def test_labels_drive_piecewise_accelerated_action_indices():
    labels = ["precision", "precision", "casual", "casual", "casual", "neutral", "precision"]

    indices = build_accelerated_action_indices(
        labels,
        start=0,
        horizon=4,
        precision_stride=1,
        neutral_stride=2,
        casual_stride=3,
    )

    assert indices.tolist() == [0, 1, 2, 5]


def test_labels_and_strides_schema_uses_precision_neutral_casual():
    scored = {
        0: {
            "precision_score": np.array([0.9, 0.1, 0.2], dtype=np.float32),
            "casualness_score": np.array([0.1, 0.9, 0.2], dtype=np.float32),
        }
    }

    labels, strides, summary = labels_and_strides_from_scores(
        scored,
        precision_quantile=0.8,
        casual_quantile=0.8,
        precision_stride=1,
        neutral_stride=2,
        casual_stride=4,
        always_precision_head_tail=0,
    )

    assert labels[0] == ["precision", "casual", "neutral"]
    assert strides[0].tolist() == [1, 4, 2]
    assert summary["strides"] == {"precision": 1, "neutral": 2, "casual": 4}
