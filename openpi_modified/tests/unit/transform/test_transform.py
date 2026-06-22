import numpy as np
import pytest

from openpi.shared.normalize import NormStats
import openpi.transforms as _transforms


@pytest.mark.pretrain
@pytest.mark.smoke
def test_semantic_delta_actions_uses_robot_specific_mask() -> None:
    transform = _transforms.SemanticDeltaActions(
        mask_by_robot={"aloha": [True, False, True]},
        default_mask=[False, False, False],
    )
    item = {
        "robot_type": "aloha",
        "state": np.array([1.0, 2.0, 3.0]),
        "actions": np.array([[5.0, 8.0, 9.0]]),
    }

    transformed = transform(item)

    assert np.allclose(transformed["actions"], np.array([[4.0, 8.0, 6.0]]))


@pytest.mark.pretrain
def test_semantic_delta_actions_wraps_angles() -> None:
    transform = _transforms.SemanticDeltaActions(
        mask_by_robot={"aloha": [True, True]},
        angle_indices=[1],
    )
    item = {
        "robot_type": "aloha",
        "state": np.array([0.0, np.pi - 0.1]),
        "actions": np.array([[1.0, -np.pi + 0.1]]),
    }

    transformed = transform(item)

    assert transformed["actions"][0, 0] == pytest.approx(1.0)
    assert transformed["actions"][0, 1] == pytest.approx(0.2, abs=1e-6)


@pytest.mark.pretrain
@pytest.mark.smoke
def test_quantile_normalize_falls_back_to_zscore_for_tiny_quantile_span() -> None:
    stats = {
        "robot": {
            "actions": NormStats(
                mean=np.array([10.0, 5.0]),
                std=np.array([2.0, 5.0]),
                q01=np.array([1.0, 0.0]),
                q99=np.array([1.0 + 1e-9, 10.0]),
            )
        }
    }
    transform = _transforms.Normalize(
        stats,
        use_quantiles=True,
        min_quantile_span=1e-3,
    )

    transformed = transform(
        {
            "robot_type": "robot",
            "actions": np.array([[12.0, 5.0]]),
        }
    )

    assert transformed["actions"][0, 0] == pytest.approx(1.0, abs=1e-6)
    assert transformed["actions"][0, 1] == pytest.approx(0.0, abs=1e-6)
