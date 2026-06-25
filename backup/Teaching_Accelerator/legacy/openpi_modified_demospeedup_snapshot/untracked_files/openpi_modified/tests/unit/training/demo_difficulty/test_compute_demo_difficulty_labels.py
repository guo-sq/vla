import numpy as np

from openpi.training.demo_difficulty.compute_demo_difficulty_labels import _direction_change


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

    turn = _direction_change(velocity)

    assert turn[1] == 0.0
    assert turn[2] == 0.0
    assert turn[3] > 0.49
    assert turn[4] > 0.49
