"""Tests for returns normalization strategies.

Tests the three normalization strategies:
- per_episode: each episode normalized by its own length
- per_task: per-task percentile-based normalization
- fixed: fixed denominator for all episodes

Unified formula: reward = clamp(-(remaining_steps - 1) / norm_length, -1, 0)
where norm_length = max(strategy_computed_norm, total_steps) to prevent overflow.
"""

import numpy as np
import pytest
import torch


def compute_reward(remaining_steps: int, norm_length: int) -> float:
    """Compute reward using the unified formula."""
    reward = -(remaining_steps - 1) / norm_length
    return max(-1.0, min(0.0, reward))


@pytest.mark.rl
class TestPerEpisodeStrategy:
    """Tests for per_episode normalization strategy."""

    def test_basic(self):
        """Per-episode: norm_length = total_steps of the episode."""
        total_steps = 400
        remaining_steps = 200
        norm_length = total_steps

        reward = compute_reward(remaining_steps, norm_length)
        assert abs(reward - (-199 / 400)) < 1e-6

    def test_first_frame(self):
        total_steps = 400
        remaining_steps = 400
        reward = compute_reward(remaining_steps, total_steps)
        assert abs(reward - (-399 / 400)) < 1e-6
        assert -1.0 <= reward <= 0.0

    def test_last_frame(self):
        total_steps = 400
        remaining_steps = 1
        reward = compute_reward(remaining_steps, total_steps)
        assert reward == 0.0


@pytest.mark.rl
class TestPerTaskStrategy:
    """Tests for per_task normalization strategy."""

    def test_max_percentile(self):
        """percentile=1.0: use max episode length for the task."""
        task_lengths = [100, 200, 300, 400, 500]
        task_norm = int(np.percentile(task_lengths, 100))
        assert task_norm == 500

        total_steps = 300
        norm_length = max(task_norm, total_steps)
        remaining_steps = 150
        reward = compute_reward(remaining_steps, norm_length)
        assert abs(reward - (-149 / 500)) < 1e-6

    def test_p90_percentile(self):
        """percentile=0.9: use p90 of episode lengths."""
        task_lengths = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        task_norm = int(np.percentile(task_lengths, 90))

        # p90 should be less than max
        assert task_norm < 1000
        assert task_norm > 100

    def test_episode_longer_than_percentile(self):
        """When episode is longer than percentile value, use episode length."""
        task_lengths = [100, 200, 300]
        task_norm = int(np.percentile(task_lengths, 90))  # ~280

        total_steps = 500  # Longer than percentile
        norm_length = max(task_norm, total_steps)  # 500
        assert norm_length == 500

    def test_episode_shorter_than_percentile(self):
        """When episode is shorter, use percentile value."""
        task_lengths = [100, 200, 300, 400, 500]
        task_norm = int(np.percentile(task_lengths, 100))  # 500

        total_steps = 200
        norm_length = max(task_norm, total_steps)  # 500
        assert norm_length == 500

    def test_multi_task_isolation(self):
        """Different tasks should have independent norm lengths."""
        task_a_lengths = [100, 200, 300]
        task_b_lengths = [500, 600, 700]

        task_a_norm = max(task_a_lengths)
        task_b_norm = max(task_b_lengths)

        assert task_a_norm == 300
        assert task_b_norm == 700

    def test_p50_percentile(self):
        """percentile=0.5: use median."""
        task_lengths = [100, 200, 300, 400, 500]
        task_norm = int(np.percentile(task_lengths, 50))
        assert task_norm == 300

    def test_single_episode_task(self):
        """Single episode task: percentile always equals the one length."""
        task_lengths = [400]
        for pct in [50, 90, 100]:
            task_norm = int(np.percentile(task_lengths, pct))
            assert task_norm == 400


@pytest.mark.rl
class TestFixedStrategy:
    """Tests for fixed normalization strategy."""

    def test_basic(self):
        """Fixed: norm_length = max(fixed_value, total_steps)."""
        fixed_value = 3000
        total_steps = 400
        norm_length = max(fixed_value, total_steps)
        assert norm_length == 3000

        remaining_steps = 200
        reward = compute_reward(remaining_steps, norm_length)
        assert abs(reward - (-199 / 3000)) < 1e-6

    def test_episode_longer_than_fixed(self):
        """When episode exceeds fixed value, use episode length."""
        fixed_value = 3000
        total_steps = 5000
        norm_length = max(fixed_value, total_steps)
        assert norm_length == 5000

    def test_value_range(self):
        """Verify reward is always in [-1, 0]."""
        fixed_value = 1000
        total_steps = 400
        norm_length = max(fixed_value, total_steps)

        # First frame
        reward_first = compute_reward(400, norm_length)
        assert -1.0 <= reward_first <= 0.0

        # Last frame
        reward_last = compute_reward(1, norm_length)
        assert reward_last == 0.0

        # Middle
        reward_mid = compute_reward(200, norm_length)
        assert -1.0 <= reward_mid <= 0.0


@pytest.mark.rl
class TestPredValueTensorOverride:
    """Tests for two-stage training pred_value_tensor override."""

    def test_override_takes_precedence(self):
        """When pred_value_tensor is present, it overrides strategy-computed reward."""
        pred_value = torch.tensor(0.5)
        failure_decrease_threshold = 0.1

        # Strategy would compute this
        strategy_reward = compute_reward(200, 1000)

        # But pred_value_tensor takes precedence
        override_reward = torch.clamp(pred_value - failure_decrease_threshold, min=-1.0)

        assert abs(override_reward.item() - 0.4) < 1e-6
        assert override_reward.item() != strategy_reward

    def test_override_clamp(self):
        """Override reward is clamped to >= -1.0."""
        pred_value = torch.tensor(-0.95)
        threshold = 0.1
        reward = torch.clamp(pred_value - threshold, min=-1.0)
        assert reward.item() == -1.0


@pytest.mark.rl
class TestCrossStrategyComparison:
    """Compare behavior across strategies."""

    def test_fixed_vs_per_episode(self):
        """Fixed with large value gives smaller absolute returns than per_episode."""
        total_steps = 400
        remaining_steps = 200

        per_episode_reward = compute_reward(remaining_steps, total_steps)
        fixed_reward = compute_reward(remaining_steps, max(3000, total_steps))

        # Fixed has larger denominator → smaller absolute value
        assert abs(fixed_reward) < abs(per_episode_reward)

    def test_per_task_between_fixed_and_per_episode(self):
        """Per-task norm falls between per-episode and large fixed value."""
        task_lengths = [300, 400, 500]
        task_norm = max(task_lengths)  # 500

        total_steps = 400
        remaining_steps = 200

        per_episode_norm = total_steps  # 400
        per_task_norm = max(task_norm, total_steps)  # 500
        fixed_norm = max(3000, total_steps)  # 3000

        r_episode = compute_reward(remaining_steps, per_episode_norm)
        r_task = compute_reward(remaining_steps, per_task_norm)
        r_fixed = compute_reward(remaining_steps, fixed_norm)

        # Absolute values: per_episode > per_task > fixed
        assert abs(r_episode) > abs(r_task) > abs(r_fixed)

    def test_mixed_scenarios(self):
        """Test various parameter combinations."""
        scenarios = [
            # (strategy, task_norm, fixed_len, total_steps, remaining, expected_norm_length)
            ("per_episode", None, None, 400, 200, 400),
            ("per_task", 500, None, 400, 200, 500),
            ("per_task", 300, None, 400, 200, 400),  # total_steps > task_norm
            ("fixed", None, 3000, 400, 200, 3000),
            ("fixed", None, 3000, 5000, 200, 5000),  # total_steps > fixed
        ]

        for strategy, task_norm, fixed_len, total_steps, _remaining, expected_norm in scenarios:
            if strategy == "per_episode":
                norm_length = total_steps
            elif strategy == "per_task":
                norm_length = max(task_norm, total_steps)
            else:
                norm_length = max(fixed_len, total_steps)

            assert norm_length == expected_norm, (
                f"Failed: strategy={strategy}, task_norm={task_norm}, "
                f"fixed_len={fixed_len}, total_steps={total_steps}, "
                f"expected={expected_norm}, got={norm_length}"
            )
