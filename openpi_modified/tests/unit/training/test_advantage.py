"""advantage.py unit tests - N-step, GAE, and NumPy implementations."""

# JAX imports
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training.advantage import _check_array_for_nan_inf
from openpi.training.advantage import clip_advantages
from openpi.training.advantage import clip_advantages_numpy
from openpi.training.advantage import compute_advantage_indicator
from openpi.training.advantage import compute_advantage_threshold
from openpi.training.advantage import compute_advantages_from_trajectories
from openpi.training.advantage import compute_gae_advantage
from openpi.training.advantage import compute_n_step_advantage
from openpi.training.advantage import compute_n_step_advantage_numpy


class TestCheckArrayForNanInf:
    def test_valid_array(self):
        values = jnp.array([1.0, 2.0, 3.0])
        _check_array_for_nan_inf(values)  # Should not raise

    def test_nan_raises(self):
        values = jnp.array([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            _check_array_for_nan_inf(values)

    def test_inf_raises(self):
        values = jnp.array([1.0, float("inf"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            _check_array_for_nan_inf(values)


class TestComputeNStepAdvantage:
    def test_simple_no_done(self):
        """No done flags: advantage = reward_sum + V(N) - V(0)."""
        rewards = jnp.array([[0.0, 0.0, 0.0]])  # [1, 3]
        values = jnp.array([[-0.5, -0.4, -0.3, -0.2]])  # [1, 4]
        dones = jnp.array([[False, False, False]])  # [1, 3]
        result = compute_n_step_advantage(rewards, values, dones, n_step=3)
        # advantage = 0 + V(3) - V(0) = -0.2 - (-0.5) = 0.3
        np.testing.assert_allclose(result, [0.3], atol=1e-5)

    def test_done_at_middle(self):
        """Done at step 1: rewards after done ignored, no bootstrap."""
        rewards = jnp.array([[1.0, 2.0, 3.0]])
        values = jnp.array([[-0.5, -0.4, -0.3, -0.2]])
        dones = jnp.array([[False, True, False]])
        result = compute_n_step_advantage(rewards, values, dones, n_step=3)
        # done at step 1: reward_mask = [1, 0, 0], reward_sum = 1.0
        # can_bootstrap = False (done not at last)
        # advantage = 1.0 + 0 - (-0.5) = 1.5
        np.testing.assert_allclose(result, [1.5], atol=1e-5)

    def test_batch_computation(self):
        """Batch of 2 samples."""
        rewards = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        values = jnp.array([[-0.8, -0.4, -0.2], [-0.6, -0.3, -0.1]])
        dones = jnp.array([[False, False], [False, False]])
        result = compute_n_step_advantage(rewards, values, dones, n_step=2)
        # Sample 0: 0 + V(2) - V(0) = -0.2 - (-0.8) = 0.6
        # Sample 1: 1.0 + V(2) - V(0) = 1.0 + (-0.1) - (-0.6) = 1.5
        np.testing.assert_allclose(result, [0.6, 1.5], atol=1e-5)


class TestComputeAdvantageIndicator:
    def test_basic(self):
        advantages = jnp.array([0.1, -0.5, 0.3, -0.1, 0.0])
        result = compute_advantage_indicator(advantages, threshold=0.0)
        np.testing.assert_array_equal(result, [True, False, True, False, False])


class TestComputeAdvantageThreshold:
    def test_percentile_30(self):
        # 100 values from 0 to 99
        advantages = jnp.arange(100, dtype=jnp.float32)
        threshold = compute_advantage_threshold(advantages, percentile=30.0)
        # Top 30% means threshold at 70th percentile
        assert float(threshold) == pytest.approx(70.0, abs=1.0)


class TestComputeAdvantagesFromTrajectories:
    def test_single_trajectory(self):
        rewards = jnp.zeros((1, 10))
        values = jnp.linspace(-1.0, 0.0, 11).reshape(1, -1)
        dones = jnp.zeros((1, 10), dtype=jnp.bool_)
        result = compute_advantages_from_trajectories(rewards, values, dones, n_step=5)
        assert result.shape == (1, 10)
        # All advantages should be positive (values increasing)
        assert jnp.all(result[:, :5] > 0)


class TestComputeGAEAdvantage:
    def test_basic_gae(self):
        rewards = jnp.array([[0.0, 0.0, 0.0]])
        values = jnp.array([[-0.8, -0.6, -0.3, 0.0]])
        dones = jnp.zeros((1, 3), dtype=jnp.bool_)
        result = compute_gae_advantage(rewards, values, dones, gamma=1.0, lambda_=1.0)
        assert result.shape == (1, 3)
        # With gamma=1, lambda=1: GAE = MC returns - V(t) = V(T) - V(t)
        # For t=0: A = 0 + 0 + 0 + V(3) - V(0) = 0 - (-0.8) = 0.8
        np.testing.assert_allclose(result[0, 0], 0.8, atol=1e-5)

    def test_shape_mismatch_raises(self):
        rewards = jnp.zeros((2, 5))
        values = jnp.zeros((2, 5))  # Should be (2, 6)
        dones = jnp.zeros((2, 5), dtype=jnp.bool_)
        with pytest.raises(ValueError, match="Values shape"):
            compute_gae_advantage(rewards, values, dones)

    def test_empty_trajectory_raises(self):
        rewards = jnp.zeros((1, 0))
        values = jnp.zeros((1, 1))
        dones = jnp.zeros((1, 0), dtype=jnp.bool_)
        with pytest.raises(ValueError, match="empty trajectory"):
            compute_gae_advantage(rewards, values, dones)

    def test_invalid_gamma_raises(self):
        rewards = jnp.zeros((1, 3))
        values = jnp.zeros((1, 4))
        dones = jnp.zeros((1, 3), dtype=jnp.bool_)
        with pytest.raises(ValueError, match="gamma"):
            compute_gae_advantage(rewards, values, dones, gamma=1.5)

    def test_done_resets_gae(self):
        """Done flag should reset GAE accumulation."""
        rewards = jnp.array([[0.0, 0.0, 0.0, 0.0]])
        values = jnp.array([[-0.8, -0.6, -0.4, -0.2, 0.0]])
        dones = jnp.array([[False, True, False, False]])
        result = compute_gae_advantage(rewards, values, dones, gamma=1.0, lambda_=1.0)
        assert result.shape == (1, 4)
        # After done at t=1, GAE should not accumulate from t=0


class TestClipAdvantages:
    def test_basic_clipping(self):
        advantages = jnp.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        result = clip_advantages(advantages, clip_percentile=10.0)
        assert float(jnp.min(result)) >= float(jnp.percentile(advantages, 10.0)) - 1e-5
        assert float(jnp.max(result)) <= float(jnp.percentile(advantages, 90.0)) + 1e-5

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            clip_advantages(jnp.array([]), clip_percentile=1.0)

    def test_single_element(self):
        result = clip_advantages(jnp.array([5.0]), clip_percentile=1.0)
        np.testing.assert_allclose(result, [5.0])

    def test_invalid_percentile_raises(self):
        with pytest.raises(ValueError, match="clip_percentile"):
            clip_advantages(jnp.array([1.0, 2.0]), clip_percentile=60.0)


class TestComputeNStepAdvantageNumpy:
    def test_matches_jax(self):
        """NumPy version should produce similar results to JAX version."""
        np.random.seed(42)
        seq_len = 50
        rewards = np.zeros(seq_len, dtype=np.float32)
        values = np.linspace(-1.0, 0.0, seq_len + 1).astype(np.float32)
        dones = np.zeros(seq_len, dtype=bool)
        n_step = 10

        np_result = compute_n_step_advantage_numpy(rewards, values, dones, n_step)

        # JAX version (single batch)
        jax_rewards = jnp.zeros((1, seq_len))
        jax_values = jnp.linspace(-1.0, 0.0, seq_len + 1).reshape(1, -1)
        jax_dones = jnp.zeros((1, seq_len), dtype=jnp.bool_)
        jax_result = compute_advantages_from_trajectories(jax_rewards, jax_values, jax_dones, n_step)

        np.testing.assert_allclose(np_result, np.array(jax_result[0]), atol=1e-4)

    def test_empty_trajectory(self):
        result = compute_n_step_advantage_numpy(np.array([]), np.array([0.0]), np.array([], dtype=bool), n_step=5)
        assert len(result) == 0

    def test_values_too_short_raises(self):
        with pytest.raises(ValueError, match="values length"):
            compute_n_step_advantage_numpy(
                np.array([1.0, 2.0]),
                np.array([0.5]),  # Too short
                np.array([False, False]),
                n_step=2,
            )


class TestClipAdvantagesNumpy:
    def test_no_clip(self):
        adv = np.array([1.0, 2.0, 3.0])
        result = clip_advantages_numpy(adv, clip_percentile=0.0)
        np.testing.assert_array_equal(result, adv)

    def test_empty(self):
        result = clip_advantages_numpy(np.array([]), clip_percentile=1.0)
        assert len(result) == 0

    def test_single_element(self):
        result = clip_advantages_numpy(np.array([5.0]), clip_percentile=1.0)
        np.testing.assert_array_equal(result, [5.0])

    def test_all_same(self):
        adv = np.array([3.0, 3.0, 3.0])
        result = clip_advantages_numpy(adv, clip_percentile=1.0)
        np.testing.assert_array_equal(result, adv)
