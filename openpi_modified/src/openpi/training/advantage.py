"""Advantage computation utilities for advantage-conditioned policy training.

This module implements the N-step advantage estimation used in π*0.6 (RECAP method).
Reference: https://arxiv.org/pdf/2511.14759

The key formula is:
    A(o_t, a_t) = Σ_{t'=t}^{t+N-1} r_{t'} + V(o_{t+N}) - V(o_t)

where:
- N is the number of steps to look ahead (default 50 as per paper)
- r_{t'} is the reward at step t'
- V(o_{t+N}) is the value at the N-th step ahead
- V(o_t) is the value at the current step
"""

import jax
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at


def _check_array_for_nan_inf(values: at.Float[at.Array, "..."], name: str = "Values") -> None:
    """Check for NaN/Inf in JAX array before JIT compilation.

    Args:
        values: JAX array to check
        name: Name for error message

    Raises:
        ValueError: If array contains NaN or Inf
    """
    try:
        values_device = jax.device_get(values)
        if np.any(np.isnan(values_device)) or np.any(np.isinf(values_device)):
            raise ValueError(
                f"{name} array contains NaN or Inf. This may indicate: "
                "1) Value model numerical instability, 2) Incorrect normalization statistics, "
                "3) Data preprocessing error."
            )
    except (TypeError, AttributeError):
        pass


def compute_n_step_advantage(
    rewards: at.Float[at.Array, "b n"],
    values: at.Float[at.Array, "b n_plus_1"],
    dones: at.Bool[at.Array, "b n"],
    n_step: int,
) -> at.Float[at.Array, "b"]:  # noqa: F821
    """Compute N-step advantage estimate.

    Args:
        rewards: Rewards for each step, shape [batch, N]
        values: Value estimates for each step (including t+N), shape [batch, N+1]
        dones: Episode termination flags, shape [batch, N]
        n_step: Number of steps to look ahead (N in the formula)

    Returns:
        Advantages for each sample in the batch, shape [batch].

    Formula:
        A(o_t, a_t) = Σ_{t'=0}^{n-1} r_{t'} + V(o_n) - V(o_0)

        When done=True at step i, rewards after step i are ignored and
        no bootstrap value is used.
    """
    batch_size = rewards.shape[0]

    first_done_idx = jnp.argmax(
        jnp.concatenate(
            [dones, jnp.ones((batch_size, 1), dtype=jnp.bool_)],
            axis=-1,
        ),
        axis=-1,
    )

    step_indices = jnp.arange(rewards.shape[1])
    reward_mask = (step_indices < first_done_idx[:, None]).astype(jnp.float32)

    reward_sum = jnp.sum(rewards * reward_mask, axis=-1)

    no_done = jnp.all(~dones, axis=-1)
    done_at_last = (first_done_idx == rewards.shape[1]) | (first_done_idx == rewards.shape[1] - 1)
    can_bootstrap = no_done | done_at_last

    bootstrap_value = jnp.where(
        can_bootstrap,
        values[:, n_step],
        jnp.zeros(batch_size),
    )

    return reward_sum + bootstrap_value - values[:, 0]


def compute_advantage_indicator(
    advantages: at.Float[at.Array, "b"],  # noqa: F821
    threshold: float,
) -> at.Bool[at.Array, "b"]:  # noqa: F821
    """Compute binary advantage indicator.

    Args:
        advantages: Advantage values, shape [batch].
        threshold: Threshold for positive advantage.

    Returns:
        Binary indicators where True indicates positive advantage.
        I_t = 1 if A(o_t, a_t) > threshold else 0
    """
    return advantages > threshold


def compute_advantage_threshold(
    advantages: at.Float[at.Array, "b"],  # noqa: F821
    percentile: float = 30.0,
) -> at.Float[at.Array, ""]:
    """Compute advantage threshold as percentile of advantages.

    Args:
        advantages: Advantage values, shape [batch].
        percentile: Percentile to use (default 30.0 as per paper).

    Returns:
        Threshold value such that approximately `percentile`% of samples
        have positive advantage.
    """
    return jnp.percentile(advantages, 100 - percentile)


def compute_advantages_from_trajectories(
    rewards: at.Float[at.Array, "b t"],
    values: at.Float[at.Array, "b t_plus_1"],
    dones: at.Bool[at.Array, "b t"],
    n_step: int = 50,
) -> at.Float[at.Array, "b t"]:
    """Compute advantages for all timesteps in a batch of trajectories.

    Vectorized implementation using sliding windows for all timesteps.

    Args:
        rewards: Rewards for each timestep, shape [batch, T]
        values: Value estimates, shape [batch, T+1]
        dones: Episode termination flags, shape [batch, T]
        n_step: Number of steps to look ahead

    Returns:
        Advantages for each timestep, shape [batch, T]
    """
    batch_size, trajectory_len = rewards.shape

    _check_array_for_nan_inf(values, "Values")

    values_padded = jnp.pad(
        values,
        ((0, 0), (0, n_step)),
        mode="edge",
    )

    rewards_padded = jnp.pad(
        rewards,
        ((0, 0), (0, n_step)),
        mode="constant",
        constant_values=0.0,
    )
    dones_padded = jnp.pad(
        dones,
        ((0, 0), (0, n_step)),
        mode="constant",
        constant_values=True,
    )

    def create_sliding_windows(x, window_size):
        indices = jnp.arange(trajectory_len)
        return jax.vmap(
            lambda t: jax.lax.dynamic_slice(
                x,
                (0, t),
                (batch_size, window_size),
            ),
        )(indices)

    rewards_windows = create_sliding_windows(rewards_padded, n_step)
    values_windows = create_sliding_windows(values_padded, n_step + 1)
    dones_windows = create_sliding_windows(dones_padded, n_step)

    advantages_per_timestep = jax.vmap(
        lambda r, v, d: compute_n_step_advantage(
            rewards=r,
            values=v,
            dones=d,
            n_step=n_step,
        )
    )(rewards_windows, values_windows, dones_windows)

    return advantages_per_timestep.T


def compute_gae_advantage(
    rewards: at.Float[at.Array, "b t"],
    values: at.Float[at.Array, "b t_plus_1"],
    dones: at.Bool[at.Array, "b t"],
    gamma: float = 0.99,
    lambda_: float = 0.95,
) -> at.Float[at.Array, "b t"]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards for each timestep, shape [batch, T]
        values: Value estimates, shape [batch, T+1]
        dones: Episode termination flags, shape [batch, T]
        gamma: Discount factor (default 0.99)
        lambda_: GAE parameter (default 0.95)

    Returns:
        Advantages for each timestep, shape [batch, T]

    Formula:
        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)  # noqa: RUF002
        A^GAE_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}  # noqa: RUF002
    """
    batch_size, trajectory_len = rewards.shape

    if values.shape != (batch_size, trajectory_len + 1):
        raise ValueError(f"Values shape {values.shape} doesn't match expected ({batch_size}, {trajectory_len + 1}).")
    if dones.shape != (batch_size, trajectory_len):
        raise ValueError(f"Dones shape {dones.shape} doesn't match rewards shape {rewards.shape}.")

    if trajectory_len == 0:
        raise ValueError("Cannot compute GAE on empty trajectory (trajectory_len=0).")

    _check_array_for_nan_inf(values, "Values")

    if not (0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0.0, 1.0], got {gamma}.")
    if not (0.0 <= lambda_ <= 1.0):
        raise ValueError(f"lambda must be in [0.0, 1.0], got {lambda_}.")

    next_values = values[:, 1:] * gamma * (1.0 - dones.astype(jnp.float32))
    td_errors = rewards + next_values - values[:, :-1]

    def gae_scan_fn(carry, t):
        gae_prev = carry
        idx = trajectory_len - 1 - t
        td_error = td_errors[:, idx]
        done = dones[:, idx]
        gae = td_error + lambda_ * gamma * (1.0 - done.astype(jnp.float32)) * gae_prev
        return gae, gae

    init_gae = jnp.zeros(batch_size)
    _, gae_results = jax.lax.scan(gae_scan_fn, init_gae, jnp.arange(trajectory_len))

    return jnp.flip(gae_results, axis=0).T


def clip_advantages(
    advantages: at.Float[at.Array, "b"],  # noqa: F821
    clip_percentile: float = 1.0,
) -> at.Float[at.Array, "b"]:  # noqa: F821
    """Clip extreme advantage values to reduce variance.

    Args:
        advantages: Advantage values, shape [batch]
        clip_percentile: Percentile for clipping (default 1.0%)

    Returns:
        Clipped advantages, shape [batch]
    """
    if advantages.size == 0:
        raise ValueError("Cannot clip empty advantages array.")

    if advantages.shape[0] == 1:
        return advantages

    if clip_percentile != 0.0 and not (0.0 < clip_percentile < 50.0):
        raise ValueError(f"clip_percentile must be 0.0 or in (0, 50), got {clip_percentile}.")

    lower = jnp.percentile(advantages, clip_percentile)
    upper = jnp.percentile(advantages, 100.0 - clip_percentile)
    return jnp.clip(advantages, lower, upper)


def compute_advantages_with_clipping(
    rewards: at.Float[at.Array, "b t"],
    values: at.Float[at.Array, "b t_plus_1"],
    dones: at.Bool[at.Array, "b t"],
    n_step: int = 50,
    clip_percentile: float = 1.0,
) -> at.Float[at.Array, "b t"]:
    """Compute N-step advantages with clipping to reduce variance.

    Args:
        rewards: Rewards for each timestep, shape [batch, T]
        values: Value estimates, shape [batch, T+1]
        dones: Episode termination flags, shape [batch, T]
        n_step: Number of steps to look ahead
        clip_percentile: Percentile for clipping (default 1.0%)

    Returns:
        Clipped advantages for each timestep, shape [batch, T]
    """
    advantages = compute_advantages_from_trajectories(rewards, values, dones, n_step)

    advantages_flat = advantages.reshape(-1)
    lower = jnp.percentile(advantages_flat, clip_percentile)
    upper = jnp.percentile(advantages_flat, 100.0 - clip_percentile)

    return jnp.clip(advantages, lower, upper)


# ============================================================================
# NumPy implementations for use in DataLoader workers (no JAX dependency)
# ============================================================================


def compute_n_step_advantage_numpy(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    n_step: int,
) -> np.ndarray:
    """Pure NumPy N-step advantage computation.

    For offline scripts and DataLoader workers where JAX is not available.

    Args:
        rewards: Shape [T] reward array
        values: Shape [T+1] value array (includes bootstrap value)
        dones: Shape [T] done flag array
        n_step: N-step lookahead window size

    Returns:
        Shape [T] advantage array
    """
    trajectory_len = len(rewards)

    if trajectory_len == 0:
        return np.array([], dtype=np.float32)

    if len(values) < trajectory_len:
        raise ValueError(f"values length ({len(values)}) must be >= rewards length ({trajectory_len})")

    rewards = rewards.astype(np.float32)
    values = values.astype(np.float32)
    dones = dones.astype(bool)

    advantages = np.zeros(trajectory_len, dtype=np.float32)

    for t in range(trajectory_len):
        window_size = min(n_step, trajectory_len - t)

        window_rewards = rewards[t : t + window_size]
        window_dones = dones[t : t + window_size]

        first_done_idx = None
        for i, done in enumerate(window_dones):
            if done:
                first_done_idx = i
                break

        reward_mask = np.ones(window_size, dtype=np.float32)
        if first_done_idx is not None:
            reward_mask[first_done_idx:] = 0.0

        reward_sum = np.sum(window_rewards * reward_mask)

        no_done = first_done_idx is None
        done_at_last = (window_size == n_step) and (first_done_idx == window_size - 1)
        can_bootstrap = no_done or done_at_last

        if can_bootstrap:
            bootstrap_idx = min(t + n_step, len(values) - 1)
            bootstrap_value = values[bootstrap_idx]
        else:
            bootstrap_value = 0.0

        advantages[t] = reward_sum + bootstrap_value - values[t]

    return advantages


def clip_advantages_numpy(
    advantages: np.ndarray,
    clip_percentile: float,
) -> np.ndarray:
    """NumPy advantage clipping, matching JAX clip_advantages behavior.

    Args:
        advantages: Advantage values array
        clip_percentile: Clipping percentile (e.g., 1.0 clips bottom/top 1%)

    Returns:
        Clipped advantage array
    """
    if len(advantages) == 0:
        return advantages

    if clip_percentile <= 0:
        return advantages

    if len(advantages) == 1:
        return advantages

    if np.all(advantages == advantages[0]):
        return advantages

    lower_pct = clip_percentile / 100
    upper_pct = 1.0 - (clip_percentile / 100)

    lower_bound = np.percentile(advantages, lower_pct * 100)
    upper_bound = np.percentile(advantages, upper_pct * 100)

    return np.clip(advantages, lower_bound, upper_bound)
