from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions, static_argnames=("run_subtask_inference",)
            )
            sample_actions_fast = getattr(model, "_sample_actions_fast", None)
            self._sample_actions_fast = (
                nnx_utils.module_jit(sample_actions_fast) if sample_actions_fast is not None else None
            )
            self._fast_action_inference = getattr(model, "fast_action_inference", False)
            # JIT compile score_observation if available
            if hasattr(model, "score_observation"):
                self._score_observation_fn = nnx_utils.module_jit(model.score_observation, static_argnames=("train",))
            else:
                self._score_observation_fn = None
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        infer_delay = int(obs.get("infer_delay", 0))
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        run_subtask_inference = bool(obs.get("run_subtask_inference", False))
        sample_kwargs["run_subtask_inference"] = run_subtask_inference
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)

        batch_size = inputs["state"].shape[0]
        action_shape = (
            batch_size,
            self._model.action_horizon,
            self._model.action_dim,
        )
        if ("actions" not in inputs) or (infer_delay == 0):
            # Create zero action_prefix with shape (batch_size, action_horizon, action_dim)
            if self._is_pytorch_model:
                action_prefix = torch.zeros(action_shape, device=self._pytorch_device)
            else:
                action_prefix = jnp.zeros(action_shape)
        else:
            action_prefix = inputs["actions"]
            assert (
                action_prefix.shape == action_shape
            ), f"action_prefix's shape should be {action_shape}, but got {action_prefix.shape}"

        start_time = time.monotonic()
        if self._fast_action_inference and self._sample_actions_fast is not None:
            result = self._sample_actions_fast(
                sample_rng_or_pytorch_device,
                observation,
                max_decoding_steps=getattr(self._model, "max_decoding_steps", 64),
                temperature=sample_kwargs.get("temperature", 0.0),
            )
            outputs = {"state": inputs["state"], **result}
        else:
            result = self._sample_actions(
                sample_rng_or_pytorch_device,
                observation,
                action_prefix,
                infer_delay,
                **sample_kwargs,
            )
            if isinstance(result, dict) and "actions" in result:
                outputs = {
                    "state": inputs["state"],
                    "actions": result["actions"],
                }
                if "subtask_tokens" in result:
                    outputs["subtask_tokens"] = result["subtask_tokens"]
            else:
                outputs = {"state": inputs["state"], "actions": result}
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        if "robot_type" in obs:
            outputs["robot_type"] = obs["robot_type"]
        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def score_observation(
        self,
        obs: dict,
        *,
        value_temperature: float | None = None,
    ) -> dict:
        """Compute value prediction for the given observation.

        Args:
            obs: Observation dictionary containing images, state, etc.
            value_temperature: Optional temperature for distributional value prediction.
                Only used when model has value_bins > 1. Higher values produce smoother
                distributions. If None, uses model's default value_temperature.

        Returns:
            Dictionary containing:
                - "value": scalar value prediction (expected value for distributional mode)
                - "value_logits": raw logits from model (np.ndarray)
                - "value_metadata": dict with value_bins, value_range, is_distributional
                - "policy_timing": timing info

        Raises:
            ValueError: If value_temperature is not positive.
            NotImplementedError: If model does not support score_observation.
        """
        if value_temperature is not None and value_temperature <= 0:
            raise ValueError(f"value_temperature must be positive, got {value_temperature}")

        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, score_rng = jax.random.split(self._rng)

            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()

            if self._score_observation_fn is None:
                raise NotImplementedError("score_observation not supported by this model")

            value_logits = self._score_observation_fn(score_rng, observation, train=False)
            model_time = time.monotonic() - start_time

            value_logits_np = np.asarray(value_logits[0], dtype=np.float32)
            is_distributional = hasattr(self._model, "value_bins") and self._model.value_bins > 1

            if is_distributional:
                value_logits_jax = jnp.asarray(value_logits[0])
                if hasattr(self._model, "value_distribution_to_scalar"):
                    expected_value = self._model.value_distribution_to_scalar(
                        value_logits_jax[None, ...], temperature=value_temperature
                    )
                    value_np = float(np.asarray(expected_value[0]))
                else:
                    value_range = self._model.value_range
                    vmin, vmax = value_range
                    bin_centers = jnp.linspace(vmin, vmax, self._model.value_bins)
                    probs = jax.nn.softmax(value_logits_jax, axis=-1)
                    value_np = float(np.asarray(jnp.sum(probs * bin_centers, axis=-1)))

                metadata = {
                    "value_bins": self._model.value_bins,
                    "value_range": self._model.value_range,
                    "is_distributional": True,
                }
            else:
                value_np = float(value_logits_np)
                metadata = {
                    "value_bins": 1,
                    "value_range": None,
                    "is_distributional": False,
                }

            return {
                "value": value_np,
                "value_logits": value_logits_np,
                "value_metadata": metadata,
                "policy_timing": {
                    "score_ms": model_time * 1000,
                },
            }

        # PyTorch model path
        if self._is_pytorch_model and hasattr(self._model, "score_observation"):
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...],
                inputs,
            )
            observation = _model.Observation.from_dict(inputs)
            start_time = time.monotonic()

            with torch.no_grad():
                value_logits = self._model.score_observation(observation)

            model_time = time.monotonic() - start_time

            value_logits_np = np.asarray(value_logits[0].detach().cpu(), dtype=np.float32)
            is_distributional = hasattr(self._model, "value_bins") and self._model.value_bins > 1

            if is_distributional:
                if hasattr(self._model, "value_distribution_to_scalar"):
                    value_logits_torch = value_logits[0].detach().cpu()
                    expected_value = self._model.value_distribution_to_scalar(
                        value_logits_torch[None, ...], temperature=value_temperature
                    )
                    value_np = float(np.asarray(expected_value[0]))
                else:
                    value_range = self._model.value_range
                    vmin, vmax = value_range
                    bin_centers = torch.linspace(vmin, vmax, self._model.value_bins)
                    probs = torch.softmax(value_logits[0], dim=-1)
                    value_np = float(torch.sum(probs * bin_centers).detach().cpu())
                metadata = {
                    "value_bins": self._model.value_bins,
                    "value_range": self._model.value_range,
                    "is_distributional": True,
                }
            else:
                value_np = float(value_logits_np)
                metadata = {
                    "value_bins": 1,
                    "value_range": None,
                    "is_distributional": False,
                }

            return {
                "value": value_np,
                "value_logits": value_logits_np,
                "value_metadata": metadata,
                "policy_timing": {
                    "score_ms": model_time * 1000,
                },
            }

        raise NotImplementedError(
            "score_observation is not supported by this model. " "Please use a model with RL value head enabled."
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
