from collections.abc import Sequence
import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"
    # Vision encoder output dimension (before projection to LLM width).
    # SigLIP-400M outputs 2048 dimensions.
    # - For Gemma-2B: 2048 → 2048 (no projection needed)
    # - For Gemma-270M: 2048 → 640 (img_proj projects 2048 to 640)
    # - For Gemma-300M: 2048 → 320 (img_proj projects 2048 to 320)
    # None means use paligemma_config.width (backward compatible).
    vision_output_dim: int | None = None  # type: ignore
    # Vision encoder input image size for training and inference (height, width).
    # Default 224x224 matches standard Pi0/Pi0.5 with 16x16 patches.
    # This is used by ModelTransformFactory to resize images before encoding.
    input_image_size: tuple[int, int] = (224, 224)
    # Image resolution used in checkpoint for pos_embedding (height, width).
    # Used to interpolate pos_embedding when loading checkpoints trained at
    # different resolutions than input_image_size.
    # Default (224, 224) means no interpolation needed.
    # Set to (896, 896) for T5Gemma 2 encoder checkpoints (64x64 patches).
    checkpoint_image_size: tuple[int, int] = (224, 224)
    # Vocabulary size for the LLM. None uses default PALIGEMMA_VOCAB_SIZE (257152).
    # Set to 262144 for T5Gemma 2 models to preserve full vocabulary.
    vocab_size: int | None = None  # type: ignore

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    image_keys: Sequence[str] = _model.IMAGE_KEYS
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # π0.5 subtask + FAST: token sequence includes Subtask + Action (FAST tokens), with CE loss on both.
    pi05_subtask_fast: bool = False
    pi05_with_subtask: bool = False
    pi05_with_fast_action: bool = False
    # When True and pi05_subtask_fast, infer actions via FAST token decoding instead of flow matching.
    fast_action_inference: bool = False
    # Loss weight for FAST action token CE loss when pi05_subtask_fast=True.
    fast_action_loss_weight: float = 0.5
    # vocab size
    # max token length for subtask tokens
    subtask_max_token_len: int = 32
    # loss weight for subtask token prediction loss. Only used when pi05_subtask is True.
    subtask_loss_weight: float = 1.0
    # loss weight for action prediction loss. Only used when pi05_subtask is True.
    action_loss_weight: float = 1.0
    # If True, subtask is used as action condition (action expert attends to it). Optional for both train and inference.
    # If False, subtask is only used for auxiliary loss; action expert does not attend to it.
    # π0.5 paper: action is conditioned on subtask; use True for paper alignment.
    subtask_as_action_cond: bool = False
    # At inference: whether to generate subtask first and insert into prompt. If subtask_as_action_cond, defaults True.
    # π0.5 paper: infer subtask first, then flow matching; use True for paper alignment.
    infer_action_with_subtask: bool = False
    # π0.5 paper: "high-level inference runs at a lower frequency than low-level action inference."
    # Number of inference calls between subtask updates. 1 = every call (default). N = every N calls.
    # Client passes run_subtask_inference=True every N calls, False otherwise; passes cached subtask when False.
    subtask_inference_interval: int = 1
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    enable_rl_value_head: bool = False
    use_joint_eef_mask: bool = False
    enable_recap: bool = False
    disable_color_aug: bool = False

    # Distributional value prediction (π*0.6)
    # value_bins=1: scalar output with MSE loss (backward compatible)
    # value_bins>1: distributional output with cross-entropy loss
    value_bins: int = 1
    # Range for distributional value bins, typically [-1, 0] for episode progress
    value_range: tuple = (-1.0, 0.0)
    # Label smoothing for value prediction (only for value_bins > 1)
    # 0.0 = no smoothing (one-hot targets)
    # 0.1 = recommended, distributes 10% probability uniformly to other bins
    value_label_smoothing: float = 0.0
    # Temperature scaling for value distribution (only for value_bins > 1)
    # T=1.0: no scaling (default, no effect on distribution)
    # T>1.0: smoother distribution, reduces temporal fluctuation in predictions
    # Recommended: 1.5-2.5 for inference to smooth value curves
    # Note: This does NOT affect training, only inference via value_distribution_to_scalar
    value_temperature: float = 1.0

    def __post_init__(self):
        use_pi05_modalities = self.pi05 or self.pi05_subtask_fast
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if use_pi05_modalities else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", use_pi05_modalities)

        # Match gemma.Module: None means PALIGEMMA_VOCAB_SIZE (needed for pi05_subtask_fast CE / one_hot).
        if self.vocab_size is None:
            object.__setattr__(self, "vocab_size", _gemma.PALIGEMMA_VOCAB_SIZE)

        # Validate image size configurations
        patch_size = 14  # SigLIP So400M/14 patch size
        for name, size in [
            ("input_image_size", self.input_image_size),
            ("checkpoint_image_size", self.checkpoint_image_size),
        ]:
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"{name} must be positive, got {size}")
            if size[0] % patch_size != 0 or size[1] % patch_size != 0:
                raise ValueError(
                    f"{name} must be divisible by patch_size ({patch_size}), got {size}. "
                    f"For SigLIP So400M/14, image dimensions must be multiples of 14."
                )

        # Validate vocab_size
        if self.vocab_size is not None and self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        # Validate π*0.6 distributional value prediction config
        if self.value_bins < 1:
            raise ValueError(f"value_bins must be >= 1, got {self.value_bins}")
        if self.enable_rl_value_head and self.value_bins > 1 and self.value_range[0] >= self.value_range[1]:
            raise ValueError(
                f"value_range must be increasing (min < max) for distributional value prediction, "
                f"got {self.value_range}"
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        if self.pi05_subtask_fast:
            return _model.ModelType.PI05_SUBTASK_FAST
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                    "third_view_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                    "third_view_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                joint_eef_dof_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.bool_)
                    if self.use_joint_eef_mask
                    else None
                ),
                token_ar_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32)
                    if self.pi05_subtask_fast
                    else None
                ),
                token_loss_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_)
                    if self.pi05_subtask_fast
                    else None
                ),
                fast_action_loss_mask=(
                    jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.bool_)
                    if self.pi05_subtask_fast
                    else None
                ),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
