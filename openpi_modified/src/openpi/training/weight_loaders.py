import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np
import scipy.ndimage

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)

# Vision encoder constants
_DEFAULT_IMAGE_SIZE = (224, 224)
_SIGLIP_PATCH_SIZE = 14  # Patch size for SigLIP So400M/14


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


def _interpolate_pos_embedding(
    pos_embedding: np.ndarray,
    target_size: tuple[int, int],
    source_size: tuple[int, int],
    patch_size: int = _SIGLIP_PATCH_SIZE,
) -> np.ndarray:
    """Interpolate positional embedding from source to target resolution.

    Uses bilinear interpolation to downsample/upsample position embeddings when
    loading checkpoints trained at different image resolutions.

    Args:
        pos_embedding: Position embeddings with shape (1, num_patches, embed_dim)
        target_size: Target image size in pixels (height, width)
        source_size: Source/checkpoint image size in pixels (height, width)
        patch_size: Patch size for the vision encoder (default 14 for SigLIP)

    Returns:
        Interpolated position embeddings with shape (1, target_patches, embed_dim)

    Raises:
        ValueError: If pos_embedding does not have the expected shape or if patch_size is invalid.
    """
    # Validate inputs
    if patch_size <= 0:
        raise ValueError(
            f"patch_size must be positive, got {patch_size}. "
            f"This should match the vision encoder patch size (14 for SigLIP So400M/14)."
        )
    if pos_embedding.ndim != 3:
        raise ValueError(
            f"pos_embedding must be 3D with shape (1, num_patches, embed_dim), "
            f"got {pos_embedding.ndim}D array with shape {pos_embedding.shape}"
        )
    if pos_embedding.shape[0] != 1:
        raise ValueError(
            f"pos_embedding first dimension must be 1 (batch size), got {pos_embedding.shape[0]}"
        )

    _, _, embed_dim = pos_embedding.shape

    # Calculate spatial dimensions in patches
    src_h, src_w = source_size[0] // patch_size, source_size[1] // patch_size
    tgt_h, tgt_w = target_size[0] // patch_size, target_size[1] // patch_size

    # Validate reshape dimensions
    expected_src_patches = src_h * src_w
    actual_src_patches = pos_embedding.shape[1]
    if expected_src_patches != actual_src_patches:
        raise ValueError(
            f"pos_embedding shape mismatch: expected {expected_src_patches} patches "
            f"({src_h}x{src_w} for resolution {source_size} with patch_size={patch_size}), "
            f"got {actual_src_patches} patches. Check checkpoint_image_size configuration."
        )

    # Reshape to 2D spatial: (h, w, embed_dim)
    pos_2d = pos_embedding.reshape(src_h, src_w, embed_dim)

    # Interpolate each channel independently using bilinear interpolation
    interpolated = np.zeros((tgt_h, tgt_w, embed_dim), dtype=pos_embedding.dtype)
    for i in range(embed_dim):
        interpolated[:, :, i] = scipy.ndimage.zoom(
            pos_2d[:, :, i],
            (tgt_h / src_h, tgt_w / src_w),
            order=1,  # Bilinear interpolation
        )

    # Check for NaN/Inf after interpolation
    if np.any(np.isnan(interpolated)) or np.any(np.isinf(interpolated)):
        raise ValueError(
            f"pos_embedding interpolation produced NaN/Inf values. "
            f"This may indicate numerical instability with zoom factors "
            f"({tgt_h/src_h:.4f}, {tgt_w/src_w:.4f})."
        )

    return interpolated.reshape(1, tgt_h * tgt_w, embed_dim)


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"

    For π*0.6 RL value head: When loading from base checkpoints that don't have
    RL value head parameters (score_out_proj, attn_pool, score_mlp_in, score_mlp_out),
    these parameters will be kept as randomly initialized.
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(
            download.maybe_download(self.params_path), restore_type=np.ndarray
        )
        # Add missing LoRA weights and RL value head weights.
        # RL value head parameters: score_out_proj, attn_pool, score_mlp_in, score_mlp_out
        return _merge_params(
            loaded_params,
            params,
            missing_regex=r"(.*lora.*|.*score_out_proj.*|.*attn_pool.*|.*score_mlp_.*)",
        )


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz",
            gs={"token": "anon"},
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {
            "PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")[
                "params"
            ]
        }
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class VLMComponentWeightLoader(WeightLoader):
    """Loads only pre-trained VLM component weights (SigLIP + Gemma) from PaliGemma checkpoint.

    Unlike CheckpointWeightLoader which loads the full π0.5 base model, this loader:
    - Loads only SigLIP vision encoder weights (PaliGemma/img/*)
    - Loads only PaliGemma language model weights (PaliGemma/llm/_0/*), excluding action expert
    - Keeps all other weights (value head, action projections, etc.) from the reference params

    This is useful for training value models from scratch with pre-trained VLM backbones,
    following the approach in https://arxiv.org/pdf/2511.14759 where the value model
    is initialized from pre-trained VLM components rather than the full π0.5 checkpoint.
    """

    paligemma_checkpoint: str = "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz"

    def load(self, params: at.Params) -> at.Params:
        # Load PaliGemma checkpoint which contains SigLIP + Gemma weights
        path = download.maybe_download(self.paligemma_checkpoint, gs={"token": "anon"})
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))

        # Extract only the VLM components (img and llm/_0)
        # The checkpoint has structure: params/PaliGemma/img/* and params/PaliGemma/llm/*
        loaded_params = {"PaliGemma": {}}

        for key, value in flat_params.items():
            # Only include VLM components:
            # - PaliGemma/img/*: SigLIP vision encoder
            # - PaliGemma/llm/_0/*: Main Gemma LLM (not action expert which is llm/_1)
            if key.startswith("params/PaliGemma/img/"):
                # Strip "params/" prefix and add to loaded_params
                new_key = key[len("params/"):]
                loaded_params["PaliGemma"][new_key[len("PaliGemma/"):]] = value
            elif key.startswith("params/PaliGemma/llm/"):
                # Check if this is the main LLM (_0) or action expert (_1)
                # We only want _0 (main LLM), not _1 (action expert)
                if "/llm/_0/" in key or "/llm/embedder" in key:
                    new_key = key[len("params/"):]
                    loaded_params["PaliGemma"][new_key[len("PaliGemma/"):]] = value

        # Convert back to nested structure
        loaded_params["PaliGemma"] = flax.traverse_util.unflatten_dict(
            loaded_params["PaliGemma"], sep="/"
        )

        # Merge with reference params, keeping all non-VLM weights from reference
        # This preserves random init for: value head, action projections, time MLPs, etc.
        return _merge_params(
            loaded_params,
            params,
            # Include all keys that are NOT in PaliGemma/img/* or PaliGemma/llm/_0/*
            # These will be taken from the reference params (random init)
            missing_regex="^(?!PaliGemma/(img/|llm/_0/)).*",
        )


@dataclasses.dataclass(frozen=True)
class SigLipGemma270MWeightLoader(WeightLoader):
    """Loads SigLIP 400M + Gemma-3-270M weights from local npz file.

    This loader extracts the backbone weights (SigLIP + Gemma-270M) and preserves
    the action expert weights (llm_1) from the initialized model.
    """

    params_path: str = "/mnt/model/siglip400m_gemma270m.npz"

    def load(self, params: at.Params) -> at.Params:
        import pathlib

        path = pathlib.Path(self.params_path)
        if not path.exists():
            raise FileNotFoundError(f"Weight file not found: {self.params_path}")

        with open(path, "rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))

        if not flat_params:
            raise ValueError(
                f"Checkpoint file is empty or contains no parameters: {self.params_path}"
            )

        logger.info(f"Loading {len(flat_params)} parameters from {self.params_path}")

        # Restructure to match OpenPI format - remove "params/" prefix
        restructured = {}
        for k, v in flat_params.items():
            if k.startswith("params/"):
                new_key = k[len("params/"):]
            else:
                new_key = k
            restructured[new_key] = v

        loaded_params = {
            "PaliGemma": flax.traverse_util.unflatten_dict(restructured, sep="/")
        }

        # Preserve action expert weights (llm layers with _1 suffix)
        return _merge_params(loaded_params, params, missing_regex=".*llm.*_1.*")


@dataclasses.dataclass(frozen=True)
class T5Gemma2EncoderWeightLoader(WeightLoader):
    """Loads T5Gemma 2 Encoder weights (Vision + SigLIP Head + Gemma 3 270M Encoder LLM).

    This loader loads the converted T5Gemma 2 encoder weights which include:
    - Vision Encoder (SigLIP 400M)
    - Vision Projection (SigLIP head: 1152 -> 640)
    - Encoder LLM (Gemma 3 270M) -> Pi0 Main LLM

    The Gemma 3 specific parameters (q_norm, k_norm, post_attention_norm, post_ffw_norm)
    are also loaded.

    Action Expert weights are preserved from the initialized model.

    If checkpoint_image_size differs from target_image_size, the pos_embedding
    will be interpolated to match the target resolution.
    """

    params_path: str = "/mnt/model/t5gemma2_encoder_openpi.npz"
    # Image resolution used in the checkpoint for pos_embedding training (height, width).
    # Default (224, 224) means no interpolation needed.
    # Set to (896, 896) for T5Gemma 2 encoder checkpoints (64x64 patches).
    checkpoint_image_size: tuple[int, int] = _DEFAULT_IMAGE_SIZE
    # Target image resolution for pos_embedding interpolation (height, width).
    # This should match the model's input_image_size configuration.
    # Default (224, 224) matches standard Pi0/Pi0.5 with 16x16 patches.
    target_image_size: tuple[int, int] = _DEFAULT_IMAGE_SIZE
    # Expected Gemma 3 parameters for validation (q_norm, k_norm per layer)
    num_layers: int = 18  # Gemma-270M has 18 layers

    def load(self, params: at.Params) -> at.Params:
        import pathlib

        path = pathlib.Path(self.params_path)
        if not path.exists():
            raise FileNotFoundError(f"Weight file not found: {self.params_path}")

        with open(path, "rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))

        if not flat_params:
            raise ValueError(
                f"Checkpoint file is empty or contains no parameters: {self.params_path}"
            )

        logger.info(f"Loading {len(flat_params)} parameters from {self.params_path}")

        # Validate Gemma 3 specific parameters
        self._validate_gemma3_params(flat_params)

        # Log parameter categories
        self._log_parameter_summary(flat_params)

        # Restructure to match OpenPI format - remove "params/" prefix
        restructured = {}
        for k, v in flat_params.items():
            if k.startswith("params/"):
                new_key = k[len("params/"):]
            else:
                new_key = k
            restructured[new_key] = v

        # Handle pos_embedding interpolation if checkpoint resolution differs from target
        pos_emb_key = "img/pos_embedding"
        if pos_emb_key in restructured and self.checkpoint_image_size != self.target_image_size:
            pos_emb = restructured[pos_emb_key]
            patch_size = _SIGLIP_PATCH_SIZE  # SigLIP So400M/14

            src_patches = pos_emb.shape[1]
            tgt_patches = (self.target_image_size[0] // patch_size) * (self.target_image_size[1] // patch_size)

            if src_patches != tgt_patches:
                logger.info(
                    f"  Interpolating pos_embedding: "
                    f"{self.checkpoint_image_size} ({src_patches} patches) -> "
                    f"{self.target_image_size} ({tgt_patches} patches)"
                )
                restructured[pos_emb_key] = _interpolate_pos_embedding(
                    pos_emb, self.target_image_size, self.checkpoint_image_size, patch_size
                )

        loaded_params = {
            "PaliGemma": flax.traverse_util.unflatten_dict(restructured, sep="/")
        }

        # Preserve action expert weights (llm layers with _1 suffix) and value head weights
        # This keeps the randomly initialized action expert from the current model
        return _merge_params(
            loaded_params,
            params,
            # Keep: action expert (_1), value head (score_*), LoRA weights
            missing_regex=r"(.*llm.*_1.*|.*score_.*|.*lora.*)",
        )

    def _validate_gemma3_params(self, flat_params: dict) -> None:
        """Validate that expected Gemma 3 parameters are present.

        Args:
            flat_params: Flattened parameter dictionary from checkpoint.
        """
        # Expected Gemma 3 specific parameters
        expected_patterns = [
            ("q_norm", r"params/llm/layers/attn/q_norm", "Query normalization"),
            ("k_norm", r"params/llm/layers/attn/k_norm", "Key normalization"),
            ("post_attention_norm", r"params/llm/layers/post_attention_norm/scale", "Post-attention norm"),
            ("post_ffw_norm", r"params/llm/layers/post_ffw_norm/scale", "Post-FFW norm"),
        ]

        missing_params = []
        for name, pattern, description in expected_patterns:
            # Check if any key matches the pattern
            found = any(re.search(pattern.replace(r"params/", r"params/\d+/"), k) for k in flat_params.keys())
            if not found:
                missing_params.append(f"  - {description} ({name})")

        if missing_params:
            logger.warning(
                f"Gemma 3 parameters may be missing from checkpoint:\n"
                + "\n".join(missing_params) +
                f"\nThis is expected if loading non-Gemma-3 weights."
            )
        else:
            logger.info("All Gemma 3 specific parameters found in checkpoint.")

    def _log_parameter_summary(self, flat_params: dict) -> None:
        """Log a summary of loaded parameters by category.

        Args:
            flat_params: Flattened parameter dictionary from checkpoint.
        """
        categories = {
            "Vision Encoder (SigLIP)": lambda k: k.startswith("params/img/"),
            "Vision Projection": lambda k: "img/head" in k,
            "LLM Embeddings": lambda k: "llm/embedder" in k,
            "LLM Layers (qkv)": lambda k: "llm/layers/attn/" in k and ("q_einsum" in k or "kv_einsum" in k),
            "LLM Layers (output)": lambda k: "llm/layers/attn/attn_vec_einsum" in k,
            "LLM Layers (MLP)": lambda k: "llm/layers/mlp/" in k,
            "Gemma 3 (q/k_norm)": lambda k: any(x in k for x in ["q_norm", "k_norm"]) and "llm/layers" in k,
            "Gemma 3 (post_norms)": lambda k: "post_" in k and "llm/layers" in k,
            "Final Norm": lambda k: "llm/final_norm" in k,
        }

        for category, predicate in categories.items():
            count = sum(1 for k in flat_params.keys() if predicate(k))
            if count > 0:
                logger.info(f"  {category}: {count} parameters")


def _merge_params(
    loaded_params: at.Params, params: at.Params, *, missing_regex: str
) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = (
                v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v
            )

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
