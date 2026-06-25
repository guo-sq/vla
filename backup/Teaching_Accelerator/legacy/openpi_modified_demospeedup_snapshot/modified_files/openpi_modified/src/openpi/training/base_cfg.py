"""See _CONFIGS for the list of available configs."""

import abc
import ast
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import numpy as np
from typing_extensions import override
import tyro

import openpi.configs.robot_cfg.base as _base_robot_cfg
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.gr00t_policy as gr00t_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.piper_policy as piper_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.frame_attributes_preprocessors as _frame_attrs_preprocessors
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


_RL_NORM_STATS_FILENAME = "rl_norm_stats.json"
_RL_NORM_STATS_FINGERPRINT_FIELDS: tuple[str, ...] = (
    "returns_norm_strategy",
    "returns_norm_percentile",
    "returns_norm_length",
    "exclude_failures",
    "failure_decrease_threshold",
)


def _rl_norm_stats_fingerprint(*, repo_id: str, value_net_cfg: dict[str, Any]) -> str:
    """Mirror of ``scripts.compute_rl_norm_stats._compute_rl_norm_stats_fingerprint``.

    Must stay in lockstep — the loader needs to reproduce the exact hash the
    script writes, so both sides MUST hash the same sorted items.
    """
    relevant = {k: value_net_cfg.get(k) for k in _RL_NORM_STATS_FINGERPRINT_FIELDS}
    payload = {"repo_id": repo_id, "value_net_cfg": sorted(relevant.items())}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_rl_norm_stats(
    *,
    assets_dir: pathlib.Path,
    asset_id: str | list | None,
    repo_id: str | list | None,
    value_net_cfg: dict[str, Any] | None,
) -> dict[str, int] | None:
    """Load precomputed per-repo raw lengths and aggregate them into a global
    ``task_to_norm_length`` dict.

    Only activates for ``returns_norm_strategy == "per_task"``. For other
    strategies the function is a no-op (returns ``None``) so ``per_episode``
    and ``fixed`` training paths are completely untouched.

    Discovery: scans ``{assets_dir}/*/rl_norm_stats.json`` — loads ALL files
    found, regardless of which repo_ids are in the current data config. This
    ensures train and val DataConfigFactory.create() calls (which have
    different repo_id lists) load the same global pinned stats.

    Cross-repo semantics exactly mirror ``MultiRLAnyverseDataset.__init__``
    merge (rl_dataset.py:379-384): raw lengths extend across repos per task,
    THEN the percentile is computed — so storing per-repo percentiles would
    silently differ from the legacy code path.
    """
    if value_net_cfg is None:
        return None
    if value_net_cfg.get("returns_norm_strategy") != "per_task":
        return None
    # repo_id and asset_id are intentionally unused for discovery: we scan
    # the assets_dir for ALL rl_norm_stats.json files so that train and val
    # configs (which have different repo_id lists) get the same pinned stats.
    del repo_id, asset_id

    root = pathlib.Path(str(assets_dir))
    if not root.is_dir():
        return None

    merged_raw_lengths: dict[str, list[int]] = {}
    found_files = 0
    for subdir in sorted(root.iterdir()):
        path = subdir / _RL_NORM_STATS_FILENAME
        if not path.exists():
            continue
        rid = subdir.name
        found_files += 1
        with path.open("r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or "fingerprint" not in payload or "task_to_raw_lengths" not in payload:
            raise ValueError(f"Malformed rl_norm_stats file at {path}: missing required keys")
        stored_fp = str(payload["fingerprint"])
        expected_fp = _rl_norm_stats_fingerprint(repo_id=rid, value_net_cfg=value_net_cfg)
        if stored_fp != expected_fp:
            raise ValueError(
                f"Fingerprint mismatch in {path}: stored={stored_fp}, expected={expected_fp}. "
                f"Re-run: python scripts/compute_rl_norm_stats.py --config <name> --force"
            )
        raw = payload["task_to_raw_lengths"]
        if not isinstance(raw, dict):
            raise ValueError(f"Malformed rl_norm_stats.task_to_raw_lengths at {path}")
        for task, lengths in raw.items():
            merged_raw_lengths.setdefault(task, []).extend(int(x) for x in lengths)

    if not merged_raw_lengths:
        return None

    percentile = float(value_net_cfg.get("returns_norm_percentile", 1.0))
    task_to_norm_length: dict[str, int] = {}
    for task, lengths in merged_raw_lengths.items():
        if not lengths:
            continue
        if percentile >= 1.0:
            task_to_norm_length[task] = int(max(lengths))
        else:
            task_to_norm_length[task] = int(np.percentile(lengths, percentile * 100))
    logging.info(
        "Loaded RL norm stats from %s: %d tasks, %d repos aggregated",
        assets_dir,
        len(task_to_norm_length),
        found_files,
    )
    return task_to_norm_length


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    root_dir: str | None = None
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    split: Literal["train", "val", "test"] = "train"
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    episode_fail: list[list[int]] | None = None
    episode: list[list[int]] | None = None
    dataset_length: list[int] | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | list[dict[str, _transforms.NormStats]] | None = None

    # Used to convert public dataset format to the training format.
    public_dataset_map_transform: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    robot_align_info: _base_robot_cfg.RobotAlignInfo = dataclasses.field(default_factory=_base_robot_cfg.RobotAlignInfo)
    align_dim: int | None = None
    unify_action_space: bool = False
    subtask_info: dict[str, Any] | None = None
    tolerance_s: float = 1e-4

    # Returns normalization configuration
    # - returns_norm_strategy: "per_episode" | "per_task" | "fixed" | "segmented"
    # - returns_norm_percentile: float in (0, 1], only for per_task (1.0=max, 0.9=p90)
    # - returns_norm_length: int, only for fixed strategy (the fixed denominator)
    # - failure_decrease_threshold: float, for two-stage training with pred_value_tensor
    # - segment_values: list[float], only for segmented strategy (e.g. [0.3, 0.7])
    # - segment_values_file: str, only for segmented strategy (json file in meta/, default "segment_values.json")
    value_net_cfg: dict[str, Any] | None = dataclasses.field(
        default_factory=lambda: {
            "returns_norm_strategy": "per_episode",
            "returns_norm_percentile": 1.0,
            "returns_norm_length": None,
            "failure_decrease_threshold": 0.0,
        }
    )

    # RECAP advantage conditioning dropout rate (for AddAdvantageToPrompt transform)
    advantage_dropout_rate: float = 0.3

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will use every episode's task to define the prompt.
    prompt_from_episode: bool = False

    # Pipeline of processors for valid_mask / sample_weight computation at dataset init.
    # When None or empty, all frames valid, weight 1, segment_id 0.
    frame_attributes_preprocessors: list[_frame_attrs_preprocessors.FrameAttributeProcessor] | None = None

    # If true, will use the state as action.
    use_state_as_action: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None
    # parquet dir, used for simultaneous training of data with different processing methods.
    parquet_dir: str = "data"
    # use multiple prompt descriptions to enhance prompt generalization
    use_generalizable_prompt: bool = False
    # Whether to use the repo_sampling_weights map to resample by repo_id
    repo_sampling_weights: dict[str, float] | list[float] | None = None
    # Frame skip interval for downsampling. 1 means no skip, 2 means take every 2nd frame
    frame_skip: int = 1
    # Whether to use lazy loading mode. When enabled, shared cache is automatically used.
    # Shared cache settings can be configured via environment variables:
    #   - OPENPI_SHARED_CACHE_DIR: cache directory (default: /dev/shm/openpi_cache)
    #   - OPENPI_SHARED_CACHE_SIZE_GB: max cache size in GB (default: 200.0)
    lazy_load: bool = False
    # Optional offline per-frame difficulty label jsonl. Relative paths are
    # resolved under each LeRobot repo root, e.g. meta/difficulty_labels.jsonl.
    difficulty_label_file: str | None = None
    difficulty_label_strict: bool = False
    # If True, enforces that sampled action sequences belong to a single continuous segment.
    enforce_segment_continuity: bool = False
    # If True, disables padding of action sequences that extend beyond episode or valid boundaries.
    disable_action_padding: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(
                                model_config.max_token_len, set_zero_state=model_config.set_zero_state
                            ),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05_SUBTASK_FAST:
                assert isinstance(model_config, pi0_config.Pi0Config)
                tokenizer = _tokenizer.FASTTokenizerWithSubtask(
                    max_len=model_config.max_token_len,
                    max_subtask_len=model_config.subtask_max_token_len,
                    encode_subtask=model_config.pi05_with_subtask,
                    encode_actions=model_config.pi05_with_fast_action,
                )
                inputs = [
                    _transforms.InjectDefaultPrompt(self.default_prompt),
                    _transforms.ResizeImages(224, 224),
                    _transforms.TokenizeFASTInputsWithSubtask(tokenizer),
                    _transforms.PadStatesAndActions(model_config.action_dim),
                ]
                outputs = []
                if model_config.pi05_with_subtask:
                    outputs.append(
                        _transforms.DecodeSubtaskFromTokens(tokenizer),
                    )
                if model_config.fast_action_inference and model_config.pi05_with_fast_action:
                    outputs.append(
                        _transforms.ExtractFASTActionsWithSubtask(
                            tokenizer,
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    )
                return _transforms.Group(inputs=inputs, outputs=outputs)
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    root_dir: str | None = None
    # The LeRobot repo id.
    repo_id: str | list[str] = tyro.MISSING
    split: Literal["train", "val", "test"] = "train"
    episode_fail: list[list[int]] | None = None
    episode: list[list[int]] | None = None
    episode_train: list[list[int]] | None = None
    episode_val: list[list[int]] | None = None
    episode_test: list[list[int]] | None = None
    dataset_length: list[int] | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        selected_episode = self._resolve_episode_by_split()
        base = self.base_config or DataConfig()
        value_net_cfg = self._inject_pinned_rl_norm_stats(
            base_value_net_cfg=base.value_net_cfg,
            assets_dir=pathlib.Path(str(self.assets.assets_dir or assets_dirs)),
            asset_id=asset_id,
            repo_id=repo_id,
        )
        return dataclasses.replace(
            base,
            root_dir=self.root_dir,
            repo_id=repo_id,
            split=self.split,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id, repo_id),
            episode=selected_episode,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
            value_net_cfg=value_net_cfg,
        )

    @staticmethod
    def _inject_pinned_rl_norm_stats(
        *,
        base_value_net_cfg: dict[str, Any] | None,
        assets_dir: pathlib.Path,
        asset_id: str | list | None,
        repo_id: str | list | None,
    ) -> dict[str, Any] | None:
        """Load precomputed per-repo raw lengths and stamp the aggregated
        ``task_to_norm_length`` into ``value_net_cfg["pinned_task_to_norm_length"]``.

        Only activates for ``per_task`` strategy and only when every repo has
        a precomputed file; otherwise the field stays unset and the downstream
        strict gate in ``data_loader_rl.create_anyverse_dataset`` will hard-error.
        """
        if base_value_net_cfg is None:
            return None
        pinned = _load_rl_norm_stats(
            assets_dir=assets_dir,
            asset_id=asset_id,
            repo_id=repo_id,
            value_net_cfg=base_value_net_cfg,
        )
        if pinned is None:
            return base_value_net_cfg
        return {**base_value_net_cfg, "pinned_task_to_norm_length": pinned}

    def _resolve_episode_by_split(self) -> list[list[int]] | None:
        if self.split == "train":
            return self.episode_train if self.episode_train is not None else self.episode
        if self.split == "val":
            return self.episode_val if self.episode_val is not None else self.episode
        if self.split == "test":
            return self.episode_test if self.episode_test is not None else self.episode
        return self.episode

    def _load_norm_stats(
        self,
        assets_dir: epath.Path,
        asset_id: str | list | None,
        repo_id: str | list | None,
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            if isinstance(asset_id, list):
                data_assets_dir = str(assets_dir / "_".join(asset_id))
                # norm_stats_list = [_normalize.load(_download.maybe_download(dir)) for dir in data_assets_dir]
                # # print(f"---norm_stats_list in _load_norm_stats------:\n {norm_stats_list}")
                # return norm_stats_list
            else:
                data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class MixDataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: list = dataclasses.field(
        # default=tyro.MISSING,  # 告诉 tyro 该字段是必填项,用户必须手动传入
        default_factory=lambda: [tyro.MISSING]  # 给 dataclass 提供不可变类允许的“列表默认生成方式”
    )
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/front_image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "observation.state",
                        "action": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class Gr00tLerobotDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False
    use_delta_joint_actions: bool = False
    use_semantic_delta_actions: bool = False
    delta_action_mask_indices: Sequence[int] = dataclasses.field(default_factory=lambda: [6, -1, 6, -1])
    delta_wrap_eef_angles: bool = False
    robot_align_info: _base_robot_cfg.RobotAlignInfo = dataclasses.field(default_factory=_base_robot_cfg.RobotAlignInfo)
    align_dim: int = 14
    public_dataset_camera_map: dict[str, str] = dataclasses.field(default_factory=dict)
    target_action_dim: list = dataclasses.field(default_factory=lambda: list(range(14)))
    unify_action_space: bool = False
    tolerance_s: float = 1e-4
    repo_sampling_weights: dict[str, float] | list[float] | None = None
    frame_skip: int = 1
    # Whether to use lazy loading mode. When enabled, shared cache is automatically used.
    # Shared cache settings can be configured via environment variables:
    #   - OPENPI_SHARED_CACHE_DIR: cache directory (default: /dev/shm/openpi_cache)
    #   - OPENPI_SHARED_CACHE_SIZE_GB: max cache size in GB (default: 200.0)
    lazy_load: bool = False
    difficulty_label_file: str | None = None
    difficulty_label_strict: bool = False
    # If True, enforces that sampled action sequences belong to a single continuous segment.
    enforce_segment_continuity: bool = False
    # If True, disables padding of action sequences that extend beyond episode boundaries.
    disable_action_padding: bool = False
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None

    @override
    def create(
        self,
        assets_dirs: pathlib.Path,
        model_config: _model.BaseModelConfig,
        unify_action_mode: bool = False,
        robot_type="bi_piper_follower",
    ) -> DataConfig:
        # Map public observations to those used during training.
        public_dataset_map_transform = _transforms.Group(
            inputs=[_transforms.PublicDatasetMapTransform(self.public_dataset_camera_map)]
        )
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_dict = {
            "observation/front_image": "observation.images.head",
            "observation/wrist_image": "observation.images.right_wrist",
            "observation/wrist_image_lf": "observation.images.left_wrist",
            "observation/third_view_image": "observation.images.third_view",
            "observation/state": "observation.state",
            "action": "action",
            "action_mask": "action_mask",
            "joint_eef_dof_mask": "joint_eef_dof_mask",
            "prompt": "task",
            "robot_type": "robot_type",
            # for episode-frame level visualization
            "frame_index": "frame_index",
            "episode_index": "episode_index",
            "optimality": "optimality",
        }
        if model_config.pi05_subtask_fast and model_config.pi05_with_subtask:
            repack_dict["subtask"] = "subtask"
        if model_config.enable_rl_value_head:
            repack_dict["returns"] = "returns"

        repack_transform = _transforms.Group(inputs=[_transforms.RepackTransform(repack_dict)])

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[
                gr00t_policy.Gr00tLerobotInputs(
                    model_type=model_config.model_type,
                    unify_action_mode=unify_action_mode,
                    robot_type=robot_type,
                )
            ],
            outputs=[gr00t_policy.Gr00tLerobotOutputs(self.target_action_dim)],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        if self.use_delta_joint_actions:
            if self.use_semantic_delta_actions and self.unify_action_space and self.robot_align_info.robot_align_info:
                semantic_masks = self._build_semantic_delta_masks(
                    self.robot_align_info.robot_align_info,
                    self.align_dim,
                )
                angle_indices = (
                    _base_robot_cfg.AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
                    _base_robot_cfg.AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
                    _base_robot_cfg.AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
                    _base_robot_cfg.AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
                    _base_robot_cfg.AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
                    _base_robot_cfg.AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
                )
                data_transforms = data_transforms.push(
                    inputs=[
                        _transforms.SemanticDeltaActions(
                            semantic_masks,
                            angle_indices=(angle_indices if self.delta_wrap_eef_angles else ()),
                        )
                    ],
                    outputs=[
                        _transforms.SemanticAbsoluteActions(
                            semantic_masks,
                            angle_indices=(angle_indices if self.delta_wrap_eef_angles else ()),
                        )
                    ],
                )
            else:
                delta_action_mask = _transforms.make_bool_mask(*self.delta_action_mask_indices)
                data_transforms = data_transforms.push(
                    inputs=[_transforms.DeltaActions(delta_action_mask)],
                    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
                )
        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            public_dataset_map_transform=public_dataset_map_transform,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            robot_align_info=self.robot_align_info,
            align_dim=self.align_dim,
            unify_action_space=self.unify_action_space,
            tolerance_s=self.tolerance_s,
            frame_skip=self.frame_skip,
            lazy_load=self.lazy_load,
            difficulty_label_file=self.difficulty_label_file,
            difficulty_label_strict=self.difficulty_label_strict,
            enforce_segment_continuity=self.enforce_segment_continuity,
            disable_action_padding=self.disable_action_padding,
        )

    def _build_semantic_delta_masks(
        self,
        robot_infos: dict[str, _base_robot_cfg.RobotInfo],
        align_dim: int,
    ) -> dict[str, tuple[bool, ...]]:
        semantic_masks: dict[str, tuple[bool, ...]] = {}
        for robot_type, robot_info in robot_infos.items():
            state_dims: set[int] = set()
            for mapping in robot_info.get_state_name_dict().values():
                state_dims.update(int(v) for v in mapping.values())

            action_dims: set[int] = set()
            for mapping in robot_info.get_action_name_dict().values():
                action_dims.update(int(v) for v in mapping.values())

            overlap_dims = {d for d in (state_dims & action_dims) if 0 <= d < int(align_dim)}
            semantic_masks[robot_type] = tuple(i in overlap_dims for i in range(int(align_dim)))

        return semantic_masks


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class PiperLerobotDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/front_image": "observation.images.head",
                        "observation/wrist_image": "observation.images.right_wrist",
                        "observation/state": "observation.state",
                        "action": "action",
                        "prompt": "task",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[piper_policy.PiperLerobotInputs(model_type=model_config.model_type)],
            outputs=[piper_policy.PiperLerobotOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    rtc_max_delay: int = 0

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 2000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Controls how many hardest action dimensions are logged to wandb.
    # 0 means logging all dimensions; >0 means only top-k by per-dim loss.
    action_dim_loss_topk: int = 0

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # --- Validation configuration ---
    # Dataset repo_id(s) for validation. If None, validation is skipped.
    validation_repo_id: str | list[str] | None = None
    # Root directory for validation dataset. Falls back to data.root_dir if None.
    validation_root_dir: str | None = None
    # Number of batches to run validation on.
    validation_num_batches: int = 10
    # How often (in steps) to run validation.
    validation_interval: int = 5000

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # Optional path to a precomputed norm_stats.json file. If provided and the file exists,
    # compute_norm_stats_fast.py will reuse it instead of recalculating from scratch.
    reuse_norm_stats_path: str | None = None

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


@dataclasses.dataclass(frozen=True)
class TestConfig:
    checkpoint_dir: str
    dataset_root: str
    config: str
    num_batches: int = 2
    batch_size: int | None = None
    vis_dir: str = "./open_loop_vis"
    repo_id: str | None = None
    sample_steps: int | None = None
    num_workers: int = 1
    eval_split: str = "val"
    bucket_fields: tuple[str, ...] = ("repo_id", "robot_type", "task_prompt")
    result_tag: str | None = None


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    targets: tuple[str, ...]
    dataset_root: str
    config: str | None = None
    repo_ids: tuple[str, ...] | None = None
    eval_split: str = "val"
    num_batches: int = 1
    batch_size: int = 1
    num_workers: int = 0
    bucket_fields: tuple[str, ...] = ("repo_id", "robot_type", "task_prompt")
    report_out: str = "outputs/eval_compare/report.json"


class Config:
    def __init__(self, cfg_dict=None):
        if cfg_dict is None:
            cfg_dict = {}
        self._cfg_dict = cfg_dict

    @staticmethod
    def fromfile(filename):
        """从文件加载配置"""
        if filename.endswith(".py"):
            cfg_dict = Config._parse_py_config(filename)
        elif filename.endswith((".yml", ".yaml")):
            import yaml

            with open(filename) as f:
                cfg_dict = yaml.safe_load(f)
        elif filename.endswith(".json"):
            import json

            with open(filename) as f:
                cfg_dict = json.load(f)
        else:
            raise TypeError("只支持 .py, .yml, .yaml, .json 格式的配置文件")

        return Config(cfg_dict)

    @staticmethod
    def _parse_py_config(filename):
        """解析.py配置文件"""
        with open(filename, encoding="utf-8") as f:
            content = f.read()

        # 解析为AST
        module = ast.parse(content)

        # 执行模块获取配置
        cfg_dict = {}
        exec(compile(module, filename=filename, mode="exec"), cfg_dict)

        # 移除不需要的键
        return {k: v for k, v in cfg_dict.items() if not k.startswith("_") and k not in ["__builtins__"]}

    def __getattr__(self, name):
        return self._cfg_dict.get(name)

    def __getitem__(self, name):
        return self._cfg_dict[name]

    def __repr__(self):
        return f"Config({self._cfg_dict})"

    def merge_from_dict(self, cfg_dict):
        """从字典合并配置"""
        self._cfg_dict.update(cfg_dict)
