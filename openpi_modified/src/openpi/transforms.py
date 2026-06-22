from collections.abc import Callable, Mapping, Sequence
import dataclasses
import logging
import random
import re
from typing import Protocol, TypeAlias, TypeVar, runtime_checkable

import flax.traverse_util as traverse_util
import jax
import numpy as np
from openpi_client import image_tools

from openpi.models import tokenizer as _tokenizer
from openpi.shared import array_typing as at
from openpi.shared import normalize as _normalize

DataDict: TypeAlias = at.PyTree
NormStats: TypeAlias = _normalize.NormStats

logger = logging.getLogger("openpi")


T = TypeVar("T")
S = TypeVar("S")


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """


@dataclasses.dataclass(frozen=True)
class Group:
    """A group of transforms."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(
        self,
        *,
        inputs: Sequence[DataTransformFn] = (),
        outputs: Sequence[DataTransformFn] = (),
    ) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class PublicDatasetMapTransform(DataTransformFn):
    """Convert a public dataset dictionary into a training format dictionary."""

    key_mapping: Mapping[str, str]

    def __call__(self, data: DataDict) -> DataDict:

        return {self.key_mapping.get(k, k): v for k, v in data.items()}


@dataclasses.dataclass(frozen=True)
class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary.

    Repacking is defined using a dictionary where the keys are the new keys and the values
    are the flattened paths to the old keys. We use '/' as the separator during flattening.

    Example:
    {
        "images": {
            "cam_high": "observation.images.top",
            "cam_low": "observation.images.bottom",
        },
        "state": "observation.state",
        "actions": "action",
    }
    """

    structure: at.PyTree[str]

    def __call__(self, data: DataDict) -> DataDict:
        flat_item = flatten_dict(data)
        flat_structure = flatten_dict(self.structure)
        structure_update = {key: value for key, value in flat_structure.items() if value in flat_item}
        return jax.tree.map(lambda k: flat_item[k], structure_update)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data


@dataclasses.dataclass(frozen=True)
class Normalize(DataTransformFn):
    norm_stats: at.PyTree[dict[str, NormStats]] | at.PyTree[NormStats] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False
    # If true, will raise an error if any of the keys in the norm stats are not present in the data.
    strict: bool = False
    # Smallest allowed quantile span (q99 - q01). If smaller, fallback to z-score for those dims.
    min_quantile_span: float = 1e-3

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data
        norm_stats = self.norm_stats
        if "robot_type" in data:
            robot_type = data.pop("robot_type")
            if robot_type in self.norm_stats:
                norm_stats = self.norm_stats[robot_type]

        return apply_tree(
            data,
            norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
        return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
        span = q99 - q01
        use_zscore = np.abs(span) < self.min_quantile_span

        # Standard quantile normalization path.
        normalized = (x - q01) / (span + 1e-6) * 2.0 - 1.0

        # Fallback for tiny quantile spans to avoid amplified values.
        if np.any(use_zscore):
            mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
            zscore = (x - mean) / (std + 1e-6)
            normalized = np.where(use_zscore, zscore, normalized)
            # logger.warning(
            #     "Normalize: quantile span below threshold %.2e on %d/%d dims; fallback to z-score.",
            #     self.min_quantile_span,
            #     int(np.sum(use_zscore)),
            #     int(use_zscore.shape[-1]),
            # )

        return normalized


@dataclasses.dataclass(frozen=True)
class Unnormalize(DataTransformFn):
    norm_stats: at.PyTree[dict[str, NormStats]] | None
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantiles: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            _assert_quantile_stats(self.norm_stats)

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        norm_stats = self.norm_stats
        # Make sure that all the keys in the norm stats are present in the data.
        if "robot_type" in data and data["robot_type"] in self.norm_stats:
            norm_stats = self.norm_stats[data["robot_type"]]
        return apply_tree(
            data,
            norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
        std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
        return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        q01, q99 = stats.q01, stats.q99
        if (dim := q01.shape[-1]) < x.shape[-1]:
            return np.concatenate(
                [(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]],
                axis=-1,
            )
        return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:
        data["image"] = {k: image_tools.resize_with_pad(v, self.height, self.width) for k, v in data["image"].items()}
        return data


@dataclasses.dataclass(frozen=True)
class SubsampleActions(DataTransformFn):
    stride: int

    def __call__(self, data: DataDict) -> DataDict:
        data["actions"] = data["actions"][:: self.stride]
        return data


@dataclasses.dataclass(frozen=True)
class DeltaActions(DataTransformFn):
    """Repacks absolute actions into delta action space."""

    # Boolean mask for the action dimensions to be repacked into delta action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class SemanticDeltaActions(DataTransformFn):
    mask_by_robot: Mapping[str, Sequence[bool]]
    default_mask: Sequence[bool] | None = None
    angle_indices: Sequence[int] = ()

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data

        robot_type = data.get("robot_type", None)
        if robot_type is None:
            if self.default_mask is None:
                return data
            mask = np.asarray(self.default_mask)
        else:
            if not isinstance(robot_type, str):
                robot_type = str(np.asarray(robot_type).item())
            mask = np.asarray(self.mask_by_robot.get(robot_type, self.default_mask))
            if mask is None:
                return data

        state, actions = data["state"], data["actions"]
        dims = mask.shape[-1]
        state_ref = np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        delta = actions[..., :dims] - state_ref

        if self.angle_indices:
            for idx in self.angle_indices:
                if 0 <= idx < dims and mask[idx]:
                    delta[..., idx] = (delta[..., idx] + np.pi) % (2 * np.pi) - np.pi

        actions[..., :dims] = delta
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteActions(DataTransformFn):
    """Repacks delta actions into absolute action space."""

    # Boolean mask for the action dimensions to be repacked into absolute action space. Length
    # can be smaller than the actual number of dimensions. If None, this transform is a no-op.
    # See `make_bool_mask` for more details.
    mask: Sequence[bool] | None

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data or self.mask is None:
            return data

        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask)
        dims = mask.shape[-1]
        actions[..., :dims] += np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        data["actions"] = actions

        return data


@dataclasses.dataclass(frozen=True)
class SemanticAbsoluteActions(DataTransformFn):
    mask_by_robot: Mapping[str, Sequence[bool]]
    default_mask: Sequence[bool] | None = None
    angle_indices: Sequence[int] = ()

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data

        robot_type = data.get("robot_type", None)
        if robot_type is None:
            if self.default_mask is None:
                return data
            mask = np.asarray(self.default_mask)
        else:
            if not isinstance(robot_type, str):
                robot_type = str(np.asarray(robot_type).item())
            mask = np.asarray(self.mask_by_robot.get(robot_type, self.default_mask))
            if mask is None:
                return data

        state, actions = data["state"], data["actions"]
        dims = mask.shape[-1]
        state_ref = np.expand_dims(np.where(mask, state[..., :dims], 0), axis=-2)
        absolute = actions[..., :dims] + state_ref

        if self.angle_indices:
            for idx in self.angle_indices:
                if 0 <= idx < dims and mask[idx]:
                    absolute[..., idx] = (absolute[..., idx] + np.pi) % (2 * np.pi) - np.pi

        actions[..., :dims] = absolute
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
        }


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputs(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state, actions = data["state"], data.get("actions")
        tokens, token_mask, ar_mask, loss_mask = self.tokenizer.tokenize(prompt, state, actions)
        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
        }


@dataclasses.dataclass(frozen=True)
class InjectEvalSubtaskFlags(DataTransformFn):
    """Inject encode_subtask=False, encode_actions=False for eval (avoids GT leakage)."""

    def __call__(self, data: DataDict) -> DataDict:
        data = dict(data)
        data["encode_subtask"] = False
        data["encode_actions"] = False
        return data


@dataclasses.dataclass(frozen=True)
class TokenizeFASTInputsWithSubtask(DataTransformFn):
    """Tokenize prompt, state, subtask, and actions for π0.5 subtask+FAST mode.

    When encode_subtask=False (e.g. eval with GT): do NOT encode subtask into prompt
    to avoid information leakage. Still add subtask_tokens/subtask_tokens_mask for metrics.
    """

    tokenizer: _tokenizer.FASTTokenizerWithSubtask

    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if not isinstance(prompt, str):
            prompt = prompt.item()

        state = data["state"]
        actions = data.get("actions")
        encode_subtask = data.pop("encode_subtask", True)
        encode_actions = data.pop("encode_actions", True)

        subtask = data.pop("subtask", None)
        if subtask is not None:
            assert isinstance(subtask, str), "Subtask must be a string."
            subtask = None if len(subtask) == 0 else subtask.strip()

        # For eval: pass subtask=None to avoid encoding into prompt (prevents leakage).
        # Tokenizer uses instance _encode_subtask; we need per-call override.
        subtask_for_encode = subtask if encode_subtask else None
        actions_for_encode = actions if encode_actions else None

        result = self.tokenizer.tokenize(
            prompt,
            state,
            subtask=subtask_for_encode,
            actions=actions_for_encode,
        )
        (
            tokens,
            token_mask,
            ar_mask,
            loss_mask,
            fast_action_loss_mask,
        ) = result
        out = {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_mask,
            "token_ar_mask": ar_mask,
            "token_loss_mask": loss_mask,
            "fast_action_loss_mask": fast_action_loss_mask,
        }
        # When not encoding subtask, add subtask_tokens for GT metrics (eval).
        # Always add these keys in eval mode to ensure consistent dict structure
        # across all samples in the batch (required by _collate_fn / jax.tree.map).
        if not encode_subtask:
            max_st = getattr(self.tokenizer, "_max_subtask_len", 32)
            pg_tok = getattr(self.tokenizer, "_paligemma_tokenizer", None)
            if pg_tok is None:
                raise RuntimeError("FASTTokenizerWithSubtask missing _paligemma_tokenizer")
            if subtask is not None:
                st_tokens = pg_tok.encode(f"Subtask: {subtask}") + pg_tok.encode("|", add_eos=True)
                if len(st_tokens) > max_st:
                    st_tokens = st_tokens[:max_st]
                st_mask = [True] * len(st_tokens)
                if len(st_mask) < max_st:
                    st_tokens = st_tokens + [0] * (max_st - len(st_tokens))
                    st_mask = st_mask + [False] * (max_st - len(st_mask))
            else:
                st_tokens = [0] * max_st
                st_mask = [False] * max_st
            out["subtask_tokens"] = np.array(st_tokens, dtype=np.int32)
            out["subtask_tokens_mask"] = np.array(st_mask, dtype=np.bool_)
        return out


@dataclasses.dataclass(frozen=True)
class ExtractFASTActions(DataTransformFn):
    tokenizer: _tokenizer.FASTTokenizer
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "actions" not in data:
            return data
        # Model outputs are saved in "actions", but for FAST models they represent tokens.
        tokens = data.pop("actions")
        actions = self.tokenizer.extract_actions(tokens.astype(np.int32), self.action_horizon, self.action_dim)
        return {
            **data,
            "actions": actions,
        }


@dataclasses.dataclass(frozen=True)
class DecodeSubtaskFromTokens(DataTransformFn):
    """Decode subtask_tokens to subtask string for client caching (π0.5 low-frequency subtask inference)."""

    tokenizer: _tokenizer.FASTTokenizerWithSubtask

    def __call__(self, data: DataDict) -> DataDict:
        if "subtask_tokens" not in data:
            return data
        tokens = np.asarray(data.pop("subtask_tokens")).astype(np.int32)
        if tokens.ndim == 1:
            tokens = tokens[np.newaxis, :]
        subtasks = []
        for b in range(tokens.shape[0]):
            row = tokens[b]
            row = row[row != 0]  # trim padding
            st = self.tokenizer.extract_subtask(row)
            subtasks.append(st)
        data["subtask"] = subtasks[0] if len(subtasks) == 1 else np.array(subtasks)
        return data


@dataclasses.dataclass(frozen=True)
class ExtractFASTActionsWithSubtask(DataTransformFn):
    """Extract actions from FAST token sequence (prefix + decoded). Handles dict from _sample_actions_fast."""

    tokenizer: _tokenizer.FASTTokenizerWithSubtask
    action_horizon: int
    action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        if "fast_action_tokens" not in data:
            return data
        raw = data.pop("fast_action_tokens")
        tokens = raw["fast_action_tokens"] if isinstance(raw, dict) else raw
        tokens = np.asarray(tokens).astype(np.int32)
        if tokens.ndim == 1:
            tokens = tokens[np.newaxis, :]
        batch_size = tokens.shape[0]
        actions_list = []
        for b in range(batch_size):
            row = tokens[b]
            act = self.tokenizer.extract_actions(row, self.action_horizon, self.action_dim)
            actions_list.append(act)
        actions = np.stack(actions_list, axis=0)
        return {**data, "fast_actions": actions}


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotTask(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    tasks: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = int(data["task_index"])
        if (prompt := self.tasks.get(task_index)) is None:
            raise ValueError(f"{task_index=} not found in task mapping: {self.tasks}")

        return {**data, "prompt": prompt}


@dataclasses.dataclass(frozen=True)
class PromptFromEpisodeTask(DataTransformFn):
    """Extracts a prompt from the current episode task."""

    # Contains the LeRobot dataset tasks (dataset.meta.tasks).
    episodes: dict[int, str]

    def __call__(self, data: DataDict) -> DataDict:
        if "episode_index" not in data:
            raise ValueError('Cannot extract prompt without "episode_index"')

        episode_index = int(data["episode_index"])
        task = self.episodes[episode_index]["tasks"][0]
        if task is None:
            raise ValueError(f"{episode_index} not found in episodes")
        data["task"] = task

        return data


@dataclasses.dataclass(frozen=True)
class AddAdvantageToPrompt(DataTransformFn):
    """Adds advantage conditioning text to prompt for RECAP training.

    Inserts "Advantage: positive" or "Advantage: negative" before "Action:"
    in the task/prompt text. Supports CFG-style dropout during training.

    Training mode: reads 'indicator' key from data dict (loaded by IndicatorPreprocessor).
    Inference mode: uses fixed_advantage parameter (True=positive, False=negative).

    Args:
        fixed_advantage: Fixed advantage for inference mode. None=use indicator from data.
        dropout_rate: Probability of dropping advantage info during training (default 0.3).
        training: Whether in training mode (affects dropout behavior).
    """

    fixed_advantage: bool | None = None
    dropout_rate: float = 0.3
    training: bool = True

    def _insert_advantage_before_action(self, text: str, advantage_text: str) -> str:
        """Insert advantage text before 'Action:' marker."""
        action_marker = "Action:"
        if action_marker in text:
            parts = text.split(action_marker, 1)
            return f"{parts[0]}\n{advantage_text}\n{action_marker}{parts[1]}"
        return f"{text}\n{advantage_text}"

    def __call__(self, data: DataDict) -> DataDict:
        is_positive = None

        if self.fixed_advantage is not None:
            is_positive = self.fixed_advantage
        elif "indicator" in data:
            indicator = data["indicator"]
            if isinstance(indicator, np.ndarray):
                is_positive = bool(indicator.item() if indicator.ndim == 0 else indicator[0])
            else:
                is_positive = bool(indicator)
        else:
            return data

        # CFG-style dropout during training (only when using indicator, not fixed)
        if self.fixed_advantage is None and self.training and random.random() < self.dropout_rate:
            return data

        advantage_text = "Advantage: positive" if is_positive else "Advantage: negative"

        if "task" in data:
            task = data["task"]
            if isinstance(task, np.ndarray):
                task = task.item() if task.ndim == 0 else task[0]
            task = str(task) if not isinstance(task, str) else task
            data["task"] = self._insert_advantage_before_action(task, advantage_text)

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, np.ndarray):
                prompt = prompt.item() if prompt.ndim == 0 else prompt[0]
            prompt = str(prompt) if not isinstance(prompt, str) else prompt
            data["prompt"] = self._insert_advantage_before_action(prompt, advantage_text)

        return data


@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(data["actions"], self.model_action_dim, axis=-1)
        if "joint_eef_dof_mask" in data:
            data["joint_eef_dof_mask"] = pad_to_dim(
                np.asarray(data["joint_eef_dof_mask"]), self.model_action_dim, axis=-1
            ).astype(bool)
        return data


def flatten_dict(tree: at.PyTree) -> dict:
    """Flatten a nested dictionary. Uses '/' as the separator."""
    return traverse_util.flatten_dict(tree, sep="/")


def unflatten_dict(tree: dict) -> at.PyTree:
    """Unflatten a flattened dictionary. Assumes that '/' was used as a separator."""
    return traverse_util.unflatten_dict(tree, sep="/")


def transform_dict(patterns: Mapping[str, str | None], tree: at.PyTree) -> at.PyTree:
    """Transform the structure of a nested dictionary using a set of patterns.

    The transformation is defined using the `patterns` dictionary. The keys are the
    input keys that should be matched and the values are the new names inside the output
    dictionary. If the value is None, the input key is removed.

    Both keys and values should represent flattened paths using '/' as the separator.
    Keys can be regular expressions and values can include backreferences to the
    matched groups (see `re.sub` for more details). Note that the regular expression
    must match the entire key.

    The order inside the `patterns` dictionary is important. Only the first pattern that
    matches the input key will be used.

    See unit tests for more examples.

    Args:
        patterns: A mapping from old keys to new keys.
        tree: The nested dictionary to transform.

    Returns:
        The transformed nested dictionary.
    """
    data = flatten_dict(tree)

    # Compile the patterns.
    compiled = {re.compile(k): v for k, v in patterns.items()}

    output = {}
    for k in data:
        for pattern, repl in compiled.items():
            if pattern.fullmatch(k):
                new_k = pattern.sub(repl, k, count=1) if repl is not None else None
                break
        else:
            # Use the original key if no match is found.
            new_k = k

        if new_k is not None:
            if new_k in output:
                raise ValueError(f"Key '{new_k}' already exists in output")
            output[new_k] = data[k]

    # Validate the output structure to make sure that it can be unflattened.
    names = sorted(output)
    for i in range(len(names) - 1):
        name, next_name = names[i : i + 2]
        if next_name.startswith(name + "/"):
            raise ValueError(f"Leaf '{name}' aliases a node of '{next_name}'")

    return unflatten_dict(output)


def apply_tree(
    tree: at.PyTree[T],
    selector: at.PyTree[S],
    fn: Callable[[T, S], T],
    *,
    strict: bool = False,
) -> at.PyTree[T]:
    tree = flatten_dict(tree)
    selector = flatten_dict(selector)

    def transform(k: str, v: T) -> T:
        if k in selector:
            return fn(v, selector[k])
        return v

    if strict:
        for k in selector:
            if k not in tree:
                raise ValueError(f"Selector key {k} not found in tree")

    return unflatten_dict({k: transform(k, v) for k, v in tree.items()})


def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x


def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)


def _assert_quantile_stats(norm_stats: at.PyTree[NormStats]) -> None:
    for k, v in flatten_dict(norm_stats).items():
        if v.q01 is None or v.q99 is None:
            raise ValueError(
                f"quantile stats must be provided if use_quantile_norm is True. Key {k} is missing q01 or q99."
            )
