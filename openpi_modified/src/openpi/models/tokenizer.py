import dataclasses
import logging
import os

import jax
import numpy as np
import orbax.checkpoint as ocp
import sentencepiece
from transformers import AutoProcessor

import openpi.models.utils.fsq_tokenizer as fsq_tokenizer
import openpi.shared.download as download


def _find_token_subsequence(tokens: np.ndarray, pattern: np.ndarray, start: int = 0) -> int:
    """Return the first index of ``pattern`` inside ``tokens`` or -1 if absent."""
    if pattern.size == 0 or tokens.size < pattern.size:
        return -1
    last = tokens.size - pattern.size + 1
    for idx in range(start, last):
        if np.array_equal(tokens[idx : idx + pattern.size], pattern):
            return idx
    return -1


@dataclasses.dataclass(frozen=True)
class FASTActionDecodeResult:
    actions: np.ndarray
    status: str
    detail: str | None = None


class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48, *, set_zero_state: bool = False):
        self._max_len = max_len
        self._set_zero_state = set_zero_state

        paligemma_tokenizer_path = os.path.join(
            os.path.expanduser("~"),
            ".cache/openpi/big_vision/paligemma_tokenizer.model",
        )
        path = download.maybe_download(paligemma_tokenizer_path, gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str, state: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        if state is not None:
            if self._set_zero_state:
                zero_state = np.zeros_like(state)
                discretized_state = np.digitize(zero_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=True)
            else:
                # This is the Pi05 format, where the state is part of the discrete language input.
                discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
                tokens = self._tokenizer.encode(full_prompt, add_bos=True)
        else:
            # This is the Pi0 format, where the state is part of the continuous action expert input.
            # tokenize "\n" separately as the "start of answer" token
            tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"full_prompt: {full_prompt}, Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(
        self,
        max_len: int = 256,
        fast_tokenizer_path: str = "physical-intelligence/fast",
    ):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


###########################################################################
## The tokenizers below are used for RoboArena baseline implementations. ##
## They are *not* used for pi0-style models.                             ##
###########################################################################


class BinningTokenizer:
    """
    Standard RT-2 / OpenVLA style binning tokenizer.
    """

    def __init__(self, max_len: int = 256, n_bins: int = 256):
        self._max_len = max_len
        self._n_bins = n_bins

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Tokenize a prompt and state into a sequence of tokens.

        Args:
            prompt: The text prompt to tokenize.
            state: The state array to discretize and tokenize.
            actions: Must be None. Action encoding is not currently supported.

        Returns:
            A tuple of (tokens, token_mask, ar_mask, targets).

        Raises:
            NotImplementedError: If actions is not None.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("BinningTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        if len(action_tokens) < action_horizon * action_dim:
            return np.zeros([action_horizon, action_dim], dtype=np.float32)
        action_tokens = action_tokens[: (action_horizon * action_dim)].reshape([action_horizon, action_dim])
        return action_tokens / self._n_bins * 2 - 1

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FSQTokenizer:
    """
    FSQ tokenizer from the FAST paper baselines.
    """

    def __init__(self, max_len: int = 256, fsq_tokenizer_path: str | None = None):
        self._max_len = max_len

        assert fsq_tokenizer_path is not None, "fsq_tokenizer_path must be provided"
        # Download tokenizer
        path = download.maybe_download(fsq_tokenizer_path)
        tok_path = os.path.join(path, os.listdir(path)[0])

        # Split step from path
        step = int(tok_path.split("/")[-1])
        base_path = tok_path.rsplit("/", 1)[0]

        mgr = ocp.CheckpointManager(
            base_path,
            item_handlers={
                "params": ocp.StandardCheckpointHandler(),
                "opt_state": ocp.StandardCheckpointHandler(),
                "config": ocp.JsonCheckpointHandler(),
            },
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )

        try:
            restored = mgr.restore(
                step,
                args=ocp.args.Composite(config=ocp.args.JsonRestore(), params=ocp.args.StandardRestore()),
            )
            config = restored["config"]
            self._params = restored["params"]
            self._fsq_tokenizer = fsq_tokenizer.FsqAttentionTokenizer(**config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load FSQ tokenizer checkpoint from {fsq_tokenizer_path}. Error: {e!s}"
            ) from e

        # Compile tokenize and detokenize functions
        self._tokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.tokenize)
        )
        self._detokenize_fn = jax.jit(
            lambda params, x: self._fsq_tokenizer.apply({"params": params}, x, method=self._fsq_tokenizer.detokenize)
        )

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        prefix = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._paligemma_tokenizer.encode(prefix, add_bos=True)

        if actions is not None:
            raise NotImplementedError("FSQTokenizer does not support encoding actions atm (only for inference use)")
        postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
        )

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        try:
            # Move computation to CPU and compile on-demand
            device = jax.devices("cpu")[0]
            with jax.default_device(device):
                detok_act = self._detokenize_fn(self._params, action_tokens[None, ...])[0]
            return detok_act[: action_horizon * action_dim].reshape([action_horizon, action_dim])
        except Exception as e:
            logging.warning(f"Error decoding FSQ: {e}")
            return np.zeros((action_horizon, action_dim))

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens


class FASTTokenizerWithSubtask:
    """Combines subtask (text) and FAST action tokens per π0.5 paper.

    Sequence format: Task: {prompt}, State: {state};\\nSubtask: {subtask}|\\nAction: {fast_tokens}|
    Supports subtask=None or actions=None for inference/training.
    """

    def __init__(
        self,
        max_len: int = 128,
        max_subtask_len: int = 32,
        *,
        set_zero_state: bool = False,
        encode_subtask: bool = True,
        encode_actions: bool = True,
        fast_tokenizer_path: str = "physical-intelligence/fast",
    ):
        self._max_len = max_len
        self._max_subtask_len = max_subtask_len
        self._set_zero_state = set_zero_state
        self._encode_subtask = encode_subtask
        self._encode_actions = encode_actions

        paligemma_tokenizer_path = os.path.join(
            os.path.expanduser("~"),
            ".cache/openpi/big_vision/paligemma_tokenizer.model",
        )

        path = download.maybe_download(paligemma_tokenizer_path, gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128
        self._action_prefix_tokens = np.asarray(self._paligemma_tokenizer.encode("\nAction: "), dtype=np.int32)
        self._action_suffix_tokens = np.asarray(self._paligemma_tokenizer.encode("|"), dtype=np.int32)

    def tokenize(
        self,
        prompt: str,
        state: np.ndarray | None = None,
        subtask: str | None = None,
        actions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Returns (tokens, token_mask, ar_mask, loss_mask, fast_action_loss_mask).

        fast_action_loss_mask is True only for FAST action token positions (for separate
        weighting). None when actions is None.

        When encode_subtask=False (e.g. eval with GT): do not encode subtask even if present.
        When encode_actions=False (e.g. eval with GT): do not encode actions even if present.
        """
        cleaned_text = prompt.lower().strip().replace("_", " ")
        if state is not None:
            if self._set_zero_state:
                zero_state = np.zeros_like(state)
                discretized_state = np.digitize(zero_state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\n"
                prefix_tokens = self._paligemma_tokenizer.encode(full_prompt, add_bos=True)
            else:
                # This is the Pi05 format, where the state is part of the discrete language input.
                discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
                state_str = " ".join(map(str, discretized_state))
                full_prompt = f"Task: {cleaned_text}, State: {state_str};\n"
                prefix_tokens = self._paligemma_tokenizer.encode(full_prompt, add_bos=True)
        else:
            prefix_tokens = self._paligemma_tokenizer.encode(
                cleaned_text, add_bos=True
            ) + self._paligemma_tokenizer.encode("\n")

        postfix_tokens: list[int] = []
        postfix_loss_mask: list[bool] = []
        postfix_ar_mask: list[int] = []
        postfix_fast_mask: list[bool] = []
        if self._encode_subtask and subtask is not None:
            subtask_tokens = self.encode_subtask_to_tokens(
                subtask, add_eos=(actions is None or not self._encode_actions)
            )
            postfix_tokens.extend(subtask_tokens)
            postfix_loss_mask.extend([True] * len(subtask_tokens))
            postfix_ar_mask.extend([1] * len(subtask_tokens))
            postfix_fast_mask.extend([False] * len(subtask_tokens))

        if self._encode_actions and actions is not None:
            action_part = self.encode_action_to_tokens(actions, add_eos=True)
            postfix_tokens.extend(action_part)
            postfix_loss_mask.extend([True] * len(action_part))
            postfix_ar_mask.extend([1] * len(action_part))
            postfix_fast_mask.extend([True] * len(action_part))

        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + postfix_ar_mask
        loss_mask = [False] * len(prefix_tokens) + postfix_loss_mask
        fast_action_loss_mask = [False] * len(prefix_tokens) + postfix_fast_mask

        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            pad_len = self._max_len - tokens_len
            tokens = tokens + [0] * pad_len
            token_mask = token_mask + [False] * pad_len
            ar_mask = ar_mask + [0] * pad_len
            loss_mask = loss_mask + [False] * pad_len
            fast_action_loss_mask = fast_action_loss_mask + [False] * pad_len
        else:
            if tokens_len > self._max_len:
                logging.warning(
                    "Token length (%d) exceeds max length (%d), truncating.",
                    tokens_len,
                    self._max_len,
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]
            fast_action_loss_mask = fast_action_loss_mask[: self._max_len]

        return (
            np.asarray(tokens),
            np.asarray(token_mask),
            np.asarray(ar_mask),
            np.asarray(loss_mask),
            np.asarray(fast_action_loss_mask),
        )

    def encode_subtask_to_tokens(self, subtask: str, *, add_eos: bool = False) -> np.ndarray:
        return self._paligemma_tokenizer.encode(f"Subtask: {subtask}") + self._paligemma_tokenizer.encode(
            "|", add_eos=add_eos
        )

    def encode_action_to_tokens(self, actions: np.ndarray, *, add_eos: bool = True) -> list[int]:
        action_tokens = self._fast_tokenizer(actions[None])[0]
        action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)
        return (
            self._action_prefix_tokens.tolist()
            + action_tokens_in_pg.tolist()
            + self._paligemma_tokenizer.encode("|", add_eos=add_eos)
        )

    def extract_actions_with_info(
        self, tokens: np.ndarray, action_horizon: int, action_dim: int
    ) -> FASTActionDecodeResult:
        expected_shape = (action_horizon, action_dim)
        zeros = np.zeros(expected_shape, dtype=np.float32)
        raw_tokens = np.asarray(tokens, dtype=np.int32).reshape(-1)
        raw_prefix_start = _find_token_subsequence(raw_tokens, self._action_prefix_tokens)
        if raw_prefix_start < 0:
            return FASTActionDecodeResult(
                actions=zeros,
                status="missing_action_prefix",
                detail=f"token_count={raw_tokens.size}",
            )
        start = raw_prefix_start + len(self._action_prefix_tokens)

        raw_suffix_start = _find_token_subsequence(raw_tokens, self._action_suffix_tokens, start)
        eos_idx_after_start = np.where(raw_tokens[start:] == 1)[0]
        first_eos_after_start = start + int(eos_idx_after_start[0]) if len(eos_idx_after_start) > 0 else -1
        if raw_suffix_start >= 0:
            end = raw_suffix_start
        elif first_eos_after_start >= 0:
            end = first_eos_after_start
        else:
            return FASTActionDecodeResult(
                actions=zeros,
                status="missing_action_suffix",
                detail=f"token_count={raw_tokens.size}",
            )
        if end <= start:
            return FASTActionDecodeResult(
                actions=zeros,
                status="invalid_action_span",
                detail=f"start={start}, end={end}",
            )

        act_tokens = self._act_tokens_to_paligemma_tokens(raw_tokens[start:end])
        try:
            result = self._fast_tokenizer.decode(
                [act_tokens.tolist()],
                time_horizon=action_horizon,
                action_dim=action_dim,
            )[0]
        except Exception as e:
            return FASTActionDecodeResult(
                actions=zeros,
                status="fast_decode_exception",
                detail=f"{type(e).__name__}: {e}",
            )

        result = np.asarray(result, dtype=np.float32)
        if result.ndim != 2:
            return FASTActionDecodeResult(
                actions=zeros,
                status="decoded_rank_mismatch",
                detail=f"shape={result.shape}",
            )
        if result.shape != expected_shape:
            padded = np.zeros(expected_shape, dtype=np.float32)
            h = min(result.shape[0], action_horizon)
            d = min(result.shape[1], action_dim)
            padded[:h, :d] = result[:h, :d]
            return FASTActionDecodeResult(
                actions=padded,
                status="decoded_shape_mismatch",
                detail=f"shape={result.shape}, expected={expected_shape}",
            )

        return FASTActionDecodeResult(actions=result, status="ok")

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        decode_result = self.extract_actions_with_info(tokens, action_horizon, action_dim)
        if decode_result.status != "ok":
            suffix = f": {decode_result.detail}" if decode_result.detail is not None else ""
            logging.warning("FAST action decode failed (%s)%s", decode_result.status, suffix)
        return decode_result.actions

    def extract_subtask(self, tokens: np.ndarray, tokens_mask: np.ndarray | None = None) -> str:
        if tokens_mask is not None:
            tokens = tokens[tokens_mask.astype(bool)]
        paligemma_eos_token = 1
        eos_idx = np.where(tokens == paligemma_eos_token)[0]
        if len(eos_idx) > 0:
            tokens = tokens[: eos_idx[0]]
        decoded = self._paligemma_tokenizer.decode(tokens.tolist())
        if "Subtask: " not in decoded:
            return "Subtask: None"
        return decoded.split("Subtask: ")[1].split("|")[0].strip()

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens
