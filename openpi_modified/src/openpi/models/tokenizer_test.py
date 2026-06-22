import io
import os

import numpy as np
import pytest
import requests

from openpi.models import tokenizer as _tokenizer


def test_tokenize():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=10)
    tokens, masks = tokenizer.tokenize("Hello, world!")

    assert tokens.shape == (10,)
    assert masks.shape == (10,)


def test_paligemma_tokenizer_uses_legacy_cache_path(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    class FakePath:
        def open(self, mode: str):
            assert mode == "rb"
            return io.BytesIO(b"fake-model")

    def fake_maybe_download(path: str, gs: dict[str, str] | None = None):
        captured["path"] = path
        captured["gs"] = gs
        return FakePath()

    monkeypatch.setattr(_tokenizer.download, "maybe_download", fake_maybe_download)
    monkeypatch.setattr(_tokenizer.sentencepiece, "SentencePieceProcessor", lambda model_proto: object())

    _tokenizer.PaligemmaTokenizer(max_len=10)

    assert captured["path"] == os.path.join(
        os.path.expanduser("~"), ".cache/openpi/big_vision/paligemma_tokenizer.model"
    )
    assert captured["gs"] == {"token": "anon"}


def test_fast_tokenizer():
    prompt = "Hello, world!"
    state = np.random.rand(5).astype(np.float32)
    action = np.random.rand(3, 2).astype(np.float32)
    try:
        tokenizer = _tokenizer.FASTTokenizer(max_len=256)
    except OSError as exc:
        pytest.skip(f"FAST tokenizer assets unavailable in this environment: {exc}")
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"FAST tokenizer Hub download unavailable in this environment: {exc}")

    tokens, token_masks, ar_masks, loss_masks = tokenizer.tokenize(prompt, state, action)

    assert tokens.shape == (256,)
    assert token_masks.shape == (256,)
    assert ar_masks.shape == (256,)
    assert loss_masks.shape == (256,)

    act = tokenizer.extract_actions(tokens, 3, 2)
    assert act.shape == (3, 2)


def test_tokenize_subtask_fast_action():
    try:
        tokenizer = _tokenizer.FASTTokenizerWithSubtask(max_len=128, max_subtask_len=32)
    except OSError as exc:
        pytest.skip(f"FAST tokenizer assets unavailable in this environment: {exc}")
    except requests.exceptions.RequestException as exc:
        pytest.skip(f"FAST tokenizer Hub download unavailable in this environment: {exc}")
    actions = np.random.rand(10, 5).astype(np.float32)
    (
        tokens,
        token_masks,
        ar_masks,
        loss_masks,
        fast_action_loss_mask,
    ) = tokenizer.tokenize(prompt="prompt", subtask="test", actions=actions)
    decode_actions = tokenizer.extract_actions(tokens, 10, 5)
    print("fast action loss: ", np.mean(np.abs(decode_actions - actions), axis=1))

    assert tokens.shape == (128,)
    assert token_masks.shape == (128,)
    assert ar_masks.shape == (128,)
    assert loss_masks.shape == (128,)
    assert fast_action_loss_mask.shape == (128,)


def test_extract_action_only_tokens_roundtrip():
    try:
        tokenizer = _tokenizer.FASTTokenizerWithSubtask(max_len=128, max_subtask_len=32)
    except OSError as exc:
        pytest.skip(f"FAST tokenizer assets unavailable in this environment: {exc}")

    actions = np.random.rand(10, 5).astype(np.float32)
    action_only_tokens = np.asarray(tokenizer.encode_action_to_tokens(actions, add_eos=True), dtype=np.int32)

    decoded_actions = tokenizer.extract_actions(action_only_tokens, 10, 5)

    assert decoded_actions.shape == (10, 5)


def test_extract_actions_ignores_subtask_eos_before_action():
    try:
        tokenizer = _tokenizer.FASTTokenizerWithSubtask(max_len=256, max_subtask_len=32)
    except OSError as exc:
        pytest.skip(f"FAST tokenizer assets unavailable in this environment: {exc}")

    actions = np.random.rand(10, 5).astype(np.float32)
    subtask_tokens = np.asarray(tokenizer.encode_subtask_to_tokens("pick cube", add_eos=True), dtype=np.int32)
    action_tokens = np.asarray(tokenizer.encode_action_to_tokens(actions, add_eos=True), dtype=np.int32)
    full_tokens = np.concatenate([subtask_tokens, action_tokens], axis=0)

    decoded = tokenizer.extract_actions_with_info(full_tokens, 10, 5)

    assert decoded.status == "ok"
    assert decoded.actions.shape == (10, 5)


if __name__ == "__main__":
    test_tokenize_subtask_fast_action()
