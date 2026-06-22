from collections.abc import Callable
import dataclasses
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import jax
import numpy as np
import pytest

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training.base_cfg import TrainConfig

REPO_ROOT = Path(__file__).resolve().parents[3]
BASELINE_PATH = REPO_ROOT / "tests/test_data/top_level_config_loss_baseline.json"
BASELINE_VALUES_DIR = REPO_ROOT / "tests/test_data/top_level_config_loss_values"
LOSS_BASELINE_INIT_SEED = 0
LOSS_BASELINE_LOSS_SEED = 7
LOSS_BASELINE_BATCH_SIZE = 1

TOP_LEVEL_CONFIGS = sorted(
    path for path in (REPO_ROOT / "src/openpi/configs").glob("*.py") if path.name not in {"__init__.py", "base.py"}
)


@pytest.fixture
def fake_dataset_listing(monkeypatch: pytest.MonkeyPatch):
    real_listdir = os.listdir

    def safe_listdir(path: str | os.PathLike[str]):
        try:
            return real_listdir(path)
        except OSError:
            return ["placeholder"]

    monkeypatch.setattr(os, "listdir", safe_listdir)


def _make_test_model_config(model_cfg: _model.BaseModelConfig) -> _model.BaseModelConfig:
    if isinstance(model_cfg, pi0_config.Pi0Config):
        return dataclasses.replace(
            model_cfg,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            vision_output_dim=None,
            vocab_size=None,
            input_image_size=(224, 224),
            checkpoint_image_size=(224, 224),
            enable_recap=False,
        )
    return model_cfg


def _compute_loss_array(
    train_cfg: TrainConfig,
    *,
    init_seed: int = LOSS_BASELINE_INIT_SEED,
    loss_seed: int = LOSS_BASELINE_LOSS_SEED,
    batch_size: int = LOSS_BASELINE_BATCH_SIZE,
) -> np.ndarray:
    model_cfg = _make_test_model_config(train_cfg.model)
    model = model_cfg.create(jax.random.key(init_seed))
    observation = model_cfg.fake_obs(batch_size)
    actions = model_cfg.fake_act(batch_size)

    if model_cfg.model_type == _model.ModelType.PI0_FAST:
        loss = model.compute_loss(jax.random.key(loss_seed), observation, actions)
    else:
        loss = model.compute_loss(
            jax.random.key(loss_seed),
            observation,
            actions,
            train_cfg.rtc_max_delay,
        )

    if isinstance(loss, tuple):
        loss = loss[0]

    return np.asarray(loss)


def _summarize_loss_array(loss: np.ndarray) -> dict[str, Any]:
    contiguous_loss = np.ascontiguousarray(loss)
    return {
        "sha256": hashlib.sha256(contiguous_loss.tobytes()).hexdigest(),
        "mean": float(contiguous_loss.mean()),
        "sum": float(contiguous_loss.sum()),
        "min": float(contiguous_loss.min()),
        "max": float(contiguous_loss.max()),
    }


def _config_key(config_path: Path) -> str:
    return config_path.name


def _loss_value_path_for_key(key: str) -> Path:
    return BASELINE_VALUES_DIR / f"{Path(key).stem}.npy"


def _is_missing_repo_id_list_error(message: str) -> bool:
    return "repo_id_lists" in message and "not found" in message


def _canonicalize_expected_xfail_message(message: str) -> str:
    marker = "REPO_ID list file not found:"
    if marker in message:
        return message[message.index(marker) :]
    return message


@lru_cache(maxsize=1)
def _load_loss_baseline() -> dict[str, Any]:
    with BASELINE_PATH.open(encoding="utf-8") as baseline_file:
        return json.load(baseline_file)


def _load_expected_loss_array(expected_entry: dict[str, Any], *, dtype: np.dtype) -> np.ndarray:
    loss_file = expected_entry.get("loss_file")
    if loss_file is None:
        return np.asarray(expected_entry["loss"], dtype=dtype)
    return np.load(REPO_ROOT / loss_file).astype(dtype, copy=False)


def _build_baseline_config_entry(config_path: Path, loss: np.ndarray) -> dict[str, Any]:
    key = _config_key(config_path)
    loss_file_path = _loss_value_path_for_key(key)
    return {
        "relative_path": str(config_path.relative_to(REPO_ROOT)),
        "shape": list(loss.shape),
        "dtype": str(loss.dtype),
        "summary": _summarize_loss_array(loss),
        "loss_file": str(loss_file_path.relative_to(REPO_ROOT)),
    }


LOSS_COMPARISON_RTOL = 1e-6
LOSS_COMPARISON_ATOL = 1e-5


def _is_float_loss_array(loss: np.ndarray) -> bool:
    return np.issubdtype(loss.dtype, np.floating)


_SUMMARY_FLOAT_KEYS = ("mean", "sum", "min", "max")


def _summaries_match(
    current_summary: dict[str, Any],
    expected_summary: dict[str, Any],
    *,
    is_float: bool,
) -> bool:
    if not is_float:
        return current_summary == expected_summary
    return np.allclose(
        [current_summary[k] for k in _SUMMARY_FLOAT_KEYS],
        [expected_summary[k] for k in _SUMMARY_FLOAT_KEYS],
        rtol=LOSS_COMPARISON_RTOL,
        atol=LOSS_COMPARISON_ATOL,
    )


def _loss_arrays_match(loss: np.ndarray, expected_loss: np.ndarray, *, is_float: bool) -> bool:
    if is_float:
        return np.allclose(
            loss,
            expected_loss,
            rtol=LOSS_COMPARISON_RTOL,
            atol=LOSS_COMPARISON_ATOL,
        )
    return np.array_equal(loss, expected_loss)


def _compare_loss_to_expected_entry(
    *,
    key: str,
    loss: np.ndarray,
    expected_entry: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []
    expected_loss = _load_expected_loss_array(expected_entry, dtype=loss.dtype)

    if list(loss.shape) != expected_entry["shape"]:
        mismatches.append(f"{key}: shape mismatch, expected {expected_entry['shape']}, got {list(loss.shape)}")
    if str(loss.dtype) != expected_entry["dtype"]:
        mismatches.append(f"{key}: dtype mismatch, expected {expected_entry['dtype']}, got {loss.dtype}")

    current_summary = _summarize_loss_array(loss)
    expected_summary = expected_entry.get("summary")
    is_float = _is_float_loss_array(loss)
    if expected_summary is not None and not _summaries_match(current_summary, expected_summary, is_float=is_float):
        mismatches.append(
            f"{key}: summary mismatch, expected sha256={expected_summary['sha256']}, got sha256={current_summary['sha256']}"
        )

    if loss.shape == expected_loss.shape and not _loss_arrays_match(loss, expected_loss, is_float=is_float):
        diff = np.abs(loss - expected_loss)
        mismatches.append(
            f"{key}: loss mismatch, max_abs_diff={float(diff.max())}, current_sha256={current_summary['sha256']}"
        )

    return mismatches


def _emit_progress(progress_logger: Callable[[str], None] | None, message: str) -> None:
    if progress_logger is not None:
        progress_logger(message)


class _ConfigProgressHeartbeat:
    def __init__(
        self,
        config_key: str,
        *,
        progress_logger: Callable[[str], None] | None,
        interval_seconds: int = 60,
    ) -> None:
        self._config_key = config_key
        self._progress_logger = progress_logger
        self._interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def __enter__(self) -> "_ConfigProgressHeartbeat":
        if self._progress_logger is None:
            return self
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, name=f"loss-drift-heartbeat-{self._config_key}", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=1)

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            elapsed = int(time.monotonic() - self._start_time)
            _emit_progress(
                self._progress_logger,
                f"heartbeat: still computing loss for {self._config_key} after {elapsed}s",
            )


def _heartbeat_interval_seconds() -> int:
    configured = os.environ.get("OPENPI_LOSS_DRIFT_HEARTBEAT_INTERVAL", "60")
    try:
        return max(1, int(configured))
    except ValueError:
        return 60


def collect_top_level_config_loss_drift(progress_logger: Callable[[str], None] | None = None) -> list[str]:
    baseline = _load_loss_baseline()
    drift_messages: list[str] = []
    total_configs = len(TOP_LEVEL_CONFIGS)

    real_listdir = os.listdir

    def safe_listdir(path: str | os.PathLike[str]):
        try:
            return real_listdir(path)
        except OSError:
            return ["placeholder"]

    os.listdir = safe_listdir
    try:
        for index, config_path in enumerate(TOP_LEVEL_CONFIGS, start=1):
            key = _config_key(config_path)
            expected_entry = baseline["configs"].get(key)
            expected_xfail = baseline["expected_xfails"].get(key)
            _emit_progress(progress_logger, f"[{index}/{total_configs}] checking {key}")

            try:
                cfg = _config.get_config(str(config_path))
            except ValueError as exc:
                message = str(exc)
                if expected_xfail is not None and _is_missing_repo_id_list_error(message):
                    if expected_xfail["reason"] not in message:
                        drift_messages.append(
                            f"{key}: xfail reason changed, expected '{expected_xfail['reason']}', got '{message}'"
                        )
                    continue
                drift_messages.append(f"{key}: unexpected load failure: {message}")
                continue

            if expected_xfail is not None:
                drift_messages.append(f"{key}: now loads successfully, but baseline still marks it as xfail")
                continue
            if expected_entry is None:
                drift_messages.append(f"{key}: missing saved loss baseline entry")
                continue

            with _ConfigProgressHeartbeat(
                key,
                progress_logger=progress_logger,
                interval_seconds=_heartbeat_interval_seconds(),
            ):
                loss = _compute_loss_array(cfg)
            if not np.isfinite(loss).all():
                drift_messages.append(f"{key}: current loss contains non-finite values")
                continue

            drift_messages.extend(
                _compare_loss_to_expected_entry(
                    key=key,
                    loss=loss,
                    expected_entry=expected_entry,
                )
            )
            _emit_progress(progress_logger, f"[{index}/{total_configs}] completed {key}")

        recorded = set(baseline["configs"]) | set(baseline["expected_xfails"])
        discovered = {_config_key(path) for path in TOP_LEVEL_CONFIGS}
        missing_from_baseline = sorted(discovered - recorded)
        extra_in_baseline = sorted(recorded - discovered)
        if missing_from_baseline:
            drift_messages.append("missing baseline coverage: " + ", ".join(missing_from_baseline))
        if extra_in_baseline:
            drift_messages.append("stale baseline coverage: " + ", ".join(extra_in_baseline))
    finally:
        os.listdir = real_listdir

    return drift_messages


def build_loss_baseline_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 2,
        "init_seed": LOSS_BASELINE_INIT_SEED,
        "loss_seed": LOSS_BASELINE_LOSS_SEED,
        "batch_size": LOSS_BASELINE_BATCH_SIZE,
        "configs": {},
        "expected_xfails": {},
    }

    BASELINE_VALUES_DIR.mkdir(parents=True, exist_ok=True)
    expected_loss_files: set[Path] = set()

    real_listdir = os.listdir

    def safe_listdir(path: str | os.PathLike[str]):
        try:
            return real_listdir(path)
        except OSError:
            return ["placeholder"]

    os.listdir = safe_listdir
    try:
        for config_path in TOP_LEVEL_CONFIGS:
            key = _config_key(config_path)
            try:
                cfg = _config.get_config(str(config_path))
            except ValueError as exc:
                message = str(exc)
                if _is_missing_repo_id_list_error(message):
                    payload["expected_xfails"][key] = {
                        "relative_path": str(config_path.relative_to(REPO_ROOT)),
                        "reason": _canonicalize_expected_xfail_message(message),
                    }
                    continue
                raise

            loss = _compute_loss_array(cfg)
            loss_file_path = _loss_value_path_for_key(key)
            np.save(loss_file_path, np.ascontiguousarray(loss))
            expected_loss_files.add(loss_file_path.resolve())
            payload["configs"][key] = _build_baseline_config_entry(config_path, loss)
    finally:
        os.listdir = real_listdir

    for stale_file in BASELINE_VALUES_DIR.glob("*.npy"):
        if stale_file.resolve() not in expected_loss_files:
            stale_file.unlink()

    return payload


@pytest.mark.config
@pytest.mark.slow
@pytest.mark.parametrize("config_path", TOP_LEVEL_CONFIGS, ids=lambda path: path.stem)
def test_top_level_configs_match_saved_training_loss_baseline(
    config_path: Path,
    fake_dataset_listing,
) -> None:
    del fake_dataset_listing
    baseline = _load_loss_baseline()
    key = _config_key(config_path)
    expected_entry = baseline["configs"].get(key)
    expected_xfail = baseline["expected_xfails"].get(key)

    try:
        cfg = _config.get_config(str(config_path))
    except ValueError as exc:
        message = str(exc)
        if expected_xfail is not None and _is_missing_repo_id_list_error(message):
            assert expected_xfail["reason"] in message
            pytest.xfail(expected_xfail["reason"])
        raise

    assert isinstance(cfg, TrainConfig)
    assert expected_xfail is None, f"{key} now loads successfully, but baseline still marks it as xfail."
    assert expected_entry is not None, f"Missing saved loss baseline for {key}."

    first_loss = _compute_loss_array(cfg)
    second_loss = _compute_loss_array(cfg)

    assert first_loss.shape == second_loss.shape
    assert first_loss.size > 0
    assert np.isfinite(first_loss).all()
    np.testing.assert_array_equal(first_loss, second_loss)
    assert not _compare_loss_to_expected_entry(
        key=key,
        loss=first_loss,
        expected_entry=expected_entry,
    )


@pytest.mark.config
def test_all_top_level_config_files_are_covered_by_saved_baseline() -> None:
    baseline = _load_loss_baseline()
    recorded = set(baseline["configs"]) | set(baseline["expected_xfails"])
    discovered = {_config_key(path) for path in TOP_LEVEL_CONFIGS}

    assert TOP_LEVEL_CONFIGS, "No top-level config files were discovered under src/openpi/configs."
    assert all(path.parent.name == "configs" for path in TOP_LEVEL_CONFIGS)
    assert discovered == recorded


@pytest.mark.config
def test_saved_baseline_has_no_drift() -> None:
    assert not collect_top_level_config_loss_drift()
