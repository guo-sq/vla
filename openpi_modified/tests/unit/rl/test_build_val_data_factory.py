"""Unit tests for ``scripts.train_rl._build_val_data_factory``.

Covers two val-side overrides:

1. ``ValueReturnsPreprocessor.exclude_failures`` True -> False.
2. ``value_net_cfg["cross_negative_rate"]`` > 0 -> 0.0 (new fix, 2026-04).

Also exercises edge cases (``value_net_cfg`` is None / empty / already 0)
and verifies no in-place mutation of the original train config.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from openpi.training.frame_attributes_preprocessors import ValueReturnsPreprocessor
from scripts.train_rl import _build_val_data_factory


@dataclasses.dataclass(frozen=True)
class _FakeBaseConfig:
    value_net_cfg: dict[str, Any] | None = None
    frame_attributes_preprocessors: list[Any] | None = None


@dataclasses.dataclass
class _FakeFactory:
    base_config: _FakeBaseConfig | None = None
    root_dir: str | None = "/train/root"
    repo_id: str | None = "train_repo"


@dataclasses.dataclass
class _FakeTrainConfig:
    data: _FakeFactory = dataclasses.field(default_factory=_FakeFactory)
    validation_repo_id: str | list[str] | None = "val_repo"
    validation_root_dir: str | None = None


def _cfg(value_net_cfg, preprocessors=None) -> _FakeTrainConfig:
    return _FakeTrainConfig(
        data=_FakeFactory(
            base_config=_FakeBaseConfig(
                value_net_cfg=value_net_cfg,
                frame_attributes_preprocessors=preprocessors,
            ),
        ),
    )


class TestCrossNegativeRateOverride:
    def test_positive_rate_forced_to_zero(self):
        cfg = _cfg({"cross_negative_rate": 0.5, "returns_norm_strategy": "per_task"})
        val = _build_val_data_factory(cfg)
        assert val.base_config.value_net_cfg["cross_negative_rate"] == 0.0

    def test_other_value_net_cfg_keys_preserved(self):
        cfg = _cfg(
            {
                "cross_negative_rate": 0.5,
                "returns_norm_strategy": "per_task",
                "returns_norm_length": 77,
            }
        )
        val = _build_val_data_factory(cfg)
        vnc = val.base_config.value_net_cfg
        assert vnc["returns_norm_strategy"] == "per_task"
        assert vnc["returns_norm_length"] == 77

    def test_original_config_not_mutated(self):
        original_vnc = {"cross_negative_rate": 0.5, "returns_norm_strategy": "per_task"}
        cfg = _cfg(original_vnc)
        _build_val_data_factory(cfg)
        assert cfg.data.base_config.value_net_cfg["cross_negative_rate"] == 0.5
        assert original_vnc["cross_negative_rate"] == 0.5


class TestExcludeFailuresNotRegressed:
    def test_exclude_failures_forced_to_false(self):
        preprocessors = [ValueReturnsPreprocessor(exclude_failures=True)]
        cfg = _cfg({"cross_negative_rate": 0.5}, preprocessors=preprocessors)
        val = _build_val_data_factory(cfg)
        vrps = [p for p in val.base_config.frame_attributes_preprocessors if isinstance(p, ValueReturnsPreprocessor)]
        assert len(vrps) == 1
        assert vrps[0].exclude_failures is False


class TestEdgeCases:
    def test_value_net_cfg_none_does_not_raise(self):
        cfg = _cfg(None)
        val = _build_val_data_factory(cfg)
        assert val.base_config.value_net_cfg is None

    def test_value_net_cfg_empty_dict_does_not_raise(self):
        cfg = _cfg({})
        val = _build_val_data_factory(cfg)
        assert val.base_config.value_net_cfg == {}

    def test_cross_negative_rate_already_zero_no_extra_replace(self):
        cfg = _cfg({"cross_negative_rate": 0.0, "returns_norm_strategy": "per_task"})
        original_base_config = cfg.data.base_config
        val = _build_val_data_factory(cfg)
        assert val.base_config is original_base_config
        assert val.base_config.value_net_cfg["cross_negative_rate"] == 0.0

    def test_missing_cross_negative_rate_key_no_override(self):
        cfg = _cfg({"returns_norm_strategy": "per_task"})
        val = _build_val_data_factory(cfg)
        assert "cross_negative_rate" not in val.base_config.value_net_cfg


class TestValRepoIdPropagation:
    def test_validation_repo_id_overrides_train_repo_id(self):
        cfg = _cfg({"cross_negative_rate": 0.5})
        cfg.validation_repo_id = "val_repo_explicit"
        val = _build_val_data_factory(cfg)
        assert val.repo_id == "val_repo_explicit"

    def test_validation_root_dir_falls_back_to_train_root(self):
        cfg = _cfg({"cross_negative_rate": 0.5})
        cfg.validation_root_dir = None
        val = _build_val_data_factory(cfg)
        assert val.root_dir == "/train/root"
