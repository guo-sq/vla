"""Integration tests for the RL norm-stats pinning pipeline (Part 1 of PR #100 follow-up).

Covers:
- Fingerprint generation / validation (B2)
- Per-repo raw_lengths storage format + cross-repo aggregation at load time
- LeRobotRLDataset strict-mode resolution helper (B3)
- MultiRLAnyverseDataset pinned short-circuit + consistency check (B4)
- data_loader_rl.create_anyverse_dataset hard-error pre-check

These tests intentionally avoid touching real on-disk LeRobot datasets — the
heavy end-to-end integration is covered by the seatbelt smoke test. Here we
exercise the NEW infra (fingerprint, pinned load path, strict gate, shortcut)
with minimal mocks so the suite stays fast and hermetic.

TDD RED phase: every test below asserts against helpers/behavior that do not
yet exist. `pytest tests/integration/test_rl_norm_stats_pinning.py -xvs`
should fail with ImportError / AttributeError / assertion mismatches before
Green phase implementation lands.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

# ---------------------------------------------------------------------------
# Fingerprint tests (B2)
# ---------------------------------------------------------------------------


class TestFingerprint:
    """B2: fingerprint ensures stale precomputed files are rejected."""

    def test_fingerprint_stored_in_precomputed_json(self, tmp_path: Path):
        """Writing rl_norm_stats.json must persist a non-empty fingerprint."""
        from scripts.compute_rl_norm_stats import _compute_rl_norm_stats_fingerprint
        from scripts.compute_rl_norm_stats import write_rl_norm_stats_file

        value_net_cfg = {
            "returns_norm_strategy": "per_task",
            "returns_norm_percentile": 1.0,
            "returns_norm_length": None,
            "exclude_failures": True,
            "failure_decrease_threshold": 0.0,
        }
        fingerprint = _compute_rl_norm_stats_fingerprint(repo_id="seatbelt/repo_a", value_net_cfg=value_net_cfg)
        assert isinstance(fingerprint, str), "fingerprint must be a string"
        assert fingerprint, "fingerprint must be non-empty"

        path = tmp_path / "rl_norm_stats.json"
        raw_lengths = {"hang_seatbelt": [100, 120, 140], "take_off_seatbelt": [80, 90]}
        write_rl_norm_stats_file(path, task_to_raw_lengths=raw_lengths, fingerprint=fingerprint)

        on_disk = json.loads(path.read_text())
        assert on_disk["fingerprint"] == fingerprint
        assert on_disk["task_to_raw_lengths"] == raw_lengths

    def test_fingerprint_deterministic_across_field_order(self):
        """Fingerprint must hash sorted(dict.items()) so key order is irrelevant."""
        from scripts.compute_rl_norm_stats import _compute_rl_norm_stats_fingerprint

        cfg_a = {
            "returns_norm_strategy": "per_task",
            "returns_norm_percentile": 0.9,
            "returns_norm_length": None,
            "exclude_failures": True,
            "failure_decrease_threshold": 0.0,
        }
        # Same content, reversed insertion order
        cfg_b = {
            "failure_decrease_threshold": 0.0,
            "exclude_failures": True,
            "returns_norm_length": None,
            "returns_norm_percentile": 0.9,
            "returns_norm_strategy": "per_task",
        }

        fp_a = _compute_rl_norm_stats_fingerprint(repo_id="repo", value_net_cfg=cfg_a)
        fp_b = _compute_rl_norm_stats_fingerprint(repo_id="repo", value_net_cfg=cfg_b)
        assert fp_a == fp_b


# ---------------------------------------------------------------------------
# Load helper tests (base_cfg._load_rl_norm_stats)
# ---------------------------------------------------------------------------


class TestLoadRlNormStats:
    """Tests for the cross-repo aggregating loader in base_cfg."""

    @staticmethod
    def _write_stub(tmp_path: Path, repo_id: str, raw_lengths: dict, value_net_cfg: dict) -> Path:
        """Create a stub assets_dir/<asset_id>/rl_norm_stats.json with a fresh fingerprint."""
        from scripts.compute_rl_norm_stats import _compute_rl_norm_stats_fingerprint
        from scripts.compute_rl_norm_stats import write_rl_norm_stats_file

        asset_dir = tmp_path / repo_id
        asset_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = _compute_rl_norm_stats_fingerprint(repo_id=repo_id, value_net_cfg=value_net_cfg)
        path = asset_dir / "rl_norm_stats.json"
        write_rl_norm_stats_file(path, task_to_raw_lengths=raw_lengths, fingerprint=fingerprint)
        return path

    def test_aggregates_raw_lengths_across_repos(self, tmp_path: Path):
        """
        The loader must extend raw_lengths across repos by task key,
        then compute the percentile over the combined list — matching
        the existing MultiRLAnyverseDataset merge semantics
        (rl_dataset.py:379-384).
        """
        from openpi.training.base_cfg import _load_rl_norm_stats

        value_net_cfg = {
            "returns_norm_strategy": "per_task",
            "returns_norm_percentile": 1.0,  # max
            "returns_norm_length": None,
            "exclude_failures": True,
            "failure_decrease_threshold": 0.0,
        }

        # repo_a: "hang" = [100, 200], "take_off" = [80]
        self._write_stub(
            tmp_path,
            repo_id="repo_a",
            raw_lengths={"hang": [100, 200], "take_off": [80]},
            value_net_cfg=value_net_cfg,
        )
        # repo_b: "hang" = [500] (dominates the percentile), "new_task" = [42]
        self._write_stub(
            tmp_path,
            repo_id="repo_b",
            raw_lengths={"hang": [500], "new_task": [42]},
            value_net_cfg=value_net_cfg,
        )

        result = _load_rl_norm_stats(
            assets_dir=tmp_path,
            asset_id=["repo_a", "repo_b"],
            repo_id=["repo_a", "repo_b"],
            value_net_cfg=value_net_cfg,
        )

        assert result is not None
        # percentile=1.0 uses max across combined lists:
        # hang: max(100, 200, 500) = 500
        # take_off: max(80) = 80
        # new_task: max(42) = 42
        assert result == {"hang": 500, "take_off": 80, "new_task": 42}

    def test_returns_none_for_non_per_task_strategy(self, tmp_path: Path):
        """Non per_task strategies must short-circuit and return None."""
        from openpi.training.base_cfg import _load_rl_norm_stats

        for strategy in ("per_episode", "fixed"):
            value_net_cfg = {
                "returns_norm_strategy": strategy,
                "returns_norm_percentile": 1.0,
                "returns_norm_length": 3000,
                "exclude_failures": True,
                "failure_decrease_threshold": 0.0,
            }
            result = _load_rl_norm_stats(
                assets_dir=tmp_path,
                asset_id="repo_a",
                repo_id="repo_a",
                value_net_cfg=value_net_cfg,
            )
            assert result is None, f"strategy={strategy} must return None"

    def test_rejects_stale_fingerprint_with_force_command_hint(self, tmp_path: Path):
        """Fingerprint mismatch must raise ValueError mentioning `--force`."""
        from openpi.training.base_cfg import _load_rl_norm_stats

        value_net_cfg = {
            "returns_norm_strategy": "per_task",
            "returns_norm_percentile": 1.0,
            "returns_norm_length": None,
            "exclude_failures": True,
            "failure_decrease_threshold": 0.0,
        }

        # Write with one fingerprint, then alter cfg to force mismatch at load.
        self._write_stub(
            tmp_path,
            repo_id="repo_a",
            raw_lengths={"hang": [100]},
            value_net_cfg=value_net_cfg,
        )
        mutated_cfg = dict(value_net_cfg)
        mutated_cfg["exclude_failures"] = False  # flips fingerprint

        with pytest.raises(ValueError, match=r"Fingerprint mismatch"):
            _load_rl_norm_stats(
                assets_dir=tmp_path,
                asset_id="repo_a",
                repo_id="repo_a",
                value_net_cfg=mutated_cfg,
            )

        # Second check: the error must tell the user how to fix it.
        with pytest.raises(ValueError, match=r"--force"):
            _load_rl_norm_stats(
                assets_dir=tmp_path,
                asset_id="repo_a",
                repo_id="repo_a",
                value_net_cfg=mutated_cfg,
            )


# ---------------------------------------------------------------------------
# LeRobotRLDataset strict-mode resolver tests (B3)
# ---------------------------------------------------------------------------


class TestResolvePerTaskNormLength:
    """B3: helper that decides pinned vs. legacy inside LeRobotRLDataset.__init__."""

    def test_strict_mode_raises_on_missing_pinned_with_script_hint(self):
        """Default strict mode: missing pinned must raise with a clear error
        that both names the missing config key AND points at the compute script.
        """
        from openpi.training.rl_dataset import _resolve_per_task_norm_length

        value_net_cfg = {
            "returns_norm_strategy": "per_task",
            "returns_norm_percentile": 1.0,
            # no pinned_task_to_norm_length
            # no strict_rl_norm_stats → defaults to True
        }
        with pytest.raises(
            ValueError,
            match=r"(?s)pinned_task_to_norm_length.*compute_rl_norm_stats",
        ):
            _resolve_per_task_norm_length(
                value_net_cfg,
                task_to_raw_lengths={"hang": [100, 200]},
                returns_norm_percentile=1.0,
            )

    def test_non_strict_mode_falls_back_with_warning(self, caplog):
        """strict_rl_norm_stats=False: falls back to legacy percentile on raw_lengths."""
        import logging

        from openpi.training.rl_dataset import _resolve_per_task_norm_length

        value_net_cfg = {
            "returns_norm_strategy": "per_task",
            "returns_norm_percentile": 1.0,
            "strict_rl_norm_stats": False,
        }

        with caplog.at_level(logging.WARNING):
            result = _resolve_per_task_norm_length(
                value_net_cfg,
                task_to_raw_lengths={"hang": [100, 200, 50]},
                returns_norm_percentile=1.0,
            )

        assert result == {"hang": 200}  # max of raw lengths
        assert any(
            "legacy" in rec.message.lower() or "fallback" in rec.message.lower() for rec in caplog.records
        ), "must log a warning about the legacy fallback"

    def test_pinned_overrides_legacy_computation(self):
        """When pinned is present, it is used as-is (no percentile recomputation)."""
        from openpi.training.rl_dataset import _resolve_per_task_norm_length

        pinned = {"hang": 777, "take_off": 333}
        value_net_cfg = {
            "returns_norm_strategy": "per_task",
            "pinned_task_to_norm_length": pinned,
        }
        # raw_lengths contains completely different numbers; must be ignored.
        result = _resolve_per_task_norm_length(
            value_net_cfg,
            task_to_raw_lengths={"hang": [1, 2, 3], "take_off": [4, 5]},
            returns_norm_percentile=1.0,
        )
        assert result == pinned
        assert result is not pinned, "should return a copy so caller mutations stay local"


# ---------------------------------------------------------------------------
# MultiRLAnyverseDataset pinned short-circuit + consistency check (B4)
# ---------------------------------------------------------------------------


class TestMultirlPinnedShortCircuit:
    """B4: MultiRL merge is short-circuited when pinned is present; missing tasks raise."""

    @staticmethod
    def _fake_subdataset(raw_lengths: dict[str, list[int]]) -> Mock:
        ds = Mock()
        ds._task_to_raw_lengths = raw_lengths  # noqa: SLF001
        return ds

    def test_short_circuits_merge_when_pinned_present(self):
        """If pinned present, helper returns the pinned dict unchanged."""
        from openpi.training.rl_dataset import _maybe_short_circuit_pinned

        pinned = {"hang": 500, "take_off": 100}
        sub_a = self._fake_subdataset({"hang": [1, 2], "take_off": [3]})
        sub_b = self._fake_subdataset({"hang": [4]})

        result = _maybe_short_circuit_pinned(
            value_net_cfg={"pinned_task_to_norm_length": pinned},
            datasets=[sub_a, sub_b],
        )
        assert result == pinned

    def test_consistency_check_raises_when_pinned_missing_task(self):
        """Sub-dataset has a task not covered by pinned → clear ValueError."""
        from openpi.training.rl_dataset import _maybe_short_circuit_pinned

        pinned = {"hang": 500}  # missing "take_off"
        sub = self._fake_subdataset({"hang": [1], "take_off": [2]})

        with pytest.raises(ValueError, match=r"take_off"):
            _maybe_short_circuit_pinned(
                value_net_cfg={"pinned_task_to_norm_length": pinned},
                datasets=[sub],
            )


# ---------------------------------------------------------------------------
# data_loader_rl pre-check
# ---------------------------------------------------------------------------


class TestCreateAnyverseDatasetPreCheck:
    """Training entrypoint must hard-error when per_task lacks pinned stats."""

    def test_hard_errors_without_pinned_for_per_task(self):
        from openpi.training.data_loader_rl import create_anyverse_dataset

        data_config = Mock()
        data_config.repo_id = "seatbelt/repo_a"
        data_config.value_net_cfg = {
            "returns_norm_strategy": "per_task",
            "returns_norm_percentile": 1.0,
            # intentionally no pinned_task_to_norm_length
        }
        model_config = Mock()

        with pytest.raises(ValueError, match=r"compute_rl_norm_stats"):
            create_anyverse_dataset(data_config, model_config)

    def test_does_not_error_for_non_per_task_strategy(self):
        """per_episode / fixed must NOT trigger the pre-check (no pinned needed)."""
        import openpi.training.data_loader_rl as data_loader_rl

        data_config = Mock()
        data_config.repo_id = "seatbelt/repo_a"
        data_config.value_net_cfg = {
            "returns_norm_strategy": "fixed",
            "returns_norm_length": 3000,
        }
        model_config = Mock()
        model_config.action_horizon = 50

        # We just need to prove the pre-check doesn't raise before MultiRL is
        # instantiated. Swap MultiRL for a Mock so the call returns without
        # touching real data.
        sentinel = Mock(name="multirl_dataset")
        original = data_loader_rl.MultiRLAnyverseDataset
        data_loader_rl.MultiRLAnyverseDataset = Mock(return_value=sentinel)
        try:
            result = data_loader_rl.create_anyverse_dataset(data_config, model_config)
        finally:
            data_loader_rl.MultiRLAnyverseDataset = original
        assert result is sentinel
