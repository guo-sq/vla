"""Integration tests for openloop evaluation consistency.

This module validates that task-specific openloop evaluation baselines remain
loadable and that cfg_2603 training configs expose the embedded test_cfg needed
for evaluation.

Usage:
    pytest tests/integration/test_openloop_eval_consistency.py -v
    pytest tests/integration/test_openloop_eval_consistency.py -m manual -v
"""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import shutil
import tempfile
from typing import Any

import pytest

from openpi.training.base_cfg import TestConfig as OpenloopTestConfig
from openpi.training.config import TrainConfig

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration

BASELINE_DIR = Path(__file__).parent.parent / "baseline_metrics"
REPO_ROOT = Path(__file__).parent.parent.parent

# Task-specific evaluation cases.
OPENLOOP_EVAL_CASES = [
    {
        "case_id": "all_public_dataset",
        "config": "src/openpi/configs/cfg_2603/cfg_pi0.5_28_dim.all_public_datasets.py",
        "baseline_file": "all_public_dataset_baseline.json",
    },
    {
        "case_id": "pack_socks",
        "config": "src/openpi/configs/cfg_2603/cfg_pi05_base_pack_socks_data_pure_recover_arx4_bs256_0318_more_upsample_recover.py",
        "baseline_file": "pack_socks_baseline.json",
    },
    {
        "case_id": "seatbelt",
        "config": "src/openpi/configs/cfg_2603/cfg_pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery.py",
        "baseline_file": "seatbelt_baseline.json",
    },
]


def baseline_path(case: dict[str, Any]) -> Path:
    """Return the baseline file path for a case."""
    return BASELINE_DIR / case["baseline_file"]


def load_baseline_metrics(case: dict[str, Any]) -> dict[str, Any] | None:
    """Load baseline metrics for an evaluation case."""
    baseline_file = baseline_path(case)
    if not baseline_file.exists():
        return None
    return json.loads(baseline_file.read_text(encoding="utf-8"))


def save_baseline_metrics(case: dict[str, Any], metrics: dict[str, Any]) -> None:
    """Save baseline metrics for an evaluation case."""
    baseline_file = baseline_path(case)
    baseline_file.parent.mkdir(parents=True, exist_ok=True)
    baseline_file.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")


def compute_metrics_from_results(preds_path: Path, gts_path: Path) -> dict[str, float]:
    """Compute metrics from saved predictions and ground truths."""
    import numpy as np

    if not preds_path.exists() or not gts_path.exists():
        raise FileNotFoundError(f"Results files not found: {preds_path} or {gts_path}")

    preds = np.load(preds_path)
    gts = np.load(gts_path)

    diff = preds - gts
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))

    if preds.ndim == 3:
        timestep_l2 = np.linalg.norm(preds - gts, axis=-1)
        per_traj_ade = np.mean(timestep_l2, axis=1)
        ade = float(np.mean(per_traj_ade))
        final_l2 = np.linalg.norm(preds[:, -1, :] - gts[:, -1, :], axis=-1)
        fde = float(np.mean(final_l2))
    else:
        ade = rmse
        fde = rmse

    return {
        "mse": mse,
        "rmse": rmse,
        "ade": ade,
        "fde": fde,
        "count": int(preds.shape[0]),
    }


def results_repo_dir(results_dir: Path, repo_id: str | None) -> Path:
    """Return the result directory used by scripts/test.py for a repo id."""
    return results_dir / (Path(repo_id) if repo_id else Path("default"))


@contextmanager
def isolated_results_dir(case_id: str):
    """Create a per-run results directory so concurrent CI jobs do not collide."""
    run_token = (
        os.getenv("CI_JOB_ID") or os.getenv("GITHUB_RUN_ID") or os.getenv("PYTEST_XDIST_WORKER") or str(os.getpid())
    )
    results_dir = Path(tempfile.mkdtemp(prefix=f"openpi_eval_results_{case_id}_{run_token}_"))
    try:
        yield results_dir
    finally:
        shutil.rmtree(results_dir, ignore_errors=True)


def load_config_symbols(case: dict[str, Any]) -> dict[str, Any]:
    """Execute a config module and return its exported symbols."""
    config_path = REPO_ROOT / case["config"]
    return runpy.run_path(str(config_path))


class TestOpenloopEvalBaselineFiles:
    """Test that baseline files exist and are valid."""

    @pytest.mark.parametrize("case", OPENLOOP_EVAL_CASES, ids=[case["case_id"] for case in OPENLOOP_EVAL_CASES])
    def test_baseline_file_exists(self, case: dict[str, Any]) -> None:
        """Test that baseline file exists for each evaluation case."""
        baseline_file = baseline_path(case)
        assert baseline_file.exists(), f"Baseline file not found: {baseline_file}"

    @pytest.mark.parametrize("case", OPENLOOP_EVAL_CASES, ids=[case["case_id"] for case in OPENLOOP_EVAL_CASES])
    def test_baseline_file_valid_json(self, case: dict[str, Any]) -> None:
        """Test that baseline file contains valid JSON."""
        baseline_file = baseline_path(case)
        if not baseline_file.exists():
            pytest.skip(f"Baseline file not found: {baseline_file}")

        content = baseline_file.read_text(encoding="utf-8")
        data = json.loads(content)

        assert "model" in data, "Baseline file must contain 'model' field"
        assert "metrics" in data, "Baseline file must contain 'metrics' field"
        assert "mse" in data["metrics"], "Metrics must contain 'mse' field"
        assert "count" in data["metrics"], "Metrics must contain 'count' field"


class TestOpenloopEvalConsistency:
    """Test that evaluation results are consistent with baseline."""

    @pytest.mark.manual
    @pytest.mark.parametrize("case", OPENLOOP_EVAL_CASES, ids=[case["case_id"] for case in OPENLOOP_EVAL_CASES])
    def test_version_consistency(self, case: dict[str, Any]) -> None:
        """Test that current evaluation results match baseline metrics.

        This test is marked as 'manual' because it requires:
        1. Access to model checkpoints
        2. Access to dataset files
        3. GPU for inference

        Run manually with: pytest tests/integration/test_openloop_eval_consistency.py -m manual -v
        """
        import subprocess
        import sys

        baseline = load_baseline_metrics(case)
        if baseline is None:
            pytest.skip(f"No baseline found for case: {case['case_id']}")

        # Skip if baseline metrics are zeros (placeholder)
        if baseline["metrics"].get("mse", 0) == 0 and baseline["metrics"].get("count", 0) == 0:
            pytest.skip(f"Baseline metrics are placeholder (mse=0, count=0) for case: {case['case_id']}")

        # Run openloop evaluation
        checkpoint_dir = baseline.get("checkpoint_dir", "")
        dataset_root = baseline.get("dataset_root", "")
        repo_id = baseline.get("repo_id", "")
        config = case["config"]

        if not Path(checkpoint_dir).exists():
            pytest.skip(f"Checkpoint not found: {checkpoint_dir}")
        if not Path(dataset_root).exists():
            pytest.skip(f"Dataset root not found: {dataset_root}")

        with isolated_results_dir(case["case_id"]) as results_dir:
            cmd = [
                sys.executable,
                str(REPO_ROOT / "scripts/test.py"),
                "--ckpt_dir",
                checkpoint_dir,
                "--dataset_root",
                dataset_root,
                "--config_name",
                config,
                "--repo_id",
                repo_id,
                "--num_batches",
                str(baseline.get("num_batches", 10)),
                "--batch_size",
                str(baseline.get("batch_size", 64)),
                "--results_dir",
                str(results_dir),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)
            if result.returncode != 0:
                pytest.fail(f"Evaluation failed: {result.stderr}")

            results_case_dir = results_repo_dir(results_dir, repo_id)
            preds_path = results_case_dir / "test_all_preds.npy"
            gts_path = results_case_dir / "test_all_gts.npy"

            current_metrics = compute_metrics_from_results(preds_path, gts_path)

        # Compare with baseline
        baseline_mse = baseline["metrics"]["mse"]
        current_mse = current_metrics["mse"]

        # Allow 5% tolerance for numerical precision
        tolerance = 0.05
        if baseline_mse > 0:
            relative_diff = abs(current_mse - baseline_mse) / baseline_mse
            assert relative_diff <= tolerance, (
                f"MSE differs from baseline by {relative_diff:.2%} "
                f"(baseline={baseline_mse:.6f}, current={current_mse:.6f})"
            )

        print(f"Case: {case['case_id']}")
        print(f"  Baseline MSE: {baseline_mse:.6f}")
        print(f"  Current MSE:  {current_mse:.6f}")
        print(f"  Difference:   {abs(current_mse - baseline_mse):.6f}")


class TestOpenloopEvalConfigIntegrity:
    """Test that embedded evaluation config files are valid."""

    @pytest.mark.parametrize("case", OPENLOOP_EVAL_CASES, ids=[case["case_id"] for case in OPENLOOP_EVAL_CASES])
    def test_config_file_exists(self, case: dict[str, Any]) -> None:
        """Test that config file exists."""
        config_path = REPO_ROOT / case["config"]
        assert config_path.exists(), f"Config file not found: {config_path}"

    @pytest.mark.parametrize("case", OPENLOOP_EVAL_CASES, ids=[case["case_id"] for case in OPENLOOP_EVAL_CASES])
    def test_config_has_test_cfg(self, case: dict[str, Any]) -> None:
        """Test that config file loads successfully and exposes a valid test_cfg."""
        symbols = load_config_symbols(case)
        cfg = symbols.get("cfg")
        test_cfg = symbols.get("test_cfg")

        assert isinstance(cfg, TrainConfig), f"Config file must export a TrainConfig as 'cfg': {case['config']}"
        assert isinstance(
            test_cfg, OpenloopTestConfig
        ), f"Config file must export a TestConfig as 'test_cfg': {case['config']}"
        assert test_cfg.config == case["config"], f"test_cfg.config must match the config path: {case['config']}"
        assert test_cfg.checkpoint_dir, f"test_cfg.checkpoint_dir must be set: {case['config']}"
        assert test_cfg.dataset_root, f"test_cfg.dataset_root must be set: {case['config']}"

    @pytest.mark.parametrize("case", OPENLOOP_EVAL_CASES, ids=[case["case_id"] for case in OPENLOOP_EVAL_CASES])
    def test_config_repo_id_sampling(self, case: dict[str, Any]) -> None:
        """Test that config implements repo_id sampling."""
        config_path = REPO_ROOT / case["config"]
        if not config_path.exists():
            pytest.skip(f"Config file not found: {config_path}")

        content = config_path.read_text(encoding="utf-8")
        assert (
            "EVAL_SAMPLE_RATE" in content or "sample_rate" in content.lower()
        ), f"Config file should define sample rate for repo_id sampling: {config_path}"
        assert (
            "_sample_repo_ids" in content or "sample_repo_ids" in content
        ), f"Config file should implement repo_id sampling function: {config_path}"


class TestOpenloopEvalBaselineRefresh:
    """Utility class for updating baseline metrics."""

    @pytest.mark.manual
    def test_update_all_baselines(self) -> None:
        """Update baseline metrics for all evaluation cases.

        Run manually with: pytest tests/integration/test_openloop_eval_consistency.py::TestOpenloopEvalBaselineRefresh::test_update_all_baselines -m manual -v
        """
        from datetime import UTC
        from datetime import datetime
        import subprocess
        import sys

        import git

        try:
            repo = git.Repo(REPO_ROOT)
            git_commit = repo.head.commit.hexsha[:8]
        except Exception:
            git_commit = "unknown"

        for case in OPENLOOP_EVAL_CASES:
            baseline = load_baseline_metrics(case)
            if baseline is None:
                print(f"Skipping {case['case_id']}: no baseline file")
                continue

            checkpoint_dir = baseline.get("checkpoint_dir", "")
            dataset_root = baseline.get("dataset_root", "")
            repo_id = baseline.get("repo_id", "")
            config = case["config"]

            if not Path(checkpoint_dir).exists():
                print(f"Skipping {case['case_id']}: checkpoint not found")
                continue
            if not Path(dataset_root).exists():
                print(f"Skipping {case['case_id']}: dataset root not found")
                continue

            with isolated_results_dir(case["case_id"]) as results_dir:
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "scripts/test.py"),
                    "--ckpt_dir",
                    checkpoint_dir,
                    "--dataset_root",
                    dataset_root,
                    "--config_name",
                    config,
                    "--repo_id",
                    repo_id,
                    "--num_batches",
                    str(baseline.get("num_batches", 10)),
                    "--batch_size",
                    str(baseline.get("batch_size", 64)),
                    "--results_dir",
                    str(results_dir),
                ]

                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT), check=False)
                if result.returncode != 0:
                    print(f"Failed to evaluate {case['case_id']}: {result.stderr}")
                    continue

                results_case_dir = results_repo_dir(results_dir, repo_id)
                preds_path = results_case_dir / "test_all_preds.npy"
                gts_path = results_case_dir / "test_all_gts.npy"

                try:
                    metrics = compute_metrics_from_results(preds_path, gts_path)
                except FileNotFoundError:
                    print(f"Skipping {case['case_id']}: results files not found")
                    continue

            updated_baseline = {
                **baseline,
                "evaluated_at": datetime.now(UTC).isoformat(),
                "git_commit": git_commit,
                "metrics": metrics,
            }

            save_baseline_metrics(case, updated_baseline)
            print(f"Updated baseline for {case['case_id']}: MSE={metrics['mse']:.6f}, count={metrics['count']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
