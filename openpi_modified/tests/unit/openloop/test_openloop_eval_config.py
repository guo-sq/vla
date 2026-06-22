"""TDD tests for openloop eval configuration generation.

These tests drive the implementation of:
1. Config collection matching config-loss-drift coverage
2. Repo_ids extraction from training configs
3. Evaluation metrics structure (ADE, FDE, RMSE)
"""

from pathlib import Path
import sys
import tempfile
import types
from typing import ClassVar

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


class TestConfigCollection:
    """Tests for config collection matching config-loss-drift."""

    @pytest.mark.posttrain
    def test_collect_top_level_configs_returns_list(self) -> None:
        """Verify that collect_top_level_configs returns a list."""
        from scripts.generate_openloop_eval_configs import collect_top_level_configs

        configs = collect_top_level_configs()
        assert isinstance(configs, list)
        assert len(configs) > 0

    @pytest.mark.posttrain
    def test_collect_top_level_configs_returns_paths(self) -> None:
        """Verify that all returned items are Path objects."""
        from scripts.generate_openloop_eval_configs import collect_top_level_configs

        configs = collect_top_level_configs()
        for config in configs:
            assert isinstance(config, Path)
            assert config.suffix == ".py"

    @pytest.mark.posttrain
    def test_collect_top_level_configs_excludes_init_and_base(self) -> None:
        """Verify that __init__.py and base.py are excluded."""
        from scripts.generate_openloop_eval_configs import collect_top_level_configs

        configs = collect_top_level_configs()
        config_names = [c.name for c in configs]
        assert "__init__.py" not in config_names
        assert "base.py" not in config_names

    @pytest.mark.posttrain
    def test_collect_top_level_configs_matches_config_loss_drift_count(self) -> None:
        """Ensure we collect same number of configs as config-loss-drift."""
        from scripts.generate_openloop_eval_configs import collect_top_level_configs

        configs = collect_top_level_configs()

        # Same logic as config-loss-drift
        expected_configs = sorted(
            path
            for path in (REPO_ROOT / "src/openpi/configs").glob("*.py")
            if path.name not in {"__init__.py", "base.py"}
        )

        assert len(configs) == len(
            expected_configs
        ), f"Config count mismatch: got {len(configs)}, expected {len(expected_configs)}"

    @pytest.mark.posttrain
    def test_all_configs_are_in_configs_directory(self) -> None:
        """Verify all configs are from src/openpi/configs directory."""
        from scripts.generate_openloop_eval_configs import collect_top_level_configs

        configs = collect_top_level_configs()
        for config in configs:
            assert config.parent.name == "configs"
            assert "openpi" in str(config)


class TestRepoIdExtraction:
    """Tests for repo_id extraction from training configs."""

    @pytest.mark.posttrain
    def test_extract_repo_ids_from_config_returns_dict(self) -> None:
        """Verify that extraction returns a dict with expected keys."""
        from scripts.generate_openloop_eval_configs import extract_repo_ids_from_config

        # Use a known config
        config_path = REPO_ROOT / "src/openpi/configs/cfg_pi0.5_14_dim_example.py"
        if config_path.exists():
            result = extract_repo_ids_from_config(config_path)
            assert isinstance(result, dict)
            assert "config_path" in result
            assert "config_name" in result
            assert "repo_ids" in result
            assert "error" in result

    @pytest.mark.posttrain
    def test_extract_repo_ids_handles_missing_config(self) -> None:
        """Verify that extraction handles missing config gracefully."""
        from scripts.generate_openloop_eval_configs import extract_repo_ids_from_config

        result = extract_repo_ids_from_config(Path("/nonexistent/config.py"))
        assert isinstance(result, dict)
        assert result.get("error") is not None


class TestEvalMetricsStructure:
    """Tests for evaluation metrics structure."""

    @pytest.mark.posttrain
    def test_ade_metric_computes_correctly(self) -> None:
        """Test that ADE metric computes average displacement error."""
        from openpi.eval.system import TrajectoryADEMetric

        metric = TrajectoryADEMetric()
        pred = np.array([[[0.0, 0.0], [1.0, 1.0]]])  # 1 traj, 2 timesteps, 2 dims
        gt = np.array([[[0.0, 0.0], [2.0, 2.0]]])  # Expected L2: [0, sqrt(2)]

        metric.update_batch(pred, gt)
        ade = metric.compute()

        # ADE = mean([0, sqrt(2)]) = sqrt(2) / 2
        expected = np.sqrt(2) / 2
        assert np.isclose(ade, expected), f"Expected ADE={expected}, got {ade}"

    @pytest.mark.posttrain
    def test_fde_metric_computes_correctly(self) -> None:
        """Test that FDE metric computes final displacement error."""
        from openpi.eval.system import TrajectoryFDEMetric

        metric = TrajectoryFDEMetric()
        pred = np.array([[[0.0, 0.0], [1.0, 1.0]]])  # 1 traj, 2 timesteps
        gt = np.array([[[0.0, 0.0], [2.0, 2.0]]])  # Final L2: sqrt(2)

        metric.update_batch(pred, gt)
        fde = metric.compute()

        expected = np.sqrt(2)
        assert np.isclose(fde, expected), f"Expected FDE={expected}, got {fde}"

    @pytest.mark.posttrain
    def test_rmse_metric_computes_correctly(self) -> None:
        """Test that RMSE metric computes root mean squared error."""
        from openpi.eval.system import TrajectoryRMSEMetric

        metric = TrajectoryRMSEMetric()
        pred = np.array([[[0.0], [0.0]]])  # 1 traj, 2 timesteps, 1 dim
        gt = np.array([[[3.0], [4.0]]])  # Errors: 3, 4 -> MSE = (9+16)/2 = 12.5

        metric.update_batch(pred, gt)
        rmse = metric.compute()

        expected = np.sqrt(12.5)
        assert np.isclose(rmse, expected), f"Expected RMSE={expected}, got {rmse}"

    @pytest.mark.posttrain
    def test_bucketed_evaluator_returns_expected_structure(self) -> None:
        """Test that BucketedEvaluator returns expected metric structure."""
        from openpi.eval.system import BucketedEvaluator
        from openpi.eval.system import TrajectoryADEMetric
        from openpi.eval.system import TrajectoryFDEMetric
        from openpi.eval.system import TrajectoryRMSEMetric

        evaluator = BucketedEvaluator(
            metrics=[
                TrajectoryADEMetric(),
                TrajectoryFDEMetric(),
                TrajectoryRMSEMetric(),
            ],
        )

        pred = np.random.randn(4, 10, 7)  # 4 trajectories, 10 timesteps, 7 dims
        gt = np.random.randn(4, 10, 7)

        evaluator.add_batch(pred, gt)
        result = evaluator.compute()

        # Verify structure
        assert "global" in result
        assert "buckets" in result

        global_metrics = result["global"]
        assert "ade" in global_metrics
        assert "fde" in global_metrics
        assert "rmse" in global_metrics
        assert "count" in global_metrics
        assert global_metrics["count"] == 4

    @pytest.mark.posttrain
    def test_metrics_handle_multiple_batches(self) -> None:
        """Test that metrics correctly aggregate across multiple batches."""
        from openpi.eval.system import TrajectoryADEMetric

        metric = TrajectoryADEMetric()

        # First batch: 1 trajectory
        pred1 = np.array([[[0.0, 0.0], [1.0, 0.0]]])
        gt1 = np.array([[[0.0, 0.0], [1.0, 0.0]]])  # ADE = 0
        metric.update_batch(pred1, gt1)

        # Second batch: 1 trajectory
        pred2 = np.array([[[0.0, 0.0], [0.0, 0.0]]])
        gt2 = np.array([[[0.0, 0.0], [1.0, 0.0]]])  # ADE = 0.5
        metric.update_batch(pred2, gt2)

        ade = metric.compute()
        # Combined ADE = (0 + 0.5) / 2 = 0.25
        assert np.isclose(ade, 0.25), f"Expected ADE=0.25, got {ade}"
        assert metric.num_samples() == 2


class TestGenerateEvalConfig:
    """Tests for eval config generation."""

    @pytest.mark.posttrain
    def test_generate_eval_config_for_train_config_returns_dict(self) -> None:
        """Verify that generate_eval_config_for_train_config returns a dict."""
        from scripts.generate_openloop_eval_configs import generate_eval_config_for_train_config

        config_path = REPO_ROOT / "src/openpi/configs/cfg_pi0.5_14_dim_example.py"
        if config_path.exists():
            result = generate_eval_config_for_train_config(config_path)
            # Result may be None if extraction fails, which is acceptable
            if result is not None:
                assert isinstance(result, dict)
                assert "config" in result
                assert "repo_ids" in result

    @pytest.mark.posttrain
    def test_generate_all_eval_configs_returns_list(self) -> None:
        """Verify that generate_all_eval_configs returns a list."""
        from scripts.generate_openloop_eval_configs import generate_all_eval_configs

        results = generate_all_eval_configs()
        assert isinstance(results, list)

    @pytest.mark.posttrain
    def test_generate_all_eval_configs_has_valid_structure(self) -> None:
        """Verify that generated configs have valid structure."""
        from scripts.generate_openloop_eval_configs import generate_all_eval_configs

        results = generate_all_eval_configs()

        for cfg in results:
            assert isinstance(cfg, dict)
            assert "config" in cfg
            assert "repo_ids" in cfg
            assert isinstance(cfg["repo_ids"], list)

    @pytest.mark.posttrain
    def test_extract_repo_ids_distinguishes_config_load_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify config loading errors are reported distinctly."""
        from scripts.generate_openloop_eval_configs import extract_repo_ids_from_config

        def raise_value_error(_: str):
            raise ValueError("bad config")

        monkeypatch.setattr("scripts.generate_openloop_eval_configs._config.get_config", raise_value_error)

        result = extract_repo_ids_from_config(Path("/tmp/missing.py"))

        assert result["error"] == "config_load_error: bad config"

    @pytest.mark.posttrain
    def test_extract_repo_ids_distinguishes_data_creation_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify repo_id extraction errors are reported distinctly."""
        from scripts.generate_openloop_eval_configs import extract_repo_ids_from_config

        class DummyFactory:
            repo_id: ClassVar[tuple[str, ...]] = ()

            def create(self, assets_dirs, model_cfg):
                del assets_dirs, model_cfg
                raise OSError("dataset unavailable")

        class DummyConfig:
            data = DummyFactory()
            assets_dirs: ClassVar[tuple[str, ...]] = ()
            model = None

        monkeypatch.setattr("scripts.generate_openloop_eval_configs._config.get_config", lambda _: DummyConfig())

        result = extract_repo_ids_from_config(Path("/tmp/mock.py"))

        assert result["error"] == "repo_id_extraction_error: dataset unavailable"


class TestRunOpenloopEval:
    """Tests for run_openloop_eval script."""

    @pytest.mark.posttrain
    def test_run_openloop_eval_has_main_function(self) -> None:
        """Verify that run_openloop_eval has a main function."""
        from scripts.run_openloop_eval import main

        assert callable(main)

    @pytest.mark.posttrain
    def test_run_openloop_eval_has_required_functions(self) -> None:
        """Verify that run_openloop_eval has required functions."""
        from scripts.run_openloop_eval import run_batch_eval
        from scripts.run_openloop_eval import run_openloop_eval_for_config

        assert callable(run_openloop_eval_for_config)
        assert callable(run_batch_eval)

    @pytest.mark.posttrain
    def test_normalize_repo_id_rejects_parent_traversal(self) -> None:
        """Reject repo_id values that escape the expected results tree."""
        from scripts.run_openloop_eval import _normalize_repo_id

        with pytest.raises(ValueError, match="must not contain traversal segments"):
            _normalize_repo_id("oss_data/../secret")

    @pytest.mark.posttrain
    def test_normalize_repo_id_rejects_absolute_path(self) -> None:
        """Reject absolute repo_id values."""
        from scripts.run_openloop_eval import _normalize_repo_id

        with pytest.raises(ValueError, match="must be a relative path"):
            _normalize_repo_id("/mnt/oss_data/demo")

    @pytest.mark.posttrain
    def test_resolve_existing_dir_requires_directory(self, tmp_path: Path) -> None:
        """Reject file paths for directory-only arguments."""
        from scripts.run_openloop_eval import _resolve_existing_dir

        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content", encoding="utf-8")

        with pytest.raises(NotADirectoryError, match="is not a directory"):
            _resolve_existing_dir(str(file_path), arg_name="dataset_root")

    @pytest.mark.posttrain
    def test_normalize_train_config_path_requires_repo_relative_file(self, tmp_path: Path) -> None:
        """Reject config paths that point outside the repository."""
        from scripts.run_openloop_eval import _normalize_train_config_path

        # CI redirects pytest temp roots under the repo checkout, so build the
        # probe file in a sibling temp dir that is guaranteed to be outside REPO_ROOT.
        with tempfile.TemporaryDirectory(dir=REPO_ROOT.parent) as temp_dir:
            external_config = Path(temp_dir) / "external.py"
            external_config.write_text("cfg = None\n", encoding="utf-8")

            with pytest.raises(ValueError, match="must stay within repo root"):
                _normalize_train_config_path(str(external_config))

    @pytest.mark.posttrain
    def test_run_openloop_eval_uses_explicit_results_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Allow callers to place raw artifacts outside read-only checkpoint trees."""
        from scripts.run_openloop_eval import run_openloop_eval_for_config

        checkpoint_dir = tmp_path / "checkpoints" / "demo_exp" / "10000"
        checkpoint_dir.mkdir(parents=True)
        dataset_root = tmp_path / "dataset"
        dataset_root.mkdir()
        results_dir = tmp_path / "artifacts"
        vis_dir = tmp_path / "vis"
        config_path = REPO_ROOT / "src/openpi/configs/cfg_pi0.5_14_dim_example.py"
        captured: dict[str, str] = {}

        def fake_main(**kwargs) -> None:
            captured.update({k: str(v) for k, v in kwargs.items()})
            repo_dir = Path(kwargs["results_dir"]) / kwargs["repo_id"]
            repo_dir.mkdir(parents=True, exist_ok=True)
            np.save(repo_dir / "test_all_preds.npy", np.zeros((2, 3), dtype=np.float32))
            np.save(repo_dir / "test_all_gts.npy", np.ones((2, 3), dtype=np.float32))

        monkeypatch.setitem(sys.modules, "test", types.SimpleNamespace(main=fake_main))

        result = run_openloop_eval_for_config(
            train_config_path=str(config_path),
            checkpoint_dir=str(checkpoint_dir),
            dataset_root=str(dataset_root),
            repo_id="robomind_2/agilex/fold_clothes/success_episodes",
            num_batches=2,
            batch_size=16,
            vis_dir=str(vis_dir),
            results_dir=str(results_dir),
        )

        assert captured["results_dir"] == str(results_dir.resolve())
        assert captured["vis_dir"] == str(vis_dir.resolve())
        assert result["count"] == 2
        assert "mse" in result
