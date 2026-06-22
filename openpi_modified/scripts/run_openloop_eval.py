#!/usr/bin/env python3
"""Openloop evaluation runner for models trained by top-level configs.

This script runs openloop evaluation for models trained from top-level configs,
producing metrics like MSE, ADE, FDE, and RMSE.

Usage:
    # Evaluate a single checkpoint with a config
    python scripts/run_openloop_eval.py \
        --checkpoint_dir /mnt/data/heyuan/openpi_modified/checkpoints/cfg_pi0.5_28_dim.robomind_2.0305/cfg_pi0.5_28_dim.robomind_2.0305_exp/15000 \
        --config src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.robomind_2.py \
        --dataset_root /mnt/ \
        --repo_id oss_data/X-Humanoid/RoboMIND2.0-Agilex_lerobot/agilex/fold_clothes/success_episodes

    # Generate eval configs first, then batch evaluate
    python scripts/generate_openloop_eval_configs.py --output eval_configs.json
    python scripts/run_openloop_eval.py \
        --eval_config eval_configs.json \
        --checkpoint_dir checkpoints/model/step \
        --dataset_root /mnt/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

# Ensure scripts directory is in path for imports
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

REPO_ROOT = Path(__file__).resolve().parents[1]


def init_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _resolve_existing_dir(path_value: str, *, arg_name: str) -> Path:
    if not path_value:
        raise ValueError(f"{arg_name} must not be empty")

    resolved_path = Path(path_value).expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{arg_name} does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        raise NotADirectoryError(f"{arg_name} is not a directory: {resolved_path}")
    return resolved_path


def _normalize_train_config_path(train_config_path: str) -> str:
    candidate = Path(train_config_path).expanduser()
    resolved_path = candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()

    if not resolved_path.exists() or not resolved_path.is_file():
        raise FileNotFoundError(f"train_config_path does not exist: {resolved_path}")

    try:
        return resolved_path.relative_to(REPO_ROOT).as_posix()
    except ValueError as exc:
        raise ValueError(f"train_config_path must stay within repo root: {resolved_path}") from exc


def _normalize_repo_id(repo_id: str) -> str:
    candidate = Path(repo_id)
    if candidate.is_absolute():
        raise ValueError(f"repo_id must be a relative path, got: {repo_id}")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"repo_id must not contain traversal segments, got: {repo_id}")

    normalized_repo_id = candidate.as_posix()
    if not normalized_repo_id:
        raise ValueError("repo_id must not be empty")
    return normalized_repo_id


def compute_metrics_from_results(preds_path: Path, gts_path: Path) -> dict[str, float]:
    """Compute ADE, FDE, RMSE from saved predictions and ground truths."""
    import numpy as np

    preds = np.load(preds_path)
    gts = np.load(gts_path)

    # MSE per element
    diff = preds - gts
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))

    # ADE: Average Displacement Error per trajectory
    # Shape: (N, T, D) -> per-timestep L2 norm, then mean per trajectory
    if preds.ndim == 3:
        timestep_l2 = np.linalg.norm(preds - gts, axis=-1)  # (N, T)
        per_traj_ade = np.mean(timestep_l2, axis=1)  # (N,)
        ade = float(np.mean(per_traj_ade))

        # FDE: Final Displacement Error
        final_l2 = np.linalg.norm(preds[:, -1, :] - gts[:, -1, :], axis=-1)  # (N,)
        fde = float(np.mean(final_l2))
    else:
        # 2D case: treat as single timestep
        ade = rmse
        fde = rmse

    return {
        "mse": mse,
        "rmse": rmse,
        "ade": ade,
        "fde": fde,
        "count": int(preds.shape[0]),
    }


def run_openloop_eval_for_config(
    train_config_path: str,
    checkpoint_dir: str,
    dataset_root: str,
    repo_id: str,
    num_batches: int = 10,
    batch_size: int = 64,
    vis_dir: str | None = None,
    results_dir: str | None = None,
) -> dict[str, Any]:
    """Run openloop evaluation for a trained model from a top-level config.

    Args:
        train_config_path: Path to the training config file.
        checkpoint_dir: Directory containing the model checkpoint.
        dataset_root: Root directory for datasets.
        repo_id: Repository ID for the dataset to evaluate.
        num_batches: Number of batches to evaluate.
        batch_size: Batch size for evaluation.
        vis_dir: Directory for visualization outputs.
        results_dir: Directory for raw prediction/ground-truth numpy outputs.

    Returns:
        Dict with evaluation metrics (MSE, ADE, FDE, RMSE, count).
    """
    import test as eval_test

    normalized_config_path = _normalize_train_config_path(train_config_path)
    resolved_checkpoint_dir = _resolve_existing_dir(checkpoint_dir, arg_name="checkpoint_dir")
    resolved_dataset_root = _resolve_existing_dir(dataset_root, arg_name="dataset_root")
    normalized_repo_id = _normalize_repo_id(repo_id)

    # Determine visualization directory
    if vis_dir is None:
        resolved_vis_dir = resolved_checkpoint_dir / "openloop_eval_vis" / normalized_repo_id.replace("/", "__")
    else:
        resolved_vis_dir = Path(vis_dir).expanduser().resolve()

    checkpoint_base_dir = resolved_checkpoint_dir.parent.parent.parent
    exp_name = resolved_checkpoint_dir.parent.name
    step = resolved_checkpoint_dir.name

    if results_dir is None:
        resolved_results_dir = (
            checkpoint_base_dir / normalized_config_path.replace(".py", "") / exp_name / step / "test_results"
        )
    else:
        resolved_results_dir = Path(results_dir).expanduser().resolve()

    # Call test.py main function directly
    eval_test.main(
        checkpoint_dir=str(resolved_checkpoint_dir),
        dataset_root=str(resolved_dataset_root),
        config_name=normalized_config_path,
        num_batches=num_batches,
        batch_size=batch_size,
        vis_dir=str(resolved_vis_dir),
        results_dir=str(resolved_results_dir),
        repo_id=normalized_repo_id,
    )

    # Load saved results and compute metrics
    repo_results_dir = resolved_results_dir / normalized_repo_id

    preds_path = repo_results_dir / "test_all_preds.npy"
    gts_path = repo_results_dir / "test_all_gts.npy"

    if preds_path.exists() and gts_path.exists():
        metrics = compute_metrics_from_results(preds_path, gts_path)
    else:
        metrics = {"error": "Results files not found"}

    return {
        "config": normalized_config_path,
        "checkpoint_dir": str(resolved_checkpoint_dir),
        "repo_id": normalized_repo_id,
        **metrics,
    }


def run_batch_eval(
    eval_configs: list[dict[str, Any]],
    checkpoint_dir: str,
    dataset_root: str,
    num_batches: int = 10,
    batch_size: int = 64,
    results_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Run batch evaluation for multiple configs.

    Args:
        eval_configs: List of eval config dicts from generate_openloop_eval_configs.
        checkpoint_dir: Directory containing checkpoints.
        dataset_root: Root directory for datasets.
        num_batches: Number of batches per evaluation.
        batch_size: Batch size for evaluation.
        results_dir: Directory for raw prediction/ground-truth numpy outputs.

    Returns:
        List of evaluation result dicts.
    """
    results: list[dict[str, Any]] = []

    for cfg in eval_configs:
        config_name = cfg.get("config", "")
        repo_ids = cfg.get("repo_ids", [])

        if not config_name:
            logging.warning("Skipping config with missing path")
            continue

        if not repo_ids:
            logging.warning(f"No repo_ids for config {config_name}, skipping")
            continue

        # Take first repo_id for evaluation
        repo_id = repo_ids[0]
        logging.info(f"Evaluating {config_name} on {repo_id}")

        try:
            result = run_openloop_eval_for_config(
                train_config_path=config_name,
                checkpoint_dir=checkpoint_dir,
                dataset_root=dataset_root,
                repo_id=repo_id,
                num_batches=num_batches,
                batch_size=batch_size,
                results_dir=results_dir,
            )
            results.append(result)
            logging.info(
                f"  MSE={result.get('mse'):.4f}, ADE={result.get('ade'):.4f}, FDE={result.get('fde'):.4f}, RMSE={result.get('rmse'):.4f}"
            )
        except Exception as e:
            logging.error(f"Failed to evaluate {config_name} on {repo_id}: {e}")
            results.append(
                {
                    "config": config_name,
                    "checkpoint_dir": checkpoint_dir,
                    "repo_id": repo_id,
                    "error": str(e),
                }
            )

    return results


def write_results(results: list[dict[str, Any]], output_path: str) -> None:
    """Write evaluation results to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_evaluations": len(results),
        "successful_evaluations": sum(1 for r in results if "error" not in r),
        "results": results,
    }

    output_file.write_text(
        json.dumps(output_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote results to {output_file}")


def main() -> int:
    init_logging()

    parser = argparse.ArgumentParser(description="Run openloop evaluation for trained models")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Path to training config file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Directory containing model checkpoint",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="",
        help="Dataset repository ID to evaluate",
    )
    parser.add_argument(
        "--eval_config",
        type=str,
        default="",
        help="Path to JSON file with eval configs (from generate_openloop_eval_configs.py)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all top-level configs",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=10,
        help="Number of batches to evaluate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default="",
        help="Directory for visualization outputs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="openloop_eval_results.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Directory for raw prediction/ground-truth numpy outputs",
    )

    args = parser.parse_args()

    # Mode 1: Evaluate all top-level configs
    if args.all:
        from generate_openloop_eval_configs import generate_all_eval_configs

        if not args.checkpoint_dir or not args.dataset_root:
            parser.error("--checkpoint_dir and --dataset_root are required with --all")

        eval_configs = generate_all_eval_configs()
        if not eval_configs:
            logging.error("No eval configs generated")
            return 1

        results = run_batch_eval(
            eval_configs=eval_configs,
            checkpoint_dir=args.checkpoint_dir,
            dataset_root=args.dataset_root,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            results_dir=args.results_dir or None,
        )
        write_results(results, args.output)
        return 0

    # Mode 2: Use eval config file
    if args.eval_config:
        if not args.checkpoint_dir or not args.dataset_root:
            parser.error("--checkpoint_dir and --dataset_root are required with --eval_config")

        with open(args.eval_config, encoding="utf-8") as f:
            config_data = json.load(f)

        eval_configs = config_data.get("configs", [])
        if not eval_configs:
            logging.error("No configs found in eval config file")
            return 1

        results = run_batch_eval(
            eval_configs=eval_configs,
            checkpoint_dir=args.checkpoint_dir,
            dataset_root=args.dataset_root,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            results_dir=args.results_dir or None,
        )
        write_results(results, args.output)
        return 0

    # Mode 3: Single config evaluation
    if args.config and args.checkpoint_dir and args.dataset_root and args.repo_id:
        result = run_openloop_eval_for_config(
            train_config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            dataset_root=args.dataset_root,
            repo_id=args.repo_id,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            vis_dir=args.vis_dir or None,
            results_dir=args.results_dir or None,
        )
        write_results([result], args.output)
        return 0

    parser.error(
        "Either provide --config, --checkpoint_dir, --dataset_root, and --repo_id, "
        "or use --eval_config with --checkpoint_dir and --dataset_root, "
        "or use --all with --checkpoint_dir and --dataset_root"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
