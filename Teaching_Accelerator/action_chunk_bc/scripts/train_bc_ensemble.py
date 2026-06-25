#!/usr/bin/env python3
"""Train a lightweight action-chunk BC ensemble for seatbelt demos."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from action_chunk_bc.data import build_dataset_arrays
from action_chunk_bc.data import fit_normalization
from action_chunk_bc.data import load_episodes
from action_chunk_bc.data import repo_ids_from_file
from action_chunk_bc.training import TrainConfig
from action_chunk_bc.training import train_one_seed


LOGGER = logging.getLogger("train_bc_ensemble")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--repo-list-file", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--max-episodes-per-repo", type=int, default=None)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.90)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _last_history_entry(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    history = ckpt.get("history", [])
    return dict(history[-1]) if history else {}


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    if args.ensemble_size < 1:
        raise SystemExit("--ensemble-size must be >= 1")
    repo_ids = repo_ids_from_file(args.repo_list_file)
    if not repo_ids:
        raise SystemExit(f"No repo ids found in {args.repo_list_file}")

    LOGGER.info("Loading episodes from %d repos", len(repo_ids))
    episodes = load_episodes(args.root_dir, repo_ids, max_episodes_per_repo=args.max_episodes_per_repo)
    arrays = build_dataset_arrays(episodes, args.horizon)
    stats = fit_normalization(arrays)
    LOGGER.info("Loaded %d episodes, %d frames", len(episodes), len(arrays.features))

    config = TrainConfig(
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        train_fraction=args.train_fraction,
    )

    checkpoint_paths: list[Path] = []
    seed_summaries: list[dict[str, Any]] = []
    for seed in range(args.seed_offset, args.seed_offset + args.ensemble_size):
        LOGGER.info("Training seed %d", seed)
        path = train_one_seed(
            arrays,
            seed=seed,
            checkpoint_dir=args.checkpoint_dir,
            config=config,
            stats=stats,
            device=args.device,
        )
        checkpoint_paths.append(path)
        seed_summaries.append(
            {
                "seed": seed,
                "checkpoint": str(path),
                "final_metrics": _last_history_entry(path),
            }
        )
        LOGGER.info("Saved %s", path)

    summary = {
        "root_dir": str(args.root_dir),
        "repo_ids": repo_ids,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "num_episodes": len(episodes),
        "num_frames": int(len(arrays.features)),
        "action_dim": 14,
        "feature_dim": int(arrays.features.shape[1]),
        "horizon": args.horizon,
        "train_config": config.__dict__,
        "seeds": seed_summaries,
    }
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.checkpoint_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote training summary to %s", summary_path)


if __name__ == "__main__":
    main()
