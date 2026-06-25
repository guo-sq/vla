#!/usr/bin/env python3
"""Train episode-level out-of-fold diffusion proxy checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_proxy.data import build_dataset_arrays
from diffusion_proxy.data import episode_folds
from diffusion_proxy.data import load_episodes
from diffusion_proxy.training import DiffusionTrainConfig
from diffusion_proxy.training import checkpoint_last_metrics
from diffusion_proxy.training import train_fold
from diffusion_proxy.utils import repo_ids_from_file
from diffusion_proxy.vision import DEFAULT_ENCODER


LOGGER = logging.getLogger("train_oof_diffusion_proxy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--repo-list-file", type=Path, required=True)
    parser.add_argument("--vision-cache", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--camera", default="observation.images.head")
    parser.add_argument("--encoder", choices=["resnet18", "grid"], default=DEFAULT_ENCODER)
    parser.add_argument("--max-episodes-per-repo", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ema-decay", type=float, default=0.995)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    repo_ids = repo_ids_from_file(args.repo_list_file)
    episodes = load_episodes(
        args.root_dir,
        repo_ids,
        vision_cache=args.vision_cache,
        camera=args.camera,
        encoder=args.encoder,
        max_episodes_per_repo=args.max_episodes_per_repo,
    )
    arrays = build_dataset_arrays(episodes, args.horizon)
    folds = episode_folds(arrays.episode_keys, args.folds, seed=args.seed)
    config = DiffusionTrainConfig(
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        blocks=args.blocks,
        diffusion_steps=args.diffusion_steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ema_decay=args.ema_decay,
    )
    checkpoint_paths = []
    fold_summaries = []
    all_keys = set(arrays.episode_keys)
    for fold_index, heldout_list in enumerate(folds):
        heldout_keys = set(heldout_list)
        train_keys = all_keys - heldout_keys
        LOGGER.info("Training fold %d: train_episodes=%d heldout_episodes=%d", fold_index, len(train_keys), len(heldout_keys))
        path = train_fold(
            arrays,
            train_keys=train_keys,
            heldout_keys=heldout_keys,
            fold_index=fold_index,
            checkpoint_dir=args.checkpoint_dir,
            config=config,
            seed=args.seed + fold_index,
            device=args.device,
        )
        checkpoint_paths.append(path)
        fold_summaries.append(
            {
                "fold_index": fold_index,
                "checkpoint": str(path),
                "heldout_keys": [[repo, int(ep)] for repo, ep in sorted(heldout_keys)],
                "final_metrics": checkpoint_last_metrics(path),
            }
        )
    summary = {
        "root_dir": str(args.root_dir),
        "repo_ids": repo_ids,
        "vision_cache": str(args.vision_cache),
        "camera": args.camera,
        "encoder": args.encoder,
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoints": [str(path) for path in checkpoint_paths],
        "num_episodes": len(episodes),
        "num_frames": int(len(arrays.condition)),
        "condition_dim": int(arrays.condition.shape[1]),
        "action_dim": 14,
        "folds": args.folds,
        "train_config": config.__dict__,
        "fold_summaries": fold_summaries,
    }
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("Wrote training summary")


if __name__ == "__main__":
    main()

