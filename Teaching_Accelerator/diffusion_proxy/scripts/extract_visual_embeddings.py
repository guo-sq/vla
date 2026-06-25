#!/usr/bin/env python3
"""Extract frozen visual embeddings for the diffusion proxy."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_proxy.utils import load_json
from diffusion_proxy.utils import load_jsonl
from diffusion_proxy.utils import repo_ids_from_file
from diffusion_proxy.vision import DEFAULT_ENCODER
from diffusion_proxy.vision import embedding_cache_path
from diffusion_proxy.vision import save_episode_embeddings


LOGGER = logging.getLogger("extract_visual_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--repo-list-file", type=Path, required=True)
    parser.add_argument("--camera", default="observation.images.head")
    parser.add_argument("--encoder", choices=["resnet18", "grid"], default=DEFAULT_ENCODER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-episodes-per-repo", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    repo_ids = repo_ids_from_file(args.repo_list_file)
    jobs = []
    for repo_id in repo_ids:
        repo_root = args.root_dir / repo_id
        episodes = load_jsonl(repo_root / "meta" / "episodes.jsonl")
        if args.max_episodes_per_repo is not None:
            episodes = episodes[: args.max_episodes_per_repo]
        load_json(repo_root / "meta" / "info.json")
        for ep in episodes:
            jobs.append((repo_id, ep))
    written = 0
    skipped = 0
    for repo_id, ep in tqdm(jobs, desc="visual embeddings"):
        episode_index = int(ep["episode_index"])
        out_path = embedding_cache_path(args.output_dir, repo_id, episode_index, args.camera, args.encoder)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue
        save_episode_embeddings(
            root_dir=args.root_dir,
            repo_id=repo_id,
            episode_meta=ep,
            cache_dir=args.output_dir,
            camera=args.camera,
            encoder=args.encoder,
            batch_size=args.batch_size,
            device=args.device,
        )
        written += 1
    summary = {
        "root_dir": str(args.root_dir),
        "repo_ids": repo_ids,
        "camera": args.camera,
        "encoder": args.encoder,
        "output_dir": str(args.output_dir),
        "written": written,
        "skipped": skipped,
        "total": len(jobs),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.encoder}_{args.camera.replace('.', '_')}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Embedding extraction complete: written=%d skipped=%d", written, skipped)


if __name__ == "__main__":
    main()

