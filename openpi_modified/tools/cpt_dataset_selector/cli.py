"""CLI: manifest, index, serve."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from .indexer import build_index
from .indexer import full_rebuild
from .manifest import load_all_public_datasets_repo_ids
from .manifest import load_repo_ids_merged
from .manifest import parse_config_paths_csv
from .manifest import write_manifest


def cmd_manifest(ns: argparse.Namespace) -> None:
    payload = write_manifest(Path(ns.out), config_paths=parse_config_paths_csv(ns.config))
    print(json.dumps({"count": payload["count"], "out": ns.out, "config_paths": payload["config_paths"]}, indent=2))


def cmd_index(ns: argparse.Namespace) -> None:
    manifest_path = Path(ns.manifest)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_ids = data["repo_ids"]
    root_dir = Path(ns.root_dir)
    db_path = Path(ns.db)
    if ns.full:
        stats = full_rebuild(root_dir, repo_ids, db_path, workers=ns.workers, progress=print)
    else:
        stats = build_index(root_dir, repo_ids, db_path, workers=ns.workers, progress=print)
    print(
        json.dumps(
            {
                "repos_scanned": stats.repos_scanned,
                "repos_ok": stats.repos_ok,
                "repos_missing_tasks": stats.repos_missing_tasks,
                "repos_errors": stats.repos_errors,
                "task_rows": stats.task_rows,
            },
            indent=2,
        )
    )


def cmd_count(ns: argparse.Namespace) -> None:
    paths = parse_config_paths_csv(ns.config)
    n = len(load_repo_ids_merged(paths)) if paths else len(load_all_public_datasets_repo_ids(None))
    print(n)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cpt_dataset_selector", description="CPT public dataset index tools")
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("manifest", help="Write manifest.json from training config(s) REPO_ID")
    m.add_argument("--out", default="tools/cpt_dataset_selector/data/manifest.json")
    m.add_argument(
        "--config",
        default=None,
        metavar="PATHS",
        help=(
            "Comma-separated paths to training config .py files (REPO_ID merged in order, deduped). "
            "Default: cfg_pi0.5_28_dim.all_public_datasets.py only."
        ),
    )
    m.set_defaults(func=cmd_manifest)

    i = sub.add_parser("index", help="Build SQLite index from tasks.jsonl")
    i.add_argument("--root-dir", default=os.environ.get("OPENPI_ROOT_DIR", "/mnt"))
    i.add_argument("--manifest", default="tools/cpt_dataset_selector/data/manifest.json")
    i.add_argument("--db", default="tools/cpt_dataset_selector/data/index.sqlite3")
    i.add_argument("--workers", type=int, default=8)
    i.add_argument("--full", action="store_true", help="Drop DB and full rebuild")
    i.set_defaults(func=cmd_index)

    c = sub.add_parser("count", help="Print merged REPO_ID count from config(s)")
    c.add_argument(
        "--config",
        default=None,
        metavar="PATHS",
        help="Same as manifest --config (comma-separated). Default: all_public_datasets only.",
    )
    c.set_defaults(func=cmd_count)

    return p


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    ns = parser.parse_args(argv)
    ns.func(ns)


if __name__ == "__main__":
    main()
