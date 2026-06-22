#!/usr/bin/env python3
"""Generate evaluation config entries from top-level training configs.

This script scans the same top-level configs covered by config-loss-drift,
extracts repo_ids from each training config, and emits JSON metadata that can
be consumed by generic openloop evaluation tooling.

Usage:
    python scripts/generate_openloop_eval_configs.py
    python scripts/generate_openloop_eval_configs.py --output eval_configs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import openpi.training.config as _config

REPO_ROOT = Path(__file__).resolve().parents[1]


def collect_top_level_configs() -> list[Path]:
    """Collect top-level config paths matching config-loss-drift coverage."""
    configs_dir = REPO_ROOT / "src/openpi/configs"
    return sorted(path for path in configs_dir.glob("*.py") if path.name not in {"__init__.py", "base.py"})


def _extract_repo_ids_from_data_factory(data_factory: Any) -> list[str]:
    """Extract repo_ids from a data factory or data config."""
    repo_ids: list[str] = []

    if hasattr(data_factory, "repo_id"):
        repo_id = data_factory.repo_id
        if isinstance(repo_id, str):
            repo_ids.append(repo_id)
        elif isinstance(repo_id, list):
            repo_ids.extend(str(r) for r in repo_id)

    if hasattr(data_factory, "base_config"):
        base = data_factory.base_config
        if hasattr(base, "repo_id"):
            repo_id = base.repo_id
            if isinstance(repo_id, str):
                repo_ids.append(repo_id)
            elif isinstance(repo_id, list):
                repo_ids.extend(str(r) for r in repo_id)

    return repo_ids


def extract_repo_ids_from_config(config_path: Path) -> dict[str, Any]:
    """Extract repo_ids and config metadata from a training config file."""
    try:
        relative_path = str(config_path.relative_to(REPO_ROOT))
    except ValueError:
        relative_path = str(config_path)

    result: dict[str, Any] = {
        "config_path": relative_path,
        "config_name": config_path.name,
        "repo_ids": [],
        "error": None,
    }

    try:
        cfg = _config.get_config(str(config_path))
    except (FileNotFoundError, ModuleNotFoundError, ValueError, OSError) as exc:
        result["error"] = f"config_load_error: {exc}"
        return result

    data_factory = getattr(cfg, "data", None)
    if data_factory is None:
        result["error"] = "No 'data' attribute in config"
        return result

    repo_ids = _extract_repo_ids_from_data_factory(data_factory)

    if not repo_ids:
        assets_dirs = getattr(cfg, "assets_dirs", [])
        model_cfg = getattr(cfg, "model", None)
        try:
            data_cfg = data_factory.create(assets_dirs, model_cfg)
        except (AttributeError, FileNotFoundError, TypeError, ValueError, OSError) as exc:
            result["error"] = f"repo_id_extraction_error: {exc}"
            return result
        repo_ids = _extract_repo_ids_from_data_factory(data_cfg)

    result["repo_ids"] = repo_ids
    if not repo_ids:
        result["error"] = "Could not extract repo_ids from config"

    return result


def generate_eval_config_for_train_config(
    train_config_path: Path,
    checkpoint_dir: str = "",
    dataset_root: str = "",
) -> dict[str, Any] | None:
    """Generate a TestConfig-like dict from a training config path."""
    info = extract_repo_ids_from_config(train_config_path)
    if info.get("error"):
        return None
    return {
        "checkpoint_dir": checkpoint_dir,
        "dataset_root": dataset_root,
        "config": str(train_config_path.relative_to(REPO_ROOT)),
        "repo_ids": info["repo_ids"],
        "config_name": train_config_path.name,
    }


def generate_all_eval_configs() -> list[dict[str, Any]]:
    """Generate eval configs for all top-level training configs."""
    configs = collect_top_level_configs()
    results: list[dict[str, Any]] = []

    for config_path in configs:
        eval_cfg = generate_eval_config_for_train_config(config_path)
        if eval_cfg is not None:
            results.append(eval_cfg)

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate eval configs for top-level training configs")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output JSON file path (default: stdout)",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include configs that failed to extract repo_ids",
    )
    args = parser.parse_args()

    configs = collect_top_level_configs()
    results: list[dict[str, Any]] = []
    for config_path in configs:
        info = extract_repo_ids_from_config(config_path)
        if args.include_errors or not info.get("error"):
            results.append(info)

    output_data = {
        "total_configs": len(configs),
        "successful_extractions": sum(1 for r in results if not r.get("error")),
        "configs": results,
    }

    output_json = json.dumps(output_data, indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_json, encoding="utf-8")
        print(f"Wrote {len(results)} config entries to {output_path}")
    else:
        print(output_json)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
