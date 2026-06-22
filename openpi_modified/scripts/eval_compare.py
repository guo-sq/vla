"""Batch evaluation orchestrator for multi-dataset / multi-checkpoint comparison.

Design goals:
- Reuse `scripts/test.py` execution logic directly (no duplicated CLI command strings).
- Keep report schema centralized and stable.
- Support both shared config and per-checkpoint config mapping.
"""

import argparse
import csv
import dataclasses
import json
from pathlib import Path
from typing import Any

import openpi.training.config as _config
import openpi.training.base_cfg as _base_cfg
import test as eval_test
from openpi.training.base_cfg import EvalConfig, TestConfig


def _parse_cli_value(raw: str):
    lowered = raw.strip().lower()
    if lowered == "none":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        import ast

        return ast.literal_eval(raw)
    except Exception:
        return raw


def _unknown_args_to_dict(args: list[str]) -> dict[str, Any]:
    alias_map = {
        "config_name": "config",
    }
    out: dict[str, Any] = {}
    i = 0
    while i < len(args):
        token = args[i]
        if not token.startswith("--"):
            i += 1
            continue

        payload = token[2:]
        if "=" in payload:
            key, raw_value = payload.split("=", 1)
            i += 1
        else:
            key = payload
            if i + 1 >= len(args) or args[i + 1].startswith("--"):
                raw_value = "true"
                i += 1
            else:
                raw_value = args[i + 1]
                i += 2

        key = alias_map.get(key, key)
        out[key] = _parse_cli_value(raw_value)
    return out


REPORT_COLUMNS = [
    "checkpoint",
    "config",
    "model",
    "step",
    "repo_id",
    "ade",
    "fde",
    "rmse",
    "count",
    "metrics_json",
    "vis_png",
]


@dataclasses.dataclass(frozen=True)
class EvalTarget:
    ckpt_dir: str
    config: str


def _as_repo_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _safe_tag(text: str) -> str:
    return text.replace("/", "__").replace(" ", "_")


def _parse_target(text: str, default_config: str | None) -> EvalTarget:
    if "::" in text:
        ckpt_dir, cfg = text.split("::", 1)
        if not cfg:
            raise ValueError(f"Invalid target '{text}': empty config name after '::'.")
        return EvalTarget(ckpt_dir=ckpt_dir, config=cfg)

    if default_config is None:
        raise ValueError(
            f"Target '{text}' has no config binding. Use --config or '<ckpt>::<config>'."
        )
    return EvalTarget(ckpt_dir=text, config=default_config)


def _load_repo_ids_from_cfg(config: str, eval_split: str) -> list[str]:
    config = _config.get_config(config)
    data_factory = config.data
    if hasattr(data_factory, "split"):
        data_factory = dataclasses.replace(data_factory, split=eval_split)
    data_cfg = data_factory.create(config.assets_dirs, config.model)
    repo_ids = _as_repo_list(data_cfg.repo_id)
    if not repo_ids:
        raise ValueError(f"No repo_id found in config: {config}")
    return repo_ids


def _run_single(
    *,
    target: EvalTarget,
    repo_id: str,
    eval_cfg: EvalConfig,
) -> dict[str, Any]:
    model_tag = _safe_tag(Path(target.ckpt_dir).parent.name)
    repo_tag = _safe_tag(repo_id)
    result_tag = f"{model_tag}__{repo_tag}"
    vis_dir = str(Path(target.ckpt_dir) / "test_results" / result_tag)

    run_result = eval_test.run_test(
        TestConfig(
            checkpoint_dir=target.ckpt_dir,
            dataset_root=eval_cfg.dataset_root,
            config=target.config,
            num_batches=eval_cfg.num_batches,
            batch_size=eval_cfg.batch_size,
            vis_dir=vis_dir,
            repo_id=repo_id,
            num_workers=eval_cfg.num_workers,
            eval_split=eval_cfg.eval_split,
            bucket_fields=eval_cfg.bucket_fields,
            result_tag=result_tag,
        )
    )

    global_metrics = run_result.get("global", {})
    return {
        "checkpoint": target.ckpt_dir,
        "config": target.config,
        "model": Path(target.ckpt_dir).parent.name,
        "step": Path(target.ckpt_dir).name,
        "repo_id": repo_id,
        "ade": global_metrics.get("ade"),
        "fde": global_metrics.get("fde"),
        "rmse": global_metrics.get("rmse"),
        "count": global_metrics.get("count"),
        "metrics_json": run_result.get("metrics_json"),
        "vis_png": run_result.get("vis_png"),
    }


def _write_reports(rows: list[dict[str, Any]], report_out: Path) -> None:
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(rows, ensure_ascii=False, indent=2))

    csv_out = report_out.with_suffix(".csv")
    with csv_out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in REPORT_COLUMNS})

    print(f"Saved compare report JSON: {report_out}")
    print(f"Saved compare report CSV: {csv_out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")
    args, unknown = parser.parse_known_args()

    base_values = dataclasses.asdict(EvalConfig(targets=(), dataset_root=""))
    if args.config:
        cfg_file = _base_cfg.Config.fromfile(args.config)
        loaded = None
        if "eval_cfg" in cfg_file._cfg_dict:
            loaded = cfg_file._cfg_dict["eval_cfg"]
        elif "cfg" in cfg_file._cfg_dict:
            loaded = cfg_file._cfg_dict["cfg"]
        elif len(cfg_file._cfg_dict) == 1:
            loaded = next(iter(cfg_file._cfg_dict.values()))

        if isinstance(loaded, dict):
            base_values.update(loaded)
        elif isinstance(loaded, EvalConfig):
            base_values.update(dataclasses.asdict(loaded))

    cfg_obj = _base_cfg.Config(base_values)
    cfg_obj.merge_from_dict(_unknown_args_to_dict(unknown))
    eval_cfg = EvalConfig(**cfg_obj._cfg_dict)

    targets = [_parse_target(item, eval_cfg.config) for item in eval_cfg.targets]

    rows: list[dict[str, Any]] = []
    for target in targets:
        repo_ids = (
            list(eval_cfg.repo_ids)
            if eval_cfg.repo_ids
            else _load_repo_ids_from_cfg(target.config, eval_cfg.eval_split)
        )
        for repo_id in repo_ids:
            rows.append(_run_single(target=target, repo_id=repo_id, eval_cfg=eval_cfg))

    _write_reports(rows, Path(eval_cfg.report_out))


if __name__ == "__main__":
    main()
