#!/usr/bin/env python3

'''
Evaluate the physical RMSE scale implied by normalized training losses and norm_stats.json.
风险扫描（按 config 聚合、保留最差样本）：
python scripts/eval_loss_physical_scale.py \
    --top-k 30 \
    --step 200 \
    --filter-config openx_embodiment \
    --aggregate-by config \
    --aggregate-pick worst \
    --sort-by rmse_eef \
    --ascending --wide \
    --md-out outputs/report.md

精准定位某类机器人：
python scripts/eval_loss_physical_scale.py \
    --filter-robot franka \
    --sort-by loss \
    --top-k 30 \
    --table-out outputs/franka_table.txt
'''

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


JOINT_DIMS = list(range(0, 16))
EEF_DIMS = list(range(16, 28))


@dataclass
class EvalRow:
    config: str
    run_id: str
    step: int
    loss: float
    robot_type: str
    use_quantile_estimate: bool
    rmse_joint_est: float | None
    rmse_eef_est: float | None
    rmse_all_est: float | None
    qspan_joint_median: float | None
    qspan_eef_median: float | None
    qspan_all_median: float | None
    std_joint_median: float | None
    std_eef_median: float | None
    std_all_median: float | None
    action_nonzero_dims: int
    state_nonzero_dims: int
    norm_stats_path: str
    output_log_path: str


def _median_nonzero(values: list[float]) -> float | None:
    vals = sorted(v for v in values if v > 0)
    if not vals:
        return None
    return vals[len(vals) // 2]


def _parse_step_loss(log_path: Path, step: int) -> float | None:
    text = log_path.read_text(errors="ignore")
    pattern = re.compile(rf"Step\s+{step}:.*?loss=([0-9eE+\.-]+)")
    match = pattern.search(text)
    if match:
        return float(match.group(1))

    fallback = re.findall(r"Step\s+(\d+):.*?loss=([0-9eE+\.-]+)", text)
    if not fallback:
        return None
    last_step, last_loss = fallback[-1]
    return float(last_loss)


def _find_wandb_log(wandb_root: Path, run_id: str) -> Path | None:
    runs = sorted(wandb_root.glob(f"run-*-{run_id}"))
    if not runs:
        return None
    path = runs[-1] / "files" / "output.log"
    return path if path.exists() else None


def _find_norm_stats(exp_dir: Path) -> Path | None:
    candidates = sorted(exp_dir.glob("*/assets/*/norm_stats.json"))
    if candidates:
        return candidates[-1]
    fallback = sorted(exp_dir.glob("assets/*/norm_stats.json"))
    if fallback:
        return fallback[-1]
    return None


def _safe_take(values: list[float], indices: list[int]) -> list[float]:
    return [values[i] for i in indices if 0 <= i < len(values)]


def _estimate_rmse(loss: float, scale: float | None, quantile: bool) -> float | None:
    if scale is None:
        return None
    if quantile:
        return math.sqrt(loss) * (scale / 2.0)
    return math.sqrt(loss) * scale


def _collect_rows(
    checkpoints_root: Path,
    wandb_root: Path,
    step: int,
    use_quantile_estimate: bool,
) -> list[EvalRow]:
    rows: list[EvalRow] = []

    for cfg_dir in sorted(checkpoints_root.glob("cfg_*")):
        exp_dirs = sorted([p for p in cfg_dir.iterdir() if p.is_dir() and p.name.endswith("_exp")])
        if not exp_dirs:
            continue
        exp_dir = exp_dirs[0]

        wandb_id_file = exp_dir / "wandb_id.txt"
        if not wandb_id_file.exists():
            continue
        run_id = wandb_id_file.read_text().strip()
        if not run_id:
            continue

        log_path = _find_wandb_log(wandb_root, run_id)
        if log_path is None:
            continue
        loss = _parse_step_loss(log_path, step)
        if loss is None:
            continue

        norm_stats_path = _find_norm_stats(exp_dir)
        if norm_stats_path is None:
            continue

        payload = json.loads(norm_stats_path.read_text())
        norm_stats = payload.get("norm_stats", {})
        if not isinstance(norm_stats, dict) or not norm_stats:
            continue

        for robot_type, robot_stats in norm_stats.items():
            actions = robot_stats.get("actions", {})
            state = robot_stats.get("state", {})
            std = [float(x) for x in actions.get("std", [])]
            q01 = [float(x) for x in actions.get("q01", [])]
            q99 = [float(x) for x in actions.get("q99", [])]
            state_std = [float(x) for x in state.get("std", [])]

            qspan = [abs(b - a) for a, b in zip(q01, q99)]

            std_joint = _median_nonzero(_safe_take(std, JOINT_DIMS))
            std_eef = _median_nonzero(_safe_take(std, EEF_DIMS))
            std_all = _median_nonzero(std)

            qspan_joint = _median_nonzero(_safe_take(qspan, JOINT_DIMS))
            qspan_eef = _median_nonzero(_safe_take(qspan, EEF_DIMS))
            qspan_all = _median_nonzero(qspan)

            scale_joint = qspan_joint if use_quantile_estimate else std_joint
            scale_eef = qspan_eef if use_quantile_estimate else std_eef
            scale_all = qspan_all if use_quantile_estimate else std_all

            rows.append(
                EvalRow(
                    config=cfg_dir.name,
                    run_id=run_id,
                    step=step,
                    loss=loss,
                    robot_type=robot_type,
                    use_quantile_estimate=use_quantile_estimate,
                    rmse_joint_est=_estimate_rmse(loss, scale_joint, use_quantile_estimate),
                    rmse_eef_est=_estimate_rmse(loss, scale_eef, use_quantile_estimate),
                    rmse_all_est=_estimate_rmse(loss, scale_all, use_quantile_estimate),
                    qspan_joint_median=qspan_joint,
                    qspan_eef_median=qspan_eef,
                    qspan_all_median=qspan_all,
                    std_joint_median=std_joint,
                    std_eef_median=std_eef,
                    std_all_median=std_all,
                    action_nonzero_dims=sum(1 for x in std if x > 0),
                    state_nonzero_dims=sum(1 for x in state_std if x > 0),
                    norm_stats_path=str(norm_stats_path),
                    output_log_path=str(log_path),
                )
            )

    return rows


def _format(x: float | None) -> str:
    return "NA" if x is None else f"{x:.6g}"


def _safe_float(value: float | None, fallback: float) -> float:
    return fallback if value is None else float(value)


def _sort_value(row: EvalRow, sort_by: str) -> float | str | None:
    if sort_by == "loss":
        return row.loss
    if sort_by == "rmse_all":
        return row.rmse_all_est
    if sort_by == "rmse_joint":
        return row.rmse_joint_est
    if sort_by == "rmse_eef":
        return row.rmse_eef_est
    if sort_by == "config":
        return row.config
    if sort_by == "robot":
        return row.robot_type
    return row.loss


def _sort_rows(rows: list[EvalRow], sort_by: str, ascending: bool) -> list[EvalRow]:
    numeric_sort = {"loss", "rmse_all", "rmse_joint", "rmse_eef"}

    if sort_by in numeric_sort:
        missing = float("inf") if ascending else float("-inf")
        return sorted(
            rows,
            key=lambda r: _safe_float(_sort_value(r, sort_by), missing),
            reverse=not ascending,
        )

    return sorted(
        rows,
        key=lambda r: str(_sort_value(r, sort_by)),
        reverse=not ascending,
    )


def _filter_rows(
    rows: list[EvalRow],
    config_filter: str,
    robot_filter: str,
    config_regex: str,
    robot_regex: str,
) -> list[EvalRow]:
    out = rows

    if config_filter:
        needle = config_filter.lower()
        out = [r for r in out if needle in r.config.lower()]

    if robot_filter:
        needle = robot_filter.lower()
        out = [r for r in out if needle in r.robot_type.lower()]

    if config_regex:
        cre = re.compile(config_regex)
        out = [r for r in out if cre.search(r.config)]

    if robot_regex:
        rre = re.compile(robot_regex)
        out = [r for r in out if rre.search(r.robot_type)]

    return out


def _aggregate_rows(
    rows: list[EvalRow],
    aggregate_by: str,
    aggregate_pick: str,
    sort_by: str,
) -> list[EvalRow]:
    if aggregate_by == "none":
        return rows

    groups: dict[str, list[EvalRow]] = {}
    for row in rows:
        if aggregate_by == "config":
            key = row.config
        else:
            key = f"{row.config}||{row.robot_type}"
        groups.setdefault(key, []).append(row)

    selected: list[EvalRow] = []
    for group_rows in groups.values():
        ordered = _sort_rows(group_rows, sort_by=sort_by, ascending=True)
        selected.append(ordered[0] if aggregate_pick == "best" else ordered[-1])

    return selected


def _clip_text(text: str, width: int) -> str:
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _render_table(
    rows: list[EvalRow],
    wide: bool,
) -> str:
    if wide:
        cfg_width = min(72, max([len("config")] + [len(r.config) for r in rows]))
        robot_width = min(48, max([len("robot_type")] + [len(r.robot_type) for r in rows]))
        columns = [
            ("config", cfg_width, "left"),
            ("loss", 12, "right"),
            ("rmse_all", 12, "right"),
            ("rmse_joint", 12, "right"),
            ("rmse_eef", 12, "right"),
            ("act_nz", 6, "right"),
            ("state_nz", 8, "right"),
            ("robot_type", robot_width, "left"),
        ]
    else:
        columns = [
            ("cfg", 16, "left"),
            ("loss", 9, "right"),
            ("rmse_all", 9, "right"),
            ("rmse_jnt", 9, "right"),
            ("rmse_eef", 9, "right"),
            ("a_nz", 4, "right"),
            ("s_nz", 4, "right"),
            ("robot", 12, "left"),
        ]

    def format_cell(value: str, width: int, align: str) -> str:
        text = _clip_text(value, width)
        if align == "right":
            return text.rjust(width)
        return text.ljust(width)

    header_cells = [format_cell(name, width, align) for name, width, align in columns]
    header = " ".join(header_cells)
    separator = " ".join("-" * width for _, width, _ in columns)

    lines = [header, separator]

    for r in rows:
        row_values = [
            r.config,
            f"{r.loss:.6g}",
            _format(r.rmse_all_est),
            _format(r.rmse_joint_est),
            _format(r.rmse_eef_est),
            str(r.action_nonzero_dims),
            str(r.state_nonzero_dims),
            r.robot_type,
        ]
        row_cells = [
            format_cell(val, width, align)
            for val, (_, width, align) in zip(row_values, columns)
        ]
        lines.append(" ".join(row_cells))

    return "\n".join(lines)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mid = n // 2
    if n % 2 == 1:
        return values[mid]
    return (values[mid - 1] + values[mid]) / 2.0


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = _mean(xs)
    my = _mean(ys)
    if mx is None or my is None:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _summary_text(all_rows: list[EvalRow], rows: list[EvalRow], step: int, estimate_mode: str) -> str:
    losses = [r.loss for r in rows]
    rmse_all = [r.rmse_all_est for r in rows if r.rmse_all_est is not None]
    rmse_joint = [r.rmse_joint_est for r in rows if r.rmse_joint_est is not None]
    rmse_eef = [r.rmse_eef_est for r in rows if r.rmse_eef_est is not None]

    corr = _pearson([r.loss for r in rows if r.rmse_all_est is not None], [r.rmse_all_est for r in rows if r.rmse_all_est is not None])

    weak_semantic = [r for r in rows if r.action_nonzero_dims <= 4 or r.state_nonzero_dims <= 4]
    worst_loss = _sort_rows(rows, sort_by="loss", ascending=False)[:3]

    lines = [
        f"Summary: total_rows={len(all_rows)}, selected_rows={len(rows)}, step={step}, estimate_mode={estimate_mode}",
        f"Stats: loss(mean/median)={_format(_mean(losses))}/{_format(_median(losses))}",
        f"Stats: rmse_all(mean/median)={_format(_mean(rmse_all))}/{_format(_median(rmse_all))}",
        f"Stats: rmse_joint(mean/median)={_format(_mean(rmse_joint))}/{_format(_median(rmse_joint))}",
        f"Stats: rmse_eef(mean/median)={_format(_mean(rmse_eef))}/{_format(_median(rmse_eef))}",
        f"Correlation: corr(loss, rmse_all)={_format(corr)}",
    ]

    if weak_semantic:
        preview = ", ".join(f"{r.config}:{r.robot_type}(a={r.action_nonzero_dims},s={r.state_nonzero_dims})" for r in weak_semantic[:5])
        lines.append(f"Potential semantic-risk rows (nonzero dims low): {preview}")

    if worst_loss:
        preview = ", ".join(f"{r.config}:{r.robot_type}(loss={r.loss:.4g})" for r in worst_loss)
        lines.append(f"Top loss rows: {preview}")

    return "\n".join(lines)


def _render_markdown_table(rows: list[EvalRow]) -> str:
    headers = ["config", "loss", "rmse_all", "rmse_joint", "rmse_eef", "act_nz", "state_nz", "robot_type"]
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        vals = [
            r.config,
            f"{r.loss:.6g}",
            _format(r.rmse_all_est),
            _format(r.rmse_joint_est),
            _format(r.rmse_eef_est),
            str(r.action_nonzero_dims),
            str(r.state_nonzero_dims),
            r.robot_type,
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _write_json(rows: list[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.__dict__ for r in rows]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_csv(rows: list[EvalRow], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "config",
                "run_id",
                "step",
                "loss",
                "robot_type",
                "use_quantile_estimate",
                "rmse_joint_est",
                "rmse_eef_est",
                "rmse_all_est",
                "qspan_joint_median",
                "qspan_eef_median",
                "qspan_all_median",
                "std_joint_median",
                "std_eef_median",
                "std_all_median",
                "action_nonzero_dims",
                "state_nonzero_dims",
                "norm_stats_path",
                "output_log_path",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r.__dict__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate physical RMSE scale from normalized training loss and norm_stats."
    )
    parser.add_argument("--checkpoints-root", default="checkpoints")
    parser.add_argument("--wandb-root", default="wandb")
    parser.add_argument("--step", type=int, default=200)
    parser.add_argument(
        "--estimate-mode",
        choices=["quantile", "zscore"],
        default="quantile",
        help="Use quantile-span scale (default) or std scale when converting normalized loss to physical RMSE estimate.",
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--csv", default="")
    parser.add_argument("--json-out", default="", help="Optional path to save selected rows in JSON format.")
    parser.add_argument(
        "--filter-config",
        default="",
        help="Case-insensitive substring filter for config name.",
    )
    parser.add_argument(
        "--filter-robot",
        default="",
        help="Case-insensitive substring filter for robot_type.",
    )
    parser.add_argument(
        "--filter-config-regex",
        default="",
        help="Regex filter for config name.",
    )
    parser.add_argument(
        "--filter-robot-regex",
        default="",
        help="Regex filter for robot_type.",
    )
    parser.add_argument(
        "--aggregate-by",
        choices=["none", "config", "config_robot"],
        default="none",
        help="Group rows before ranking; useful to de-duplicate by config.",
    )
    parser.add_argument(
        "--aggregate-pick",
        choices=["best", "worst"],
        default="worst",
        help="Within each group, keep best or worst row by sort key.",
    )
    parser.add_argument(
        "--wide",
        action="store_true",
        help="Print a wider, less-truncated table for easier reading.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["loss", "rmse_all", "rmse_joint", "rmse_eef", "config", "robot"],
        default="loss",
        help="Sort key for table output.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default is descending).",
    )
    parser.add_argument(
        "--table-out",
        default="",
        help="Optional path to save the formatted table text output.",
    )
    parser.add_argument(
        "--md-out",
        default="",
        help="Optional path to save a Markdown report.",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Disable summary stats and risk hints.",
    )

    args = parser.parse_args()

    rows = _collect_rows(
        checkpoints_root=Path(args.checkpoints_root),
        wandb_root=Path(args.wandb_root),
        step=args.step,
        use_quantile_estimate=(args.estimate_mode == "quantile"),
    )

    if not rows:
        print("No valid runs found. Check checkpoints/wandb roots and step number.")
        return

    filtered_rows = _filter_rows(
        rows,
        config_filter=args.filter_config,
        robot_filter=args.filter_robot,
        config_regex=args.filter_config_regex,
        robot_regex=args.filter_robot_regex,
    )
    aggregated_rows = _aggregate_rows(
        filtered_rows,
        aggregate_by=args.aggregate_by,
        aggregate_pick=args.aggregate_pick,
        sort_by=args.sort_by,
    )
    sorted_rows = _sort_rows(aggregated_rows, sort_by=args.sort_by, ascending=args.ascending)
    if args.top_k > 0:
        sorted_rows = sorted_rows[: args.top_k]

    table_text = _render_table(sorted_rows, wide=args.wide)

    if not args.no_summary:
        print(_summary_text(rows, sorted_rows, step=args.step, estimate_mode=args.estimate_mode))
        print()
    print(table_text)

    if args.table_out:
        out_path = Path(args.table_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(table_text)
        print(f"\nSaved table: {out_path}")

    if args.md_out:
        out_path = Path(args.md_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        md_sections = [
            f"# Loss Physical Scale Report",
            "",
            f"- step: {args.step}",
            f"- estimate_mode: {args.estimate_mode}",
            f"- sort_by: {args.sort_by}",
            f"- ascending: {args.ascending}",
            f"- top_k: {args.top_k}",
            f"- filter_config: {args.filter_config or '(none)'}",
            f"- filter_robot: {args.filter_robot or '(none)'}",
            f"- aggregate_by: {args.aggregate_by}",
            f"- aggregate_pick: {args.aggregate_pick}",
            "",
        ]
        if not args.no_summary:
            md_sections.extend([
                "## Summary",
                "",
                "```",
                _summary_text(rows, sorted_rows, step=args.step, estimate_mode=args.estimate_mode),
                "```",
                "",
            ])
        md_sections.extend([
            "## Table",
            "",
            _render_markdown_table(sorted_rows),
            "",
        ])
        out_path.write_text("\n".join(md_sections))
        print(f"Saved Markdown: {out_path}")

    if args.csv:
        out = Path(args.csv)
        _write_csv(sorted_rows, out)
        print(f"\nSaved CSV: {out}")

    if args.json_out:
        out = Path(args.json_out)
        _write_json(sorted_rows, out)
        print(f"Saved JSON: {out}")


if __name__ == "__main__":
    main()
