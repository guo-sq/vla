"""Build multi-task benchmark v0409 - fold + flatten tasks.

基于 Step 4 的分类结果构建两个独立评测任务:

- **Task 1: fold (叠好衣服) ** -- 目标配额: TP≥80, TN≥60, FP≥30, edge≥10
- **Task 2: flatten (叠平衣服) ** -- 目标配额: TP≥30, TN≥20

**输入**:
- `episode_classification.json` (任务 D + DLC 结果, Step 4e 产出)
- `flatten_classification.json` (任务 4c-2 已产出)
- `exclusion_list.json` (任务 B 已产出)

**输出**:
- `test_results/split/clothes_v0409/fold/manifest.json` - Task 1
- `test_results/split/clothes_v0409/fold/repo_list.txt`
- `test_results/split/clothes_v0409/flatten/manifest.json` - Task 2
- `test_results/split/clothes_v0409/flatten/repo_list.txt`
- `test_results/split/clothes_v0409/split_report.md` - 总报告

**使用**:

    # Full run (after DLC + task D completes)
    python scripts/benchmark/build_benchmark_v0409.py \\
      --episode-classification test_results/data_audit/episode_classification.json \\
      --flatten-classification test_results/data_audit/flatten_classification.json \\
      --exclusion-list test_results/data_audit/exclusion_list.json \\
      --output-dir test_results/split/clothes_v0409 \\
      --seed 42

    # Dry run (无 episode_classification.json 时, 仅检查 repo-level 切分可行性)
    python scripts/benchmark/build_benchmark_v0409.py --dry-run --output-dir test_results/split/clothes_v0409_dryrun
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import random

# ---------------------------------------------------------------------------
# 配额配置
# ---------------------------------------------------------------------------

FOLD_TASK_QUOTAS: dict[str, int] = {
    "TP": 80,  # fold_success
    "TN": 60,  # shuffle_success
    "FP": 30,  # fold_failure
    "edge": 10,  # intervention_recovery
}

FLATTEN_TASK_QUOTAS: dict[str, int] = {
    "TP": 30,  # flatten_success
    "TN": 20,  # shuffle_success (与 fold 任务共享同一批 TN 源)
}

# category -> quadrant 映射
FOLD_CATEGORY_TO_QUADRANT: dict[str, str] = {
    "fold_success": "TP",
    "shuffle_success": "TN",
    "fold_failure": "FP",
    "intervention_recovery": "edge",
}

FLATTEN_CATEGORY_TO_QUADRANT: dict[str, str] = {
    "flatten_success": "TP",
    "shuffle_success": "TN",
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    episode_key: str  # 如 "record.xxx:0"
    repo_id: str
    episode_idx: int
    category: str  # 2D 分类结果
    confidence: str  # high/medium/low
    flatten_label: str  # fold/flatten/bimodal/disarrange/non_task/unknown
    head_pred: float | None = None
    tail_pred: float | None = None

    @property
    def full_repo_id(self) -> str:
        return self.repo_id


@dataclass
class TaskSplit:
    task_name: str  # "fold" or "flatten"
    quadrants: dict[str, list[Episode]] = field(default_factory=dict)
    quotas: dict[str, int] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(len(eps) for eps in self.quadrants.values())

    def meets_quotas(self) -> bool:
        return all(len(self.quadrants.get(q, [])) >= n for q, n in self.quotas.items())


# ---------------------------------------------------------------------------
# 加载器
# ---------------------------------------------------------------------------


def load_exclusion_set(path: Path) -> set[str]:
    """读 exclusion_list.json, 返回永久排除的 repo_id 集合 (不含 temporary) ."""
    with open(path) as f:
        data = json.load(f)
    excluded: set[str] = set()
    for cat in ("permanent_non_task", "permanent_data_quality", "permanent_structural"):
        for entry in data.get(cat, []):
            excluded.add(entry["repo_id"])
    return excluded


def load_flatten_labels(path: Path) -> dict[str, str]:
    """返回 repo_id -> final_label 映射."""
    with open(path) as f:
        entries = json.load(f)
    return {e["repo_id"]: e["final_label"] for e in entries}


def load_episodes(
    episode_classification_path: Path,
    flatten_labels: dict[str, str],
    excluded: set[str],
) -> list[Episode]:
    """加载所有分类后的 episodes, 应用排除清单和 flatten 标签."""
    with open(episode_classification_path) as f:
        data = json.load(f)

    episodes_data = data.get("episodes", data) if isinstance(data, dict) else data
    episodes: list[Episode] = []

    for ep in episodes_data:
        key = ep["episode_key"]
        # 解析 "repo_id:episode_idx"
        if ":" in key:
            repo_id, idx_str = key.rsplit(":", 1)
            try:
                episode_idx = int(idx_str)
            except ValueError:
                episode_idx = -1
        else:
            repo_id, episode_idx = key, -1

        if repo_id in excluded:
            continue

        episodes.append(
            Episode(
                episode_key=key,
                repo_id=repo_id,
                episode_idx=episode_idx,
                category=ep.get("category", "unknown"),
                confidence=ep.get("confidence", "unknown"),
                flatten_label=flatten_labels.get(repo_id, "unknown"),
                head_pred=ep.get("head_pred"),
                tail_pred=ep.get("tail_pred"),
            )
        )

    return episodes


# ---------------------------------------------------------------------------
# 切分逻辑
# ---------------------------------------------------------------------------


def build_fold_task(
    episodes: list[Episode],
    quotas: dict[str, int],
    rng: random.Random,
    *,
    high_conf_only: bool = True,
) -> TaskSplit:
    """Task 1: fold -- 选高置信的 fold_success/shuffle_success/fold_failure/intervention.

    排除规则:
    - flatten_label == "flatten" 的 repos 不进入 fold 任务
    - flatten_label == "disarrange" 的 repos 只能作 TN
    - flatten_label == "bimodal" 的 repos 需要 episode 级别的 category 判定
    """
    task = TaskSplit(task_name="fold", quotas=dict(quotas))

    for quadrant in quotas:
        task.quadrants[quadrant] = []

    # 按 category 分桶
    buckets: dict[str, list[Episode]] = {q: [] for q in quotas}

    for ep in episodes:
        # 对 fold 任务, flatten 类 repos 必须排除 (避免交叉污染)
        if ep.flatten_label == "flatten":
            continue
        # disarrange repos 只能作为 TN
        if ep.flatten_label == "disarrange" and ep.category != "shuffle_success":
            continue

        quadrant = FOLD_CATEGORY_TO_QUADRANT.get(ep.category)
        if quadrant is None:
            continue
        if high_conf_only and ep.confidence not in ("high", "medium"):
            continue
        buckets[quadrant].append(ep)

    # 按配额抽样
    for quadrant, target in quotas.items():
        pool = buckets[quadrant]
        rng.shuffle(pool)
        task.quadrants[quadrant] = pool[:target]

    return task


def build_flatten_task(
    episodes: list[Episode],
    quotas: dict[str, int],
    rng: random.Random,
    *,
    high_conf_only: bool = True,
) -> TaskSplit:
    """Task 2: flatten -- TP 来源: category=flatten_success OR (flatten_label=flatten AND category=fold_success)."""
    task = TaskSplit(task_name="flatten", quotas=dict(quotas))
    for quadrant in quotas:
        task.quadrants[quadrant] = []

    buckets: dict[str, list[Episode]] = {q: [] for q in quotas}

    for ep in episodes:
        # TP 规则: 明确的 flatten_success 或在 flatten repos 中被 2D 分类为 "已完成状态"
        is_flatten_tp = ep.category == "flatten_success" or (
            ep.flatten_label == "flatten" and ep.category == "fold_success"
        )
        # TN 规则: 和 fold 任务共享 shuffle_success
        is_flatten_tn = ep.category == "shuffle_success"

        if is_flatten_tp:
            if high_conf_only and ep.confidence not in ("high", "medium"):
                continue
            buckets["TP"].append(ep)
        elif is_flatten_tn:
            if high_conf_only and ep.confidence not in ("high", "medium"):
                continue
            buckets["TN"].append(ep)

    for quadrant, target in quotas.items():
        pool = buckets[quadrant]
        rng.shuffle(pool)
        task.quadrants[quadrant] = pool[:target]

    return task


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------


def write_task_output(task: TaskSplit, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "task_name": task.task_name,
        "quotas": task.quotas,
        "actual_counts": {q: len(eps) for q, eps in task.quadrants.items()},
        "meets_quotas": task.meets_quotas(),
        "total_episodes": task.total,
        "episodes": {q: [ep.episode_key for ep in eps] for q, eps in task.quadrants.items()},
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # repo_list.txt - 去重的 repo_id 列表 (供 run_benchmark.sh 使用)
    repo_ids: set[str] = set()
    for eps in task.quadrants.values():
        for ep in eps:
            repo_ids.add(ep.repo_id)
    with open(output_dir / "repo_list.txt", "w") as f:
        for rid in sorted(repo_ids):
            f.write(rid + "\n")


def write_report(
    fold_task: TaskSplit,
    flatten_task: TaskSplit,
    output_path: Path,
    metadata: dict,
) -> None:
    lines: list[str] = [
        "# Benchmark v0409 构建报告",
        "",
        f"**生成时间**: {metadata.get('timestamp', 'unknown')}",
        f"**随机种子**: {metadata.get('seed')}",
        f"**输入 episodes**: {metadata.get('total_input_episodes', 0)}",
        f"**永久排除 repos**: {metadata.get('excluded_count', 0)}",
        "",
        "---",
        "",
        "## Task 1: Fold (叠好衣服) ",
        "",
        "| 象限 | 目标配额 | 实际数量 | 达标 |",
        "|------|---------|---------|------|",
    ]
    for q, target in fold_task.quotas.items():
        actual = len(fold_task.quadrants.get(q, []))
        mark = "✅" if actual >= target else "❌"
        lines.append(f"| {q} | ≥{target} | {actual} | {mark} |")

    lines.extend(
        [
            "",
            f"**Task 1 总 episodes**: {fold_task.total}",
            f"**Task 1 涉及 repos**: {len({ep.repo_id for eps in fold_task.quadrants.values() for ep in eps})}",
            "",
            "---",
            "",
            "## Task 2: Flatten (叠平衣服) ",
            "",
            "| 象限 | 目标配额 | 实际数量 | 达标 |",
            "|------|---------|---------|------|",
        ]
    )
    for q, target in flatten_task.quotas.items():
        actual = len(flatten_task.quadrants.get(q, []))
        mark = "✅" if actual >= target else "❌"
        lines.append(f"| {q} | ≥{target} | {actual} | {mark} |")

    lines.extend(
        [
            "",
            f"**Task 2 总 episodes**: {flatten_task.total}",
            f"**Task 2 涉及 repos**: {len({ep.repo_id for eps in flatten_task.quadrants.values() for ep in eps})}",
            "",
            "---",
            "",
            "## 使用方式",
            "",
            "两个任务的 repo 列表分别在: ",
            "- `fold/repo_list.txt` - 输入到 `run_benchmark.sh` 的 `--repos` 参数",
            "- `flatten/repo_list.txt` - 同上",
            "",
            "两个任务的 prompt 不同: ",
            '- **fold**: `"You are a two-armed piper robot ... Your task is to fold a T-shirt ..."`',
            '- **flatten**: `"You are a two-armed piper robot ... Your task is to straighten/lay flat a T-shirt ..."`',
            "",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build multi-task benchmark v0409")
    parser.add_argument(
        "--episode-classification", type=Path, help="Path to episode_classification.json (from task D + DLC)"
    )
    parser.add_argument(
        "--flatten-classification", type=Path, default=Path("test_results/data_audit/flatten_classification.json")
    )
    parser.add_argument("--exclusion-list", type=Path, default=Path("test_results/data_audit/exclusion_list.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--dry-run", action="store_true", help="Skip episode loading, only check flatten + exclusion feasibility"
    )
    args = parser.parse_args()

    import datetime

    rng = random.Random(args.seed)
    metadata = {
        "timestamp": datetime.datetime.now(tz=datetime.UTC).isoformat(),
        "seed": args.seed,
    }

    # 加载必要输入
    excluded = load_exclusion_set(args.exclusion_list)
    flatten_labels = load_flatten_labels(args.flatten_classification)
    metadata["excluded_count"] = len(excluded)
    print(f"[v0409] Excluded repos: {len(excluded)}")
    print(f"[v0409] Flatten-labeled repos: {len(flatten_labels)}")

    # Dry-run: 只输出 flatten-only repos 和排除摘要
    if args.dry_run:
        print("[v0409] DRY-RUN MODE")
        print(
            f" flatten_success candidates (repo-level): " f"{sum(1 for v in flatten_labels.values() if v == 'flatten')}"
        )
        print(f" bimodal candidates: " f"{sum(1 for v in flatten_labels.values() if v == 'bimodal')}")
        print(
            f" disarrange candidates (shuffle source): "
            f"{sum(1 for v in flatten_labels.values() if v == 'disarrange')}"
        )
        return

    if not args.episode_classification or not args.episode_classification.exists():
        print(f"[v0409] ERROR: episode_classification.json not found at {args.episode_classification}")
        print(" Run episode_classifier_2d.py first (after DLC completes)")
        return

    episodes = load_episodes(args.episode_classification, flatten_labels, excluded)
    metadata["total_input_episodes"] = len(episodes)
    print(f"[v0409] Loaded {len(episodes)} classified episodes (after exclusion)")

    # 构建两个任务
    fold_task = build_fold_task(episodes, FOLD_TASK_QUOTAS, rng)
    flatten_task = build_flatten_task(episodes, FLATTEN_TASK_QUOTAS, rng)

    print(f"[v0409] Fold task: {fold_task.total} eps, meets quotas: {fold_task.meets_quotas()}")
    for q, eps in fold_task.quadrants.items():
        print(f" {q}: {len(eps)}/{fold_task.quotas[q]}")

    print(f"[v0409] Flatten task: {flatten_task.total} eps, meets quotas: {flatten_task.meets_quotas()}")
    for q, eps in flatten_task.quadrants.items():
        print(f" {q}: {len(eps)}/{flatten_task.quotas[q]}")

    # 写出
    write_task_output(fold_task, args.output_dir / "fold")
    write_task_output(flatten_task, args.output_dir / "flatten")
    write_report(fold_task, flatten_task, args.output_dir / "split_report.md", metadata)

    print(f"[v0409] Written to {args.output_dir}")


if __name__ == "__main__":
    main()
