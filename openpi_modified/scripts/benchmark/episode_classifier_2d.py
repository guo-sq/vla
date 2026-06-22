"""2D Episode Classifier - 基于 (head_pred, tail_pred) 的 5 类状态分类器.

根据 value model 对每个 episode 的首帧 (head_pred) 和末帧 (tail_pred) 预测,
将 episode 分类为以下 5 类之一:

- fold_success 叠好 (成功完成 fold 任务)
- flatten_success 叠平 (仅铺平未折叠)
- shuffle_success 打乱 (disarrange 任务成功)
- fold_failure 失败 (试图 fold 但未完成)
- intervention_recovery 接管纠错 (从失败状态被纠正回成功)

**阈值状态**:
- tail_pred 轴: 基于 1D 分析 (`test_results/data_audit/tail_pred_1d_analysis.json`) ,
  fast_mode 模型的三类分离度 100% (0% 重叠) , 阈值高置信.
- head_pred 轴: **待 DLC 完成后填充**. 当前 head_pred 维度留空,
  `classify_episode` 会自动降级为 1D 分类 (只看 tail_pred) .

**使用方式**:

    # Mode 1: 分类单个 episode
    label, confidence = classify_episode(head_pred=-0.85, tail_pred=-0.003)

    # Mode 2: 批量分类 (读 episode_details.json)
    results = classify_all(
        details_path="test_results/benchmark/xxx/metrics/episode_details.json",
        output_path="test_results/data_audit/episode_classification.json",
    )

**待办** (DLC 完成后) :
1. 运行 `tail_pred_analysis.py` 的 2D 扩展版, 提取 head_pred 分布
2. 填充 `HEAD_PRED_RANGES` 常量
3. 在 58 个 bimodal repos 上验证细化分类是否合理
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path

# ---------------------------------------------------------------------------
# 常量: 从 1D 分析迁移的 tail_pred 分布 (fast_mode 模型)
# ---------------------------------------------------------------------------

# 源: test_results/data_audit/tail_pred_1d_analysis.json (per_category, fast_mode)
TAIL_PRED_STATS: dict[str, dict[str, float]] = {
    "fold_success": {"mean": -0.0035, "std": 0.0007, "p5": -0.0047, "p95": -0.0027},
    "intervention_recovery": {"mean": -0.0041, "std": 0.0022, "p5": -0.0074, "p95": -0.0030},
    "fold_failure": {"mean": -0.7668, "std": 0.0867, "p5": -0.8659, "p95": -0.6044},
    "shuffle_success": {"mean": -0.8514, "std": 0.0233, "p5": -0.8912, "p95": -0.8144},
    # flatten_success 尚无 ground-truth 1D 数据, 需要从确认 flatten repos 上单独测
    "flatten_success": {"mean": None, "std": None, "p5": None, "p95": None},
}

# Midpoint thresholds from separation analysis (fast_mode, 0% overlap)
# fold_success 中心 ≈ 0, shuffle_success 中心 ≈ -0.85, fold_failure 中心 ≈ -0.77
TAIL_PRED_THRESHOLDS: dict[str, float] = {
    "success_vs_failure": -0.385,  # fold_success_vs_fold_failure midpoint
    "success_vs_shuffle": -0.427,  # fold_success_vs_shuffle_success midpoint
    "flatten_vs_success": -0.10,  # 估计: flatten 是 "未完成的 fold", tail 比 success 低
    "flatten_vs_failure": -0.50,  # 估计: flatten 应在 success 和 failure 之间
    "failure_vs_shuffle": -0.82,  # 估计: shuffle 比 failure 更负
}

# ---------------------------------------------------------------------------
# 常量: head_pred 范围 -- **待 DLC 完成后填充**
# ---------------------------------------------------------------------------
# 当前全部为 None, classify_episode 会自动降级到 1D 模式
HEAD_PRED_RANGES: dict[str, tuple[float, float] | None] = {
    "fold_success": None,  # TODO: 随机起点应该有较宽的分布, 期望 p5-p95 覆盖 [-0.9, -0.3]
    "flatten_success": None,  # TODO: 初始状态可能较乱 -> head 较负
    "shuffle_success": None,  # TODO: 起点是"叠好"状态 -> head 预计 ≈ 0
    "fold_failure": None,  # TODO: 随机起点 -> head 分布应与 fold_success 类似
    "intervention_recovery": None,  # TODO: 起点为失败态 -> head 应偏负, tail 偏 0
}


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


class Category(str, Enum):
    FOLD_SUCCESS = "fold_success"
    FLATTEN_SUCCESS = "flatten_success"
    SHUFFLE_SUCCESS = "shuffle_success"
    FOLD_FAILURE = "fold_failure"
    INTERVENTION_RECOVERY = "intervention_recovery"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    HIGH = "high"  # 距离类中心 < 1 std
    MEDIUM = "medium"  # 距离 1-2 std
    LOW = "low"  # 距离 > 2 std, 或 head_pred 缺失降级到 1D


@dataclass
class Region:
    """2D 分类区域 (矩形边界) . 每个 category 对应一个 Region."""

    category: Category
    tail_min: float
    tail_max: float
    head_min: float | None = None  # None -> 不做 head 维度检查 (1D 模式)
    head_max: float | None = None
    center_tail: float = 0.0
    center_head: float = 0.0
    std_tail: float = 1.0
    std_head: float = 1.0

    def contains(self, head_pred: float | None, tail_pred: float) -> bool:
        if not (self.tail_min <= tail_pred <= self.tail_max):
            return False
        return not (
            self.head_min is not None and head_pred is not None and not (self.head_min <= head_pred <= self.head_max)
        )

    def normalized_distance(self, head_pred: float | None, tail_pred: float) -> float:
        """返回 episode 到类中心的归一化距离 (越小越置信) ."""
        dt = (tail_pred - self.center_tail) / max(self.std_tail, 1e-6)
        if head_pred is None or self.head_min is None:
            return abs(dt)
        dh = (head_pred - self.center_head) / max(self.std_head, 1e-6)
        return math.sqrt(dt * dt + dh * dh)


@dataclass
class ClassificationResult:
    episode_key: str
    head_pred: float | None
    tail_pred: float
    category: Category
    confidence: Confidence
    distance: float
    mode: str  # "1d" or "2d"
    # 可选: 原始 benchmark 元信息
    original_quadrant: str | None = None
    original_role: str | None = None
    n_frames: int | None = None


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------


def _head_bounds(
    category_key: str,
    override: dict[str, tuple[float, float] | None] | None,
) -> tuple[float | None, float | None]:
    """Resolve the ``(head_min, head_max)`` tuple for a category.

    Priority: ``override[category_key]`` if present, else the module-level
    :data:`HEAD_PRED_RANGES`. A value of ``None`` at either layer means
    "no head constraint" -> the region stays on 1D tail_pred fallback.
    """
    if override is not None and category_key in override:
        bounds = override[category_key]
    else:
        bounds = HEAD_PRED_RANGES.get(category_key)
    if bounds is None:
        return None, None
    return bounds[0], bounds[1]


def build_default_regions(
    head_pred_ranges: dict[str, tuple[float, float] | None] | None = None,
) -> list[Region]:
    """基于 TAIL_PRED_STATS 构造默认 5 个分类区域.

    tail 轴用 p5-p95 作为 min/max, 中心用 mean, std 用 std 值.
    head 轴默认从模块级 :data:`HEAD_PRED_RANGES` 读取 (启动时全 None,
    1D 模式) . 传入 ``head_pred_ranges`` 可覆盖(例如从
    ``fill_head_pred_ranges.py`` 产出的 JSON 加载而来), 覆盖值不会修改
    全局常量. 覆盖 dict 里没有的 category 会 fallback 到模块默认.
    """
    regions: list[Region] = []

    # fold_success: tail ∈ [-0.05, 0] (严格: std 太小, 扩展一点)
    head_min, head_max = _head_bounds("fold_success", head_pred_ranges)
    regions.append(
        Region(
            category=Category.FOLD_SUCCESS,
            tail_min=TAIL_PRED_THRESHOLDS["flatten_vs_success"],
            tail_max=0.1,
            center_tail=TAIL_PRED_STATS["fold_success"]["mean"],
            std_tail=max(TAIL_PRED_STATS["fold_success"]["std"], 0.02),  # floor to avoid ultra-tight
            head_min=head_min,
            head_max=head_max,
        )
    )

    # intervention_recovery: 与 fold_success 在 tail 上重合, 用 head 区分
    head_min, head_max = _head_bounds("intervention_recovery", head_pred_ranges)
    regions.append(
        Region(
            category=Category.INTERVENTION_RECOVERY,
            tail_min=TAIL_PRED_THRESHOLDS["flatten_vs_success"],
            tail_max=0.1,
            center_tail=TAIL_PRED_STATS["intervention_recovery"]["mean"],
            std_tail=max(TAIL_PRED_STATS["intervention_recovery"]["std"], 0.02),
            head_min=head_min,
            head_max=head_max,
        )
    )

    # flatten_success: tail 在中间区间 [-0.5, -0.1]
    head_min, head_max = _head_bounds("flatten_success", head_pred_ranges)
    regions.append(
        Region(
            category=Category.FLATTEN_SUCCESS,
            tail_min=TAIL_PRED_THRESHOLDS["flatten_vs_failure"],
            tail_max=TAIL_PRED_THRESHOLDS["flatten_vs_success"],
            center_tail=(TAIL_PRED_THRESHOLDS["flatten_vs_failure"] + TAIL_PRED_THRESHOLDS["flatten_vs_success"]) / 2,
            std_tail=0.1,  # 估计值, 待实测
            head_min=head_min,
            head_max=head_max,
        )
    )

    # fold_failure: tail ∈ [-0.82, -0.5]
    head_min, head_max = _head_bounds("fold_failure", head_pred_ranges)
    regions.append(
        Region(
            category=Category.FOLD_FAILURE,
            tail_min=TAIL_PRED_THRESHOLDS["failure_vs_shuffle"],
            tail_max=TAIL_PRED_THRESHOLDS["flatten_vs_failure"],
            center_tail=TAIL_PRED_STATS["fold_failure"]["mean"],
            std_tail=TAIL_PRED_STATS["fold_failure"]["std"],
            head_min=head_min,
            head_max=head_max,
        )
    )

    # shuffle_success: tail < -0.82
    head_min, head_max = _head_bounds("shuffle_success", head_pred_ranges)
    regions.append(
        Region(
            category=Category.SHUFFLE_SUCCESS,
            tail_min=-1.5,
            tail_max=TAIL_PRED_THRESHOLDS["failure_vs_shuffle"],
            center_tail=TAIL_PRED_STATS["shuffle_success"]["mean"],
            std_tail=TAIL_PRED_STATS["shuffle_success"]["std"],
            head_min=head_min,
            head_max=head_max,
        )
    )

    return regions


def load_head_pred_ranges_from_json(
    path: Path,
) -> dict[str, tuple[float, float] | None]:
    """Load head_pred_ranges.json (output of ``fill_head_pred_ranges.py``)
    into the ``{category: (head_min, head_max) | None}`` format expected by
    :func:`build_default_regions`.

    The JSON entry schema contains both the active bounds ``head_min`` /
    ``head_max`` and the summary stats ``p5`` / ... / ``p95``. We deliberately
    read the bounds, not the summary stats, so that calling the fill script
    with wider percentiles (e.g. lower=1, upper=99) actually widens the
    classifier regions. Categories with ``None`` (insufficient samples) stay
    ``None`` so the classifier falls back to 1D tail_pred for them. A
    malformed entry missing the bounds raises ``KeyError``.
    """
    with open(path) as f:
        raw = json.load(f)

    result: dict[str, tuple[float, float] | None] = {}
    for category, entry in raw.items():
        if entry is None:
            result[category] = None
            continue
        result[category] = (float(entry["head_min"]), float(entry["head_max"]))
    return result


def classify_episode(
    head_pred: float | None,
    tail_pred: float,
    regions: list[Region] | None = None,
) -> tuple[Category, Confidence, float, str]:
    """单个 episode 分类.

    Returns:
        (category, confidence, distance, mode)

    Mode:
        - "2d": head_pred 非 None 且至少一个 region 设置了 head_min/head_max
        - "1d": 降级, 仅看 tail_pred
    """
    if regions is None:
        regions = build_default_regions()

    has_head_ranges = any(r.head_min is not None for r in regions)
    mode = "2d" if (head_pred is not None and has_head_ranges) else "1d"

    # 1. 先过滤出 tail_pred 落在区间内的候选
    candidates = [r for r in regions if r.contains(head_pred, tail_pred)]

    if not candidates:
        # tail 不落在任何区间 (边界情况) , 取最近的区间
        distances = [(r.normalized_distance(head_pred, tail_pred), r) for r in regions]
        distances.sort(key=lambda x: x[0])
        dist, region = distances[0]
        return region.category, Confidence.LOW, dist, mode

    # 2. 落入多个区间时 (例如 fold_success 和 intervention_recovery 在 tail 上重合)
    # 选 normalized_distance 最小的
    candidates_with_dist = [(r.normalized_distance(head_pred, tail_pred), r) for r in candidates]
    candidates_with_dist.sort(key=lambda x: x[0])
    dist, region = candidates_with_dist[0]

    # 3. 置信度
    if dist < 1.0:
        conf = Confidence.HIGH
    elif dist < 2.0:
        conf = Confidence.MEDIUM
    else:
        conf = Confidence.LOW

    # 1D 模式下置信度降一级 (head_pred 缺失 -> 本质是欠定)
    if mode == "1d":
        conf = Confidence.LOW if conf == Confidence.MEDIUM else conf
        if conf == Confidence.HIGH:
            conf = Confidence.MEDIUM

    return region.category, conf, dist, mode


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------


def classify_all(
    details_path: Path,
    output_path: Path | None = None,
    regions: list[Region] | None = None,
) -> list[ClassificationResult]:
    """读取 episode_details.json, 分类每个 episode, 可选写出 JSON."""
    details_path = Path(details_path)
    with open(details_path) as f:
        episodes = json.load(f)

    if regions is None:
        regions = build_default_regions()

    results: list[ClassificationResult] = []
    for ep in episodes:
        head_pred = ep.get("head_pred")  # 可能为 None (旧 benchmark)
        tail_pred = ep.get("tail_pred")
        if tail_pred is None:
            continue  # 跳过没有 tail_pred 的 (不应该发生)

        category, confidence, distance, mode = classify_episode(
            head_pred=head_pred,
            tail_pred=tail_pred,
            regions=regions,
        )

        results.append(
            ClassificationResult(
                episode_key=ep.get("episode_key", "unknown"),
                head_pred=head_pred,
                tail_pred=tail_pred,
                category=category,
                confidence=confidence,
                distance=distance,
                mode=mode,
                original_quadrant=ep.get("quadrant"),
                original_role=ep.get("role"),
                n_frames=ep.get("n_frames"),
            )
        )

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metadata": {
                "source": str(details_path),
                "n_episodes": len(results),
                "mode_counts": _count_field(results, "mode"),
                "category_counts": _count_field(results, "category"),
                "confidence_counts": _count_field(results, "confidence"),
                "thresholds_used": TAIL_PRED_THRESHOLDS,
                "head_pred_ranges_filled": any(v is not None for v in HEAD_PRED_RANGES.values()),
            },
            "episodes": [_result_to_dict(r) for r in results],
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    return results


def _count_field(results: list[ClassificationResult], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        key = str(getattr(r, field))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _result_to_dict(r: ClassificationResult) -> dict:
    d = asdict(r)
    d["category"] = r.category.value
    d["confidence"] = r.confidence.value
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="2D episode classifier")
    parser.add_argument("--details", required=True, help="Path to episode_details.json")
    parser.add_argument("--output", required=True, help="Path to output classification JSON")
    parser.add_argument(
        "--head-pred-ranges",
        type=Path,
        default=None,
        help="Optional path to head_pred_ranges.json from "
        "fill_head_pred_ranges.py. When set, promotes the "
        "classifier from 1D fallback mode to full 2D dispatch "
        "for categories with sufficient samples.",
    )
    args = parser.parse_args()

    regions = None
    if args.head_pred_ranges is not None:
        head_pred_ranges = load_head_pred_ranges_from_json(args.head_pred_ranges)
        regions = build_default_regions(head_pred_ranges=head_pred_ranges)
        active = sum(1 for c, v in head_pred_ranges.items() if v is not None)
        print(
            f"[classifier] Loaded head_pred_ranges from {args.head_pred_ranges} "
            f"({active}/{len(head_pred_ranges)} categories with head bounds)"
        )

    results = classify_all(
        details_path=Path(args.details),
        output_path=Path(args.output),
        regions=regions,
    )

    # Print summary
    print(f"Classified {len(results)} episodes")
    print(f"Mode: {_count_field(results, 'mode')}")
    print(f"Categories: {_count_field(results, 'category')}")
    print(f"Confidence: {_count_field(results, 'confidence')}")


if __name__ == "__main__":
    main()
