"""Synthesize L2 (5-class) episode labels for clothes multi-task value training.

Merges multi-source labels from PR #102 data audit outputs into a single
episode-level mapping used by value_returns_preprocessor l2_labels mode.

Priority (highest first):
  1. exclusion_list.json hits -> drop (all categories)
  2. flatten_classification bimodal repos -> excluded_bimodal_p0 (P0 no train)
  3. ground_truth_labels.episode_labels -> adopt as-is (self_play_qc 29 eps)
  4. episode_classification.json HIGH confidence -> adopt
  5. episode_classification.json MEDIUM confidence -> adopt (flag confidence)
  6. episode_classification.json LOW confidence -> exclude
  7. ground_truth_labels.repo_labels -> repo-level fallback for uncovered eps

Output schema:
  {"<repo_id>:<episode_index>": {
      "l2": "fold_success|flatten_success|shuffle_success|fold_failure|intervention_recovery",
      "task_group": "fold|flatten",
      "confidence": "high|medium|low",
      "source": "self_play_qc|episode_classification|repo_label|excluded_bimodal_p0"
  }}
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

FIVE_CLASSES = {
    "fold_success",
    "fold_failure",
    "flatten_success",
    "shuffle_success",
    "intervention_recovery",
}

TASK_GROUP_BY_L2 = {
    "fold_success": "fold",
    "fold_failure": "fold",
    "flatten_success": "flatten",
}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def collect_exclusion_repos(exclusion_list: dict[str, Any]) -> set[str]:
    repos: set[str] = set()
    for key in (
        "permanent_non_task",
        "permanent_data_quality",
        "permanent_structural",
        "temporary_upload_pending",
    ):
        for entry in exclusion_list.get(key, []):
            repos.add(entry["repo_id"])
    return repos


def collect_repo_task_groups(flatten_classification: list[dict]) -> tuple[set[str], dict[str, str]]:
    """Return (bimodal_repos, repo_task_group_map).

    repo_task_group_map only contains repos whose final_label unambiguously
    resolves to 'fold' or 'flatten'. Bimodal / non_task / disarrange repos
    are returned in bimodal_repos (excluded from training in P0).
    """
    bimodal: set[str] = set()
    task_group: dict[str, str] = {}
    for entry in flatten_classification:
        repo_id = entry["repo_id"]
        label = entry["final_label"]
        if label == "bimodal":
            bimodal.add(repo_id)
        elif label == "fold":
            task_group[repo_id] = "fold"
        elif label == "flatten":
            task_group[repo_id] = "flatten"
        # non_task / disarrange: leave out of task_group map so
        # shuffle_success / intervention_recovery episodes from these
        # repos fall back to 'fold' (default bias in P0).
    return bimodal, task_group


def resolve_task_group(l2: str, repo_id: str, repo_task_group: dict[str, str]) -> str:
    direct = TASK_GROUP_BY_L2.get(l2)
    if direct is not None:
        return direct
    return repo_task_group.get(repo_id, "fold")


def synthesize(
    episode_classification: dict[str, Any],
    ground_truth: dict[str, Any],
    exclusion_list: dict[str, Any],
    flatten_classification: list[dict],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    """Return (episode_labels, repo_fallback_labels, stats)."""
    excluded_repos = collect_exclusion_repos(exclusion_list)
    bimodal_repos, repo_task_group = collect_repo_task_groups(flatten_classification)

    labels: dict[str, dict[str, Any]] = {}
    repo_fallback: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()

    gt_episode = ground_truth.get("episode_labels", {})
    gt_repo = ground_truth.get("repo_labels", {})

    for ep in episode_classification.get("episodes", []):
        episode_key = ep["episode_key"]
        repo_id = episode_key.rsplit(":", 1)[0]

        if repo_id in excluded_repos:
            stats["dropped_exclusion"] += 1
            continue

        if repo_id in bimodal_repos:
            labels[episode_key] = {
                "l2": None,
                "task_group": None,
                "confidence": "high",
                "source": "excluded_bimodal_p0",
            }
            stats["excluded_bimodal_p0"] += 1
            continue

        # Priority 3: self_play_qc episode label
        gt_entry = gt_episode.get(episode_key)
        if gt_entry is not None:
            l2 = gt_entry["category"]
            if l2 not in FIVE_CLASSES:
                stats["gt_unknown_class"] += 1
                continue
            labels[episode_key] = {
                "l2": l2,
                "task_group": resolve_task_group(l2, repo_id, repo_task_group),
                "confidence": gt_entry.get("confidence", "high"),
                "source": "self_play_qc",
            }
            stats[f"self_play_qc_{l2}"] += 1
            continue

        # Priority 4-6: episode_classification by confidence
        confidence = ep.get("confidence", "low")
        l2 = ep.get("category")
        if l2 not in FIVE_CLASSES:
            stats["ep_unknown_class"] += 1
            continue
        if confidence == "low":
            stats["dropped_low_confidence"] += 1
            continue
        labels[episode_key] = {
            "l2": l2,
            "task_group": resolve_task_group(l2, repo_id, repo_task_group),
            "confidence": confidence,
            "source": "episode_classification",
        }
        stats[f"episode_classification_{confidence}_{l2}"] += 1

    # Priority 7: repo_label fallback for repos not yet covered.
    # Stored in a separate top-level bucket so downstream consumers can
    # broadcast to all episodes of the repo explicitly.
    covered_repos = {k.rsplit(":", 1)[0] for k in labels}
    for repo_id, repo_entry in gt_repo.items():
        if repo_id in excluded_repos or repo_id in bimodal_repos:
            continue
        if repo_id in covered_repos:
            continue
        l2 = repo_entry.get("category")
        if l2 not in FIVE_CLASSES:
            continue
        repo_fallback[repo_id] = {
            "l2": l2,
            "task_group": resolve_task_group(l2, repo_id, repo_task_group),
            "confidence": repo_entry.get("confidence", "medium"),
            "source": "repo_label",
        }
        stats[f"repo_label_{l2}"] += 1

    return labels, repo_fallback, dict(stats)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-audit-dir", type=Path, default=Path("test_results/data_audit"))
    parser.add_argument("--output", type=Path, default=Path("test_results/data_audit/l2_labels_v0410.json"))
    args = parser.parse_args()

    d = args.data_audit_dir
    episode_classification = load_json(d / "episode_classification.json")
    ground_truth = load_json(d / "ground_truth_labels.json")
    exclusion_list = load_json(d / "exclusion_list.json")
    flatten_classification = load_json(d / "flatten_classification.json")

    labels, repo_fallback, stats = synthesize(
        episode_classification=episode_classification,
        ground_truth=ground_truth,
        exclusion_list=exclusion_list,
        flatten_classification=flatten_classification,
    )

    n_valid = sum(1 for v in labels.values() if v.get("l2") is not None)
    n_bimodal = sum(1 for v in labels.values() if v.get("source") == "excluded_bimodal_p0")

    output = {
        "metadata": {
            "source": "scripts/benchmark/synthesize_l2_labels.py",
            "n_entries": len(labels),
            "n_valid_l2": n_valid,
            "n_excluded_bimodal": n_bimodal,
            "n_repo_fallback_labels": len(repo_fallback),
            "stats": stats,
        },
        "labels": labels,
        "repo_fallback_labels": repo_fallback,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    # Class distribution warning
    class_counts: Counter[str] = Counter()
    for v in labels.values():
        l2 = v.get("l2")
        if l2 is not None:
            class_counts[l2] += 1
    total_valid = sum(class_counts.values())
    print(f"[synthesize_l2_labels] wrote {len(labels)} entries -> {args.output}")
    print(f"  valid (with l2): {n_valid}, excluded_bimodal: {n_bimodal}")
    print("  class distribution:")
    for l2 in sorted(FIVE_CLASSES):
        n = class_counts.get(l2, 0)
        pct = (100.0 * n / total_valid) if total_valid else 0.0
        warn = "  !! <3%" if pct < 3.0 and n > 0 else ""
        print(f"    {l2:24s} {n:5d} ({pct:5.1f}%){warn}")

    if n_valid < 2800:
        print(f"[WARNING] n_valid={n_valid} < 2800 (plan expected HIGH 1921 + MEDIUM 929 - exclusion)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
