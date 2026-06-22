"""Fuse external label sources into ground-truth category labels for the 2D
classifier's HEAD_PRED_RANGES prior.

Two sources, both already produced by earlier steps:

1. **self_play_label_qc.json** (Step 2, ``self_play_label_qc.py``) - per-episode
   QC'd labels from self-play metadata. Each entry has ``role`` ("builder",
   "destroyer", "unknown"), ``success``, and ``intervention_count``.

2. **flatten_classification.json** (Step 4c-2, ``flatten_classifier.py``) -
   per-repo rule-based labels on task descriptions. Each entry has
   ``final_label`` ("fold", "flatten", "disarrange", "bimodal", "non_task")
   and ``confidence`` ("high", "medium", "low").

**Strict mode**: only labels we trust as ground truth are kept; ambiguous
entries (``role=unknown``, ``final_label=bimodal``, ``final_label=fold`` without
per-episode disambiguation) are dropped. Self-classified tail_pred labels are
deliberately **not** consulted - those are the very thing HEAD_PRED_RANGES is
supposed to improve.

Output structure:
    {
        "episode_labels": {
            "<episode_key>": {"category": ..., "confidence": ..., "source": ...}
        },
        "repo_labels": {
            "<repo_id>": {"category": ..., "confidence": ..., "source": ...}
        }
    }

The downstream consumer (:mod:`fill_head_pred_ranges`) looks up an episode's
category by first checking ``episode_labels[episode_key]`` and falling back to
``repo_labels[repo_id]``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Category names must match openpi.training.episode_classifier_2d.Category.
CATEGORY_FOLD_SUCCESS = "fold_success"
CATEGORY_FLATTEN_SUCCESS = "flatten_success"
CATEGORY_SHUFFLE_SUCCESS = "shuffle_success"
CATEGORY_FOLD_FAILURE = "fold_failure"
CATEGORY_INTERVENTION_RECOVERY = "intervention_recovery"


def _classify_self_play_entry(entry: dict[str, Any]) -> str | None:
    """Return the ground-truth category for a self_play_label_qc row, or None.

    ``role=unknown`` and missing role fall through to ``None`` - we refuse to
    guess. ``builder + success + intervention>0`` is the trickiest case: the
    episode did finish successfully, but only because a human operator stepped
    in. For value-model benchmarking this is its own category
    (``intervention_recovery``) distinct from clean successes.
    """
    role = entry.get("role")
    if role not in {"builder", "destroyer"}:
        return None

    success = entry.get("success", False)
    intervention = entry.get("intervention_count", 0) or 0

    if role == "destroyer":
        return CATEGORY_SHUFFLE_SUCCESS if success else None

    # role == "builder"
    if not success:
        return CATEGORY_FOLD_FAILURE
    if intervention > 0:
        return CATEGORY_INTERVENTION_RECOVERY
    return CATEGORY_FOLD_SUCCESS


def _classify_flatten_entry(entry: dict[str, Any]) -> str | None:
    """Return the ground-truth category for a flatten_classification row, or None.

    Accepted labels:
    - ``flatten`` high/medium -> flatten_success
    - ``fold`` high -> fold_success (broadens fold coverage from ~15
      self_play episodes to hundreds of record.* episodes; safe because
      flatten_classifier's high-confidence ``fold`` label is a string-rule
      match on ``task_text`` without value-model input, so it is not circular)
    - ``disarrange`` high -> shuffle_success

    Dropped:
    - ``fold`` medium/low - rule precision too low for per-episode dist
    - ``bimodal`` - explicitly "don't know"
    - ``non_task`` - already on exclusion list
    """
    final_label = entry.get("final_label")
    confidence = entry.get("confidence")

    if final_label == "flatten" and confidence in {"high", "medium"}:
        return CATEGORY_FLATTEN_SUCCESS
    if final_label == "fold" and confidence == "high":
        return CATEGORY_FOLD_SUCCESS
    if final_label == "disarrange" and confidence == "high":
        return CATEGORY_SHUFFLE_SUCCESS
    return None


def merge_labels(
    self_play_qc: list[dict[str, Any]],
    flatten_classification: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, str]]]:
    """Merge per-episode and per-repo external labels into a single ground-truth dict.

    Args:
        self_play_qc: Rows from ``self_play_label_qc.json``; each entry is
            keyed by ``episode_key``.
        flatten_classification: Rows from ``flatten_classification.json``; each
            entry is keyed by ``repo_id``.

    Returns:
        ``{"episode_labels": {...}, "repo_labels": {...}}``. Entries that cannot
        be classified with confidence are omitted from both dicts.
    """
    episode_labels: dict[str, dict[str, str]] = {}
    for entry in self_play_qc:
        episode_key = entry.get("episode_key")
        if not episode_key:
            continue
        category = _classify_self_play_entry(entry)
        if category is None:
            continue
        episode_labels[episode_key] = {
            "category": category,
            "confidence": "high",
            "source": "self_play_qc",
        }

    repo_labels: dict[str, dict[str, str]] = {}
    for entry in flatten_classification:
        repo_id = entry.get("repo_id")
        if not repo_id:
            continue
        category = _classify_flatten_entry(entry)
        if category is None:
            continue
        repo_labels[repo_id] = {
            "category": category,
            "confidence": str(entry.get("confidence", "unknown")),
            "source": "flatten_classification",
        }

    return {"episode_labels": episode_labels, "repo_labels": repo_labels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-play-qc", type=Path, default=Path("test_results/data_audit/self_play_label_qc.json"))
    parser.add_argument(
        "--flatten-classification", type=Path, default=Path("test_results/data_audit/flatten_classification.json")
    )
    parser.add_argument("--output", type=Path, default=Path("test_results/data_audit/ground_truth_labels.json"))
    args = parser.parse_args()

    with open(args.self_play_qc) as f:
        self_play = json.load(f)
    with open(args.flatten_classification) as f:
        flatten = json.load(f)

    result = merge_labels(self_play, flatten)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"[merge] Episode labels: {len(result['episode_labels'])}")
    print(f"[merge] Repo labels: {len(result['repo_labels'])}")
    print(f"[merge] Written to {args.output}")


if __name__ == "__main__":
    main()
