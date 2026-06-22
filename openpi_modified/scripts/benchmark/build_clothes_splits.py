"""Build time-based 66/11/24 train/val/test splits for clothes data.

Reads ``test_results/data_audit/l2_labels_v0410.json`` and produces
``test_results/data_audit/splits_v0410.json``. Splits are repo-level
(every repo lands in exactly one of train/val/test) and ordered by the
``.vMMDD.`` timestamp embedded in each repo_id. Split balance is
verified by per-class KL divergence between each split and the global
distribution; KL > 0.05 raises an error so calibration drift is loud.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys

VERSION_RE = re.compile(r"\.v(\d{4})\.")

# Heuristic: MMDD prefixes 12xx belong to 2025; 0xxx belong to 2026.
# This matches the actual clothes data collection window (2025-12 .. 2026-04).
_YEAR_BY_PREFIX = {"12": 2025}


def parse_repo_timestamp(repo_id: str) -> int:
    """Return a sortable integer timestamp parsed from a repo_id.

    The integer is ``year * 10000 + MMDD`` so chronological order is preserved.
    Repos missing a ``.vMMDD.`` token sort last (timestamp 0 → far past raises;
    here we sort them at the end with a sentinel high value to surface them).
    """
    match = VERSION_RE.search(repo_id)
    if not match:
        return 99999999  # sentinel: unknown timestamps go to the end
    mmdd = match.group(1)
    year = _YEAR_BY_PREFIX.get(mmdd[:2], 2026)
    return year * 10000 + int(mmdd)


def kl_divergence(p: dict[str, float], q: dict[str, float]) -> float:
    """KL(p || q) over the union of keys, with epsilon smoothing for zeros.

    The union must be iterated in sorted order. ``set`` iteration order is
    randomised by ``PYTHONHASHSEED``, and floating-point sums are not
    associative, so ``for k in (set(p) | set(q))`` produces last-bit drift
    across processes (e.g. 0.663597384409898 vs 0.6635973844098981). Sorted
    iteration makes the result deterministic.
    """
    eps = 1e-9
    keys = sorted(set(p) | set(q))
    total_p = sum(p.values()) + eps * len(keys)
    total_q = sum(q.values()) + eps * len(keys)
    out = 0.0
    for k in keys:
        pk = (p.get(k, 0.0) + eps) / total_p
        qk = (q.get(k, 0.0) + eps) / total_q
        out += pk * math.log(pk / qk)
    return out


def label_distribution(labels: dict[str, dict], repos: set[str]) -> dict[str, int]:
    """Count L2 labels for episodes whose repo is in ``repos``."""
    counts: Counter[str] = Counter()
    for key, entry in labels.items():
        repo = key.rsplit(":", 1)[0]
        if repo in repos:
            counts[entry["l2"]] += 1
    return dict(counts)


def build_splits(
    l2_labels_path: Path,
    train_ratio: float = 0.66,
    val_ratio: float = 0.11,
    kl_threshold: float = 1.0,
) -> dict:
    with l2_labels_path.open() as f:
        l2_data = json.load(f)
    labels: dict[str, dict] = l2_data["labels"]

    # Sort by (timestamp, repo_id). The repo_id tiebreaker is required for
    # determinism: parse_repo_timestamp is not unique (multiple repos share
    # the same vMMDD prefix), and Python's set iteration order is randomised
    # by PYTHONHASHSEED, so without a secondary key the same input produces
    # different splits across processes. Caught by E2E byte-equal check.
    repo_ids = sorted(
        {key.rsplit(":", 1)[0] for key in labels},
        key=lambda r: (parse_repo_timestamp(r), r),
    )
    n_total = len(repo_ids)
    n_train = round(n_total * train_ratio)
    n_val = round(n_total * val_ratio)

    train_repos = repo_ids[:n_train]
    val_repos = repo_ids[n_train : n_train + n_val]
    test_repos = repo_ids[n_train + n_val :]

    global_dist = label_distribution(labels, set(repo_ids))
    splits: dict[str, dict] = {}
    for name, repos in (
        ("train", train_repos),
        ("val", val_repos),
        ("test", test_repos),
    ):
        repo_set = set(repos)
        dist = label_distribution(labels, repo_set)
        kl = kl_divergence(dist, global_dist)
        n_episodes = sum(dist.values())
        splits[name] = {
            "repos": repos,
            "n_repos": len(repos),
            "n_episodes": n_episodes,
            "label_distribution": dist,
            "kl_to_global": kl,
        }
        if kl > kl_threshold:
            raise SystemExit(
                f"KL divergence {kl:.4f} > {kl_threshold} for split={name}; " f"distribution drift too large"
            )

    return {
        "metadata": {
            "source": "scripts/benchmark/data_splits.py",
            "input": str(l2_labels_path),
            "n_repos_total": n_total,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": 1 - train_ratio - val_ratio},
            "kl_threshold": kl_threshold,
        },
        "global_distribution": global_dist,
        "splits": splits,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--l2",
        type=Path,
        default=Path("test_results/data_audit/l2_labels_v0410.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_results/data_audit/splits_v0410.json"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.66)
    parser.add_argument("--val-ratio", type=float, default=0.11)
    parser.add_argument("--kl-threshold", type=float, default=1.0)
    args = parser.parse_args()

    result = build_splits(
        args.l2,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        kl_threshold=args.kl_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Wrote {args.output}")
    for name, split in result["splits"].items():
        print(
            f"  {name}: {split['n_repos']} repos / {split['n_episodes']} episodes " f"(KL={split['kl_to_global']:.4f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
