#!/usr/bin/env python3
"""Rule-based classifier for flatten vs fold clothes-folding data repos.

Classifies each repo by combining English catalog task_text with Chinese
feishu_task labels. Produces a JSON classification file and a Markdown
viability report.

Usage:
    PYTHONPATH=src:. python scripts/benchmark/flatten_classifier.py \
        --catalog test_results/data_audit/clothes_data_catalog.json \
        --feishu test_results/data_audit/feishu_labels.json \
        --output_dir test_results/data_audit
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class FlattenLabel(str, Enum):
    FLATTEN = "flatten"
    FOLD = "fold"
    BIMODAL = "bimodal"
    DISARRANGE = "disarrange"
    NON_TASK = "non_task"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RepoClassification:
    repo_id: str
    task_text: str
    feishu_task: str
    feishu_annotation: str
    rule_classification: str  # FlattenLabel value
    feishu_label: str
    final_label: str  # FlattenLabel value
    confidence: str  # Confidence value
    reason: str
    n_episodes: int
    has_conflict: bool
    conflict_detail: str
    in_catalog: bool
    in_feishu: bool
    matched_rule: int


# ---------------------------------------------------------------------------
# Non-task feishu labels (Rule 9)
# ---------------------------------------------------------------------------

_NON_TASK_LABELS: frozenset[str] = frozenset(
    {
        "model_test",
        "reset_arm",
        "intervention_correction",
        "failure",
        "format_error",
        "low_quality",
    }
)


# ---------------------------------------------------------------------------
# Rule classifier
# ---------------------------------------------------------------------------


def rule_classify(
    task_text: str,
    feishu_task: str,
) -> tuple[FlattenLabel, Confidence, str, int]:
    """Apply classification rules in priority order.

    Returns:
        (label, confidence, reason, matched_rule_number)
    """
    # Rule 1: English "Straighten" -> flatten
    if re.search(r"(?i)\bstraighten\b", task_text):
        return (
            FlattenLabel.FLATTEN,
            Confidence.MEDIUM,
            "English task contains 'Straighten'",
            1,
        )

    # Rule 2: English "Lay...flat" -> flatten
    if re.search(r"(?i)\blay\b.*\bflat\b", task_text):
        return (
            FlattenLabel.FLATTEN,
            Confidence.MEDIUM,
            "English task contains 'Lay...flat'",
            2,
        )

    # Rule 3: English "disarrange" -> disarrange
    if re.search(r"(?i)\bdisarrange\b", task_text):
        return (
            FlattenLabel.DISARRANGE,
            Confidence.HIGH,
            "English task contains 'disarrange'",
            3,
        )

    # Rule 4: feishu_task contains "铺平或叠好" -> bimodal
    if "铺平或叠好" in feishu_task:
        return (
            FlattenLabel.BIMODAL,
            Confidence.LOW,
            "feishu_task contains '铺平或叠好' (either flatten OR fold)",
            4,
        )

    # Rule 5: feishu_task contains "铺平情况下叠好" or "斜铺平情况下叠好" -> fold
    if "铺平情况下叠好" in feishu_task or "斜铺平情况下叠好" in feishu_task:
        return (
            FlattenLabel.FOLD,
            Confidence.HIGH,
            "feishu_task indicates flatten is prerequisite, goal is fold",
            5,
        )

    # Rule 6: feishu_task contains "到铺平" -> fold (milestone)
    if "到铺平" in feishu_task:
        return (
            FlattenLabel.FOLD,
            Confidence.MEDIUM,
            "feishu_task uses '到铺平' as milestone, not goal",
            6,
        )

    # Rule 7: feishu_task contains "非铺平" -> fold
    if "非铺平" in feishu_task:
        return (
            FlattenLabel.FOLD,
            Confidence.LOW,
            "feishu_task contains '非铺平' (not flat = starting condition)",
            7,
        )

    # Rule 8: feishu_task contains "铺平" AND NOT "叠"/"或"/"非铺平"/"到铺平"
    if "铺平" in feishu_task:
        excluders = ("叠", "或", "非铺平", "到铺平")
        if not any(ex in feishu_task for ex in excluders):
            return (
                FlattenLabel.FLATTEN,
                Confidence.HIGH,
                "feishu_task contains '铺平' with no fold/bimodal qualifiers",
                8,
            )

    # Rule 10: Default -> fold (rule 9 is handled at the cross_validate stage)
    return (
        FlattenLabel.FOLD,
        Confidence.HIGH,
        "Default: no flatten keywords found",
        10,
    )


# ---------------------------------------------------------------------------
# Confidence manipulation
# ---------------------------------------------------------------------------

_CONFIDENCE_ORDER: list[Confidence] = [Confidence.LOW, Confidence.MEDIUM, Confidence.HIGH]


def _lower_confidence(conf: Confidence) -> Confidence:
    idx = _CONFIDENCE_ORDER.index(conf)
    return _CONFIDENCE_ORDER[max(0, idx - 1)]


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------


def cross_validate(
    rule_label: FlattenLabel,
    rule_confidence: Confidence,
    feishu_label: str,
    feishu_annotation: str,
) -> tuple[FlattenLabel, Confidence, bool, str]:
    """Cross-validate rule classification against feishu label.

    Returns:
        (final_label, final_confidence, has_conflict, conflict_detail)
    """
    # No feishu data -> use rule, lower confidence
    if not feishu_label:
        return (
            rule_label,
            _lower_confidence(rule_confidence),
            False,
            "No feishu data - confidence lowered",
        )

    # Non-task labels pass through
    if rule_label == FlattenLabel.NON_TASK:
        return rule_label, rule_confidence, False, ""

    # Disarrange pass through
    if rule_label == FlattenLabel.DISARRANGE:
        return rule_label, rule_confidence, False, ""

    # Bimodal + fold is expected (59 repos)
    if rule_label == FlattenLabel.BIMODAL and feishu_label == "fold":
        return rule_label, rule_confidence, False, ""

    # Bimodal + flatten: no conflict
    if rule_label == FlattenLabel.BIMODAL and feishu_label == "flatten":
        return rule_label, rule_confidence, False, ""

    # Rule flatten + feishu flatten: agreement
    if rule_label == FlattenLabel.FLATTEN and feishu_label == "flatten":
        return rule_label, rule_confidence, False, ""

    # Rule fold + feishu fold: agreement
    if rule_label == FlattenLabel.FOLD and feishu_label == "fold":
        return rule_label, rule_confidence, False, ""

    # Rule flatten + feishu fold: conflict
    if rule_label == FlattenLabel.FLATTEN and feishu_label == "fold":
        return (
            rule_label,
            _lower_confidence(rule_confidence),
            True,
            "Rule says flatten but feishu says fold",
        )

    # Rule fold + feishu flatten: conflict, trust feishu if annotation="衣服铺平"
    if rule_label == FlattenLabel.FOLD and feishu_label == "flatten":
        if "衣服铺平" in feishu_annotation or "铺平" in feishu_annotation:
            return (
                FlattenLabel.FLATTEN,
                Confidence.MEDIUM,
                True,
                "Rule says fold but feishu says flatten (annotation confirms flatten)",
            )
        return (
            rule_label,
            _lower_confidence(rule_confidence),
            True,
            "Rule says fold but feishu says flatten (no confirming annotation)",
        )

    # Any other mismatch
    if rule_label.value != feishu_label:
        return (
            rule_label,
            _lower_confidence(rule_confidence),
            True,
            f"Rule says {rule_label.value} but feishu says {feishu_label}",
        )

    return rule_label, rule_confidence, False, ""


# ---------------------------------------------------------------------------
# Weird-key filter
# ---------------------------------------------------------------------------

_WEIRD_KEY_RE = re.compile(r"[\u4e00-\u9fff]")  # contains Chinese characters


def _is_weird_feishu_key(key: str) -> bool:
    """Return True if the key looks like a note rather than a real repo_id."""
    return bool(_WEIRD_KEY_RE.search(key))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def classify_repos(
    catalog_path: Path,
    feishu_path: Path,
) -> list[RepoClassification]:
    """Run full classification pipeline on catalog + feishu data.

    Returns list of RepoClassification, one per unique repo_id.
    """
    catalog_raw: dict[str, dict] = json.loads(catalog_path.read_text())
    feishu_raw: dict[str, dict] = json.loads(feishu_path.read_text())

    # Collect all repo_ids (union of both sources, excluding weird keys)
    all_repo_ids: set[str] = set(catalog_raw.keys())
    for key in feishu_raw:
        if not _is_weird_feishu_key(key):
            all_repo_ids.add(key)

    results: list[RepoClassification] = []

    for repo_id in sorted(all_repo_ids):
        cat = catalog_raw.get(repo_id, {})
        fei = feishu_raw.get(repo_id, {})

        in_catalog = repo_id in catalog_raw
        in_feishu = repo_id in feishu_raw and not _is_weird_feishu_key(repo_id)

        task_text = cat.get("task_text", "")
        feishu_task = fei.get("feishu_task", "")
        feishu_label_raw = fei.get("feishu_label", "")
        feishu_annotation = fei.get("feishu_annotation", "")
        n_episodes = cat.get("n_episodes", 0)

        # Rule 9 check: non-task feishu labels
        if feishu_label_raw in _NON_TASK_LABELS:
            rule_label = FlattenLabel.NON_TASK
            rule_conf = Confidence.HIGH
            reason = f"feishu_label is '{feishu_label_raw}' (non-task category)"
            matched_rule = 9
        else:
            rule_label, rule_conf, reason, matched_rule = rule_classify(task_text, feishu_task)

        final_label, final_conf, has_conflict, conflict_detail = cross_validate(
            rule_label, rule_conf, feishu_label_raw, feishu_annotation
        )

        results.append(
            RepoClassification(
                repo_id=repo_id,
                task_text=task_text,
                feishu_task=feishu_task,
                feishu_annotation=feishu_annotation,
                rule_classification=rule_label.value,
                feishu_label=feishu_label_raw,
                final_label=final_label.value,
                confidence=final_conf.value,
                reason=reason,
                n_episodes=n_episodes,
                has_conflict=has_conflict,
                conflict_detail=conflict_detail,
                in_catalog=in_catalog,
                in_feishu=in_feishu,
                matched_rule=matched_rule,
            )
        )

    return results


# ---------------------------------------------------------------------------
# Viability assessment
# ---------------------------------------------------------------------------


def compute_viability(results: list[RepoClassification]) -> dict:
    """Compute viability stats for flatten data extraction."""
    confirmed_flatten = [r for r in results if r.final_label == FlattenLabel.FLATTEN and r.in_catalog]
    bimodal = [r for r in results if r.final_label == FlattenLabel.BIMODAL and r.in_catalog]

    confirmed_eps = sum(r.n_episodes for r in confirmed_flatten)
    bimodal_eps = sum(r.n_episodes for r in bimodal)

    return {
        "confirmed_flatten_repos": len(confirmed_flatten),
        "confirmed_flatten_episodes": confirmed_eps,
        "bimodal_repos": len(bimodal),
        "bimodal_episodes": bimodal_eps,
        "min_flatten_tp_estimate": confirmed_eps,
        "max_flatten_tp_estimate": confirmed_eps + bimodal_eps,
        "viable": confirmed_eps >= 30,
        "verdict": "VIABLE" if confirmed_eps >= 30 else "NEEDS_MORE_DATA",
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def _generate_report(
    results: list[RepoClassification],
    viability: dict,
) -> str:
    """Generate Markdown report."""
    lines: list[str] = []

    # --- Summary ---
    lines.append("# Flatten Classifier Report\n")
    lines.append("## Summary\n")

    label_counts: Counter[str] = Counter()
    for r in results:
        label_counts[r.final_label] += 1

    lines.append("| Label | Repos |")
    lines.append("|-------|-------|")
    lines.extend(f"| {label} | {label_counts[label]} |" for label in sorted(label_counts.keys()))
    lines.append(f"| **Total** | **{len(results)}** |")
    lines.append("")

    # --- Confidence breakdown ---
    lines.append("## Confidence Breakdown\n")

    conf_matrix: dict[str, Counter[str]] = defaultdict(Counter)
    for r in results:
        conf_matrix[r.final_label][r.confidence] += 1

    lines.append("| Label | High | Medium | Low |")
    lines.append("|-------|------|--------|-----|")
    lines.extend(
        f"| {label} | {conf_matrix[label].get('high', 0)} | "
        f"{conf_matrix[label].get('medium', 0)} | {conf_matrix[label].get('low', 0)} |"
        for label in sorted(conf_matrix.keys())
    )
    lines.append("")

    # --- Conflict report ---
    lines.append("## Conflict Report\n")

    conflicts = [r for r in results if r.has_conflict]
    if conflicts:
        lines.append(f"Found {len(conflicts)} repos with rule/feishu conflicts:\n")
        lines.append("| Repo | Rule | Feishu | Final | Detail |")
        lines.append("|------|------|--------|-------|--------|")
        lines.extend(
            f"| {r.repo_id} | {r.rule_classification} | {r.feishu_label} " f"| {r.final_label} | {r.conflict_detail} |"
            for r in conflicts
        )
    else:
        lines.append("No conflicts found.")
    lines.append("")

    # --- Viability ---
    lines.append("## Flatten Viability Assessment\n")
    lines.append(f"- Confirmed flatten repos: **{viability['confirmed_flatten_repos']}**")
    lines.append(f"- Confirmed flatten episodes: **{viability['confirmed_flatten_episodes']}**")
    lines.append(f"- Bimodal repos: **{viability['bimodal_repos']}**")
    lines.append(f"- Bimodal episodes: **{viability['bimodal_episodes']}**")
    lines.append(f"- Min flatten TP estimate: **{viability['min_flatten_tp_estimate']}**")
    lines.append(f"- Max flatten TP estimate: **{viability['max_flatten_tp_estimate']}**")
    lines.append(f"- Verdict: **{viability['verdict']}**")
    lines.append("")

    # --- Bimodal repos needing resolution ---
    bimodal = [r for r in results if r.final_label == FlattenLabel.BIMODAL]
    if bimodal:
        lines.append("## Repos Needing Value Model Resolution\n")
        lines.append("These bimodal repos may contain flatten episodes:\n")
        lines.append("| Repo | Episodes | feishu_task |")
        lines.append("|------|----------|-------------|")
        lines.extend(f"| {r.repo_id} | {r.n_episodes} | {r.feishu_task} |" for r in bimodal)
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Classify clothes-folding repos as flatten/fold.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("test_results/data_audit/clothes_data_catalog.json"),
        help="Path to catalog JSON.",
    )
    parser.add_argument(
        "--feishu",
        type=Path,
        default=Path("test_results/data_audit/feishu_labels.json"),
        help="Path to feishu labels JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("test_results/data_audit"),
        help="Output directory.",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = classify_repos(args.catalog, args.feishu)
    viability = compute_viability(results)

    # Write classification JSON
    classification_path = output_dir / "flatten_classification.json"
    classification_path.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2))

    # Write report
    report_path = output_dir / "flatten_report.md"
    report_path.write_text(_generate_report(results, viability))

    # Print summary to stdout
    label_counts: Counter[str] = Counter()
    for r in results:
        label_counts[r.final_label] += 1

    print("=== Flatten Classifier Summary ===")
    print(f"Total repos classified: {len(results)}")
    for label in sorted(label_counts.keys()):
        print(f" {label}: {label_counts[label]}")
    print()
    print(
        f"Confirmed flatten: {viability['confirmed_flatten_repos']} repos, "
        f"{viability['confirmed_flatten_episodes']} episodes"
    )
    print(
        f"Bimodal (needs resolution): {viability['bimodal_repos']} repos, " f"{viability['bimodal_episodes']} episodes"
    )
    print(f"Verdict: {viability['verdict']}")
    print()
    conflicts = [r for r in results if r.has_conflict]
    if conflicts:
        print(f"Conflicts: {len(conflicts)} repos")
    print(f"Output: {classification_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
