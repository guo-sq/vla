#!/usr/bin/env python3
"""Audit all clothes-folding data repos across workspace and OSS paths.

Produces:
  - clothes_data_catalog.json: structured inventory of every repo
  - clothes_data_report.md: human-readable summary

Designed to be idempotent - re-run after new data uploads to get updated results.

Usage:
    PYTHONPATH=src:. python scripts/benchmark/data_auditor.py \
        --roots /path/to/workspace /path/to/oss \
        --output_dir test_results/data_audit
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
import json
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EpisodeLabel:
    episode_index: int
    success: bool | None = None
    role: str | None = None  # builder/destroyer/folder/disturber/unknown
    end_reason: str | None = None
    intervention_count: int = 0
    is_edge_case: bool = False
    edge_case_type: str | None = None  # "intervention", "exit_early", "failure"


@dataclass
class RepoAudit:
    repo_id: str
    paths: list[str] = field(default_factory=list)
    date: str = ""
    data_type: str = ""  # record/policy/r_policy/error/sfp/self_play/record_infer
    n_episodes: int = 0
    n_frames: int = 0
    schema_version: str = ""  # V1(7col)/V2(8col)/V3(9col)
    task_text: str = ""
    has_labels: bool = False
    label_summary: dict = field(default_factory=dict)
    episode_labels: list[dict] = field(default_factory=list)
    structural_issues: list[str] = field(default_factory=list)
    edge_case_flags: list[str] = field(default_factory=list)
    has_t5_embedding: bool = False


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\.v(\d{4})\.")


def extract_date(repo_id: str) -> str:
    m = _DATE_RE.search(repo_id)
    return f"v{m.group(1)}" if m else "unknown"


def classify_type(repo_id: str) -> str:
    if repo_id.startswith("self_play."):
        return "self_play"
    if repo_id.startswith("record_infer."):
        return "record_infer"
    if ".r.policy." in repo_id:
        return "r_policy"
    if ".error." in repo_id:
        return "error"
    if ".sfp." in repo_id:
        return "sfp"
    if ".policy." in repo_id or ".policy" in repo_id.split(".")[-1]:
        return "policy"
    return "record"


_ROLE_MAP = {"folder": "builder", "disturber": "destroyer"}


def normalize_role(role: str | None) -> str | None:
    if role is None:
        return None
    return _ROLE_MAP.get(role, role)


def quadrant(role: str | None, success: bool | None) -> str:
    if role == "builder" and success is True:
        return "TP"
    if role == "destroyer" and success is True:
        return "TN"
    if role == "builder" and success is False:
        return "FP"
    if role == "destroyer" and success is False:
        return "FN"
    if success is True:
        return "success_unknown_role"
    if success is False:
        return "failure_unknown_role"
    return "unlabeled"


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


def detect_schema(repo_path: Path) -> str:
    """Detect parquet schema version by reading first parquet file."""
    chunk_dir = repo_path / "data" / "chunk-000"
    if not chunk_dir.is_dir():
        return "no_data"
    parquets = sorted(chunk_dir.glob("*.parquet"))
    if not parquets:
        return "no_parquet"
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(str(parquets[0]))
        cols = set(schema.names)
        if "is_human_intervention" in cols and "sub_task_index" in cols:
            return "V3"
        if "is_human_intervention" in cols:
            return "V2"
        return "V1"
    except Exception:
        return "read_error"


# ---------------------------------------------------------------------------
# Repo scanning
# ---------------------------------------------------------------------------


def scan_repo(repo_id: str, repo_path: Path) -> RepoAudit:
    audit = RepoAudit(repo_id=repo_id)
    audit.date = extract_date(repo_id)
    audit.data_type = classify_type(repo_id)

    # Structural checks
    meta_dir = repo_path / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"

    if not meta_dir.exists():
        audit.structural_issues.append("missing_meta_dir")
        return audit
    if not info_path.exists():
        audit.structural_issues.append("missing_info_json")
    if not episodes_path.exists():
        audit.structural_issues.append("missing_episodes_jsonl")
        return audit

    # Read info.json
    if info_path.exists():
        try:
            with open(info_path) as f:
                info = json.load(f)
            audit.n_episodes = info.get("total_episodes", 0)
            audit.n_frames = info.get("total_frames", 0)
        except Exception:
            audit.structural_issues.append("info_json_parse_error")

    # Read episodes.jsonl
    episodes = []
    try:
        with open(episodes_path) as f:
            for raw_line in f:
                line = raw_line.strip()
                if line:
                    episodes.append(json.loads(line))
    except Exception:
        audit.structural_issues.append("episodes_jsonl_parse_error")
        return audit

    if not episodes:
        audit.structural_issues.append("empty_episodes_jsonl")
        return audit

    # Correct episode count from jsonl if info.json was missing
    if audit.n_episodes == 0:
        audit.n_episodes = len(episodes)

    # Task text
    tasks = episodes[0].get("tasks", [])
    audit.task_text = tasks[0][:100] if tasks else ""

    # T5 embedding
    audit.has_t5_embedding = "t5_embedding_path" in episodes[0]

    # Labels
    first_ep = episodes[0]
    has_success = "success" in first_ep
    has_role = "role" in first_ep
    audit.has_labels = has_success or has_role

    # Parse episode labels
    label_counts: dict[str, int] = defaultdict(int)
    for ep in episodes:
        role_raw = ep.get("role")
        role = normalize_role(role_raw)
        success = ep.get("success")
        end_reason = ep.get("end_reason", "")
        intervention = ep.get("intervention_count", 0) or 0

        q = quadrant(role, success)
        label_counts[q] += 1

        # Edge case detection
        is_edge = False
        edge_type = None
        if intervention > 0:
            is_edge = True
            edge_type = "intervention"
            audit.edge_case_flags.append(f"ep{ep['episode_index']}:intervention={intervention}")
        if end_reason == "exit_early":
            is_edge = True
            edge_type = "exit_early"
            audit.edge_case_flags.append(f"ep{ep['episode_index']}:exit_early")
        if success is False:
            is_edge = True
            edge_type = edge_type or "failure"
            audit.edge_case_flags.append(f"ep{ep['episode_index']}:failure(role={role or 'unknown'})")

        el = EpisodeLabel(
            episode_index=ep["episode_index"],
            success=success,
            role=role,
            end_reason=end_reason or None,
            intervention_count=intervention,
            is_edge_case=is_edge,
            edge_case_type=edge_type,
        )
        audit.episode_labels.append(asdict(el))

    audit.label_summary = dict(label_counts)

    # Schema
    audit.schema_version = detect_schema(repo_path)

    return audit


def scan_roots(roots: list[str]) -> dict[str, RepoAudit]:
    """Scan multiple root directories, merging repos that appear in multiple roots."""
    all_audits: dict[str, RepoAudit] = {}

    for root_str in roots:
        root = Path(root_str)
        if not root.is_dir():
            print(f"WARNING: root {root_str} not a directory, skipping", flush=True)
            continue

        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            repo_id = entry.name
            # Skip nested heyuan1993 directory
            if repo_id == "heyuan1993":
                continue

            if repo_id in all_audits:
                # Already scanned from another root, just add path
                all_audits[repo_id].paths.append(root_str)
                continue

            print(f" Scanning {repo_id}...", end="\r", flush=True)
            audit = scan_repo(repo_id, entry)
            audit.paths.append(root_str)
            all_audits[repo_id] = audit

    print(f" Scanned {len(all_audits)} unique repos. ", flush=True)
    return all_audits


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(audits: dict[str, RepoAudit]) -> str:
    lines = ["# 叠衣服数据审计报告\n"]
    lines.append(f"扫描时间: {__import__('datetime').datetime.now().isoformat()}\n")

    # Summary
    total_repos = len(audits)
    total_eps = sum(a.n_episodes for a in audits.values())
    total_frames = sum(a.n_frames for a in audits.values())
    labeled = sum(1 for a in audits.values() if a.has_labels)
    broken = sum(1 for a in audits.values() if a.structural_issues)

    lines.append("## 1. 总量\n")
    lines.append("| 维度 | 数值 |")
    lines.append("|------|------|")
    lines.append(f"| Repos | {total_repos} |")
    lines.append(f"| Episodes | {total_eps:,} |")
    lines.append(f"| Frames | {total_frames:,} |")
    lines.append(f"| 有标签 repos | {labeled} ({labeled/total_repos*100:.1f}%) |")
    lines.append(f"| 结构异常 repos | {broken} |")
    lines.append("")

    # By type
    type_stats: dict[str, dict] = defaultdict(lambda: {"repos": 0, "eps": 0, "frames": 0})
    for a in audits.values():
        t = a.data_type
        type_stats[t]["repos"] += 1
        type_stats[t]["eps"] += a.n_episodes
        type_stats[t]["frames"] += a.n_frames

    lines.append("## 2. 按类型分布\n")
    lines.append("| 类型 | repos | episodes | frames |")
    lines.append("|------|-------|----------|--------|")
    for t in ["record", "policy", "r_policy", "error", "sfp", "self_play", "record_infer"]:
        s = type_stats.get(t, {"repos": 0, "eps": 0, "frames": 0})
        if s["repos"] > 0:
            lines.append(f"| {t} | {s['repos']} | {s['eps']:,} | {s['frames']:,} |")
    lines.append("")

    # Schema distribution
    schema_counts: dict[str, int] = defaultdict(int)
    for a in audits.values():
        schema_counts[a.schema_version] += 1

    lines.append("## 3. Schema 版本\n")
    lines.append("| Schema | repos |")
    lines.append("|--------|-------|")
    lines.extend(f"| {sv} | {schema_counts[sv]} |" for sv in sorted(schema_counts.keys()))
    lines.append("")

    # Task distribution
    task_counts: dict[str, int] = defaultdict(int)
    for a in audits.values():
        short = a.task_text[:60] + "..." if len(a.task_text) > 60 else a.task_text
        task_counts[short] += 1

    lines.append("## 4. Task 描述分布\n")
    lines.append("| Task | repos |")
    lines.append("|------|-------|")
    for task, count in sorted(task_counts.items(), key=lambda x: -x[1]):
        if count > 0:
            lines.append(f"| {task} | {count} |")
    lines.append("")

    # Labeled data quadrants
    global_quads: dict[str, int] = defaultdict(int)
    for a in audits.values():
        for q, n in a.label_summary.items():
            global_quads[q] += n

    if any(v > 0 for v in global_quads.values()):
        lines.append("## 5. 标签数据象限分布\n")
        lines.append("| 象限 | episodes |")
        lines.append("|------|----------|")
        lines.extend(
            f"| {q} | {global_quads[q]} |"
            for q in ["TP", "TN", "FP", "FN", "success_unknown_role", "failure_unknown_role", "unlabeled"]
            if global_quads.get(q, 0) > 0
        )
        lines.append("")

    # Edge cases
    all_edges = [f"{a.repo_id}:{flag}" for a in audits.values() for flag in a.edge_case_flags]

    if all_edges:
        lines.append("## 6. 边界 Case\n")
        lines.extend(f"- {edge}" for edge in sorted(all_edges))
        lines.append("")

    # Structural issues
    broken_repos = [(a.repo_id, a.structural_issues) for a in audits.values() if a.structural_issues]
    if broken_repos:
        lines.append("## 7. 结构异常 Repos\n")
        for repo_id, issues in sorted(broken_repos):
            lines.append(f"- **{repo_id}**: {', '.join(issues)}")
        lines.append("")

    # Path coverage
    both = sum(1 for a in audits.values() if len(a.paths) > 1)
    ws_only = sum(1 for a in audits.values() if len(a.paths) == 1 and "workspace" in a.paths[0])
    oss_only = sum(1 for a in audits.values() if len(a.paths) == 1 and "oss" in a.paths[0])

    lines.append("## 8. 路径覆盖\n")
    lines.append("| 位置 | repos |")
    lines.append("|------|-------|")
    lines.append(f"| 仅 workspace | {ws_only} |")
    lines.append(f"| 仅 OSS | {oss_only} |")
    lines.append(f"| 两者都有 | {both} |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Audit clothes-folding data repos.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="Data root directories to scan",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for catalog JSON and report MD",
    )
    args = parser.parse_args()

    print(f"Scanning {len(args.roots)} root(s):", flush=True)
    for r in args.roots:
        print(f" {r}", flush=True)
    print(flush=True)

    audits = scan_roots(args.roots)

    # Write JSON catalog
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    catalog = {repo_id: asdict(audit) for repo_id, audit in sorted(audits.items())}
    catalog_path = out / "clothes_data_catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nCatalog written to {catalog_path}", flush=True)

    # Write report
    report = generate_report(audits)
    report_path = out / "clothes_data_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report written to {report_path}", flush=True)

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(
        f"Total: {len(audits)} repos, "
        f"{sum(a.n_episodes for a in audits.values()):,} episodes, "
        f"{sum(a.n_frames for a in audits.values()):,} frames",
        flush=True,
    )
    print(f"Labeled: {sum(1 for a in audits.values() if a.has_labels)} repos", flush=True)
    print(f"Issues: {sum(1 for a in audits.values() if a.structural_issues)} repos", flush=True)
    print(f"Edge cases: {sum(len(a.edge_case_flags) for a in audits.values())} episodes", flush=True)


if __name__ == "__main__":
    main()
