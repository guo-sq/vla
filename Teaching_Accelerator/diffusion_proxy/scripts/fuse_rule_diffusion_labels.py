#!/usr/bin/env python3
"""Fuse rule-based and diffusion proxy labels."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from diffusion_proxy.fusion import fuse_records
from diffusion_proxy.sidecar import read_jsonl
from diffusion_proxy.sidecar import write_jsonl
from diffusion_proxy.sidecar import write_summary


LOGGER = logging.getLogger("fuse_rule_diffusion_labels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-labels", type=Path, required=True)
    parser.add_argument("--diffusion-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--rule-weight", type=float, default=0.50)
    parser.add_argument("--diffusion-weight", type=float, default=0.30)
    parser.add_argument("--event-weight", type=float, default=0.20)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")
    rule_records = read_jsonl(args.rule_labels)
    diffusion_records = read_jsonl(args.diffusion_labels)
    records, summary = fuse_records(
        rule_records,
        diffusion_records,
        rule_weight=args.rule_weight,
        diffusion_weight=args.diffusion_weight,
        event_weight=args.event_weight,
    )
    summary = {
        **summary,
        "rule_labels": str(args.rule_labels),
        "diffusion_labels": str(args.diffusion_labels),
        "output": str(args.output),
    }
    write_jsonl(records, args.output, kind="fusion")
    summary_path = args.summary or args.output.with_name("diffusion_fused_summary.json")
    write_summary(summary, summary_path)
    LOGGER.info("Wrote fused labels to %s", args.output)
    LOGGER.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
