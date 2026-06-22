#!/usr/bin/env python3

# ruff: noqa: E402

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.unit.config.test_top_level_config_loss_consistency import BASELINE_PATH
from tests.unit.config.test_top_level_config_loss_consistency import build_loss_baseline_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the saved top-level config loss baseline.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BASELINE_PATH,
        help="Where to write the generated JSON baseline.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_loss_baseline_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(
        "Wrote top-level config loss baseline:",
        args.output,
    )
    print(f"Saved {len(payload['configs'])} config baselines and {len(payload['expected_xfails'])} expected xfails.")


if __name__ == "__main__":
    main()
