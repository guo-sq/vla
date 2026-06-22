#!/usr/bin/env python3

# ruff: noqa: E402

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.unit.config.test_top_level_config_loss_consistency import TOP_LEVEL_CONFIGS
from tests.unit.config.test_top_level_config_loss_consistency import _load_loss_baseline
from tests.unit.config.test_top_level_config_loss_consistency import collect_top_level_config_loss_drift


def _progress(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    print(f"Starting top-level config loss drift check for {len(TOP_LEVEL_CONFIGS)} config(s).", flush=True)
    drift_messages = collect_top_level_config_loss_drift(progress_logger=_progress)
    if not drift_messages:
        baseline = _load_loss_baseline()
        print(
            f"Validated {len(baseline['configs'])} saved top-level config baselines; "
            f"{len(baseline['expected_xfails'])} config(s) remain expected xfail; "
            f"{len(TOP_LEVEL_CONFIGS)} top-level config(s) covered.",
            flush=True,
        )
        print("No top-level config loss drift detected.", flush=True)
        return 0

    print("Detected top-level config loss drift:", flush=True)
    for message in drift_messages:
        print(f"- {message}", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
