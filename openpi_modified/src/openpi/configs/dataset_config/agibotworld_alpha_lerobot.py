from pathlib import Path


ROOT_DIR = "/mnt/"
SUB_PATH = "oss_data/AI-ModelScope/AgiBotWorld-Alpha_lerobot/agibotworld"

_dataset_root = Path(ROOT_DIR) / SUB_PATH
REPO_ID = [
    f"{SUB_PATH}/{task_dir.name}"
    for task_dir in sorted(_dataset_root.iterdir())
    if task_dir.is_dir() and (task_dir / "meta" / "info.json").exists()
]

if not REPO_ID:
    raise ValueError(f"No valid tasks found under {_dataset_root}")

SAMPLE_1M_REPO_ID = REPO_ID[::5]
print(f"Total tasks: {len(REPO_ID)}, Sample 1M tasks: {len(SAMPLE_1M_REPO_ID)}")
