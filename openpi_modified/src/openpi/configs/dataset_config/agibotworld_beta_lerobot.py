from pathlib import Path


ROOT_DIR = "/mnt/"
SUB_PATH = "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta"

_dataset_root = Path(ROOT_DIR) / SUB_PATH
REPO_ID = [
    f"{SUB_PATH}/{task_dir.name}"
    for task_dir in sorted(_dataset_root.iterdir())
    if task_dir.is_dir() and (task_dir / "meta" / "info.json").exists()
]

if not REPO_ID:
    raise ValueError(f"No valid tasks found under {_dataset_root}")

SAMPLE_1M_REPO_ID = [
    "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_503",
    # "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_709",
    "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_722",
    # "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_707",
    "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_708",
    # "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_725",
    "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_786",
    # "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_698",
    "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_773",
    # "oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_621",
]

DEBUG_REPO_ID = ["oss_data/agibot_world/agibot_world_beta_lerobot/agibotworld_beta/task_351"]
