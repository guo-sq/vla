from pathlib import Path


ROOT_DIR = "/mnt/"
SUB_PATH = "oss_data/agibot_world/AgiBotWorld2026"

_dataset_root = Path(ROOT_DIR) / SUB_PATH
REPO_ID = [
    str(info_path.parent.parent.relative_to(_dataset_root))
    for info_path in sorted(_dataset_root.glob("**/data/meta/info.json"))
]

if not REPO_ID:
    raise ValueError(f"No valid tasks found under {_dataset_root}")

DEBUG_REPO_ID = ["ImitationLearning/CommercialSpaces/task_3400/313498_314085/data"]

SAMPLE_1M_REPO_ID = [
    "ImitationLearning/CommercialSpaces/task_3400/346206_347749/data",
    "ImitationLearning/CommercialSpaces/task_3401/352507_353983/data",
    "ImitationLearning/CommercialSpaces/task_3401/357385_358520/data",
    "ImitationLearning/CommercialSpaces/task_3402/380892_384933/data",
    "ImitationLearning/CommercialSpaces/task_3404/305627_321730/data",
]
