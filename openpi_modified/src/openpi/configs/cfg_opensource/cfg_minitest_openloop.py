"""Minitest configuration for fast openloop evaluation in CI.

This config is fully self-contained and does not depend on OSS paths.
All dataset paths are local to /mnt/workspace/openpi_minitest/

IMPORTANT: The config name must match the checkpoint directory structure.
"""

from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *

DEBUG_MODE = False

# Use the same name as the original checkpoint to ensure correct path resolution
TASK_NAME = "cfg_pi0.5_28_dim.all_public_datasets.with_updated_robomind2"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260307"

# Dataset settings - use local minitest data
ROOT_DIR = "/mnt/workspace/openpi_minitest/"

# Explicitly define REPO_ID list (no OSS access needed)
REPO_ID = [
    "robomind_2/agilex/fold_clothes/success_episodes",
]

ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = True
DELTA_ACTION_MASK_INDICES = [7, -1, 7, -1, 12]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
# Match the TARGET_ACTION_DIM from the original robomind_2 training config
TARGET_ACTION_DIM = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

ROBOT_ALIGN_INFO = _utils.ROBOT_ALIGN_INFO

# TrainConfig - minimal settings for evaluation only
BATCH_SIZE = 16
NUM_WORKERS = 4
TRAIN_STEPS = 0  # Evaluation only
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        pi05=True,
        max_token_len=256,
        enable_rl_value_head=False,
        use_joint_eef_mask=True,
        # Use non-LoRA variants since the checkpoint has full weights (merged LoRA or full fine-tuning)
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        max_token_len=256,
        enable_rl_value_head=False,
        use_joint_eef_mask=True,
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
    ).get_freeze_filter(),
    ema_decay=None,
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=True,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
        ),
        extra_delta_transform=False,
        use_delta_joint_actions=True,
        delta_action_mask_indices=DELTA_ACTION_MASK_INDICES,
        public_dataset_camera_map=_utils.PUBLIC_DATASET_MAP,
        align_dim=ALIGN_DIM,
        unify_action_space=UNIFY_ACTION_SPACE,
        target_action_dim=TARGET_ACTION_DIM,
        robot_align_info=RobotAlignInfo(robot_align_info=ROBOT_ALIGN_INFO),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_WEIGHT_PATH),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=100,
    save_interval=10000,
    keep_period=10000,
    overwrite=True,
)
