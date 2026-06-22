from openpi.configs.base import *
from openpi.configs.robot_cfg.base import *
import openpi.training.utils as _utils

"""
本地 AgiBotWorld-Alpha LeRobot 数据集。

- 数据根目录: /mnt/oss_data/AI-ModelScope/AgiBotWorld-Alpha_lerobot/agibotworld
- 有效 task 数: 35
- task_422 仅有 videos，缺少 meta/info.json，已在 dataset_config 中自动跳过
"""

DEBUG_MODE = False

TASK_NAME = "cfg_pi0.5_28_dim.agibotworld_alpha_lerobot.sample1m"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "260408"

# Dataset settings
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = True
DELTA_ACTION_MASK_INDICES = [7, -1, 7, -1, 12]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

from openpi.configs.dataset_config.agibotworld_alpha_lerobot import SAMPLE_1M_REPO_ID as agibotworld_alpha_lerobot
from openpi.configs.dataset_config.agibotworld_alpha_lerobot import ROOT_DIR

REPO_ID = agibotworld_alpha_lerobot

ROBOT_ALIGN_INFO = _utils.ROBOT_ALIGN_INFO
# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 16
TRAIN_STEPS = 50000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"
LOG_INTERVAL = 50
SAVE_INTERVAL = 5000
KEEP_PERIOD = 10000

if DEBUG_MODE:
    BATCH_SIZE = 1
    NUM_WORKERS = 0
    TRAIN_STEPS = 2
    LOG_INTERVAL = 1
    SAVE_INTERVAL = 1
    KEEP_PERIOD = 1
    REPO_ID = ["agibotworld/task_352"]
    ASSET_ID = "agibotworld_alpha_lerobot_debug"

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        pi05=True,
        max_token_len=256,
        enable_rl_value_head=False,
        # image_keys=[
        #     "base_0_rgb",
        #     "left_wrist_0_rgb",
        #     "right_wrist_0_rgb",
        #     "third_view_0_rgb",
        # ],
        use_joint_eef_mask=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        max_token_len=256,
        enable_rl_value_head=False,
        use_joint_eef_mask=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    # Turn off EMA for LoRA finetuning.
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
        use_semantic_delta_actions=False,
        delta_wrap_eef_angles=False,
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
    log_interval=LOG_INTERVAL,
    # How often (in steps) to save checkpoints.
    save_interval=SAVE_INTERVAL,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=KEEP_PERIOD,
    overwrite=True,
)
