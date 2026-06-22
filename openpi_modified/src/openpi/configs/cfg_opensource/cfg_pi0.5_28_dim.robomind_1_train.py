from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *

"""
按机器人类型详细统计:
  ALOHA:
    - 数据集数: 99
    - Frame数: 9,336,033
    - 时长: 86.44 小时
  h5_franka_2rgb:
    - 数据集数: 90
    - Frame数: 1,753,203
    - 时长: 16.23 小时
"""

DEBUG_MODE = False

TASK_NAME = "cfg_pi0.5_28_dim.robomind_1_train"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260207"

# Dataset settings
ROOT_DIR = "/mnt/"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = True
DELTA_ACTION_MASK_INDICES = [7, -1, 7, -1, 12]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

# verifed
from openpi.configs.dataset_config.aloha import REPO_ID as aloha
from openpi.configs.dataset_config.intern_a1_real import REPO_ID as intern_a1_real
from openpi.configs.dataset_config.rdt import REPO_ID as rdt
from openpi.configs.dataset_config.rhos_ai_gm100 import REPO_ID as rhos_ai_gm100
from openpi.configs.dataset_config.robocoin import REPO_ID as robocoin

from openpi.configs.dataset_config.agibotworld_alpha_lerobot import (
    REPO_ID as agibotworld_alpha_lerobot,
)
from openpi.configs.dataset_config.galaxea import REPO_ID as galaxea
from openpi.configs.dataset_config.robochallenge import REPO_ID as robochallenge
from openpi.configs.dataset_config.robomind_1_train import REPO_ID as robomind_1_train
from openpi.configs.dataset_config.robomind_1_val import REPO_ID as robomind_1_val
from openpi.configs.dataset_config.robomind_2 import REPO_ID as robomind_2

REPO_ID = robomind_1_train  # galaxea + agibotworld_alpha_lerobot + aloha + galaxea + intern_a1_real + rdt + rhos_ai_gm100 + robocoin

ROBOT_ALIGN_INFO = _utils.ROBOT_ALIGN_INFO
# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 16
TRAIN_STEPS = 100000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[::20]

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
    # How often (in steps) to save checkpoints.
    save_interval=5000,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=10000,
    overwrite=True,
)
