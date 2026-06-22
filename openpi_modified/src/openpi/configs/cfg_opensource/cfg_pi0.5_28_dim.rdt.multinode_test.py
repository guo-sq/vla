from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *

"""
单机模拟多节点分布式训练测试用 config。
与 cfg_pi0.5_28_dim.rdt_local 相同，但将 num_train_steps 设为极小值以快速验证。
"""

DEBUG_MODE = False

TASK_NAME = "cfg_pi0.5_28_dim.rdt_local.multinode_test"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260224"

# Dataset settings
ROOT_DIR = "/mnt/"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = True
DELTA_ACTION_MASK_INDICES = [7, -1, 7, -1, 12]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

from openpi.configs.dataset_config.rdt import REPO_ID as rdt

REPO_ID = rdt[:2]

ROBOT_ALIGN_INFO = _utils.ROBOT_ALIGN_INFO

# 使用极小的步数用于快速验证分布式流程
BATCH_SIZE = 4
NUM_WORKERS = 2
TRAIN_STEPS = 10
FRAME_SKIP = 10
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        pi05=True,
        max_token_len=256,
        enable_rl_value_head=False,
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
        frame_skip=FRAME_SKIP,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_WEIGHT_PATH),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=1,
    save_interval=100,
    keep_period=100,
    overwrite=True,
)
