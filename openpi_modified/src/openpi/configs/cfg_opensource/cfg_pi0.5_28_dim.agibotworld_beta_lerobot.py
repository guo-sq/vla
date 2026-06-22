from openpi.configs.base import *
from openpi.configs.robot_cfg.base import *
import openpi.training.utils as _utils


DEBUG_MODE = False

TASK_NAME = "cfg_pi0.5_28_dim.agibotworld_beta_lerobot.sample10"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "agibotworld_beta_lerobot_sample10"

ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = True
DELTA_ACTION_MASK_INDICES = [7, -1, 7, -1, 12]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

from openpi.configs.dataset_config.agibotworld_beta_lerobot import DEBUG_REPO_ID, SAMPLE_1M_REPO_ID
from openpi.configs.dataset_config.agibotworld_beta_lerobot import REPO_ID as agibotworld_beta_lerobot
from openpi.configs.dataset_config.agibotworld_beta_lerobot import ROOT_DIR


ROBOT_ALIGN_INFO = _utils.ROBOT_ALIGN_INFO
REPO_ID = SAMPLE_1M_REPO_ID

BATCH_SIZE = 16
NUM_WORKERS = 8
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
    REPO_ID = DEBUG_REPO_ID
    ASSET_ID = "agibotworld_beta_task351_debug"

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
    save_interval=SAVE_INTERVAL,
    keep_period=KEEP_PERIOD,
    overwrite=True,
)
