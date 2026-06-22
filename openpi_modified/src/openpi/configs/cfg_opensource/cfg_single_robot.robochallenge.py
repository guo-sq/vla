from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *
import openpi.training.optimizer as _optimizer

from openpi.configs.dataset_config.robochallenge import REPO_ID

DEBUG_MODE = False

TASK_NAME = "cfg_pi0.5_28_dim.robochallenge"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260207"

# Dataset settings
ROOT_DIR = "/mnt/"
ACTION_SEQUENCE_KEYS = (
    "action",
    "action.left_gripper",
    "action.right_gripper",
    "action.left_arm",
    "action.right_arm",
    "actions.joint.position",
    "actions.effector.position",
    "action.arm.position",
    "action.effector.position",
    "eef_sim_pose_action",
)
UNIFY_ACTION_SPACE = True
DELTA_ACTION_MASK_INDICES = [7, -1, 7, -1, 12]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

ROBOT_ALIGN_INFO = _utils.ROBOT_ALIGN_INFO
# TrainConfig
BATCH_SIZE = 128
NUM_WORKERS = 16
TRAIN_STEPS = 10000
TOLERANCE_S = 0.02  # TODO(JY)
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

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
        # paligemma_variant="gemma_2b_lora",
        # action_expert_variant="gemma_300m_lora",
    ),
    # freeze_filter=pi0_config.Pi0Config(
    #     pi05=True,
    #     max_token_len=256,
    #     enable_rl_value_head=False,
    #     use_joint_eef_mask=True,
    #     paligemma_variant="gemma_2b_lora",
    #     action_expert_variant="gemma_300m_lora",
    # ).get_freeze_filter(),
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
        tolerance_s=TOLERANCE_S,
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
    keep_period=5000,
    overwrite=True,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=TRAIN_STEPS // 15,
        peak_lr=2e-5 * max(BATCH_SIZE // 64, 1),
        decay_steps=TRAIN_STEPS,
        decay_lr=2e-6,
    ),
)
