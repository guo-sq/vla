from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *
import openpi.training.optimizer as _optimizer
from openpi.configs.dataset_config.anyverse_tasks.pour_water import REPO_ID, ROOT_DIR
from openpi.training.frame_attributes_preprocessors import (
    FrameWeightByDimThresholdProcessor,
    VelocityBasedStaticDetector,
)

DEBUG_MODE = False

TASK_NAME = "pour_water_14_dim_0303"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260127"

# Dataset settings
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 12
TRAIN_STEPS = 70000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"  # "/mnt/oss_models/pretrained_models/pour_water/fr_cpt_0308/pour_water_0120_0130_cpt_base_finetune_0120_0306_date_0308_adust_lr_exp/49999/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

FRAME_ATTRS_PREPROCESSORS = [
    VelocityBasedStaticDetector(
        fps=30,
        joint_velocity_threshold=0.01,
        gripper_velocity_threshold=0.01,
        smoothing_half_window=30,
    ),
    FrameWeightByDimThresholdProcessor(
        dim_thresh_config=[
            (5, [(-100, -10)]),
        ],
        repeat_weight=3,
    ),
]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(pi05=True, enable_rl_value_head=False, max_token_len=128),
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=True,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
            parquet_dir="data_refractor",
            frame_attributes_preprocessors=FRAME_ATTRS_PREPROCESSORS,
        ),
        extra_delta_transform=False,
        use_delta_joint_actions=True,
        delta_action_mask_indices=DELTA_ACTION_MASK_INDICES,
        public_dataset_camera_map=_utils.PUBLIC_DATASET_MAP,
        align_dim=ALIGN_DIM,
        target_action_dim=TARGET_ACTION_DIM,
        unify_action_space=UNIFY_ACTION_SPACE,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_WEIGHT_PATH),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=TRAIN_STEPS // 15,
        peak_lr=2e-5 * max(BATCH_SIZE // 64, 1),
        decay_steps=TRAIN_STEPS,
        decay_lr=2e-6,
    ),
    log_interval=100,
    # How often (in steps) to save checkpoints.
    save_interval=10000,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=10000,
    overwrite=True,
)
