"""Example config: seatbelt task with frame-attribute preprocessors.

Demonstrates:
  - VelocityBasedStaticDetector for is_static detection
  - PruneHeadTailStaticValidMaskPreprocessor with prune_trailing=False (keep tail)
    and prune_trailing=True for avoid/wandering repos
  - RepoNameMatchSampleWeightPreprocessor for 2x upweighting "insert" repos
"""

from openpi.configs.base import *
import openpi.training.optimizer as _optimizer
import openpi.training.utils as _utils
from openpi.configs.dataset_config.anyverse_tasks.seatbelt import REPO_ID, ROOT_DIR
from openpi.training.frame_attributes_preprocessors import (
    GripperCountRule,
    GripperCountSampleWeightPreprocessor,
    GripperCountSampleWeightRule,
    GripperCountValidMaskPreprocessor,
    PruneHeadTailStaticValidMaskPreprocessor,
    RepoNameMatchSampleWeightPreprocessor,
    ValidMaskGroupParams,
    VelocityBasedStaticDetector,
)

DEBUG_MODE = False

TASK_NAME = "seatbelt_14_dim_0401"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260401"

# Dataset settings
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)

# TrainConfig
BATCH_SIZE = 256
NUM_WORKERS = 16
TRAIN_STEPS = 75000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

# Preprocessor pipeline:
#   1. Static detection -> is_static
#   2. Valid mask from static boundaries (head-only for most, head+tail for avoid/wandering)
#   3. Gripper-count valid mask for recover_2_n_move (invalidate after 1st right close)
#   4. 2x sample weight for "insert" repos
#   5. Gripper-count sample weight for insert (4s before last right open, weight 2 -> total 4x)
FRAME_ATTRS_PREPROCESSORS = [
    VelocityBasedStaticDetector(
        fps=30,
        joint_velocity_threshold=0.1,
        gripper_velocity_threshold=0.2,
        smoothing_half_window=2,
    ),
    PruneHeadTailStaticValidMaskPreprocessor(
        fps=30,
        groups=[
            ValidMaskGroupParams(
                name="avoid_and_wandering",
                match=["*avoid*", "*wandering*"],
                head_margin_s=0.3,
                prune_trailing=True,
            ),
            ValidMaskGroupParams(
                name="self_play",
                match=["*self_play*", "*infer*"],
                prune_trailing=False,
                skip_static_processing=True,
            ),
            # Default: prune head static only, keep tail
            ValidMaskGroupParams(
                name="default",
                head_margin_s=0.3,
                prune_trailing=False,
            ),
        ],
    ),
    # recover_2_n_move: starts closed -> opens -> closes again.
    # Invalidate everything after the first right gripper close.
    GripperCountValidMaskPreprocessor(
        open_threshold=2.0,
        close_threshold=3.0,
        fps=30,
        rules=[
            GripperCountRule(
                batch_contains="recover_2_n_move",
                gripper="right",
                event="close",
                count=1,
                invalidate="after",
            ),
        ],
    ),
    # insert batches: weight=2 for all insert frames (combined with GripperCount below -> 2*2=4)
    RepoNameMatchSampleWeightPreprocessor(substring="insert", weight=2),
    # insert batches: weight=2 for 4 seconds before the last right gripper open
    GripperCountSampleWeightPreprocessor(
        open_threshold=2.0,
        close_threshold=3.0,
        fps=30,
        rules=[
            GripperCountSampleWeightRule(
                batch_contains="insert",
                gripper="right",
                event="open",
                count=-1,
                region="before",
                duration_s=4.0,
                weight=2,
            ),
        ],
    ),
]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(pi05=True, enable_rl_value_head=False, max_token_len=128, action_horizon=50),
    rtc_max_delay=15,
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=False,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
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
    log_interval=200,
    save_interval=5000,
    keep_period=TRAIN_STEPS // 2,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=int(TRAIN_STEPS * 0.05),
        peak_lr=2e-5,
        decay_steps=TRAIN_STEPS,
        decay_lr=2e-6,
    ),
    overwrite=True,
)
