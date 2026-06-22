from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *
from openpi.training.frame_attributes_preprocessors import (
    ValuePredictionPreprocessor,
)

DEBUG_MODE = False

TASK_NAME = "pi05_base_finetune_box_value_stage2"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20251215"

# Dataset settings
ROOT_DIR = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/fold_box_from_scratch/"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
REPO_ID = [
    "total_steps/fold_box_scratch.all.102s.20260302.batch.11",
    "total_steps/fold_box_scratch.all.102s.20260302.batch.12",
    "total_steps/fold_box_scratch.all.102s.20260302.batch.13",
    "total_steps/fold_box_scratch.all.102s.20260302.batch.14",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.1",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.2",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.3",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.4",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.5",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.6",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.7",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.8",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.9",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.10",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.11",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.12",
    "total_steps/fold_box_scratch.all.102s.20260303.batch.13",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.10",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.11",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.12",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.13",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.2",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.4",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.5",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.6",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.7",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.8",
    "total_steps/fold_box_scratch.all.102s.20260304.batch.9",
    "total_steps/fold_box_scratch.all.102s.20260306.batch.8",
    "total_steps/fold_box_scratch.all.102s.20260306.batch.9",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.1",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.2",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.3",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.4",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.5",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.6",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.7",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.8",
    "total_steps/fold_box_scratch.all.102s.20260309.batch.9",
    "total_steps/fold_box_scratch.all.105s.20260310.batch.1",
    "total_steps/fold_box_scratch.all.105s.20260310.batch.2",
    "total_steps/fold_box_scratch.all.105s.20260310.batch.3",
    "total_steps/fold_box_scratch.all.105s.20260311.batch.1",
    "total_steps/fold_box_scratch_green.all.105s.20260311.batch.1",
    "total_steps/fold_box_scratch_purple.all.105s.20260311.batch.1",
    "total_steps/fold_box_scratch_silver.all.105s.20260311.batch.1",
    "total_steps/fold_box_scratch_yellow.all.105s.20260311.batch.1",
    # 失败
    "bad/fold_box_scratch.bad2-8.10s.20260310.batch.1",
    "bad/fold_box_scratch.bad2-8.10s.20260310.batch.2",
    "bad/fold_box_scratch.bad2-8.10s.20260310.batch.3",
    "bad/fold_box_scratch.bad2-8.10s.20260311.batch.0",
    "bad/fold_box_scratch.bad2-8.10s.20260311.batch.1",
    "bad/fold_box_scratch.bad2-8.10s.20260311.batch.2",
    "bad/fold_box_scratch.bad2-8.10s.20260311.batch.3",
    "bad/fold_box_scratch.bad2-8.10s.20260311.batch.4",
    "bad/fold_box_scratch.bad23.10s.20260311.batch.1",
    "bad/fold_box_scratch.bad23.10s.20260311.batch.2",
    "bad/fold_box_scratch.bad31.10s.20260311.batch.1",
    "bad/fold_box_scratch.bad31.10s.20260311.batch.2",
    "bad/fold_box_scratch.bad37.10s.20260311.batch.1",
    "bad/fold_box_scratch.bad37.10s.20260311.batch.2",
]

# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 12
TRAIN_STEPS = 5000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

FRAME_ATTRS_PREPROCESSORS = [
    ValuePredictionPreprocessor(
        value_pred_dir="value_pred",
        auto_discover_chunks=True,
        validate_episode_count=True,
    ),
]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(pi05=True, enable_rl_value_head=True, max_token_len=128),
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=True,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
            value_net_cfg={
                "returns_norm_strategy": "fixed",
                "returns_norm_length": 3180,
                "failure_decrease_threshold": 0.1,
            },
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
    log_interval=10,
    # How often (in steps) to save checkpoints.
    save_interval=1000,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=1000,
    overwrite=True,
)
