"""Seatbelt value self-play fixed (production) config.

Fixes v3 GT distribution skew (constant -1 output at 5k steps) via:
1. state_confirmation_by_role: builder→BOTH, destroyer→END (was all END)
2. exclude_failures=True: failure episodes (GT=-1) excluded from training
3. Unified prompt: both pos/neg use hang prompt (align with inference)
4. Train/val split: 0325-0401 train, 0402.batch.1-6 val

Data: scripted (129 hang + 24 take_off) + self-play train (33 batches, 0325-0401).
Val: 6 self-play batches (0402.batch.1-6).
Excluded: 0402.batch.7-13, 0403.* (uncorrected policy).
"""

from openpi.configs.base import *
import os
import numpy as np
import openpi.configs as _configs_pkg
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *
from openpi.training.frame_attributes_preprocessors import (
    GripperCountRule,
    GripperCountValidMaskPreprocessor,
    PruneHeadTailStaticValidMaskPreprocessor,
    ValidMaskGroupParams,
    ValueReturnsPreprocessor,
    VelocityBasedStaticDetector,
)

DEBUG_MODE = False

TASK_NAME = "pi06_seatbelt_value_selfplay_fixed"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20251215"

ROOT_DIR = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt"

DATASET_TYPE_CONFIG_PATH = os.path.join(os.path.dirname(_configs_pkg.__file__), "dataset_types_seatbelt.yaml")

# ---------------------------------------------------------------------------
# Positive: scripted hang datasets (129 batches)
# ---------------------------------------------------------------------------
HANG_REPO_ID = [
    "seatbelt.single.hang.baichenglong.20260205.batch.10",
    "seatbelt.single.hang.baichenglong.20260205.batch.11",
    "seatbelt.single.hang.baichenglong.20260205.batch.12",
    "seatbelt.single.hang.baichenglong.20260205.batch.13",
    "seatbelt.single.hang.baichenglong.20260205.batch.14",
    "seatbelt.single.hang.baichenglong.20260205.batch.15",
    "seatbelt.single.hang.baichenglong.20260205.batch.5",
    "seatbelt.single.hang.baichenglong.20260205.batch.6",
    "seatbelt.single.hang.baichenglong.20260205.batch.7",
    "seatbelt.single.hang.baichenglong.20260205.batch.8",
    "seatbelt.single.hang.baichenglong.20260205.batch.9",
    "seatbelt.single.hang.baichenglong.20260206.batch.10",
    "seatbelt.single.hang.baichenglong.20260206.batch.11",
    "seatbelt.single.hang.baichenglong.20260206.batch.12",
    "seatbelt.single.hang.baichenglong.20260206.batch.13",
    "seatbelt.single.hang.baichenglong.20260206.batch.14",
    "seatbelt.single.hang.baichenglong.20260206.batch.15",
    "seatbelt.single.hang.baichenglong.20260206.batch.16",
    "seatbelt.single.hang.baichenglong.20260206.batch.17",
    "seatbelt.single.hang.baichenglong.20260206.batch.18",
    "seatbelt.single.hang.baichenglong.20260206.batch.19",
    "seatbelt.single.hang.baichenglong.20260206.batch.9",
    "seatbelt.single.hang.baichenglong.20260209.batch.15",
    "seatbelt.single.hang.baichenglong.20260209.batch.16",
    "seatbelt.single.hang.baichenglong.20260209.batch.17",
    "seatbelt.single.hang.baichenglong.20260209.batch.18",
    "seatbelt.single.hang.baichenglong.20260210.batch.14",
    "seatbelt.single.hang.baichenglong.20260210.batch.15",
    "seatbelt.single.hang.baichenglong.20260210.batch.16",
    "seatbelt.single.hang.baichenglong.20260210.batch.17",
    "seatbelt.single.hang.baichenglong.20260211.batch.10",
    "seatbelt.single.hang.baichenglong.20260211.batch.11",
    "seatbelt.single.hang.baichenglong.20260211.batch.8",
    "seatbelt.single.hang.baichenglong.20260211.batch.9",
    "seatbelt.single.hang.zhangyu.20260206.batch.1",
    "seatbelt.single.hang.zhangyu.20260206.batch.2",
    "seatbelt.single.hang.zhangyu.20260206.batch.3",
    "seatbelt.single.hang.zhangyu.20260206.batch.4",
    "seatbelt.single.hang.zhangyu.20260206.batch.5",
    "seatbelt.single.hang.zhangyu.20260206.batch.6",
    "seatbelt.single.hang.zhangyu.20260206.batch.7",
    "seatbelt.single.hang.zhangyu.20260206.batch.8",
    "seatbelt.single.hang.zhangyu.20260207.batch.1",
    "seatbelt.single.hang.zhangyu.20260207.batch.10",
    "seatbelt.single.hang.zhangyu.20260207.batch.11",
    "seatbelt.single.hang.zhangyu.20260207.batch.12",
    "seatbelt.single.hang.zhangyu.20260207.batch.2",
    "seatbelt.single.hang.zhangyu.20260207.batch.3",
    "seatbelt.single.hang.zhangyu.20260207.batch.4",
    "seatbelt.single.hang.zhangyu.20260207.batch.5",
    "seatbelt.single.hang.zhangyu.20260207.batch.6",
    "seatbelt.single.hang.zhangyu.20260207.batch.7",
    "seatbelt.single.hang.zhangyu.20260207.batch.8",
    "seatbelt.single.hang.zhangyu.20260207.batch.9",
    "seatbelt.single.hang.zhangyu.20260208.batch.1",
    "seatbelt.single.hang.zhangyu.20260208.batch.10",
    "seatbelt.single.hang.zhangyu.20260208.batch.11",
    "seatbelt.single.hang.zhangyu.20260208.batch.2",
    "seatbelt.single.hang.zhangyu.20260208.batch.3",
    "seatbelt.single.hang.zhangyu.20260208.batch.4",
    "seatbelt.single.hang.zhangyu.20260208.batch.5",
    "seatbelt.single.hang.zhangyu.20260208.batch.6",
    "seatbelt.single.hang.zhangyu.20260208.batch.7",
    "seatbelt.single.hang.zhangyu.20260208.batch.8",
    "seatbelt.single.hang.zhangyu.20260208.batch.9",
    "seatbelt.single.hang.zhangyu.20260209.batch.1",
    "seatbelt.single.hang.zhangyu.20260209.batch.2",
    "seatbelt.single.hang.zhangyu.20260209.batch.3",
    "seatbelt.single.hang.zhangyu.20260209.batch.4",
    "seatbelt.single.hang.zhangyu.20260209.batch.5",
    "seatbelt.single.hang.zhangyu.20260209.batch.6",
    "seatbelt.single.hang.zhangyu.20260209.batch.7",
    "seatbelt.single.hang.zhangyu.20260210.batch.1",
    "seatbelt.single.hang.zhangyu.20260210.batch.10",
    "seatbelt.single.hang.zhangyu.20260210.batch.11",
    "seatbelt.single.hang.zhangyu.20260210.batch.12",
    "seatbelt.single.hang.zhangyu.20260210.batch.13",
    "seatbelt.single.hang.zhangyu.20260210.batch.2",
    "seatbelt.single.hang.zhangyu.20260210.batch.3",
    "seatbelt.single.hang.zhangyu.20260210.batch.4",
    "seatbelt.single.hang.zhangyu.20260210.batch.8",
    "seatbelt.single.hang.zhangyu.20260210.batch.9",
    "seatbelt.single.hang.zhangyu.20260211.batch.1",
    "seatbelt.single.hang.zhangyu.20260211.batch.2",
    "seatbelt.single.hang.zhangyu.20260211.batch.3",
    "seatbelt.single.hang.zhangyu.20260211.batch.7",
    "seatbelt.single.hang_continuous_5_move.baichenglong.20260224.batch.10",
    "seatbelt.single.hang_continuous_5_move.baichenglong.20260224.batch.6",
    "seatbelt.single.hang_continuous_5_move.baichenglong.20260224.batch.7",
    "seatbelt.single.hang_continuous_5_move.baichenglong.20260224.batch.8",
    "seatbelt.single.hang_continuous_5_move.baichenglong.20260224.batch.9",
    "seatbelt.single.hang_continuous_5_move.baichenglong.20260225.batch.1",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.10",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.5",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.6",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.7",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.8",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.9",
    "seatbelt.single.hang_move.baichenglong.20260213.batch.1",
    "seatbelt.single.hang_move.baichenglong.20260213.batch.2",
    "seatbelt.single.hang_move.baichenglong.20260213.batch.3",
    "seatbelt.single.hang_move.baichenglong.20260213.batch.4",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.10",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.11",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.7",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.8",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.9",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.1",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.10",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.11",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.2",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.3",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.4",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.5",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.6",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.7",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.8",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.9",
    "seatbelt.single.hang_move.panjinlong.20260303.batch.4",
    "seatbelt.single.hang_move.panjinlong.20260303.batch.5",
    "seatbelt.single.hang_move.panjinlong.20260303.batch.6",
    "seatbelt.single.hang_move.panjinlong.20260303.batch.7",
    "seatbelt.single.hang_move.zhaoshuai.20260304.batch.1",
    "seatbelt.single.hang_move.zhaoshuai.20260304.batch.2",
    "seatbelt.single.hang_move.zhaoshuai.20260304.batch.7",
    "seatbelt.single.hang_move.zhaoshuai.20260310.batch.3",
    "seatbelt.single.hang_move.zhaoshuai.20260310.batch.4",
    "seatbelt.single.hang_move.zhaoshuai.20260310.batch.5",
    "seatbelt.single.hang_move.zhaoshuai.20260310.batch.6",
]

# ---------------------------------------------------------------------------
# Negative: scripted take_off datasets (24 batches)
# ---------------------------------------------------------------------------
TAKE_OFF_REPO_ID = [
    "seatbelt.single.take_off.haoshuailing.20260309.batch.5.cpt",
    "seatbelt.single.take_off.haoshuailing.20260309.batch.6.cpt",
    "seatbelt.single.take_off_move.haoshuailing.20260309.batch.7.cpt",
    "seatbelt.single.take_off_move.haoshuailing.20260311.batch.10",
    "seatbelt.single.take_off_move.haoshuailing.20260311.batch.11",
    "seatbelt.single.take_off_move.haoshuailing.20260311.batch.9",
    "seatbelt.single.take_off_move.panjinlong.20260228.batch.13",
    "seatbelt.single.take_off_move.panjinlong.20260228.batch.14",
    "seatbelt.single.take_off_move.panjinlong.20260228.batch.15",
    "seatbelt.single.take_off_move.panjinlong.20260228.batch.16",
    "seatbelt.single.take_off_move.panjinlong.20260228.batch.17",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.1",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.10",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.2",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.3",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.4",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.5",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.6",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.7",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.8",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.9",
    "seatbelt.single.take_off_move.panjinlong.20260303.batch.1",
    "seatbelt.single.take_off_move.panjinlong.20260303.batch.2",
    "seatbelt.single.take_off_move.panjinlong.20260303.batch.3",
]

# ---------------------------------------------------------------------------
# Self-play TRAIN: 0325-0401 (33 batches)
# ---------------------------------------------------------------------------
_PREFIX = "seatbelt.single.self_play_record_cleaned.0205_0312_self_play_recovery"
SELFPLAY_TRAIN_REPO_ID = (
    [f"{_PREFIX}.20260325.batch.{b}" for b in [2, 3, 4, 5]]
    + [f"{_PREFIX}.20260326.batch.{b}" for b in [1, 2, 3, 4]]
    + [f"{_PREFIX}.20260327.batch.{b}" for b in [1, 3, 4, 5, 7, 8, 9]]
    + [f"{_PREFIX}.20260328.batch.{b}" for b in [1, 2, 3, 4, 5, 6, 7]]
    + [f"{_PREFIX}.20260330.batch.{b}" for b in [1, 2, 3, 4, 5, 6]]
    + [f"{_PREFIX}.20260331.batch.{b}" for b in [1, 2]]
    + [f"{_PREFIX}.20260401.batch.{b}" for b in [1, 2, 3]]
)

# ---------------------------------------------------------------------------
# Self-play VAL: 0402.batch.1-6 (6 batches)
# ---------------------------------------------------------------------------
VAL_REPO_ID = [f"{_PREFIX}.20260402.batch.{b}" for b in [1, 2, 3, 4, 5, 6]]

# ---------------------------------------------------------------------------
# Training repo_ids = scripted + self-play train
# ---------------------------------------------------------------------------
TRAIN_REPO_ID = HANG_REPO_ID + TAKE_OFF_REPO_ID + SELFPLAY_TRAIN_REPO_ID

ACTION_SEQUENCE_KEYS = ("action",)
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]
ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
UNIFY_ACTION_SPACE = False

BATCH_SIZE = 1024
NUM_WORKERS = 12
TRAIN_STEPS = 20000

VALUE_BINS = 1
VALUE_RANGE = (-1.0, 0.0)

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    TRAIN_REPO_ID = HANG_REPO_ID[:1] + TAKE_OFF_REPO_ID[:1] + SELFPLAY_TRAIN_REPO_ID[:1]
    VAL_REPO_ID = VAL_REPO_ID[:1]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_3_270m",
        action_expert_variant="gemma_270m",
        vision_output_dim=640,
        input_image_size=(224, 224),
        checkpoint_image_size=(896, 896),
        vocab_size=262_144,
        enable_rl_value_head=True,
        value_bins=VALUE_BINS,
        value_range=VALUE_RANGE,
        max_token_len=256,
    ),
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=TRAIN_REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=False,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
            frame_attributes_preprocessors=[
                VelocityBasedStaticDetector(fps=30),
                PruneHeadTailStaticValidMaskPreprocessor(
                    fps=30,
                    groups=[ValidMaskGroupParams(name="default", head_margin_s=0.3)],
                ),
                GripperCountValidMaskPreprocessor(
                    open_threshold=2.0,
                    close_threshold=3.0,
                    fps=30,
                    rules=[
                        GripperCountRule(
                            batch_contains="self_play_record_cleaned",
                            gripper="right",
                            event="open",
                            count=-1,
                            invalidate="after",
                        ),
                        GripperCountRule(
                            batch_contains="self_play_record_cleaned",
                            gripper="left",
                            event="open",
                            count=-1,
                            invalidate="after",
                        ),
                    ],
                ),
                ValueReturnsPreprocessor(
                    classification_mode="auto",
                    dataset_type_config_path=DATASET_TYPE_CONFIG_PATH,
                    state_confirmation="auto",
                    state_confirmation_by_role={
                        "builder": "both",
                        "destroyer": "end_only",
                    },
                    negative_roles=["destroyer"],
                    positive_prompt="Hang the seatbelt with right hand under 20 seconds.",
                    # Intentionally identical to positive_prompt — the deployed
                    # policy is prompt-free, so train-time alignment matters more
                    # than a distinct negative prompt.
                    negative_prompt="Hang the seatbelt with right hand under 20 seconds.",
                    exclude_failures=True,
                ),
            ],
            value_net_cfg={
                "returns_norm_strategy": "per_task",
                "dataset_type_config_path": DATASET_TYPE_CONFIG_PATH,
            },
        ),
        extra_delta_transform=False,
        use_delta_joint_actions=True,
        delta_action_mask_indices=DELTA_ACTION_MASK_INDICES,
        public_dataset_camera_map=_utils.PUBLIC_DATASET_MAP,
        align_dim=ALIGN_DIM,
        target_action_dim=TARGET_ACTION_DIM,
        unify_action_space=UNIFY_ACTION_SPACE,
    ),
    weight_loader=weight_loaders.T5Gemma2EncoderWeightLoader(
        params_path="/mnt/oss_models/pretrained_models/t5gemma2_encoder_openpi.npz",
        checkpoint_image_size=(896, 896),
        target_image_size=(224, 224),
    ),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=100,
    save_interval=1000,
    keep_period=5000,
    overwrite=True,
    # Validation
    validation_repo_id=VAL_REPO_ID,
    validation_interval=500,
    validation_num_batches=10,
)
