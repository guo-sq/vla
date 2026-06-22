from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *

DEBUG_MODE = False

TASK_NAME = "pi06_train_distributional_value_model_t5gemma270M_seatbelt_data_0212_0304"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20251215"

# Dataset settings
ROOT_DIR = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt"
REPO_ID = [
    "seatbelt.single.hang.baichenglong.20260205.batch.5",
    "seatbelt.single.hang.baichenglong.20260205.batch.6",
    "seatbelt.single.hang.baichenglong.20260205.batch.7",
    "seatbelt.single.hang.baichenglong.20260205.batch.8",
    "seatbelt.single.hang.baichenglong.20260205.batch.9",
    "seatbelt.single.hang.baichenglong.20260205.batch.10",
    "seatbelt.single.hang.baichenglong.20260205.batch.11",
    "seatbelt.single.hang.baichenglong.20260205.batch.12",
    "seatbelt.single.hang.baichenglong.20260205.batch.13",
    "seatbelt.single.hang.baichenglong.20260205.batch.14",
    "seatbelt.single.hang.baichenglong.20260205.batch.15",
    "seatbelt.single.hang.zhangyu.20260206.batch.1",
    "seatbelt.single.hang.zhangyu.20260206.batch.2",
    "seatbelt.single.hang.zhangyu.20260206.batch.3",
    "seatbelt.single.hang.zhangyu.20260206.batch.4",
    "seatbelt.single.hang.zhangyu.20260206.batch.5",
    "seatbelt.single.hang.zhangyu.20260206.batch.6",
    "seatbelt.single.hang.zhangyu.20260206.batch.7",
    "seatbelt.single.hang.zhangyu.20260206.batch.8",
    "seatbelt.single.hang.baichenglong.20260206.batch.9",
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
    "seatbelt.single.hang.zhangyu.20260207.batch.1",
    "seatbelt.single.hang.zhangyu.20260207.batch.2",
    "seatbelt.single.hang.zhangyu.20260207.batch.3",
    "seatbelt.single.hang.zhangyu.20260207.batch.4",
    "seatbelt.single.hang.zhangyu.20260207.batch.5",
    "seatbelt.single.hang.zhangyu.20260207.batch.6",
    "seatbelt.single.hang.zhangyu.20260207.batch.7",
    "seatbelt.single.hang.zhangyu.20260207.batch.8",
    "seatbelt.single.hang.zhangyu.20260207.batch.9",
    "seatbelt.single.hang.zhangyu.20260207.batch.10",
    "seatbelt.single.hang.zhangyu.20260207.batch.11",
    "seatbelt.single.hang.zhangyu.20260207.batch.12",
    "seatbelt.single.hang.zhangyu.20260208.batch.1",
    "seatbelt.single.hang.zhangyu.20260208.batch.2",
    "seatbelt.single.hang.zhangyu.20260208.batch.3",
    "seatbelt.single.hang.zhangyu.20260208.batch.4",
    "seatbelt.single.hang.zhangyu.20260208.batch.5",
    "seatbelt.single.hang.zhangyu.20260208.batch.6",
    "seatbelt.single.hang.zhangyu.20260208.batch.7",
    "seatbelt.single.hang.zhangyu.20260208.batch.8",
    "seatbelt.single.hang.zhangyu.20260208.batch.9",
    "seatbelt.single.hang.zhangyu.20260208.batch.10",
    "seatbelt.single.hang.zhangyu.20260208.batch.11",
    "seatbelt.single.hang.zhangyu.20260209.batch.1",
    "seatbelt.single.hang.zhangyu.20260209.batch.2",
    "seatbelt.single.hang.zhangyu.20260209.batch.3",
    "seatbelt.single.hang.zhangyu.20260209.batch.4",
    "seatbelt.single.hang.zhangyu.20260209.batch.5",
    "seatbelt.single.hang.zhangyu.20260209.batch.6",
    "seatbelt.single.hang.zhangyu.20260209.batch.7",
    "seatbelt.single.hang.baichenglong.20260209.batch.15",
    "seatbelt.single.hang.baichenglong.20260209.batch.16",
    "seatbelt.single.hang.baichenglong.20260209.batch.17",
    "seatbelt.single.hang.baichenglong.20260209.batch.18",
    "seatbelt.single.hang.zhangyu.20260210.batch.1",
    "seatbelt.single.hang.zhangyu.20260210.batch.2",
    "seatbelt.single.hang.zhangyu.20260210.batch.3",
    "seatbelt.single.hang.zhangyu.20260210.batch.4",
    "seatbelt.single.hang.zhangyu.20260210.batch.8",
    "seatbelt.single.hang.zhangyu.20260210.batch.9",
    "seatbelt.single.hang.zhangyu.20260210.batch.10",
    "seatbelt.single.hang.zhangyu.20260210.batch.11",
    "seatbelt.single.hang.zhangyu.20260210.batch.12",
    "seatbelt.single.hang.zhangyu.20260210.batch.13",
    "seatbelt.single.hang.baichenglong.20260210.batch.14",
    "seatbelt.single.hang.baichenglong.20260210.batch.15",
    "seatbelt.single.hang.baichenglong.20260210.batch.16",
    "seatbelt.single.hang.baichenglong.20260210.batch.17",
    "seatbelt.single.hang.zhangyu.20260211.batch.1",
    "seatbelt.single.hang.zhangyu.20260211.batch.2",
    "seatbelt.single.hang.zhangyu.20260211.batch.3",
    "seatbelt.single.hang.zhangyu.20260211.batch.7",
    "seatbelt.single.hang.baichenglong.20260211.batch.8",
    "seatbelt.single.hang.baichenglong.20260211.batch.9",
    "seatbelt.single.hang.baichenglong.20260211.batch.10",
    "seatbelt.single.hang.baichenglong.20260211.batch.11",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.6",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.7",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.8",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.9",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.10",
    "seatbelt.single.hang_move.baichenglong.20260213.batch.1",
    "seatbelt.single.hang_move.baichenglong.20260213.batch.2",
    "seatbelt.single.hang_move.baichenglong.20260213.batch.3",
    "seatbelt.single.hang_move.baichenglong.20260213.batch.4",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.7",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.8",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.9",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.10",
    "seatbelt.single.hang_move.baichenglong.20260227.batch.11",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.1",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.2",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.3",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.4",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.5",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.6",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.7",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.8",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.9",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.10",
    "seatbelt.single.hang_move.baichenglong.20260228.batch.11",
    "seatbelt.single.hang_move.zhaoshuai.20260304.batch.1",
    "seatbelt.single.hang_move.zhaoshuai.20260304.batch.2",
    "seatbelt.single.hang_move.zhaoshuai.20260304.batch.7",
]
ACTION_SEQUENCE_KEYS = ("action",)

DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]
ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
UNIFY_ACTION_SPACE = False

# TrainConfig
BATCH_SIZE = 1024
NUM_WORKERS = 12
TRAIN_STEPS = 60000

# Value model parameters
VALUE_BINS = 1
VALUE_RANGE = (-1.0, 0.0)

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_3_270m",  # Use Gemma 3 270M from T5Gemma 2 as backbone
        action_expert_variant="gemma_270m",  # Action expert uses Gemma 2 (preserved from current model)
        vision_output_dim=640,  # Vision encoder output dimension (matches gemma_270m width)
        input_image_size=(224, 224),  # Input image resolution (224x224 = 16x16 patches)
        checkpoint_image_size=(
            896,
            896,
        ),  # Checkpoint training resolution (896x896 = 64x64 patches, requires interpolation)
        vocab_size=262_144,  # T5Gemma 2 vocabulary size (preserve full embedding)
        enable_rl_value_head=True,
        value_bins=VALUE_BINS,
        value_range=VALUE_RANGE,
        max_token_len=256,
    ),
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=True,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
            value_net_cfg={
                "returns_norm_strategy": "per_episode",
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
        params_path="/mnt/oss_models/pretrained_models/t5gemma2_encoder_openpi.npz",  # Pretrained T5Gemma 2 encoder weights
        checkpoint_image_size=(896, 896),  # T5Gemma 2 checkpoint trained at 896x896
        target_image_size=(224, 224),  # Target resolution (matches input_image_size)
    ),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=100,
    save_interval=1000,
    keep_period=5000,
    overwrite=True,
)
