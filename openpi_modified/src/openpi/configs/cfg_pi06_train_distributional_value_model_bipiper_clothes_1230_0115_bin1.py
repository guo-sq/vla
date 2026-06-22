from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *

DEBUG_MODE = False

TASK_NAME = "pi06_train_distributional_value_model_bipiper_clothes_1230_0115_bin1"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20251215"

# Dataset settings
ROOT_DIR = "/mnt/workspace/shared/3090_14_26/cache_huggingface/lerobot/heyuan1993/bipiper_clothes"
REPO_ID = [
    "record.clothes.bipiper.v1230.1",
    "record.clothes.bipiper.v1230.2",
    "record.clothes.bipiper.v1230.3",
    "record.clothes.bipiper.v1230.4",
    "record.clothes.bipiper.v1230.5",
    "record.clothes.bipiper.v1230.6",
    "record.clothes.bipiper.v1230.7",
    "record.clothes.bipiper.v1230.8",
    "record.clothes.bipiper.v1231.1",
    "record.clothes.bipiper.v1231.2",
    "record.clothes.bipiper.v1231.3",
    "record.clothes.bipiper.v1231.4",
    "record.clothes.bipiper.v1231.5",
    "record.clothes.bipiper.v1231.6",
    "record.clothes.bipiper.v1231.7",
    "record.clothes.bipiper.v0104.1",
    "record.clothes.bipiper.v0104.2",
    "record.clothes.bipiper.v0104.3",
    "record.clothes.bipiper.v0104.4",
    "record.clothes.bipiper.v0104.5",
    "record.clothes.bipiper.v0104.6",
    "record.clothes.bipiper.v0104.7",
    "record.clothes.bipiper.v0105.1",
    "record.clothes.bipiper.v0105.2",
    "record.clothes.bipiper.v0105.3",
    "record.clothes.bipiper.v0105.4",
    "record.clothes.bipiper.v0105.5",
    "record.clothes.bipiper.v0106.1",
    "record.clothes.bipiper.v0106.2",
    "record.clothes.bipiper.v0108.1",
    "record.clothes.bipiper.v0108.2",
    "record.clothes.bipiper.v0108.3",
    "record.clothes.bipiper.v0108.4",
    "record.clothes.bipiper.v0109.1",
    "record.clothes.bipiper.v0109.2",
    "record.clothes.bipiper.v0109.3",
    "record.clothes.bipiper.v0109.4",
    "record.clothes.bipiper.v0109.5",
    "record.clothes.bipiper.v0109.6",
    "record.clothes.bipiper.v0109.7",
    "record.clothes.bipiper.v0109.8",
    "record.clothes.bipiper.v0109.9",
    "record.clothes.bipiper.v0112.1",
    "record.clothes.bipiper.v0112.2",
    "record.clothes.bipiper.v0112.3",
    "record.clothes.bipiper.v0112.4",
    "record.clothes.bipiper.v0112.5",
    "record.clothes.bipiper.v0113.1",
    "record.clothes.bipiper.v0113.2",
    "record.clothes.bipiper.v0113.3",
    "record.clothes.bipiper.v0113.4",
    "record.clothes.bipiper.v0113.6",
    "record.clothes.bipiper.v0113.7",
    "record.clothes.bipiper.v0113.8",
    "record.clothes.bipiper.v0113.9",
    "record.clothes.bipiper.v0113.10",
    "record.clothes.bipiper.v0114.1",
    "record.clothes.bipiper.v0114.2",
    "record.clothes.bipiper.v0114.3",
    "record.clothes.bipiper.v0114.4",
    "record.clothes.bipiper.v0114.5",
    "record.clothes.bipiper.v0114.6",
    "record.clothes.bipiper.v0114.7",
    "record.clothes.bipiper.v0115.1",
    "record.clothes.bipiper.v0115.2",
    "record.clothes.bipiper.v0115.3",
    "record.clothes.bipiper.v0115.4",
    "record.clothes.bipiper.v0115.5",
    "record.clothes.bipiper.v0115.6",
    "record.clothes.bipiper.v0115.7",
    "record.clothes.bipiper.v0115.8",
    "record.clothes.bipiper.v0115.9",
    "record.clothes.bipiper.v0115.10",
    "record.clothes.bipiper.v0115.11",
    "record.clothes.bipiper.v0115.12",
]
ACTION_SEQUENCE_KEYS = ("action",)

DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]
ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
UNIFY_ACTION_SPACE = False

# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 12
TRAIN_STEPS = 60000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

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
        pi05=True,
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
            prompt_from_episode=False,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
            value_net_cfg={
                "returns_norm_strategy": "fixed",
                "returns_norm_length": 3000,
            },
        ),
        default_prompt="You are a two-armed piper robot with a total of three perspectives. Your task is to fold a T-shirt. First, look for the collar of the T-shirt. If you can't see it, pick up the T-shirt and let it fall naturally until the collar is visible. Then, grab the collar to lay the T-shirt flat. Next, simultaneously grab the collar and the bottom hem to fold it in half and lay it flat. Finally, lay it flat with the collar facing down, then fold it up and place it in the fixed position on the right.",
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
    log_interval=100,
    save_interval=1000,
    keep_period=5000,
    overwrite=True,
)
