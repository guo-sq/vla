from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *

DEBUG_MODE = False

TASK_NAME = "pi05_base_finetune_clothes_recap_qirl"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260123"

# Dataset settings
ROOT_DIR = "/mnt/workspace/shared/3090_14_26/cache_huggingface/lerobot/"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
REPO_ID = [
    # [24,37,59,95]
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.1",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.2",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.3",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.4",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.5",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.6",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.7",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.8",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1230.9",
    # [24,37,59,95]
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1231.1",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1231.2",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1231.3",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1231.4",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1231.5",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1231.6",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v1231.7",
    # [24,37,59,95]
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v0104.1",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v0104.2",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v0104.3",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v0104.4",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v0104.5",
    # "heyuan1993/bipiper_clothes/record.clothes.bipiper.v0104.6",
    "heyuan1993/bipiper_clothes/record.clothes.bipiper.v0104.7",
]
# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 12
TRAIN_STEPS = 20000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(pi05=True, enable_rl_value_head=True, max_token_len=192),
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
        target_action_dim=TARGET_ACTION_DIM,
        unify_action_space=UNIFY_ACTION_SPACE,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_WEIGHT_PATH),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=100,
    # How often (in steps) to save checkpoints.
    save_interval=10000,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=10000,
    overwrite=True,
)