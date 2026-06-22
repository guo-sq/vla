from openpi.configs.base import *
import openpi.training.utils as _utils

DEBUG_MODE = False

TASK_NAME = "example_14_dim"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260115"

# Dataset settings
ROOT_DIR = "/mnt/workspace/shared/datasets/"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
REPO_ID = [
    "anyverse_pour_water_record/record.pourwater.bipiper.0122.4",
    "anyverse_pour_water_record/record.pourwater.bipiper.0122.5",
]
# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 8
TRAIN_STEPS = 50000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        pi05=True, enable_rl_value_head=False, max_token_len=128
    ),
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
