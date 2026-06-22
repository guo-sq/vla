from openpi.configs.base import *
import openpi.training.utils as _utils

DEBUG_MODE = False

TASK_NAME = "pi05_random_fold_T_1215163031_0145678121314151619_ftb0_3b_0117_mns_B3w"
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
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1215.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1215.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1216.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1216.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1216.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1216.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1216.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1216.6",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.6",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.7",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.8",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1230.9",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1231.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1231.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1231.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1231.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1231.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1231.6",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v1231.7",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0104.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0104.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0104.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0104.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0104.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0104.6",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0104.7",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0105.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0105.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0105.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0105.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0105.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0105.6",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0106.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0106.2",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0106.3",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0106.4",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0106.5",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0106.6",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0106.7",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.1",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.2",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.3",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.4",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.5",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.6",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.7",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.8",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.9",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.10",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.11",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.12",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0107.13",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0108.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0108.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0108.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0108.4",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.1",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.2",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.3",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.4",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.5",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.6",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.7",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.8",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0109.9",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.policy.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.policy.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.policy.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.policy.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0112.policy.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.6",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.7",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.8",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.9",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0113.10",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0114.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0114.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0114.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0114.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0114.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0114.6",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0114.7",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.6",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.7",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.8",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.9",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.10",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.11",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0115.12",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.4",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.5",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.6",
    # "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.7",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.8",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.9",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.10",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.policy.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0116.policy.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0119.policy.1",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0119.policy.2",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0119.policy.3",
    "anyverse/bipiper_clothes/record.clothes.bipiper.v0119.policy.4",
]
# TrainConfig
BATCH_SIZE = 128
NUM_WORKERS = 16
TRAIN_STEPS = 30000
PRETRAINED_WEIGHT_PATH = "checkpoints/pi05_general_cloth_basev0_3b_0117/pi05_general_cloth_basev0_3b_0117_exp0117/50000/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        pi05=True, enable_rl_value_head=False, max_token_len=256
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
    log_interval=200,
    # How often (in steps) to save checkpoints.
    save_interval=10000,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=10000,
    overwrite=True,
)
