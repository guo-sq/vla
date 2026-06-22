import os
from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.training.frame_attributes_preprocessors import HfColumnIsValidPreprocessor

DEBUG_MODE = False

TASK_NAME = "pi05_base_pickandplace_base"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260320"

# Dataset settings
ROOT_DIR = "/mnt/oss_data/anyverse/bipiper/pick_place"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
REPO_ID = []
black_repo_list = [
    "new_record.pick.place.onelyunzip_container.bipiper.v0304.3",
    "record.pick.place.move.newobj.bipiper.v0126.3",
    "record.pick.place.move.withmiss.bipiper.v0114.5",
    "record.pick.place.onelyunzip_container.bipiper.v0304.3",
    "record.pick.place.pushotherout.bipiper.v0225.3",
    "record.pick.place.pushotherout.bipiper.v0226.1",
    "record.pick.place.pushotherout.bipiper.v0226.2",
    "record.pick.place.pushotherout.bipiper.v0226.4",
    "record.pick.place.scheme2.otherbackgrounds.withmiss.bipiper.v0112.12",
]
data_sources = ["anyverse_pickAndplace_record", "anyverse_pickAndplace_record_reverse"]

for data_source in data_sources:
    for repo in os.listdir(os.path.join(ROOT_DIR, data_source)):
        if repo in black_repo_list:
            continue
        REPO_ID.append(os.path.join(data_source, repo))
# TrainConfig
BATCH_SIZE = 256
NUM_WORKERS = 8
TRAIN_STEPS = 100000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

FRAME_ATTRS_PREPROCESSORS = [
    HfColumnIsValidPreprocessor(column_name="is_valid"),
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
            use_generalizable_prompt=False,
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
    # How often (in steps) to save checkpoints.
    save_interval=TRAIN_STEPS // 2,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=TRAIN_STEPS // 2,
    overwrite=False,
    resume=True,
)
