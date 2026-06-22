"""Multi-task fine-tuning configuration with distributional value head.

This configuration enables distributional value prediction for RL training.
Value is predicted as a distribution over 201 bins instead of a single scalar.
"""

from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *

DEBUG_MODE = False

TASK_NAME = "pi05_base_finetune_multi_task_distributional_value"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20251215"

# Dataset settings
ROOT_DIR = "/mnt/workspace/shared/datasets/anyverse_human_data_record/arxx5_bimanual"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
REPO_ID = [
    # 叠随机形状的毛巾
    "fold_towel/fold_towel.random.35s.1110.batch.3",
    "fold_towel/fold_towel.random.40s.1110.batch.4",
    "fold_towel/fold_towel.random.40s.1110.batch.5",
    "fold_towel/fold_towel.random.40s.1111.batch.1",
    # "fold_towel/fold_towel.random.40s.1111.batch.2",
    # 叠毛巾
    "fold_towel/fold_towel.40s.1104.batch.5",  # 20条数据
    "fold_towel/fold_towel.40s.1105.batch.1",
    "fold_towel/fold_towel.40s.1105.batch.2",
    "fold_towel/fold_towel.40s.1106.batch.1",
    # "fold_towel/fold_towel.40s.1209.batch.1",
    # 叠t恤
    "fold_shirt/fold_small_t_shirt.fast_format.35s.1117.batch.3",
    "fold_shirt/fold_small_t_shirt.fast_format.35s.1117.batch.4",
    "fold_shirt/fold_small_t_shirt.fast_format.35s.1118.batch.1",
    "fold_shirt/fold_small_t_shirt.fast_format.35s.1118.batch.2",
    # "fold_shirt/fold_small_t_shirt.fast_format.35s.1118.batch.3",
    # 叠格子t恤
    "fold_shirt/fold_small_t_shirt.fast_format.green_grid.35s.1120.batch.1",
    "fold_shirt/fold_small_t_shirt.fast_format.green_grid.35s.1120.batch.2",
    "fold_shirt/fold_small_t_shirt.fast_format.green_grid.35s.1120.batch.3",
    "fold_shirt/fold_small_t_shirt.fast_format.green_grid.35s.1120.batch.4",
    # "fold_shirt/fold_small_t_shirt.fast_format.green_grid.35s.1120.batch.5",
    # 夹管+插管
    "insert_tube/grab_and_attach_tube.full_tray_52_tube.40s.1210.batch.4",
    "insert_tube/grab_and_attach_tube.full_tray_52_tube.40s.1211.batch.2",
    "insert_tube/grab_and_attach_tube.full_tray_52_tube.40s.1211.batch.3",
    "insert_tube/grab_and_attach_tube.full_tray_52_tube.40s.1211.batch.4",
    # "insert_tube/grab_and_attach_tube.full_tray_52_tube.40s.1219.batch.1",
    # 桌面收纳
    "static_sort/static_sort.45s.1114.batch.4",
    # "static_sort/static_sort.45s.1114.batch.5",
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
    model=pi0_config.Pi0Config(
        pi05=True,
        max_token_len=144,
        # Enable distributional value head for RL training
        enable_rl_value_head=True,
        value_bins=201,  # Distributional mode: 201 bins
        value_range=(-1.0, 0.0),  # Value range for distribution
        value_label_smoothing=0.1,  # Label smoothing for cross-entropy loss
        value_temperature=2.0,  # Temperature for soft labels
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
