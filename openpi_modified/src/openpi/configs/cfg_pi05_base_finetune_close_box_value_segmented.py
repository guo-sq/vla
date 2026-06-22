from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *

DEBUG_MODE = False

TASK_NAME = "pi05_base_finetune_close_box_value_segmented"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260123"

# Dataset settings
ROOT_DIR = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/close_the_flap/"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
REPO_ID = [
    # 0331 新增数据
    "total_steps/close_the_flap.all.Cylinder.85s.20260331.batch.1",
    "total_steps/close_the_flap.all.Hook.85s.20260331.batch.1",
    "total_steps/close_the_flap.all.Hook.85s.20260331.batch.2",
    # "total_steps/close_the_flap.all.Hook.85s.20260331.batch.3",
    # "total_steps/close_the_flap.all.Hook.85s.20260331.batch.4",
    "total_steps/close_the_flap.all.Hook.85s.20260331.batch.5",
    "total_steps/close_the_flap.all.Hook.85s.20260331.batch.6",
    "total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.1",
    "total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.2",
    "total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.3",
    # "total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.4",
    "total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.5",
    "total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.1",
    "total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.2",
    "total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.3",
    "total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.4",
    "total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.5",
    "total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.6",
    # 0401
    "total_steps/close_the_flap.all.pen.85s.20260401.batch.1",
    "total_steps/close_the_flap.all.pen.85s.20260401.batch.2",
    "total_steps/close_the_flap.all.pen.85s.20260401.batch.3",
    "total_steps/close_the_flap.all.pen.85s.20260401.batch.4",
    "total_steps/close_the_flap.all.round.85s.20260401.batch.1",
    "total_steps/close_the_flap.all.round.85s.20260401.batch.2",
    "total_steps/close_the_flap.all.round.85s.20260401.batch.3",
    # "total_steps/close_the_flap.all.round.85s.20260401.batch.4",
    "total_steps/close_the_flap.all.round.85s.20260401.batch.5",
    "total_steps/close_the_flap.all.usb_typec.85s.20260401.batch.1",
    "total_steps/close_the_flap.all.usb_typec.85s.20260401.batch.2",
    # "total_steps/close_the_flap.all.usb_typec_new.85s.20260401.batch.1",
    # 0402
    "total_steps/close_the_flap.all.Sticker.85s.20260402.batch.1",
    "total_steps/close_the_flap.all.Sticker.85s.20260402.batch.2",
    "total_steps/close_the_flap.all.Sticker.85s.20260402.batch.3",
    "total_steps/close_the_flap.all.Sticker.85s.20260402.batch.4",
    # "total_steps/close_the_flap.all.Sticker.85s.20260402.batch.5",
    "total_steps/close_the_flap.all.model_new.85s.20260402.batch.1",
    # "total_steps/close_the_flap.all.model_new.85s.20260402.batch.2",
    "total_steps/close_the_flap.all.model_new.85s.20260402.batch.3",
    "total_steps/close_the_flap.all.model_new.85s.20260402.batch.4",
    "total_steps/close_the_flap.all.pen.85s.20260402.batch.1",
    "total_steps/close_the_flap.all.pen.85s.20260402.batch.2",
    # "total_steps/close_the_flap.all.pen.85s.20260402.batch.3",
    "total_steps/close_the_flap.all.pen.85s.20260402.batch.4",
    "total_steps/close_the_flap.all.pen.85s.20260402.batch.5",
]

# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 12
TRAIN_STEPS = 10000
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

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
                "returns_norm_strategy": "segmented",
                "segment_values": [0.3, 0.7],
                "segment_values_file": "segment_values.json",
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
    weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_WEIGHT_PATH),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=10,
    save_interval=5000,
    keep_period=5000,
    overwrite=False,
    resume=True,
)
