from pathlib import Path
from openpi.configs.base import *
import openpi.training.utils as _utils
import openpi.training.optimizer as _optimizer
from openpi.training.frame_attributes_preprocessors import (
    PruneHeadTailStaticValidMaskPreprocessor,
    StaticRatioSampleWeightPreprocessor,
    ValidMaskGroupParams,
    VelocityBasedStaticDetector,
)

DEBUG_MODE = False

TASK_NAME = "cfg_pi05_base_pack_socks_0106_0310_wo_pjl_trtc6"
EXP_NAME = TASK_NAME + "_exp_0311"
ASSET_ID = "202603011"
RTC_MAX_DELAY = 6

# Dataset settings
ROOT_DIR = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/pack_socks"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

# TrainConfig
BATCH_SIZE = 512
NUM_WORKERS = 16
TRAIN_STEPS = 40000
WARMUP_STEPS = 2000
PEAK_LR = 6e-5
DECAY_LR = 6e-6

SAVE_INTERVAL = 5000
KEEP_PERIOD = 10000

PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)

# 从 txt 文件加载 REPO_ID，每行一个 repo 文件夹名，支持 # 注释和空行
REPO_ID_LIST_PATH = Path("src/openpi/configs/repo_id_lists/pack_socks/repo_id_0106_0310_wo_pjl.txt")
assert REPO_ID_LIST_PATH.exists(), f"REPO_ID list file not found: {REPO_ID_LIST_PATH}"
REPO_ID = [
    line.strip()
    for line in REPO_ID_LIST_PATH.read_text(encoding="utf-8").splitlines()
    if (s := line.strip()) and not s.startswith("#")
]

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 2
    REPO_ID = REPO_ID[:1]

FRAME_ATTRS_PREPROCESSORS = [
    VelocityBasedStaticDetector(
        fps=30,
        joint_velocity_threshold=0.15,
        gripper_velocity_threshold=0.2,
        smoothing_half_window=2,
    ),
    PruneHeadTailStaticValidMaskPreprocessor(
        fps=30,
        groups=[
            ValidMaskGroupParams(
                name="static_action",
                match=["*static*"],
                skip_static_processing=True,
            ),
            ValidMaskGroupParams(name="default", head_margin_s=0.3, trailing_margin_s=0.0),
            ValidMaskGroupParams(
                name="recover", match=["*recover*"], head_margin_s=0.0, trailing_margin_s=0.0
            ),
            ValidMaskGroupParams(
                name="takeover",
                match=["*takeover*"],
                head_margin_s=0.0, 
                trailing_margin_s=0.0,
                use_human_intervention_mask=True,
            ),
            ValidMaskGroupParams(
                name="need_reset",
                match=["*s2.*", "*s2_long.*", "*s2_move.", "*pair.*", "*pair_not_found*"],
                trailing_margin_s=0.5,
            ),
        ],
    ),
    StaticRatioSampleWeightPreprocessor(ratio_thre=0.5),
]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        pi05=True,
        enable_rl_value_head=False,
        disable_color_aug=True,
        max_token_len=128,
        action_horizon=30,
    ),
    rtc_max_delay=RTC_MAX_DELAY,
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=False,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
            frame_attributes_preprocessors=FRAME_ATTRS_PREPROCESSORS,
        ),
        extra_delta_transform=False,
        use_delta_joint_actions=True,
        delta_action_mask_indices=DELTA_ACTION_MASK_INDICES,
        public_dataset_camera_map=_utils.PUBLIC_DATASET_MAP,
        align_dim=ALIGN_DIM,
        target_action_dim=TARGET_ACTION_DIM,
        unify_action_space=UNIFY_ACTION_SPACE,
        enforce_segment_continuity=True,
        disable_action_padding=True,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_WEIGHT_PATH),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=100,
    # How often (in steps) to save checkpoints.
    save_interval=SAVE_INTERVAL,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=KEEP_PERIOD,
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=WARMUP_STEPS,
        peak_lr=PEAK_LR,
        decay_steps=TRAIN_STEPS,
        decay_lr=DECAY_LR,
    ),
    overwrite=True,
)
