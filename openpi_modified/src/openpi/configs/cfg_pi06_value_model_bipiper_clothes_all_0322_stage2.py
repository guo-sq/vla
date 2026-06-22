from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *
from openpi.training.frame_attributes_preprocessors import (
    ValuePredictionPreprocessor,
)
from openpi.configs.dataset_config.bipiper_clothes_rl import ROOT_DIR, REPO_ID_ALL_0322 as REPO_ID

DEBUG_MODE = False

TASK_NAME = "pi06_value_model_bipiper_clothes_t5gemma270M_all_0322_stage2"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260323"

FRAME_ATTRS_PREPROCESSORS = [
    ValuePredictionPreprocessor(
        value_pred_dir="value_pred",
        auto_discover_chunks=True,
        validate_episode_count=True,
    ),
]

ACTION_SEQUENCE_KEYS = ("action",)

DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]
ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
UNIFY_ACTION_SPACE = False

# TrainConfig
BATCH_SIZE = 1024
NUM_WORKERS = 12
TRAIN_STEPS = 30000

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

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
        value_bins=1,
        value_range=(-1.0, 0.0),
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
                "returns_norm_strategy": "per_task",
                "returns_norm_percentile": 0.9,
                "returns_norm_length": None,
                "failure_decrease_threshold": 0.0,
            },
            frame_attributes_preprocessors=FRAME_ATTRS_PREPROCESSORS,
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
    weight_loader=weight_loaders.T5Gemma2EncoderWeightLoader(
        params_path="/mnt/model/t5gemma2_encoder_openpi.npz",
        checkpoint_image_size=(896, 896),
        target_image_size=(224, 224),
    ),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=100,
    save_interval=1000,
    keep_period=5000,
)
