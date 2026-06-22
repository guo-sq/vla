"""RECAP training verification config for clothes-folding scenario.

Uses indicator path (AddAdvantageToPrompt text injection), NOT optimality embedding path.
Two RECAP paths are independent and must NOT be used together:
  - optimality: enable_recap=True, learned embedding in prefix
  - indicator: IndicatorPreprocessor + AddAdvantageToPrompt in prompt text
"""

import dataclasses
import pathlib

from openpi.configs.base import *
import openpi.training.utils as _utils
from openpi.configs.robot_cfg.base import *
import openpi.transforms as _transforms
from openpi.training.frame_attributes_preprocessors import IndicatorPreprocessor
from openpi.configs.dataset_config.bipiper_clothes_rl import ROOT_DIR, REPO_ID_1215_0227 as REPO_ID

DEBUG_MODE = False

TASK_NAME = "pi06_recap_indicator_bipiper_clothes_1215_0227"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20251215"

ACTION_SEQUENCE_KEYS = ("action",)

DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]
ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)
UNIFY_ACTION_SPACE = False

BATCH_SIZE = 64
NUM_WORKERS = 12

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1

FRAME_ATTRS_PREPROCESSORS = [
    IndicatorPreprocessor(
        indicator_dir="indicators",
        auto_discover_chunks=True,
        validate_episode_count=True,
    ),
]


class IndicatorRecapDataConfig(Gr00tLerobotDataConfig):
    """Extends Gr00tLerobotDataConfig to inject AddAdvantageToPrompt into data_transforms.

    AddAdvantageToPrompt reads 'indicator' key (loaded by IndicatorPreprocessor)
    and inserts "Advantage: positive/negative" text into prompt before tokenization.
    """

    def create(self, assets_dirs, model_config, **kwargs):
        data_config = super().create(assets_dirs, model_config, **kwargs)
        # Insert AddAdvantageToPrompt into data_transforms (before model_transforms/TokenizePrompt)
        new_data_transforms = data_config.data_transforms.push(
            inputs=[
                _transforms.AddAdvantageToPrompt(
                    dropout_rate=data_config.advantage_dropout_rate,
                    training=True,
                )
            ],
        )
        return dataclasses.replace(data_config, data_transforms=new_data_transforms)


cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        pi05=True,
        # NOTE: enable_recap=False — using indicator text injection path, not optimality embedding
        enable_recap=False,
        max_token_len=256,
    ),
    data=IndicatorRecapDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=False,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
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
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/mnt/workspace/gwl/playground/openpi_modified/checkpoints/pi05_general_cloth_basev0_3c_0206/pi05_general_cloth_basev0_3c_0206_exp0206/50000/params"
    ),
    num_train_steps=30000,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=10,
    save_interval=5000,
    keep_period=5000,
    overwrite=True,
)
