"""Config for all_public_dataset model (2603).

Training config for pi0.5 28-dim model on all public datasets.
Includes TestConfig for openloop evaluation with 1/100 repo_id sampling.
"""

# ruff: noqa: N999

from openpi.configs.dataset_config.aloha import REPO_ID as ALOHA_REPO_IDS
from openpi.configs.dataset_config.bridge_v2 import REPO_ID as BRIDGE_V2_REPO_IDS
from openpi.configs.robot_cfg.base import RobotAlignInfo
from openpi.models.pi0_config import Pi0Config
from openpi.training.base_cfg import AssetsConfig
from openpi.training.base_cfg import DataConfig
from openpi.training.base_cfg import Gr00tLerobotDataConfig
from openpi.training.base_cfg import TestConfig
from openpi.training.config import TrainConfig
from openpi.training.utils import PUBLIC_DATASET_MAP
from openpi.training.utils import ROBOT_ALIGN_INFO

# Dataset settings
ROOT_DIR = "/mnt/"
ASSET_ID = "20260215"
ACTION_SEQUENCE_KEYS = (
    "action",
    "action.left_gripper",
    "action.right_gripper",
    "action.left_arm",
    "action.right_arm",
    "actions.joint.position",
    "actions.effector.position",
    "action.arm.position",
    "action.effector.position",
    "eef_sim_pose_action",
)
UNIFY_ACTION_SPACE = True
DELTA_ACTION_MASK_INDICES = [7, -1, 7, -1, 12]

ALIGN_DIM = sum([abs(x) for x in DELTA_ACTION_MASK_INDICES])
TARGET_ACTION_DIM = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

# Sample repo_ids for evaluation (1/100 rate, at least 1)
EVAL_SAMPLE_RATE = 100

ALL_REPO_IDS = ALOHA_REPO_IDS + BRIDGE_V2_REPO_IDS


# Sample repo_ids for evaluation
def _sample_repo_ids(repo_ids: list, sample_rate: int = 100) -> list:
    """Sample 1/sample_rate of repo_ids, ensuring at least 1."""
    if not repo_ids:
        return []
    sampled = repo_ids[::sample_rate]
    return sampled if sampled else [repo_ids[0]]


EVAL_REPO_IDS = _sample_repo_ids(ALL_REPO_IDS, EVAL_SAMPLE_RATE)

cfg = TrainConfig(
    name="cfg_pi0.5_28_dim.all_public_datasets",
    exp_name="cfg_pi0.5_28_dim.all_public_datasets_exp",
    model=Pi0Config(
        pi05=True,
        max_token_len=256,
    ),
    ema_decay=None,
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=ALL_REPO_IDS,
        base_config=DataConfig(
            prompt_from_episode=True,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
        ),
        extra_delta_transform=False,
        use_delta_joint_actions=True,
        delta_action_mask_indices=DELTA_ACTION_MASK_INDICES,
        public_dataset_camera_map=PUBLIC_DATASET_MAP,
        align_dim=ALIGN_DIM,
        use_semantic_delta_actions=True,
        delta_wrap_eef_angles=True,
        tolerance_s=0.02,
        unify_action_space=UNIFY_ACTION_SPACE,
        target_action_dim=TARGET_ACTION_DIM,
        robot_align_info=RobotAlignInfo(robot_align_info=ROBOT_ALIGN_INFO),
        frame_skip=1,
        lazy_load=True,
    ),
    batch_size=64,
    num_workers=2,
)

# TestConfig for openloop evaluation
test_cfg = TestConfig(
    checkpoint_dir="/mnt/oss_models/models_deploy/2603/all_public_dataset/cfg_pi0.5_28_dim.all_public_datasets/cfg_pi0.5_28_dim.all_public_datasets_exp_0216/99999",
    dataset_root=ROOT_DIR,
    config="src/openpi/configs/cfg_2603/cfg_pi0.5_28_dim.all_public_datasets.py",
    repo_id=EVAL_REPO_IDS[0] if EVAL_REPO_IDS else None,
    num_batches=10,
    batch_size=64,
    num_workers=2,
    eval_split="val",
    vis_dir="/mnt/oss_models/models_deploy/2603/all_public_dataset/openloop_eval_vis",
)
