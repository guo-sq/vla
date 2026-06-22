"""Config for seatbelt model (2603).

Training config for pi05 model on seatbelt dataset.
Includes TestConfig for openloop evaluation with 1/100 repo_id sampling.
"""

from openpi.models.pi0_config import Pi0Config
from openpi.training.base_cfg import AssetsConfig
from openpi.training.base_cfg import DataConfig
from openpi.training.base_cfg import Gr00tLerobotDataConfig
from openpi.training.base_cfg import TestConfig
from openpi.training.config import TrainConfig
from openpi.training.utils import PUBLIC_DATASET_MAP

TASK_NAME = "pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery"
ASSET_ID = "20260312"

# Dataset settings
ROOT_DIR = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum([abs(x) for x in DELTA_ACTION_MASK_INDICES])
TARGET_ACTION_DIM = range(14)

# Sample repo_ids for evaluation (1/100 rate, at least 1)
EVAL_SAMPLE_RATE = 100

# Full repo_ids for training
ALL_REPO_IDS = [
    "seatbelt.single.hang.baichenglong.20260205.batch.5",
    "seatbelt.single.hang.baichenglong.20260205.batch.6",
    "seatbelt.single.hang.baichenglong.20260205.batch.7",
    "seatbelt.single.hang.baichenglong.20260205.batch.8",
    "seatbelt.single.hang.baichenglong.20260205.batch.9",
    "seatbelt.single.hang.baichenglong.20260205.batch.10",
    "seatbelt.single.hang.baichenglong.20260205.batch.11",
    "seatbelt.single.hang.baichenglong.20260205.batch.12",
    "seatbelt.single.hang.baichenglong.20260205.batch.13",
    "seatbelt.single.hang.baichenglong.20260205.batch.14",
    "seatbelt.single.hang.baichenglong.20260205.batch.15",
    "seatbelt.single.hang.zhangyu.20260206.batch.1",
    "seatbelt.single.hang.zhangyu.20260206.batch.2",
    "seatbelt.single.hang.zhangyu.20260206.batch.3",
    "seatbelt.single.hang.zhangyu.20260206.batch.4",
    "seatbelt.single.hang.zhangyu.20260206.batch.5",
    "seatbelt.single.hang.zhangyu.20260206.batch.6",
    "seatbelt.single.hang.zhangyu.20260206.batch.7",
    "seatbelt.single.hang.zhangyu.20260206.batch.8",
    "seatbelt.single.hang.baichenglong.20260206.batch.9",
    "seatbelt.single.hang.baichenglong.20260206.batch.10",
    "seatbelt.single.hang.baichenglong.20260206.batch.11",
    "seatbelt.single.hang.baichenglong.20260206.batch.12",
    "seatbelt.single.hang.baichenglong.20260206.batch.13",
    "seatbelt.single.hang.baichenglong.20260206.batch.14",
    "seatbelt.single.hang.baichenglong.20260206.batch.15",
    "seatbelt.single.hang.baichenglong.20260206.batch.16",
    "seatbelt.single.hang.baichenglong.20260206.batch.17",
    "seatbelt.single.hang.baichenglong.20260206.batch.18",
    "seatbelt.single.hang.baichenglong.20260206.batch.19",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.1",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.2",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.3",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.4",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.5",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.6",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.7",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.8",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.9",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.10",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.11",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.12",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.13",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.14",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.15",
    "seatbelt.single.take_off_move.panjinlong.20260228.batch.16",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.1",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.2",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.3",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.4",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.5",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.6",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.7",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.8",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.9",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.10",
]


# Sample repo_ids for evaluation
def _sample_repo_ids(repo_ids: list, sample_rate: int = 100) -> list:
    """Sample 1/sample_rate of repo_ids, ensuring at least 1."""
    if not repo_ids:
        return []
    sampled = repo_ids[::sample_rate]
    return sampled if sampled else [repo_ids[0]]


EVAL_REPO_IDS = _sample_repo_ids(ALL_REPO_IDS, EVAL_SAMPLE_RATE)

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=TASK_NAME + "_exp",
    model=Pi0Config(
        pi05=True,
        max_token_len=128,
        action_horizon=50,
    ),
    rtc_max_delay=10,
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=ALL_REPO_IDS,
        base_config=DataConfig(
            prompt_from_episode=False,
            action_sequence_keys=ACTION_SEQUENCE_KEYS,
        ),
        extra_delta_transform=False,
        use_delta_joint_actions=True,
        delta_action_mask_indices=DELTA_ACTION_MASK_INDICES,
        public_dataset_camera_map=PUBLIC_DATASET_MAP,
        align_dim=ALIGN_DIM,
        target_action_dim=TARGET_ACTION_DIM,
        unify_action_space=UNIFY_ACTION_SPACE,
    ),
    batch_size=64,
    num_workers=2,
)

# TestConfig for openloop evaluation
test_cfg = TestConfig(
    checkpoint_dir="/mnt/oss_models/models_deploy/2603/seatbelt/pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery/pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery_exp/84999",
    dataset_root=ROOT_DIR,
    config="src/openpi/configs/cfg_2603/cfg_pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery.py",
    repo_id=EVAL_REPO_IDS[0] if EVAL_REPO_IDS else None,
    num_batches=10,
    batch_size=64,
    num_workers=2,
    eval_split="val",
    vis_dir="/mnt/oss_models/models_deploy/2603/seatbelt/openloop_eval_vis",
)
