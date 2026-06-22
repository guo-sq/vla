"""Config for pack_socks model (2603).

Training config for pi05 model on pack_socks dataset.
Includes TestConfig for openloop evaluation with 1/100 repo_id sampling.
"""

from openpi.configs.dataset_config.anyverse_tasks.pack_socks import tasks as PACK_SOCKS_REPO_IDS
from openpi.models.pi0_config import Pi0Config
from openpi.training.base_cfg import AssetsConfig
from openpi.training.base_cfg import DataConfig
from openpi.training.base_cfg import Gr00tLerobotDataConfig
from openpi.training.base_cfg import TestConfig
from openpi.training.config import TrainConfig
from openpi.training.utils import PUBLIC_DATASET_MAP

TASK_NAME = "cfg_pi05_base_pack_socks_data_pure_recover_arx4_bs256_0318_more_upsample_recover"
ASSET_ID = "20260318"
RTC_MAX_DELAY = 6

# Dataset settings
ROOT_DIR = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/pack_socks"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]

ALIGN_DIM = sum([abs(x) for x in DELTA_ACTION_MASK_INDICES])
TARGET_ACTION_DIM = range(14)

# Sample repo_ids for evaluation (1/100 rate, at least 1)
EVAL_SAMPLE_RATE = 100

ALL_REPO_IDS = list(PACK_SOCKS_REPO_IDS)

if not ALL_REPO_IDS:
    raise ValueError("PACK_SOCKS_REPO_IDS is empty. Please ensure pack_socks dataset config exports repo ids.")


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
    exp_name=TASK_NAME + "_exp_0318",
    model=Pi0Config(
        pi05=True,
        max_token_len=128,
        action_horizon=30,
    ),
    rtc_max_delay=RTC_MAX_DELAY,
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
        enforce_segment_continuity=True,
        disable_action_padding=True,
    ),
    batch_size=64,
    num_workers=2,
)

# TestConfig for openloop evaluation
test_cfg = TestConfig(
    checkpoint_dir="/mnt/oss_models/models_deploy/2603/pack_socks/cfg_pi05_base_pack_socks_data_pure_recover_arx4_bs256_0318_more_upsample_recover/cfg_pi05_base_pack_socks_data_pure_recover_arx4_bs256_0318_more_upsample_recover_exp_0318/40000",
    dataset_root=ROOT_DIR,
    config="src/openpi/configs/cfg_2603/cfg_pi05_base_pack_socks_data_pure_recover_arx4_bs256_0318_more_upsample_recover.py",
    repo_id=EVAL_REPO_IDS[0] if EVAL_REPO_IDS else None,
    num_batches=10,
    batch_size=64,
    num_workers=2,
    eval_split="val",
    vis_dir="/mnt/oss_models/models_deploy/2603/pack_socks/openloop_eval_vis",
)
