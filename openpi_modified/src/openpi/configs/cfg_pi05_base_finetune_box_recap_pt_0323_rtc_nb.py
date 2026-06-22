from openpi.configs.base import *
from openpi.configs.robot_cfg.base import *
from openpi.training.frame_attributes_preprocessors import OptimalityProcessor
from openpi.training.frame_attributes_preprocessors import TemporalWeightProcessor
import openpi.training.utils as _utils

DEBUG_MODE = False

TASK_NAME = "pi05_base_finetune_box_recap_pt_0323_rtc_nb"
EXP_NAME = TASK_NAME + "_exp"
ASSET_ID = "20260123"

# Dataset settings
ROOT_DIR = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/"
ACTION_SEQUENCE_KEYS = ("action",)
UNIFY_ACTION_SPACE = False
DELTA_ACTION_MASK_INDICES = [6, -1, 6, -1]
RTC_MAX_DELAY = 20

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = range(14)

GOOD_REPO_ID_WITH_WEIGHT = [
    # 最新流程
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.1", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.2", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.3", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.4", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.5", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.6", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.7", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.8", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.9", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.10", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.11", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.12", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260303.batch.13", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.10", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.11", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.12", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.13", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.2", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.4", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.5", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.6", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.7", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.8", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260304.batch.9", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260306.batch.8", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260306.batch.9", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.1", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.2", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.3", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.4", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.5", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.6", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.7", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.8", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/total_steps/fold_box_scratch.all.102s.20260309.batch.9", [(0, 30, 4, 30.0)]),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260310.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260310.batch.2",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260310.batch.3",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260311.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260316.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260316.batch.2",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260316.batch.3",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260316.batch.4",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260317.batch.3",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.2",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.3",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.4",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.5",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.6",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.7",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.8",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.9",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.10",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.11",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260319.batch.12",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260323.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.all.105s.20260323.batch.2",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch_green.all.105s.20260311.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch_purple.all.105s.20260311.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch_silver.all.105s.20260311.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch_yellow.all.105s.20260311.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.purple.105s.20260313.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.silver.105s.20260313.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.yellow.105s.20260313.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.yellow.105s.20260313.batch.2",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch_green.105s.20260313.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.green.105s.20260318.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.purple.105s.20260318.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.silver.105s.20260318.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    (
        "fold_box_from_scratch/total_steps/fold_box_scratch.yellow.105s.20260318.batch.1",
        [(0, 30, 4, 30.0), (80, 100, 2, 30.0)],
    ),
    # 前半流程
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.1", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.2", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.3", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.4", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.5", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.6", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.7", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.8", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260316.batch.9", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260317.batch.1", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260317.batch.2", [(0, 30, 4, 30.0)]),
    ("fold_box_from_scratch/first_half_steps/fold_box_scratch.all.31s.20260317.batch.3", [(0, 30, 4, 30.0)]),
    # 后半流程
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.67s.20260305.batch.1", []), # 动作太快
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.67s.20260305.batch.2", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260305.batch.1", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260305.batch.2", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260305.batch.3", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260306.batch.1", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260306.batch.2", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260306.batch.3", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260306.batch.4", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260306.batch.5", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.79s.20260306.batch.6", []),
    ("fold_box_from_scratch/second_half_steps/fold_box_scratch.all.85s.20260323.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_green.all.82s.20260311.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_purple.all.82s.20260311.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_silver.all.82s.20260311.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_silver.all.82s.20260311.batch.2", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_green.all.82s.20260312.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_green.all.82s.20260312.batch.2", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_purple.all.82s.20260312.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_purple.all.82s.20260312.batch.2", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_silver.all.82s.20260312.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_silver.all.82s.20260312.batch.2", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch_yellow.all.82s.20260312.batch.3", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.purple.82s.20260313.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.silve.82s.20260313.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.silve.82s.20260313.batch.2", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.yellow.82s.20260313.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.yellow.82s.20260313.batch.2", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.purple.82s.20260317.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.purple.82s.20260317.batch.2", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.purple.82s.20260317.batch.3", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.purple.82s.20260317.batch.4", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.purple.82s.20260317.batch.5", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.silver.82s.20260317.batch.1", []),
    # ("fold_box_from_scratch/second_half_steps/fold_box_scratch.green.85s.20260323.batch.1", []),
    # step2-8 夹防尘翼
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_2.3s.20260314.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_2.5s.20260304.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_2.5s.20260304.batch.2", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_2.5s.20260312.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_2.5s.20260314.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_2.5s.20260314.batch.2", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_2.5s.20260317.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_3.3s.20260314.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_3.5s.20260312.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_3.5s.20260304.batch.4", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_3.5s.20260314.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_3.5s.20260314.batch.2", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_4.3s.20260314.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_4.5s.20260304.batch.3", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_4.5s.20260312.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_5.5s.20260305.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_5.5s.20260305.batch.2", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_5.5s.20260305.batch.3", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_5.5s.20260305.batch.4", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.3s.20260314.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.5s.20260304.batch.2", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.5s.20260304.batch.5", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.5s.20260305.batch.5", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.5s.20260306.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.5s.20260306.batch.2", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.5s.20260306.batch.3", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.5s.20260306.batch.4", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_6.5s.20260306.batch.5", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_7.3s.20260314.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_7.5s.20260304.batch.7", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_7.5s.20260304.batch.8", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_7.5s.20260309.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_7.5s.20260309.batch.2", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_7.5s.20260312.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_7.5s.20260314.batch.2", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_8.3s.20260314.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_8.5s.20260312.batch.1", [(0, 10, 10, 30.0)]),
    ("fold_box_from_scratch/recover2-8/fold_box_scratch.recover_8.5s.20260304.batch.6", [(0, 10, 10, 30.0)]),
    # step9
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.2s.20260318.batch.2", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.2s.20260318.batch.3", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.2s.20260318.batch.4", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.3s.20260318.batch.1", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.5s.20260317.batch.1", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.5s.20260318.batch.5", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.5s.20260318.batch.6", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.5s.20260318.batch.7", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover8.2s.20260319.batch.1", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover8.2s.20260319.batch.2", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.2s.20260319.batch.1", [(0, 10, 2, 30.0)]),
    ("fold_box_from_scratch/recover9/fold_box_scratch.recover9.2s.20260319.batch.2", [(0, 10, 2, 30.0)]),
    # step 13-14 reinforce
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_13.3s.20260314.batch.1", [(0, 10, 3, 30.0)]),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_14.3s.20260314.batch.1", [(0, 10, 3, 30.0)]),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_14.5s.20260320.batch.1", [(0, 10, 3, 30.0)]),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_13.5s.20260320.batch.1", [(0, 10, 3, 30.0)]),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_14.5s.20260320.batch.2", [(0, 10, 3, 30.0)]),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_13.5s.20260320.batch.2", [(0, 10, 3, 30.0)]),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_14.5s.20260320.batch.3", [(0, 10, 3, 30.0)]),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_13.5s.20260320.batch.3", [(0, 10, 3, 30.0)]),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.green.rein_13.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.green.rein_14.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.purple.rein_13.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.purple_rein_14.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.silver.rein_13.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.silver.rein_14.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.yellow_rein_13.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.yellow_rein_14.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein.31.3s.20260314.batch.1", []),
    ("fold_box_from_scratch/step_13_14/fold_box_scratch.rein_38.3s.20260314.batch.1", []),
    # step 24-25 reinforce
    ("fold_box_from_scratch/step_24_25/fold_box_scratch.new24.5s.20260317.batch.1", []),
    ("fold_box_from_scratch/step_24_25/fold_box_scratch.new24.5s.20260317.batch.2", []),
    ("fold_box_from_scratch/step_24_25/fold_box_scratch.new24.5s.20260317.batch.3", []),
    ("fold_box_from_scratch/step_24_25/fold_box_scratch.new24.5s.20260317.batch.4", []),
    ("fold_box_from_scratch/step_24_25/fold_box_scratch.new25.15s.20260317.batch.1", []),
    ("fold_box_from_scratch/step_24_25/fold_box_scratch.new25.15s.20260317.batch.2", []),
    ("fold_box_from_scratch/step_24_25/fold_box_scratch.new25.15s.20260317.batch.3", []),
    ("fold_box_from_scratch/step_24_25/fold_box_scratch.new25.15s.20260317.batch.4", []),
    # step 31 recover
    ("fold_box_from_scratch/recover31/fold_box_scratch.recover_31.10s.20260311.batch.1", []),
    ("fold_box_from_scratch/recover31/fold_box_scratch.recover_31.5s.20260311.batch.2", []),
    ("fold_box_from_scratch/recover31/fold_box_scratch.recover_31.5s.20260311.batch.3", []),
    ("fold_box_from_scratch/recover31/fold_box_scratch.recover_31.5s.20260313.batch.1", []),
    ("fold_box_from_scratch/recover31/fold_box_scratch.recover_31.5s.20260313.batch.2", []),
    # step 37 recover
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_1.5s.20260317.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_1.5s.20260317.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_1.5s.20260317.batch.3", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_1.5s.20260318.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_1.5s.20260318.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_1.5s.20260318.batch.3", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_1.5s.20260318.batch.4", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_1.5s.20260318.batch.5", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_2.3s.20260317.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_2.3s.20260318.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_2.3s.20260318.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_2.3s.20260318.batch.3", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_2.3s.20260318.batch.4", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_2.3s.20260318.batch.5", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_2.5s.20260317.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_2.5s.20260304.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_3.3s.20260317.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_3.3s.20260317.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_3.5s.20260318.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_3.5s.20260318.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_3.5s.20260318.batch.3", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_3.5s.20260318.batch.4", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_3.5s.20260318.batch.5", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_4.5s.20260317.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_4.5s.20260318.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_4.5s.20260318.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_4.5s.20260318.batch.3", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_4.5s.20260318.batch.4", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover37_4.5s.20260318.batch.5", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover_37.5s.20260311.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover_37.5s.20260313.batch.1", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover_37.5s.20260313.batch.2", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover_37.5s.20260313.batch.3", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover_37.5s.20260313.batch.4", []),
    ("fold_box_from_scratch/recover37/fold_box_scratch.recover_37.5s.20260313.batch.5", []),
]

BAD_REPO_ID_WITH_WEIGHT = [
    # 失败数据
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad2-8.10s.20260310.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad2-8.10s.20260310.batch.2", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad2-8.10s.20260310.batch.3", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad2-8.10s.20260311.batch.0", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad2-8.10s.20260311.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad2-8.10s.20260311.batch.2", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad2-8.10s.20260311.batch.3", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad2-8.10s.20260311.batch.4", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad23.10s.20260311.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad23.10s.20260311.batch.2", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad31.10s.20260311.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad31.10s.20260311.batch.2", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad37.10s.20260311.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad37.10s.20260311.batch.2", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad37_1.5s.20260317.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad37_2.5s.20260317.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad37_3.5s.20260317.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad37_4.5s.20260317.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad8.5s.20260318.batch.1", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad8.5s.20260318.batch.2", []),
    # ("fold_box_from_scratch/bad/fold_box_scratch.bad9.5s.20260318.batch.1", []),
]

REPO_ID_WITH_WEIGHT = GOOD_REPO_ID_WITH_WEIGHT + BAD_REPO_ID_WITH_WEIGHT

REPO_ID = [item[0] for item in REPO_ID_WITH_WEIGHT]

# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 8
TRAIN_STEPS = 50000
PRETRAINED_WEIGHT_PATH = (
    "checkpoints/pi05_base_finetune_box_recap_cpt_0320/pi05_base_finetune_box_recap_cpt_0320_exp.0313_1947/79999/params"
)

FRAME_ATTRS_PREPROCESSORS = [
    TemporalWeightProcessor(repo_id_with_weights=REPO_ID_WITH_WEIGHT),
    OptimalityProcessor(bad_repo_id_with_weight=BAD_REPO_ID_WITH_WEIGHT),
]

if DEBUG_MODE:
    BATCH_SIZE = 2
    NUM_WORKERS = 1
    REPO_ID = REPO_ID[:1]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(pi05=True, enable_recap=True, max_token_len=128),
    data=Gr00tLerobotDataConfig(
        assets=AssetsConfig(asset_id=ASSET_ID),
        root_dir=ROOT_DIR,
        repo_id=REPO_ID,
        base_config=DataConfig(
            prompt_from_episode=True,
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
    ),
    rtc_max_delay=RTC_MAX_DELAY,
    weight_loader=weight_loaders.CheckpointWeightLoader(PRETRAINED_WEIGHT_PATH),
    num_train_steps=TRAIN_STEPS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    log_interval=100,
    # How often (in steps) to save checkpoints.
    save_interval=10000,
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period=10000,
    overwrite=False,
    resume=True,
)
