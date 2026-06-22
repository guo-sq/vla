from openpi.configs.base import *
from openpi.configs.robot_cfg.base import *
from openpi.training.frame_attributes_preprocessors import OptimalityProcessor
from openpi.training.frame_attributes_preprocessors import StaleHeadFramesValidMaskPreprocessor
from openpi.training.frame_attributes_preprocessors import TemporalWeightProcessor
import openpi.training.utils as _utils

DEBUG_MODE = False

TASK_NAME = "pi05_base_finetune_box_recap_cpt_0419_close"
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
    # 盖上盖子
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.90s.20260326.batch.1", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.2", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.3", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.4", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.5", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.6", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.7", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.8", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.9", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.95s.20260326.batch.10", [(18, 39, 0, 30.0)]),
    # 0327 新增数据
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.1", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.2", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.3", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.4", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.5", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.6", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.7", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.8", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.9", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.10", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.11", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.12", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.13", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.14", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.15", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.red_shirt.85s.20260327.batch.16", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.keychain.85s.20260327.batch.17", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.keychain.85s.20260327.batch.18", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.keychain.85s.20260327.batch.19", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.keychain.85s.20260327.batch.20", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.keychain.85s.20260327.batch.22", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.keychain.85s.20260327.batch.23", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.keychain.85s.20260327.batch.24", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.keychain.85s.20260327.batch.26", [(18, 39, 0, 30.0)]),
    # 0330 新增数据
    ("close_the_flap/total_steps/close_the_flap.all.black_obj.85s.20260330.batch.1", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.black_obj.85s.20260330.batch.2", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.black_obj.85s.20260330.batch.3", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.black_obj.85s.20260330.batch.4", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.pen.85s.20260330.batch.1", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.pen.85s.20260330.batch.2", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.pen.85s.20260330.batch.3", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.pen.85s.20260330.batch.4", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.pen.85s.20260330.batch.5", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260330.batch.1", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260330.batch.2", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260330.batch.3", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.whiteT.85s.20260330.batch.1", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.whiteT.85s.20260330.batch.2", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.whiteT.85s.20260330.batch.3", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.whiteT.85s.20260330.batch.4", [(18, 39, 0, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.whiteT.85s.20260330.batch.5", [(18, 39, 0, 30.0)]),
    # 0331 新增数据
    (
        "close_the_flap/total_steps/close_the_flap.all.Cylinder.85s.20260331.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Hook.85s.20260331.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Hook.85s.20260331.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Hook.85s.20260331.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Hook.85s.20260331.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Hook.85s.20260331.batch.5",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Hook.85s.20260331.batch.6",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.screwdriver.85s.20260331.batch.5",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.5",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260331.batch.6",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    # 0401 新增数据
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260401.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260401.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260401.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260401.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.round.85s.20260401.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.round.85s.20260401.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.round.85s.20260401.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.round.85s.20260401.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.round.85s.20260401.batch.5",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260401.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec.85s.20260401.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.usb_typec_new.85s.20260401.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (38, 85, 2, 30.0)],
    ),
    # 0401 rein21-24 数据
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.usb_typec_new.7s.20260401.batch.1", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.usb_typec_new.7s.20260401.batch.2", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.usb_typec_new.7s.20260401.batch.3", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.usb_typec_new.7s.20260401.batch.4", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.usb_typec_new.7s.20260401.batch.5", [(0, 7, 3, 30.0)]),
    # 0402 新增数据
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260402.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260402.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260402.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260402.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260402.batch.5",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.model_new.85s.20260402.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.model_new.85s.20260402.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.model_new.85s.20260402.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.model_new.85s.20260402.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260402.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260402.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260402.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260402.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.pen.85s.20260402.batch.5",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    # 0402 rein21-24 数据
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.1", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.2", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.3", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.4", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.5", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.6", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.7", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.8", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.9", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.10", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.11", [(0, 7, 3, 30.0)]),
    ("close_the_flap/rein21-24/close_the_flap.rein21-24.model_new.7s.20260402.batch.12", [(0, 7, 3, 30.0)]),
    # 0403 新增数据
    (
        "close_the_flap/total_steps/close_the_flap.all.Cylinder.85s.20260403.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Cylinder.85s.20260403.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Cylinder.85s.20260403.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Cylinder.85s.20260403.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Cylinder.85s.20260403.batch.5",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260403.batch.1",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260403.batch.2",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260403.batch.3",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260403.batch.4",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260403.batch.5",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.all.Sticker.85s.20260403.batch.6",
        [(0, 22, 2, 30.0), (22, 38, 0, 30.0), (35, 85, 2, 30.0)],
    ),
    # 0403 recover14 数据
    ("close_the_flap/recover14/close_the_flap.recover14.Any.5s.20260403.batch.1", [(0, 3, 3, 30.0), (3, 6, 0, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14.Any.5s.20260403.batch.2", [(0, 3, 3, 30.0), (3, 6, 0, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14.Cup.5s.20260403.batch.1", [(0, 3, 3, 30.0), (3, 6, 0, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14.Cup.5s.20260403.batch.2", [(0, 3, 3, 30.0), (3, 6, 0, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14.Cup.5s.20260403.batch.3", [(0, 3, 3, 30.0), (3, 6, 0, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14.Cup.5s.20260403.batch.4", [(0, 3, 3, 30.0), (3, 6, 0, 30.0)]),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Cylinder.5s.20260403.batch.1",
        [(0, 3, 3, 30.0), (3, 6, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Cylinder.5s.20260403.batch.2",
        [(0, 3, 3, 30.0), (3, 6, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Cylinder.5s.20260403.batch.3",
        [(0, 3, 3, 30.0), (3, 6, 0, 30.0)],
    ),
    # 0403 recover23 数据
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260403.batch.1", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260403.batch.2", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260403.batch.3", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260403.batch.4", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_2.Any.3s.20260403.batch.1", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    # 0407 新增数据
    (
        "close_the_flap/total_steps/close_the_flap.Mold.85s.20260407.batch.6",
        [(0, 25, 2, 30.0), (25, 30, 1, 30.0), (30, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.Mold.85s.20260407.batch.7",
        [(0, 25, 2, 30.0), (25, 30, 1, 30.0), (30, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.Stuff.85s.20260407.batch.1",
        [(0, 25, 2, 30.0), (25, 30, 1, 30.0), (30, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.Stuff.85s.20260407.batch.2",
        [(0, 25, 2, 30.0), (25, 30, 1, 30.0), (30, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.Stuff.85s.20260407.batch.3",
        [(0, 25, 2, 30.0), (25, 30, 1, 30.0), (30, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.Stuff.85s.20260407.batch.4",
        [(0, 25, 2, 30.0), (25, 30, 1, 30.0), (30, 85, 2, 30.0)],
    ),
    (
        "close_the_flap/total_steps/close_the_flap.Stuff.85s.20260407.batch.5",
        [(0, 25, 2, 30.0), (25, 30, 1, 30.0), (30, 85, 2, 30.0)],
    ),
    # 0407 recover14 数据
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.13s.20260407.batch.1",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260407.batch.2",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260407.batch.3",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260407.batch.4",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260407.batch.5",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    # 0407 recover23 数据
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260407.batch.1", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260407.batch.2", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260407.batch.3", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260407.batch.4", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_2.Any.3s.20260407.batch.1", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_2.Any.3s.20260407.batch.2", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_2.Any.3s.20260407.batch.3", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    # 0408 recover14 数据
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260408.batch.1",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260408.batch.2",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260408.batch.3",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260408.batch.4",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260408.batch.5",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260408.batch.6",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260408.batch.7",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260408.batch.8",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    # 0408 recover23 数据
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260408.batch.1", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260408.batch.2", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_1.Any.3s.20260408.batch.3", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_2.Any.3s.20260408.batch.1", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_2.Any.3s.20260408.batch.2", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    ("close_the_flap/recover23/close_the_flap.recover23_2.Any.3s.20260408.batch.3", [(0, 1, 3, 30.0), (1, 3, 0, 30.0)]),
    # 0409 recover14
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.1",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.2",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.3",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.4",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.5",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.6",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.7",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.8",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.9",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.10",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    (
        "close_the_flap/recover14/close_the_flap.recover14.Any.14s.20260409.batch.11",
        [(0, 12, 3, 30.0), (12, 14, 0, 30.0)],
    ),
    ("close_the_flap/recover14/close_the_flap.recover14_2.Any.16s.20260409.batch.1", [(0, 16, 3, 30.0)]),
    # recover20
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260409.batch.1", [(0, 8, 2, 30.0)]),
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260409.batch.2", [(0, 8, 2, 30.0)]),
    # ("close_the_flap/recover20/close_the_flap.recover20_1.Any.3s.20260409.batch.1", [(0, 8, 2, 30.0)]), # 需要转移一下
    # ("close_the_flap/recover20/close_the_flap.recover20_1.Any.3s.20260409.batch.2", [(0, 8, 2, 30.0)]),
    # ("close_the_flap/recover20/close_the_flap.recover20_1.Any.3s.20260409.batch.3", [(0, 8, 2, 30.0)]),
    # ("close_the_flap/recover20/close_the_flap.recover20_1.Any.3s.20260409.batch.4", [(0, 8, 2, 30.0)]),
    # ("close_the_flap/recover20/close_the_flap.recover20_2.Any.3s.20260409.batch.1", [(0, 8, 2, 30.0)]),
    # 0410 recover14 数据
    ("close_the_flap/recover14/close_the_flap.recover14_2.Any.16s.20260410.batch.1", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_2.Any.16s.20260410.batch.2", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_2.Any.16s.20260410.batch.3", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_2.Any.16s.20260410.batch.4", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_2.Any.16s.20260410.batch.5", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_2.Any.16s.20260410.batch.6", [(0, 16, 3, 30.0)]),
    # 0410 recover20 数据
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260410.batch.1", [(0, 8, 2, 30.0)]),
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260410.batch.2", [(0, 8, 2, 30.0)]),
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260410.batch.3", [(0, 8, 2, 30.0)]),
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260410.batch.4", [(0, 8, 2, 30.0)]),
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260410.batch.5", [(0, 8, 2, 30.0)]),
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260410.batch.6", [(0, 8, 2, 30.0)]),
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260410.batch.7", [(0, 8, 2, 30.0)]),
    ("close_the_flap/recover20/close_the_flap.recover20.Any.8s.20260410.batch.8", [(0, 8, 2, 30.0)]),
    # 0413 新增数据
    ("close_the_flap/total_steps/close_the_flap.all.Any.85s.20260413.batch.1", [(0, 85, 2, 30.0)]),
    # ("close_the_flap/total_steps/close_the_flap.all.Any.85s.20260413.batch.2", [(0, 85, 2, 30.0)]), # episode2是我乱录的
    ("close_the_flap/total_steps/close_the_flap.all.Any.85s.20260413.batch.4", [(0, 85, 2, 30.0)]),
    ("close_the_flap/total_steps/close_the_flap.all.Any.85s.20260413.batch.5", [(0, 85, 2, 30.0)]),
    # 0413 recover14 数据
    # ("close_the_flap/recover14/close_the_flap.recover14_3.Any.16s.20260413.batch.1", [(0, 16, 3, 30.0)]), # 做评测集
    ("close_the_flap/recover14/close_the_flap.recover14_3.Any.16s.20260413.batch.2", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_3.Any.16s.20260413.batch.3", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_3.Any.16s.20260413.batch.4", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_3.Any.16s.20260413.batch.5", [(0, 16, 3, 30.0)]),
    ("close_the_flap/recover14/close_the_flap.recover14_3.Any.16s.20260413.batch.6", [(0, 16, 3, 30.0)]),
    # 0414 first_half (shuhao)
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.1", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.2", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.3", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.4", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.5", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.6", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.7", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.8", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.9", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.10", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.11", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.12", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.13", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.14", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.15", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.16", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.17", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.18", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.19", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.20", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.21", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.22", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.23", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.24", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.25", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.26", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.27", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.28", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.29", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.30", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.shuhao.50s.20260414.batch.31", [(0, 50, 2, 30.0)]),
    # 0415 first_half (zhongyou)
    ("close_the_flap/first_half/close_the_flap.first_half.mold.zhongyou.50s.20260415.batch.1", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.zhongyou.50s.20260415.batch.2", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.zhongyou.50s.20260415.batch.3", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.zhongyou.50s.20260415.batch.4", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.zhongyou.50s.20260415.batch.5", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.zhongyou.50s.20260415.batch.6", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.zhongyou.50s.20260415.batch.7", [(0, 50, 2, 30.0)]),
    ("close_the_flap/first_half/close_the_flap.first_half.mold.zhongyou.50s.20260415.batch.9", [(0, 50, 2, 30.0)]),
    # 0416 first_half (zhongyou)
    (
        "close_the_flap/first_half/close_the_flap.first_half.screwdriver.zhongyou.20s.20260416.batch.1",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/first_half/close_the_flap.first_half.screwdriver.zhongyou.20s.20260416.batch.2",
        [(0, 20, 2, 30.0)],
    ),
    # 0416 second_half (zhongyou) # review了一下，不太好
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.20s.20260416.batch.3", []),
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.20s.20260416.batch.4", []),
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.20s.20260416.batch.5", []),
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.20s.20260416.batch.6", []),
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.20s.20260416.batch.7", []),
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.20s.20260416.batch.8", []),
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.20s.20260416.batch.9", []),
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.23s.20260416.batch.10", []),
    ("close_the_flap/second_half/close_the_flap.second_half.screwdriver.zhongyou.23s.20260416.batch.11", []),
    # 0416 step19-27 (zhongyou)
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.10s.20260416.batch.1", [(0, 10, 2, 30.0)]),
    # 0417 step19-27 (shuhao)
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.1", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.2", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.3", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.4", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.5", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.6", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.7", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.8", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.9", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.10", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.11", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.12", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.13", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.14", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.15", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.16", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.17", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.18", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.19", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.20", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.22", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.23", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.24", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.25", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.26", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.shuhao.8s.20260417.batch.27", [(0, 8, 2, 30.0)]),
    # 0417 step19-27 (zhongyou)
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.1", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.2", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.3", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.4", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.5", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.6", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.7", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.8", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.9", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.10", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.11", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.12", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.13", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.14", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.16", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.17", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.18", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.19", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.20", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.21", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.22", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.23", [(0, 8, 2, 30.0)]),
    ("close_the_flap/step19-27/close_the_flap.step19-27.screwdriver.zhongyou.8s.20260417.batch.24", [(0, 8, 2, 30.0)]),
    # 0418 second_half (shuhao)
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.20s.20260418.batch.1",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.20s.20260418.batch.2",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.20s.20260418.batch.3",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.20s.20260418.batch.4",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.20s.20260418.batch.5",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.20s.20260418.batch.6",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.20s.20260418.batch.7",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.20s.20260418.batch.8",
        [(0, 20, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.1",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.2",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.3",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.4",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.5",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.6",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.7",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.8",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.9",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.10",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.11",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.12",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.13",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.14",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.15",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.16",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.17",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.18",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.19",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.20",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.21",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.22",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.23",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.24",
        [(0, 22, 2, 30.0)],
    ),
    (
        "close_the_flap/second_half/close_the_flap.second_half.screwdriver.shuhao.22s.20260418.batch.25",
        [(0, 22, 2, 30.0)],
    ),
]

BAD_REPO_ID_WITH_WEIGHT = [
    # 失败数据
    # ("close_the_flap/infer/infer_value_labeled_bad/close_the_box_infer.origin.pi05_base_finetune_box_recap_pt_0330_close.2w.6000s.20260331.batch.2.ep0.flat", []),
    # ("close_the_flap/infer/infer_value_labeled_bad/close_the_box_infer.origin.pi05_base_finetune_box_recap_pt_0330_close.2w.6000s.20260331.batch.2.ep1.flat", []),
]

REPO_ID_WITH_WEIGHT = GOOD_REPO_ID_WITH_WEIGHT + BAD_REPO_ID_WITH_WEIGHT

REPO_ID = [item[0] for item in REPO_ID_WITH_WEIGHT]

# TrainConfig
BATCH_SIZE = 64
NUM_WORKERS = 8
TRAIN_STEPS = 50000
PRETRAINED_WEIGHT_PATH = "checkpoints/pi05_base_finetune_box_recap_cpt_0410_close/pi05_base_finetune_box_recap_cpt_0410_close_exp/79999/params"

FRAME_ATTRS_PREPROCESSORS = [
    TemporalWeightProcessor(repo_id_with_weights=REPO_ID_WITH_WEIGHT),
    OptimalityProcessor(bad_repo_id_with_weight=BAD_REPO_ID_WITH_WEIGHT),
    StaleHeadFramesValidMaskPreprocessor(first_n=3),
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
