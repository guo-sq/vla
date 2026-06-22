from openpi.configs.base import *
from openpi.configs.robot_cfg.base import *
import openpi.training.optimizer as _optimizer
import openpi.training.utils as _utils

DEBUG_MODE = False

ASSET_ID = "20260401"
TASK_NAME = "cfg_pi0.5_28_dim.all_public_datasets_" + ASSET_ID
EXP_NAME = TASK_NAME + "_exp"


# Dataset settings
ROOT_DIR = "/mnt/"
# TODO(heyuan:)
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

ALIGN_DIM = sum(np.abs(DELTA_ACTION_MASK_INDICES).tolist())
TARGET_ACTION_DIM = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15]

# verifed
from openpi.configs.dataset_config.aloha import REPO_ID as aloha

# from openpi.configs.dataset_config.openx_embodiment_bc_z import (
#     REPO_ID as openx_embodiment_bc_z,
# )
# from openpi.configs.dataset_config.openx_embodiment_fractal import (
#     REPO_ID as openx_embodiment_fractal,
# )
# from openpi.configs.dataset_config.openx_embodiment_language_table import (
#     REPO_ID as openx_embodiment_language_table,
# )
from openpi.configs.dataset_config.anyverse_tasks.fold_box import REPO_ID as fold_box
from openpi.configs.dataset_config.anyverse_tasks.fold_clothes import REPO_ID as fold_clothes
from openpi.configs.dataset_config.anyverse_tasks.insert_tube import REPO_ID as insert_tube
from openpi.configs.dataset_config.anyverse_tasks.pack_socks import REPO_ID as pack_socks
from openpi.configs.dataset_config.anyverse_tasks.pick_place import REPO_ID as pick_place
from openpi.configs.dataset_config.anyverse_tasks.pour_water import REPO_ID as pour_water
from openpi.configs.dataset_config.anyverse_tasks.seatbelt import REPO_ID as seatbelt

# from openpi.configs.dataset_config.openx_embodiment import REPO_ID as openx_embodiment
# from openpi.configs.dataset_config.anyverse import REPO_ID as anyverse
from openpi.configs.dataset_config.galaxea import REPO_ID as galaxea
from openpi.configs.dataset_config.intern_a1_real import REPO_ID as intern_a1_real
from openpi.configs.dataset_config.rdt import REPO_ID as rdt
from openpi.configs.dataset_config.rhos_ai_gm100 import REPO_ID as rhos_ai_gm100
from openpi.configs.dataset_config.robocoin import REPO_ID as robocoin
from openpi.configs.dataset_config.robomind_1_train import REPO_ID as robomind_1_train
from openpi.configs.dataset_config.robomind_1_val import REPO_ID as robomind_1_val
from openpi.configs.dataset_config.robomind_2 import REPO_ID as robomind_2

REPO_ID = (
    aloha
    + intern_a1_real
    + rdt
    + rhos_ai_gm100
    + robocoin
    # + galaxea
    # + openx_embodiment
    # + bridge_v2
    # + bridge_orig
    # + openx_embodiment_bc_z
    # + openx_embodiment_fractal
    # + openx_embodiment_language_table
    + robomind_2
    + robomind_1_train
    + robomind_1_val
    + fold_box
    + fold_clothes
    + insert_tube
    + pack_socks
    + pick_place
    + pour_water
    + seatbelt
    # + robochallenge
    # + droid
    # + anyverse
)

ROBOT_ALIGN_INFO = _utils.ROBOT_ALIGN_INFO
TOLERANCE_S = 0.02  # TODO(JY)
# TrainConfig
BATCH_SIZE = 512
NUM_WORKERS = 16
TRAIN_STEPS = 20000
FRAME_SKIP = 1
PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"
REUSE_NORM_STATES = "src/openpi/configs/norm_jsons/all_public_dataset_norm_stats.json"
LAZY_LOAD = True

if DEBUG_MODE:
    BATCH_SIZE = 4
    NUM_WORKERS = 2
    TRAIN_STEPS = 10
    EXP_NAME = "DEBUG"
    REPO_ID = REPO_ID[:1]

cfg = TrainConfig(
    name=TASK_NAME,
    exp_name=EXP_NAME,
    model=pi0_config.Pi0Config(
        pi05=True,
        max_token_len=256,
        enable_rl_value_head=False,
        # image_keys=[
        #     "base_0_rgb",
        #     "left_wrist_0_rgb",
        #     "right_wrist_0_rgb",
        #     "third_view_0_rgb",
        # ],
        use_joint_eef_mask=True,
        # paligemma_variant="gemma_2b_lora",
        # action_expert_variant="gemma_300m_lora",
    ),
    # freeze_filter=pi0_config.Pi0Config(
    #     pi05=True,
    #     max_token_len=256,
    #     enable_rl_value_head=False,
    #     use_joint_eef_mask=True,
    #     paligemma_variant="gemma_2b_lora",
    #     action_expert_variant="gemma_300m_lora",
    # ).get_freeze_filter(),
    # Turn off EMA for LoRA finetuning.
    ema_decay=None,
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
        # use_semantic_delta_actions=True,
        # delta_wrap_eef_angles=True,
        tolerance_s=TOLERANCE_S,
        unify_action_space=UNIFY_ACTION_SPACE,
        target_action_dim=TARGET_ACTION_DIM,
        robot_align_info=RobotAlignInfo(robot_align_info=ROBOT_ALIGN_INFO),
        frame_skip=FRAME_SKIP,  # 跳帧间隔，1表示不跳帧，2表示每2帧取1帧，3表示每3帧取1帧
        lazy_load=LAZY_LOAD,  # lazy_load=True 时自动启用共享缓存
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
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=TRAIN_STEPS // 15,
        peak_lr=2e-5 * max(BATCH_SIZE // 64, 1),
        decay_steps=TRAIN_STEPS,
        decay_lr=2e-6,
    ),
    overwrite=True,
    # resume=True,
    # Path to precomputed norm_stats.json. If exists, will be reused instead of recalculating.
    reuse_norm_stats_path=REUSE_NORM_STATES,
)
