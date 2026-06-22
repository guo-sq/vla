from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.configs.robot_cfg import *
from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


PUBLIC_DATASET_MAP = {
    # Common
    "observation.images.cam_left_wrist_rgb": "observation.images.left_wrist",
    "observation.images.cam_right_wrist_rgb": "observation.images.right_wrist",
    # AgiBot-g1(ruantong_a2d)
    # AIRBOT_MMK2(discover_robotics_aitbot_mmk2)
    # Galbot_g1(yinhe)
    # opengalaxea: r1lite
    "observation.images.head_rgb": "observation.images.head",
    "observation.images.head_right_rgb": "observation.images.third_view",
    "observation.images.left_wrist_rgb": "observation.images.left_wrist",
    "observation.images.right_wrist_rgb": "observation.images.right_wrist",
    # Split_aloha(agilex_cobot_decoupled_magic)
    "observation.images.cam_high_rgb": "observation.images.head",
    # TODO(heyuan): why same mapping of cam_high_rgb and cam_high_left_rgb
    "observation.images.cam_high_left_rgb": "observation.images.third_view",  # R1_Lite
    # alpha_bot_2(alpha_bot_2)
    "observation.images.cam_head_rgb": "observation.images.head",
    # Cobot_Magic(agilex_cobot_decoupled_magic)
    "observation.images.cam_front_rgb": "observation.images.head",
    "observation.images.cam_left_wrist_rgb_rgb": "observation.images.left_wrist",
    "observation.images.cam_right_wrist_rgb_rgb": "observation.images.right_wrist",
    # G1edu-u3(unitree_g1)
    # "observation.images.cam_high_rgb": "observation.images.head",
    "observation.images.color_left_wrist": "observation.images.left_wrist",
    "observation.images.color_right_wrist": "observation.images.right_wrist",
    # leju_robot(leju_robot)
    "observation.images.camera_head_rgb": "observation.images.head",
    "observation.images.camera_left_wrist_rgb": "observation.images.left_wrist",
    "observation.images.camera_right_wrist_rgb": "observation.images.right_wrist",
    # ALOHA
    "observation.images.camera_front": "observation.images.head",
    "observation.images.cam_high": "observation.images.head",
    "observation.images.cam_left_wrist": "observation.images.left_wrist",
    "observation.images.cam_right_wrist": "observation.images.right_wrist",
    "observation.images.camera_left_wrist": "observation.images.left_wrist",
    "observation.images.camera_right_wrist": "observation.images.right_wrist",
    # INTERNDATA_Genie1
    "observation.images.hand_right": "observation.images.right_wrist",
    "observation.images.hand_left": "observation.images.left_wrist",
    # AGIBOT_G1_ALPHA_LEROBOT
    "observation.images.top_head": "observation.images.head",
    "observation.images.head_center_fisheye": "observation.images.third_view",
    # GLM
    "observation.images.camera_top": "observation.images.head",
    "observation.images.camera_wrist_left": "observation.images.left_wrist",
    "observation.images.camera_wrist_right": "observation.images.right_wrist",
    # H5_FRANKA_2RGB
    "observation.images.camera_left": "observation.images.left_wrist",
    "observation.images.camera_right": "observation.images.right_wrist",
    # OPENX_EMBODIMENT
    "observation.images.image": "observation.images.head",
    "observation.images.wrist_image": "observation.images.left_wrist",
    "observation.images.hand_image": "observation.images.left_wrist",
    "observation.images.top_image": "observation.images.third_view",
    "observation.images.wrist45_image": "observation.images.left_wrist",
    "observation.images.wrist225_image": "observation.images.right_wrist",
    "observation.images.image_additional_view": "observation.images.third_view",
    "observation.images.agentview_rgb": "observation.images.third_view",
    "observation.images.eye_in_hand_rgb": "observation.images.left_wrist",
    "observation.images.front_rgb": "observation.images.head",
    "observation.images.rgb": "observation.images.head",
    # oss_data/IPEC-COMMUNITY: has 4 camera, 3 third view
    "observation.images.image_0": "observation.images.head",
    "observation.images.image_1": "observation.images.third_view",
    "observation.images.image_2": "observation.images.left_wrist",
    "observation.images.image_3": "observation.images.right_wrist",
    "observation.images.image_side_2": "observation.images.third_view",
    "observation.images.image_side_1": "observation.images.head",
    "observation.images.image_wrist_2": "observation.images.right_wrist",
    "observation.images.image_wrist_1": "observation.images.left_wrist",
    # DROID
    "observation.images.exterior_1_left": "observation.images.head",
    "observation.images.exterior_2_left": "observation.images.third_view",
    "observation.images.rgb_gripper": "observation.images.left_wrist",
    "observation.images.rgb_static": "observation.images.head",
}

ROBOT_ALIGN_INFO = {
    # Fangzhou
    "arxx5_bimanual": FANGZHOU,
    "bi_piper_follower": BI_PIPER,  # anyverse bi_piper
    # MMK2
    "discover_robotics_aitbot_mmk2": AIRBOT_MMK2,
    # Cobot_Magic/Split_aloha
    "agilex_cobot_decoupled_magic": COBOT_DECOUPLED_MAGIC,
    # R1_Lite
    "galaxea_r1_lite": R1_LITE,
    # R1_LITE_OPEN
    "r1lite": R1_LITE_OPEN,
    "r1pro": R1_LITE_PRO,
    # AgiBot-g1
    "ruantong_a2d": AGIBOT_G1,  # robocoin agibot_g1
    "a2d": INTERNDATA_GENIE_1,  # intern genie1
    "g2a": AGIBOT_G2A,
    "AGIBOT_G1_ALPHA_LEROBOT": AGIBOT_G1_ALPHA_LEROBOT,  # agibot world alpha lerobot
    # leju_robot
    "leju_robot": LEJU_ROBOT,
    # ALPHA_BOT_2
    "alpha_bot_2": ALPHA_BOT_2,
    # Galbot_g1
    "yinhe": GALBOT_G1,
    # ALOHA
    "ALOHA": ALOHA,
    "aloha": ALOHA_v2,  # lerobot/aloha_mobile_cabinet
    "songling_selfcollect": SONGLING_SELFCOLLECT,
    "cobot_magic": SONGLING_SELFCOLLECT,
    "h5_franka_2rgb": H5_FRANKA_2RGB,
    # RoboChallenge
    "ALOHA_ROBOCHALLENGE": ALOHA_ROBOCHALLENGE,
    "ARX5_SINGLE_ARM_ROBOCHALLENGE": ARX5_SINGLE_ARM_ROBOCHALLENGE,
    "FRANKA_SINGLE_ARM_ROBOCHALLENGE": FRANKA_SINGLE_ARM_ROBOCHALLENGE,
    "UR5_SINGLE_ARM_ROBOCHALLENGE": UR5_SINGLE_ARM_ROBOCHALLENGE,
    # RoboMIND_1
    "H5_AGILEX_3RGB_ROBOMIND_1": H5_AGILEX_3RGB_ROBOMIND_1,
    "H5_FRANKA_1RGB_ROBOMIND_1": H5_FRANKA_1RGB_ROBOMIND_1,
    "H5_FRANKA_2RGB_ROBOMIND_1": H5_FRANKA_2RGB_ROBOMIND_1,
    "H5_TIENKUNG_GELLO_1RGB_ROBOMIND_1": H5_TIENKUNG_GELLO_1RGB_ROBOMIND_1,
    "H5_TIENKUNG_XSENS_1RGB_ROBOMIND_1": H5_TIENKUNG_XSENS_1RGB_ROBOMIND_1,
    # ROBOMIND_2
    "AGILE_X_BIMANUAL_ROBOMIND_2": AGILE_X_BIMANUAL_ROBOMIND_2,
    "AGILE_X_MOBILE_BIMANUAL_ROBOMIND_2": AGILE_X_MOBILE_BIMANUAL_ROBOMIND_2,
    "ARX_BIMANUAL_ROBOMIND_2": ARX_BIMANUAL_ROBOMIND_2,
    "ARX_BIMANUAL_MOBILE_ROBOMIND_2": ARX_BIMANUAL_MOBILE_ROBOMIND_2,
    "TIENYI_ROBOMIND_2": TIENYI_ROBOMIND_2,
    "TIENKUNG_ROBOMIND_2": TIENKUNG_ROBOMIND_2,
    "TIENKUNG_DEX_ROBOMIND_2": TIENKUNG_DEX_ROBOMIND_2,
    "UR5_ROBOMIND_2": UR5_ROBOMIND_2,
    "UR5_DEX_ROBOMIND_2": UR5_DEX_ROBOMIND_2,
    "FRANKA_ROBOMIND_2": FRANKA_ROBOMIND_2,
    # OPENX_EMBODIMENT
    "AUSTIN_BUDS_OPENX_FRANKA": AUSTIN_BUDS_OPENX_FRANKA,
    "AUSTIN_SAILOR_OPENX_FRANKA": AUSTIN_SAILOR_OPENX_FRANKA,
    "AUSTIN_SIRIUS_OPENX_FRANKA": AUSTIN_SIRIUS_OPENX_FRANKA,
    "BC_Z_OPENX_GOOGLE_ROBOT": BC_Z_OPENX_GOOGLE_ROBOT,
    "BERKELEY_AUTOLAB_OPENX_UR5": BERKELEY_AUTOLAB_OPENX_UR5,
    "BERKELEY_CABLE_ROUTING_OPENX_FRANKA": BERKELEY_CABLE_ROUTING_OPENX_FRANKA,
    "BERKELEY_FANUC_MANIPULATION_OPENX_FANUC_MATE": BERKELEY_FANUC_MANIPULATION_OPENX_FANUC_MATE,
    "BERKELEY_MVP_OPENX_XARM": BERKELEY_MVP_OPENX_XARM,
    "BERKELEY_RPT_OPENX_FRANKA": BERKELEY_RPT_OPENX_FRANKA,
    "BRIDGE_ORIG_OPENX_WINDOWX": BRIDGE_ORIG_OPENX_WINDOWX,
    "CMU_PLAY_FUSION_OPENX_FRANKA": CMU_PLAY_FUSION_OPENX_FRANKA,
    "CMU_STRETCH_OPENX_STRETCH": CMU_STRETCH_OPENX_STRETCH,
    "DLR_EDAN_SHARED_CONTROL_OPENX": DLR_EDAN_SHARED_CONTROL_OPENX,
    "DOBBE_OPENX_STRETCH": DOBBE_OPENX_STRETCH,
    "FMB_OPENX_FRANKA": FMB_OPENX_FRANKA,
    "FRACTRAL_OPENX_GOOGLE_ROBOT": FRACTRAL_OPENX_GOOGLE_ROBOT,
    "FURNITURE_BENCH_OPENX_FRANKA": FURNITURE_BENCH_OPENX_FRANKA,
    "IAMLAB_CMU_PICKUP_INSERT_OPENX_FRANKA": IAMLAB_CMU_PICKUP_INSERT_OPENX_FRANKA,
    "JACO_PLAY_OPENX_JACO": JACO_PLAY_OPENX_JACO,
    "KUKA_OPENX": KUKA_OPENX,
    "LANGUAGE_TABLE_OPENX": LANGUAGE_TABLE_OPENX,
    "NYU_DOOR_OPEN_OPENX_STRETCH": NYU_DOOR_OPEN_OPENX_STRETCH,
    "NYU_FRANKA_PLAY_OPENX": NYU_FRANKA_PLAY_OPENX,
    "ROBOTURK_OPENX_FRANKA": ROBOTURK_OPENX_FRANKA,
    "STANFORD_HYDRA_OPENX_FRANKA": STANFORD_HYDRA_OPENX_FRANKA,
    "TACO_PLAY_OPENX_FRANKA": TACO_PLAY_OPENX_FRANKA,
    "TOTO_OPENX_FRANKA": TOTO_OPENX_FRANKA,
    "USCD_KITCHEN_OPENX_XARM": USCD_KITCHEN_OPENX_XARM,
    "UTAUSTIN_MUTEX_OPENX_FRANKA": UTAUSTIN_MUTEX_OPENX_FRANKA,
    "VIOLA_OPENX_FRANKA": VIOLA_OPENX_FRANKA,
    # BRIDGE_V2
    "WINDOWX": WINDOWX,
    # DROID
    "DROID": DROID,
}

TEST_ROBOT_ALIGN_INFO = {
    "bi_piper_follower": TEST_BI_PIPER,
    "arxx5_bimanual": TEST_FANGZHOU,
}
