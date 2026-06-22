from openpi.configs.robot_cfg.AGIBOT_G1 import AGIBOT_G1
from openpi.configs.robot_cfg.AGIBOT_G1 import INTERNDATA_GENIE_1
from openpi.configs.robot_cfg.AGIBOT_G2A import AGIBOT_G2A
from openpi.configs.robot_cfg.AGIBOT_G1_ALPHA_LEROBOT import AGIBOT_G1_ALPHA_LEROBOT
from openpi.configs.robot_cfg.AIRBOT_MMK2 import AIRBOT_MMK2
from openpi.configs.robot_cfg.ALOHA import ALOHA
from openpi.configs.robot_cfg.ALOHA import ALOHA_ROBOCHALLENGE
from openpi.configs.robot_cfg.ALOHA import ALOHA_v2
from openpi.configs.robot_cfg.ALPHA_BOT_2 import ALPHA_BOT_2
from openpi.configs.robot_cfg.BI_PIPER import BI_PIPER
from openpi.configs.robot_cfg.BI_PIPER import TEST_BI_PIPER
from openpi.configs.robot_cfg.BRIDGE_V2 import WINDOWX
from openpi.configs.robot_cfg.COBOT_MAGIC import COBOT_DECOUPLED_MAGIC
from openpi.configs.robot_cfg.COBOT_MAGIC import SONGLING_SELFCOLLECT
from openpi.configs.robot_cfg.DROID import DROID
from openpi.configs.robot_cfg.FANGZHOU import ARX5_SINGLE_ARM_ROBOCHALLENGE
from openpi.configs.robot_cfg.FANGZHOU import FANGZHOU
from openpi.configs.robot_cfg.FANGZHOU import TEST_FANGZHOU
from openpi.configs.robot_cfg.FRANKA import FRANKA_SINGLE_ARM_ROBOCHALLENGE
from openpi.configs.robot_cfg.FRANKA import H5_FRANKA_2RGB
from openpi.configs.robot_cfg.GALBOT_G1 import GALBOT_G1
from openpi.configs.robot_cfg.LEJU_ROBOT import LEJU_ROBOT
from openpi.configs.robot_cfg.OPENX_EMBODIMENT import *
from openpi.configs.robot_cfg.R1_LITE import R1_LITE
from openpi.configs.robot_cfg.R1_LITE import R1_LITE_OPEN
from openpi.configs.robot_cfg.R1_LITE import R1_LITE_PRO
from openpi.configs.robot_cfg.ROBOMIND_1 import H5_AGILEX_3RGB_ROBOMIND_1
from openpi.configs.robot_cfg.ROBOMIND_1 import H5_FRANKA_1RGB_ROBOMIND_1
from openpi.configs.robot_cfg.ROBOMIND_1 import H5_FRANKA_2RGB_ROBOMIND_1
from openpi.configs.robot_cfg.ROBOMIND_1 import H5_TIENKUNG_GELLO_1RGB_ROBOMIND_1
from openpi.configs.robot_cfg.ROBOMIND_1 import H5_TIENKUNG_XSENS_1RGB_ROBOMIND_1
from openpi.configs.robot_cfg.ROBOMIND_2 import AGILE_X_BIMANUAL_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import AGILE_X_MOBILE_BIMANUAL_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import ARX_BIMANUAL_MOBILE_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import ARX_BIMANUAL_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import FRANKA_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import TIENKUNG_DEX_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import TIENKUNG_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import TIENYI_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import UR5_DEX_ROBOMIND_2
from openpi.configs.robot_cfg.ROBOMIND_2 import UR5_ROBOMIND_2
from openpi.configs.robot_cfg.UR5 import UR5_SINGLE_ARM_ROBOCHALLENGE

__all__ = [
    "BI_PIPER",
    "TEST_BI_PIPER",
    "AGIBOT_G1_ALPHA_LEROBOT",
    "AGIBOT_G1",
    "AGIBOT_G2A",
    "AIRBOT_MMK2",
    "ALPHA_BOT_2",
    "FANGZHOU",
    "ARX5_SINGLE_ARM_ROBOCHALLENGE",
    "GALBOT_G1",
    "R1_LITE",
    "R1_LITE_OPEN",
    "LEJU_ROBOT",
    "COBOT_DECOUPLED_MAGIC",
    "TEST_FANGZHOU",
    "ALOHA",
    "ALOHA_v2",
    "ALOHA_ROBOCHALLENGE",
    "INTERNDATA_GENIE_1",
    "SONGLING_SELFCOLLECT",
    "H5_FRANKA_2RGB",
    "R1_LITE_PRO",
    "FRANKA_SINGLE_ARM_ROBOCHALLENGE",
    "UR5_SINGLE_ARM_ROBOCHALLENGE",
    "H5_AGILEX_3RGB_ROBOMIND_1",
    "H5_FRANKA_1RGB_ROBOMIND_1",
    "H5_FRANKA_2RGB_ROBOMIND_1",
    "H5_TIENKUNG_GELLO_1RGB_ROBOMIND_1",
    "H5_TIENKUNG_XSENS_1RGB_ROBOMIND_1",
    "AGILE_X_BIMANUAL_ROBOMIND_2",
    "AGILE_X_MOBILE_BIMANUAL_ROBOMIND_2",
    "ARX_BIMANUAL_MOBILE_ROBOMIND_2",
    "ARX_BIMANUAL_ROBOMIND_2",
    "TIENYI_ROBOMIND_2",
    "TIENKUNG_ROBOMIND_2",
    "TIENKUNG_DEX_ROBOMIND_2",
    "UR5_ROBOMIND_2",
    "UR5_DEX_ROBOMIND_2",
    "FRANKA_ROBOMIND_2",
    # OPENX embodiment
    "AUSTIN_BUDS_OPENX_FRANKA",
    "AUSTIN_SAILOR_OPENX_FRANKA",
    "AUSTIN_SIRIUS_OPENX_FRANKA",
    "BC_Z_OPENX_GOOGLE_ROBOT",
    "BERKELEY_AUTOLAB_OPENX_UR5",
    "BERKELEY_CABLE_ROUTING_OPENX_FRANKA",
    "BERKELEY_FANUC_MANIPULATION_OPENX_FANUC_MATE",
    "BERKELEY_MVP_OPENX_XARM",
    "BERKELEY_RPT_OPENX_FRANKA",
    "BRIDGE_ORIG_OPENX_WINDOWX",
    "CMU_PLAY_FUSION_OPENX_FRANKA",
    "CMU_STRETCH_OPENX_STRETCH",
    "DLR_EDAN_SHARED_CONTROL_OPENX",
    "DOBBE_OPENX_STRETCH",
    "FMB_OPENX_FRANKA",
    "FRACTRAL_OPENX_GOOGLE_ROBOT",
    "FURNITURE_BENCH_OPENX_FRANKA",
    "IAMLAB_CMU_PICKUP_INSERT_OPENX_FRANKA",
    "JACO_PLAY_OPENX_JACO",
    "KUKA_OPENX",
    "LANGUAGE_TABLE_OPENX",
    "NYU_DOOR_OPEN_OPENX_STRETCH",
    "NYU_FRANKA_PLAY_OPENX",
    "ROBOTURK_OPENX_FRANKA",
    "STANFORD_HYDRA_OPENX_FRANKA",
    "TACO_PLAY_OPENX_FRANKA",
    "TOTO_OPENX_FRANKA",
    "USCD_KITCHEN_OPENX_XARM",
    "UTAUSTIN_MUTEX_OPENX_FRANKA",
    "VIOLA_OPENX_FRANKA",
    # bridge_v2
    "WINDOWX",
    "DROID",
]
