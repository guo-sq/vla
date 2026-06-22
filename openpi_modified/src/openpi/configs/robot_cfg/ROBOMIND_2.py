from .base import AlignActionDim
from .base import RobotInfo

AGILE_X_BIMANUAL_ROBOMIND_2 = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state",
        "state_left_gripper_map_dict": "observation.state",
        "state_left_eef_map_dict": "observation.state",
        "state_right_joint_map_dict": "observation.state",
        "state_right_gripper_map_dict": "observation.state",
        "state_right_eef_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action",
        "action_left_gripper_map_dict": "action",
        "action_left_eef_map_dict": "action",
        "action_right_joint_map_dict": "action",
        "action_right_gripper_map_dict": "action",
        "action_right_eef_map_dict": "action",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            "state_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "state_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "state_2": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "state_3": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "state_4": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "state_5": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            "state_6": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_joint_map_dict={
            "state_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "state_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "state_9": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "state_10": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "state_11": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "state_12": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_right_gripper_map_dict={
            "state_13": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        state_left_eef_map_dict={
            "state_14": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "state_15": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "state_16": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "state_17": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "state_18": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "state_19": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        state_right_eef_map_dict={
            "state_21": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
            "state_22": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
            "state_23": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
            "state_24": AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
            "state_25": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
            "state_26": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
        },
        action_left_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            "motor_6": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_joint_map_dict={
            "motor_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "motor_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "motor_9": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "motor_10": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "motor_11": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "motor_12": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_right_gripper_map_dict={
            "motor_13": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        action_left_eef_map_dict={
            "motor_14": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "motor_15": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "motor_16": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "motor_17": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "motor_18": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "motor_19": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        action_right_eef_map_dict={
            "motor_21": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
            "motor_22": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
            "motor_23": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
            "motor_24": AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
            "motor_25": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
            "motor_26": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
        },
    ),
)
AGILE_X_MOBILE_BIMANUAL_ROBOMIND_2 = AGILE_X_BIMANUAL_ROBOMIND_2

ARX_BIMANUAL_ROBOMIND_2 = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state",
        "state_left_gripper_map_dict": "observation.state",
        "state_left_eef_map_dict": "observation.state",
        "state_right_joint_map_dict": "observation.state",
        "state_right_gripper_map_dict": "observation.state",
        "state_right_eef_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action",
        "action_left_gripper_map_dict": "action",
        "action_left_eef_map_dict": "action",
        "action_right_joint_map_dict": "action",
        "action_right_gripper_map_dict": "action",
        "action_right_eef_map_dict": "action",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            "state_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "state_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "state_2": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "state_3": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "state_4": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "state_5": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            "state_6": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_joint_map_dict={
            "state_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "state_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "state_9": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "state_10": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "state_11": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "state_12": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_right_gripper_map_dict={
            "state_13": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        state_left_eef_map_dict={
            "state_14": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "state_15": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "state_16": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            # "state_17": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            # "state_18": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            # "state_19": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        state_right_eef_map_dict={
            "state_21": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
            "state_22": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
            "state_23": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
            # "state_24": AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
            # "state_25": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
            # "state_26": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
        },
        action_left_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            "motor_6": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_joint_map_dict={
            "motor_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "motor_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "motor_9": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "motor_10": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "motor_11": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "motor_12": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_right_gripper_map_dict={
            "motor_13": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        action_left_eef_map_dict={
            "motor_14": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "motor_15": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "motor_16": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            # "motor_17": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            # "motor_18": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            # "motor_19": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        action_right_eef_map_dict={
            "motor_21": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
            "motor_22": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
            "motor_23": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
            # "motor_24": AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
            # "motor_25": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
            # "motor_26": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
        },
    ),
)

ARX_BIMANUAL_MOBILE_ROBOMIND_2 = ARX_BIMANUAL_ROBOMIND_2


TIENYI_ROBOMIND_2 = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state",
        "state_left_gripper_map_dict": "observation.state",
        "state_right_joint_map_dict": "observation.state",
        "state_right_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action",
        "action_left_gripper_map_dict": "action",
        "action_right_joint_map_dict": "action",
        "action_right_gripper_map_dict": "action",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            "state_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "state_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "state_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "state_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "state_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "state_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "state_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            "state_7": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_joint_map_dict={
            "state_8": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "state_9": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "state_10": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "state_11": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "state_12": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "state_13": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "state_14": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_right_gripper_map_dict={
            "state_15": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # TODO(heyuan): add state twist, head pose, etc.
        action_left_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            "motor_7": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_joint_map_dict={
            "motor_8": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "motor_9": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "motor_10": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "motor_11": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "motor_12": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "motor_13": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "motor_14": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_right_gripper_map_dict={
            "motor_15": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # TODO(heyuan): add action twist, head pose, etc.
    ),
)

TIENKUNG_ROBOMIND_2 = TIENYI_ROBOMIND_2

TIENKUNG_DEX_ROBOMIND_2 = TIENYI_ROBOMIND_2

UR5_ROBOMIND_2 = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state",
        "state_left_gripper_map_dict": "observation.state",
        "state_right_joint_map_dict": "observation.state",
        "state_right_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action",
        "action_left_gripper_map_dict": "action",
        "action_right_joint_map_dict": "action",
        "action_right_gripper_map_dict": "action",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            "state_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "state_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "state_2": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "state_3": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "state_4": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "state_5": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            "state_6": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_joint_map_dict={
            "state_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "state_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "state_9": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "state_10": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "state_11": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "state_12": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_right_gripper_map_dict={
            "state_13": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # TODO(heyuan): add eef
        action_left_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            "motor_6": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_joint_map_dict={
            "motor_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "motor_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "motor_9": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "motor_10": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "motor_11": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "motor_12": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_right_gripper_map_dict={
            "motor_13": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # TODO(heyuan): add eef
    ),
)


UR5_DEX_ROBOMIND_2 = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state",
        "state_left_gripper_map_dict": "observation.state",
        "state_right_joint_map_dict": "observation.state",
        "state_right_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action",
        "action_left_gripper_map_dict": "action",
        "action_right_joint_map_dict": "action",
        "action_right_gripper_map_dict": "action",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            "state_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "state_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "state_2": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "state_3": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "state_4": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "state_5": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            "state_6": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
            # ... # TODO(heyuan): add left hand state mapping if needed
            # "state_17": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_joint_map_dict={
            "state_18": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "state_19": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "state_20": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "state_21": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "state_22": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "state_23": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_right_gripper_map_dict={
            "state_24": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
            # ... # TODO(heyuan): add right hand state mapping if needed
            # "state_35": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # TODO(heyuan): add eef
        action_left_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            "motor_6": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
            # ... # TODO(heyuan): add left hand action mapping if needed
            # "motor_17": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_joint_map_dict={
            "motor_18": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "motor_19": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "motor_20": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "motor_21": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "motor_22": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "motor_23": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_right_gripper_map_dict={
            "motor_24": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
            # ... # TODO(heyuan): add right hand action mapping if needed
            # "motor_35": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # TODO(heyuan): add eef
    ),
)


FRANKA_ROBOMIND_2 = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state",
        "state_left_gripper_map_dict": "observation.state",
        "state_left_eef_map_dict": "observation.state",
        "state_right_joint_map_dict": "observation.state",
        "state_right_gripper_map_dict": "observation.state",
        "state_right_eef_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action",
        "action_left_gripper_map_dict": "action",
        "action_left_eef_map_dict": "action",
        "action_right_joint_map_dict": "action",
        "action_right_gripper_map_dict": "action",
        "action_right_eef_map_dict": "action",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            "state_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "state_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "state_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "state_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "state_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "state_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "state_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            "state_7": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_joint_map_dict={
            "state_9": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "state_10": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "state_11": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "state_12": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "state_13": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "state_14": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "state_15": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_right_gripper_map_dict={
            "state_16": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        state_left_eef_map_dict={
            "state_18": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "state_19": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "state_20": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
        },
        state_right_eef_map_dict={
            "state_25": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
            "state_26": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
            "state_27": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
        },
        action_left_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            "motor_7": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_joint_map_dict={
            "motor_9": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "motor_10": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "motor_11": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "motor_12": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "motor_13": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "motor_14": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "motor_15": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_right_gripper_map_dict={
            "motor_16": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        action_left_eef_map_dict={
            "motor_18": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "motor_19": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "motor_20": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
        },
        action_right_eef_map_dict={
            "motor_25": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
            "motor_26": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
            "motor_27": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
        },
    ),
)
