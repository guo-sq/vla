from .base import AlignActionDim
from .base import RobotInfo

ALOHA_ROBOCHALLENGE = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "left_joint_map_dict": "state",
        "left_gripper_map_dict": "state",
        "right_joint_map_dict": "state",
        "right_gripper_map_dict": "state",
        "left_eef_map_dict": "state",
        "right_eef_map_dict": "state",
    },
    action_meta_source_dict={
        "left_joint_map_dict": "action",
        "left_gripper_map_dict": "action",
        "right_joint_map_dict": "action",
        "right_gripper_map_dict": "action",
        "left_eef_map_dict": "action",
        "right_eef_map_dict": "action",
    },
    name_mapping_dict=dict(
        left_joint_map_dict={
            # action
            "left_joint_1": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "left_joint_2": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "left_joint_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "left_joint_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "left_joint_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "left_joint_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        left_gripper_map_dict={
            "left_gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        right_joint_map_dict={
            "right_joint_1": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "right_joint_2": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "right_joint_3": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "right_joint_4": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "right_joint_5": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "right_joint_6": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        right_gripper_map_dict={
            "right_gripper": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        left_eef_map_dict={
            "left_ee_x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "left_ee_y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "left_ee_z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "left_ee_roll": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "left_ee_pitch": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "left_ee_yaw": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        right_eef_map_dict={
            "right_ee_x": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
            "right_ee_y": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
            "right_ee_z": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
            "right_ee_roll": AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
            "right_ee_pitch": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
            "right_ee_yaw": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
        },
    ),
    # meta_mapping_dict={
    #     "action/names/motors": "action/names",
    #     "observation.state/names/motors": "observation.state/names",
    # },
)


ARX5_SINGLE_ARM_ROBOCHALLENGE = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "left_eef_map_dict": "state",
        "left_gripper_map_dict": "state",
    },
    action_meta_source_dict={
        "left_eef_map_dict": "action",
        "left_gripper_map_dict": "action",
    },
    name_mapping_dict=dict(
        left_eef_map_dict={
            "ee_x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "ee_y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "ee_z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "ee_roll": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "ee_pitch": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "ee_yaw": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        left_gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
    ),
)


FRANKA_SINGLE_ARM_ROBOCHALLENGE = RobotInfo(
    state_meta_source_dict={
        "left_joint_map_dict": "state",
        "left_gripper_map_dict": "state",
    },
    action_meta_source_dict={
        "left_joint_map_dict": "action",
        "left_gripper_map_dict": "action",
    },
    name_mapping_dict=dict(
        left_joint_map_dict={
            # action
            "joint_1": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "joint_2": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "joint_3": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "joint_4": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "joint_5": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "joint_6": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "joint_7": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        left_gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
    ),
)


UR5_SINGLE_ARM_ROBOCHALLENGE = RobotInfo(
    state_meta_source_dict={
        "left_joint_map_dict": ("action", 0),
        "left_gripper_map_dict": ("action", 0),
    },
    action_meta_source_dict={
        "left_joint_map_dict": "action",
        "left_gripper_map_dict": "action",
    },
    name_mapping_dict=dict(
        left_joint_map_dict={
            # action
            "joint_1": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "joint_2": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "joint_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "joint_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "joint_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "joint_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        left_gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
    ),
)
