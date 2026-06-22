from .base import AlignActionDim
from .base import RobotInfo

# refer to
# https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND/blob/main/static/all_robot_h5_info.md

"""
AgileX 3RGB
end_effector (7d): [x, y, z, r, p, y, gripper]

joint_position (7d): [base link, ..., end_effector link, gripper]
"""

H5_AGILEX_3RGB_ROBOMIND_1 = RobotInfo(
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

"""
Franka 1RGB
end_effector (6d): [x, y, z, r, p, y]

joint_position (8d): [base link, ..., end_effector link, gripper]
"""
H5_FRANKA_1RGB_ROBOMIND_1 = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "state_joint_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
        "state_eef_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_joint_map_dict": "action",
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    name_mapping_dict=dict(
        state_joint_map_dict={
            # state
            "state_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "state_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "state_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "state_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "state_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "state_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "state_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_gripper_map_dict={
            "state_7": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_eef_map_dict={
            "state_8": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "state_9": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "state_10": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "state_11": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "state_12": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "state_13": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        action_joint_map_dict={
            # action
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_gripper_map_dict={
            "motor_7": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_eef_map_dict={
            "motor_8": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "motor_9": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "motor_10": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "motor_11": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "motor_12": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "motor_13": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
    ),
)

H5_FRANKA_2RGB_ROBOMIND_1 = H5_FRANKA_1RGB_ROBOMIND_1

"""
Tien Kung Gello 1RGB
joint_position (16d): [left arm (7d), left hand closure (1d), right arm (7d), right hand closure (1d)]

arm (7d): [base link, ..., end_effector link]

final: [left arm (7d), left hand closure (1d), right arm (7d), right hand closure (1d)]
"""
H5_TIENKUNG_GELLO_1RGB_ROBOMIND_1 = RobotInfo(
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
    ),
)


"""
Tien Kung Xsens 1RGB
end_effector (12d): [left hand (6d), right hand (6d)]

    
hand (6d): [little finger, ring finger, middle finger, index finger, thumb0 for bending, thumb1 for rotation]

joint_position (14d): [left arm (7d), right arm (7d)]

arm (7d): [base link, ..., end_effector link]


final:  [left arm (7d), right arm (7d), left hand (6d), right hand (6d)]

"""
H5_TIENKUNG_XSENS_1RGB_ROBOMIND_1 = RobotInfo(
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
        state_right_joint_map_dict={
            "state_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "state_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "state_9": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "state_10": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "state_11": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "state_12": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "state_13": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            # left hand: index finger closure
            "state_17": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_gripper_map_dict={
            # right hand: index finger closure
            "state_23": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
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
        action_right_joint_map_dict={
            "motor_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "motor_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "motor_9": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "motor_10": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "motor_11": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "motor_12": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "motor_13": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            # left hand: index finger closure
            "motor_17": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_gripper_map_dict={
            # right hand: index finger closure
            "motor_23": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
    ),
)
