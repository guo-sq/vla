from .base import RobotInfo, AlignActionDim

ALOHA = RobotInfo(
    # has_eef=False,
    # NOTE(heyuan): because robochallenge state is one time step behind action,
    # so we directly use action as state for consistency.
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
            # state
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
            # state
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
        action_left_joint_map_dict={
            # action
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
            # action
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
    ),
)


ALOHA_v2 = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "left_joint_map_dict": "observation.state",
        "left_gripper_map_dict": "observation.state",
        "right_joint_map_dict": "observation.state",
        "right_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "left_joint_map_dict": "action",
        "left_gripper_map_dict": "action",
        "right_joint_map_dict": "action",
        "right_gripper_map_dict": "action",
    },
    name_mapping_dict=dict(
        left_joint_map_dict={
            # action
            "left_waist": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "left_shoulder": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "left_elbow": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "left_forearm_roll": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "left_wrist_angle": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "left_wrist_rotate": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        left_gripper_map_dict={
            "left_gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        right_joint_map_dict={
            "right_waist": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "right_shoulder": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "right_elbow": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "right_forearm_roll": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "right_wrist_angle": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "right_wrist_rotate": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        right_gripper_map_dict={
            "right_gripper": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
    ),
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
)


ALOHA_ROBOCHALLENGE = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "left_joint_map_dict": ("action", 0),
        "left_gripper_map_dict": ("action", 0),
        "right_joint_map_dict": ("action", 0),
        "right_gripper_map_dict": ("action", 0),
        "left_eef_map_dict": ("action", 0),
        "right_eef_map_dict": ("action", 0),
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
