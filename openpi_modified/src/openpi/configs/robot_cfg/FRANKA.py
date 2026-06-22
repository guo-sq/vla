from .base import RobotInfo, AlignActionDim

H5_FRANKA_2RGB = RobotInfo(
    # has_eef=False,
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state",
        "state_right_joint_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action",
        "action_right_joint_map_dict": "action",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            # state
            "state_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "state_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "state_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "state_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "state_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "state_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "state_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_right_joint_map_dict={
            # state
            "state_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "state_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "state_9": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "state_10": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "state_11": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "state_12": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "state_13": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_left_joint_map_dict={
            # action
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_right_joint_map_dict={
            # action
            "motor_7": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "motor_8": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "motor_9": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "motor_10": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "motor_11": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "motor_12": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "motor_13": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
    ),
)



FRANKA_SINGLE_ARM_ROBOCHALLENGE = RobotInfo(
    state_meta_source_dict={
        "left_joint_map_dict": "action",
        "left_gripper_map_dict": "action",
    },
    action_meta_source_dict={
        "left_joint_map_dict": "action",
        "left_gripper_map_dict": "action",
    },
    name_mapping_dict = dict(
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