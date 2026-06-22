from .base import RobotInfo, AlignActionDim


DROID = RobotInfo(
    # fps = 10
    # has_eef=False,
    state_meta_source_dict={
        "state_joint_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action.original",
        "action_eef_map_dict": "action.original",
    },
    meta_mapping_dict={
        "action.original/names/axes": "action.original/names",
        "observation.state/names/axes": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_joint_map_dict={
            "joint_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "joint_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "joint_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "joint_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "joint_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "joint_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "joint_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_eef_map_dict={
            "x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "roll": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "pitch": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "yaw": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        action_gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
    ),
)


DROID_EEF = RobotInfo(
    # fps = 10
    # has_eef=False,
    state_meta_source_dict={
        "joint_map_dict": "observation.state",
        "gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "gripper_map_dict": "action",
        "joint_map_dict": "action",
    },
    meta_mapping_dict={
        "action.original/names/axes": "action.original/names",
        "observation.state/names/axes": "observation.state/names",
    },
    name_mapping_dict=dict(
        joint_map_dict={
            "joint_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "joint_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "joint_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "joint_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "joint_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "joint_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "joint_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        # action_eef_map_dict={
        #     "x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
        #     "y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
        #     "z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
        #     "roll": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
        #     "pitch": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
        #     "yaw": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        # },
        # action_gripper_map_dict={
        #     "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        # },
    ),
)
