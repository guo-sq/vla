from .base import RobotInfo, AlignActionDim

WINDOWX = RobotInfo(
    # has_eef=False,
    # fps = 20
    state_meta_source_dict={
        "eef_map_dict": "observation.state",
        "gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "gripper_map_dict": "action",
        "eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        eef_map_dict={
            # state
            "x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "roll": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "pitch": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "yaw": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
    ),
)
