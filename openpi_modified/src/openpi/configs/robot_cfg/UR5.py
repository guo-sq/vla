from .base import RobotInfo, AlignActionDim

UR5_SINGLE_ARM_ROBOCHALLENGE = RobotInfo(
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