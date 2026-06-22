from .base import AlignActionDim
from .base import RobotInfo


AGIBOT_G2A = RobotInfo(
    state_meta_source_dict={
        "left_joint_map_dict": "observation.state",
        "left_gripper_map_dict": "observation.state",
        "right_joint_map_dict": "observation.state",
        "right_gripper_map_dict": "observation.state",
        "left_eef_map_dict": "observation.state",
        "right_eef_map_dict": "observation.state",
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
            "left_arm_joint_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "left_arm_joint_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "left_arm_joint_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "left_arm_joint_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "left_arm_joint_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "left_arm_joint_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "left_arm_joint_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        left_gripper_map_dict={
            "left_gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        right_joint_map_dict={
            "right_arm_joint_0": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "right_arm_joint_1": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "right_arm_joint_2": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "right_arm_joint_3": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "right_arm_joint_4": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "right_arm_joint_5": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "right_arm_joint_6": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        right_gripper_map_dict={
            "right_gripper": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        left_eef_map_dict={
            "left_eef_pos_x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "left_eef_pos_y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "left_eef_pos_z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "left_eef_ori_x": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "left_eef_ori_y": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "left_eef_ori_z": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        },
        right_eef_map_dict={
            "right_eef_pos_x": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
            "right_eef_pos_y": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
            "right_eef_pos_z": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
            "right_eef_ori_x": AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
            "right_eef_ori_y": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
            "right_eef_ori_z": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
        },
    ),
)
