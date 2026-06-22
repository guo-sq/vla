from .base import RobotInfo, AlignActionDim

# TODO(heyuan): add hand
LEJU_ROBOT = RobotInfo(
    # has_eef=True,
    state_meta_source_dict={
        "left_joint_map_dict": "observation.state",
        # "left_gripper_map_dict": "observation.state",

        "right_joint_map_dict": "observation.state",
        # "right_gripper_map_dict": "observation.state",

        "left_eef_map_dict": "eef_sim_pose_state",
        "right_eef_map_dict": "eef_sim_pose_state",
    },
    action_meta_source_dict={
        "left_joint_map_dict": "action",
        # "left_gripper_map_dict": "action",

        "right_joint_map_dict": "action",
        # "right_gripper_map_dict": "action",

        "left_eef_map_dict": "eef_sim_pose_action",
        "right_eef_map_dict": "eef_sim_pose_action",
    },
    name_mapping_dict = dict(
        left_joint_map_dict={
            "left_arm_joint_1_rad": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "left_arm_joint_2_rad": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "left_arm_joint_3_rad": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "left_arm_joint_4_rad": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "left_arm_joint_5_rad": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "left_arm_joint_6_rad": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "left_arm_joint_7_rad": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        right_joint_map_dict={
            "right_arm_joint_1_rad": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "right_arm_joint_2_rad": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "right_arm_joint_3_rad": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "right_arm_joint_4_rad": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "right_arm_joint_5_rad": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "right_arm_joint_6_rad": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "right_arm_joint_7_rad": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
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
