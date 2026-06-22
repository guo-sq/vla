from .base import RobotInfo, AlignActionDim

# R1_Lite
R1_LITE = RobotInfo(
    # has_eef=True,
    state_meta_source_dict={
        "left_joint_map_dict": "observation.state",
        "left_gripper_map_dict": "observation.state",
        "right_joint_map_dict": "observation.state",
        "right_gripper_map_dict": "observation.state",
        "left_eef_map_dict": "eef_sim_pose_state",
        "right_eef_map_dict": "eef_sim_pose_state",
    },
    action_meta_source_dict={
        "left_joint_map_dict": "action",
        "left_gripper_map_dict": "action",
        "right_joint_map_dict": "action",
        "right_gripper_map_dict": "action",
        "left_eef_map_dict": "eef_sim_pose_action",
        "right_eef_map_dict": "eef_sim_pose_action",
    },
    name_mapping_dict=dict(
        left_joint_map_dict={
            "left_arm_joint_1_rad": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "left_arm_joint_2_rad": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "left_arm_joint_3_rad": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "left_arm_joint_4_rad": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "left_arm_joint_5_rad": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "left_arm_joint_6_rad": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        left_gripper_map_dict={
            "left_gripper_open": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        right_joint_map_dict={
            "right_arm_joint_1_rad": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "right_arm_joint_2_rad": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "right_arm_joint_3_rad": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "right_arm_joint_4_rad": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "right_arm_joint_5_rad": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "right_arm_joint_6_rad": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        right_gripper_map_dict={
            "right_gripper_open": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
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

# R1_LITE_OPEN
R1_LITE_OPEN = RobotInfo(
    # has_eef=True,
    # state key-mapping
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state.left_arm",
        # "state_left_gripper_map_dict": "observation.state.left_gripper",
        "state_right_joint_map_dict": "observation.state.right_arm",
        # "state_right_gripper_map_dict": "observation.state.right_gripper",
        # "left_eef_map_dict": "observation.state.left_ee_pose",
        # "right_eef_map_dict": "observation.state.right_ee_pose",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action.left_arm",
        # "action_left_gripper_map_dict": "action.left_gripper",
        "action_right_joint_map_dict": "action.right_arm",
        # "action_right_gripper_map_dict": "action.right_gripper",
        # "left_eef_map_dict": "action.left_ee_pose",
        # "right_eef_map_dict": "action.right_ee_pose",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            "/hdas/feedback_arm_left.position[0]": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "/hdas/feedback_arm_left.position[1]": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "/hdas/feedback_arm_left.position[2]": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "/hdas/feedback_arm_left.position[3]": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "/hdas/feedback_arm_left.position[4]": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "/hdas/feedback_arm_left.position[5]": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            "/hdas/feedback_gripper_left.position[0]": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_joint_map_dict={
            "/hdas/feedback_arm_right.position[0]": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "/hdas/feedback_arm_right.position[1]": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "/hdas/feedback_arm_right.position[2]": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "/hdas/feedback_arm_right.position[3]": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "/hdas/feedback_arm_right.position[4]": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "/hdas/feedback_arm_right.position[5]": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_right_gripper_map_dict={
            "/hdas/feedback_gripper_right.position[0]": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # action keys
        action_left_joint_map_dict={
            "/motion_target/target_joint_state_arm_left.position[0]": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "/motion_target/target_joint_state_arm_left.position[1]": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "/motion_target/target_joint_state_arm_left.position[2]": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "/motion_target/target_joint_state_arm_left.position[3]": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "/motion_target/target_joint_state_arm_left.position[4]": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "/motion_target/target_joint_state_arm_left.position[5]": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            "/motion_target/target_position_gripper_left.position[0]": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_joint_map_dict={
            "/motion_target/target_joint_state_arm_right.position[0]": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "/motion_target/target_joint_state_arm_right.position[1]": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "/motion_target/target_joint_state_arm_right.position[2]": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "/motion_target/target_joint_state_arm_right.position[3]": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "/motion_target/target_joint_state_arm_right.position[4]": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "/motion_target/target_joint_state_arm_right.position[5]": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_right_gripper_map_dict={
            "/motion_target/target_position_gripper_right.position[0]": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # action_left_eef_map_dict={
        #     "/motion_control/pose_ee_arm_left.pose.position.x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
        #     "/motion_control/pose_ee_arm_left.pose.position.y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
        #     "/motion_control/pose_ee_arm_left.pose.position.z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
        #     "/motion_control/pose_ee_arm_left.pose.orientation.x": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
        #     "/motion_control/pose_ee_arm_left.pose.orientation.y": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
        #     "/motion_control/pose_ee_arm_left.pose.orientation.z": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        #     "/motion_control/pose_ee_arm_left.pose.orientation.w": AlignActionDim.LEFT_ALIGN_EEF_ROT_W.value,
        # },
        # action_right_eef_map_dict={
        #     "/motion_control/pose_ee_arm_right.pose.position.x": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
        #     "/motion_control/pose_ee_arm_right.pose.position.y": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
        #     "/motion_control/pose_ee_arm_right.pose.position.z": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
        #     "/motion_control/pose_ee_arm_right.pose.orientation.x": AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
        #     "/motion_control/pose_ee_arm_right.pose.orientation.y": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
        #     "/motion_control/pose_ee_arm_right.pose.orientation.z": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
        #     "/motion_control/pose_ee_arm_right.pose.orientation.w": AlignActionDim.RIGHT_ALIGN_EEF_ROT_W.value,
        # },
    ),
)

# R1_LITE_OPEN
R1_LITE_PRO = RobotInfo(
    # has_eef=True,
    # state key-mapping
    state_meta_source_dict={
        "state_left_joint_map_dict": "observation.state.left_arm",
        # "state_left_gripper_map_dict": "observation.state.left_gripper",
        "state_right_joint_map_dict": "observation.state.right_arm",
        # "state_right_gripper_map_dict": "observation.state.right_gripper",
        # "left_eef_map_dict": "observation.state.left_ee_pose",
        # "right_eef_map_dict": "observation.state.right_ee_pose",
    },
    action_meta_source_dict={
        "action_left_joint_map_dict": "action.left_arm",
        # "action_left_gripper_map_dict": "action.left_gripper",
        "action_right_joint_map_dict": "action.right_arm",
        # "action_right_gripper_map_dict": "action.right_gripper",
        # "left_eef_map_dict": "action.left_ee_pose",
        # "right_eef_map_dict": "action.right_ee_pose",
    },
    name_mapping_dict=dict(
        state_left_joint_map_dict={
            "/hdas/feedback_arm_left.position[0]": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "/hdas/feedback_arm_left.position[1]": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "/hdas/feedback_arm_left.position[2]": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "/hdas/feedback_arm_left.position[3]": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "/hdas/feedback_arm_left.position[4]": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "/hdas/feedback_arm_left.position[5]": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "/hdas/feedback_arm_left.position[6]": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_left_gripper_map_dict={
            "/hdas/feedback_gripper_left.position[0]": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        state_right_joint_map_dict={
            "/hdas/feedback_arm_right.position[0]": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "/hdas/feedback_arm_right.position[1]": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "/hdas/feedback_arm_right.position[2]": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "/hdas/feedback_arm_right.position[3]": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "/hdas/feedback_arm_right.position[4]": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "/hdas/feedback_arm_right.position[5]": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "/hdas/feedback_arm_right.position[6]": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        state_right_gripper_map_dict={
            "/hdas/feedback_gripper_right.position[0]": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # action keys
        action_left_joint_map_dict={
            "/motion_target/target_joint_state_arm_left.position[0]": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "/motion_target/target_joint_state_arm_left.position[1]": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "/motion_target/target_joint_state_arm_left.position[2]": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "/motion_target/target_joint_state_arm_left.position[3]": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "/motion_target/target_joint_state_arm_left.position[4]": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "/motion_target/target_joint_state_arm_left.position[5]": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "/motion_target/target_joint_state_arm_left.position[6]": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_left_gripper_map_dict={
            "/motion_target/target_position_gripper_left.position[0]": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_right_joint_map_dict={
            "/motion_target/target_joint_state_arm_right.position[0]": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "/motion_target/target_joint_state_arm_right.position[1]": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "/motion_target/target_joint_state_arm_right.position[2]": AlignActionDim.RIGHT_ALIGN_JOINT_3.value,
            "/motion_target/target_joint_state_arm_right.position[3]": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "/motion_target/target_joint_state_arm_right.position[4]": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "/motion_target/target_joint_state_arm_right.position[5]": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "/motion_target/target_joint_state_arm_right.position[6]": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        action_right_gripper_map_dict={
            "/motion_target/target_position_gripper_right.position[0]": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
        # action_left_eef_map_dict={
        #     "/motion_control/pose_ee_arm_left.pose.position.x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
        #     "/motion_control/pose_ee_arm_left.pose.position.y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
        #     "/motion_control/pose_ee_arm_left.pose.position.z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
        #     "/motion_control/pose_ee_arm_left.pose.orientation.x": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
        #     "/motion_control/pose_ee_arm_left.pose.orientation.y": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
        #     "/motion_control/pose_ee_arm_left.pose.orientation.z": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
        #     "/motion_control/pose_ee_arm_left.pose.orientation.w": AlignActionDim.LEFT_ALIGN_EEF_ROT_W.value,
        # },
        # action_right_eef_map_dict={
        #     "/motion_control/pose_ee_arm_right.pose.position.x": AlignActionDim.RIGHT_ALIGN_EEF_POS_X.value,
        #     "/motion_control/pose_ee_arm_right.pose.position.y": AlignActionDim.RIGHT_ALIGN_EEF_POS_Y.value,
        #     "/motion_control/pose_ee_arm_right.pose.position.z": AlignActionDim.RIGHT_ALIGN_EEF_POS_Z.value,
        #     "/motion_control/pose_ee_arm_right.pose.orientation.x": AlignActionDim.RIGHT_ALIGN_EEF_ROT_X.value,
        #     "/motion_control/pose_ee_arm_right.pose.orientation.y": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Y.value,
        #     "/motion_control/pose_ee_arm_right.pose.orientation.z": AlignActionDim.RIGHT_ALIGN_EEF_ROT_Z.value,
        #     "/motion_control/pose_ee_arm_right.pose.orientation.w": AlignActionDim.RIGHT_ALIGN_EEF_ROT_W.value,
        # },
    ),
)
