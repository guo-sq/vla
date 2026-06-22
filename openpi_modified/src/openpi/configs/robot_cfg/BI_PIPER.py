from .base import RobotInfo, AlignActionDim

BI_PIPER = RobotInfo(
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
    name_mapping_dict = dict(
        left_joint_map_dict={
            "left_joint_0.pos": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "left_joint_1.pos": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "left_joint_2.pos": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "left_joint_3.pos": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "left_joint_4.pos": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "left_joint_5.pos": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        left_gripper_map_dict={
            "left_joint_6.pos": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        right_joint_map_dict={
            "right_joint_0.pos": AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            "right_joint_1.pos": AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            "right_joint_2.pos": AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            "right_joint_3.pos": AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            "right_joint_4.pos": AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            "right_joint_5.pos": AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        right_gripper_map_dict={
            "right_joint_6.pos": AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
    ),
)

TEST_BI_PIPER = RobotInfo(
    # has_eef=False,
    name_mapping_dict = dict(
        left_joint_map_dict={
            0: AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            1: AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            2: AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            3: AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            4: AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            5: AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        left_gripper_map_dict={
            6: AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        right_joint_map_dict={
            7: AlignActionDim.RIGHT_ALIGN_JOINT_1.value,
            8: AlignActionDim.RIGHT_ALIGN_JOINT_2.value,
            9: AlignActionDim.RIGHT_ALIGN_JOINT_4.value,
            10: AlignActionDim.RIGHT_ALIGN_JOINT_5.value,
            11: AlignActionDim.RIGHT_ALIGN_JOINT_6.value,
            12: AlignActionDim.RIGHT_ALIGN_JOINT_7.value,
        },
        right_gripper_map_dict={
            13: AlignActionDim.RIGHT_ALIGN_GRIPPER.value,
        },
    ),
)
