from .base import RobotInfo, AlignActionDim

AUSTIN_BUDS_OPENX_FRANKA = RobotInfo(
    # has_eef=False,
    # fps = 20
    state_meta_source_dict={
        "state_joint_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_joint_map_dict={
            # state
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
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

AUSTIN_SAILOR_OPENX_FRANKA = RobotInfo(
    # has_eef=False,
    # fps = 20
    state_meta_source_dict={
        "state_eef_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_eef_map_dict={
            "x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "rx": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "ry": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "rz": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
            # TODO(heyuan): need code support ROT_W
            # "rw": AlignActionDim.LEFT_ALIGN_EEF_ROT_W.value,
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


AUSTIN_SIRIUS_OPENX_FRANKA = AUSTIN_SAILOR_OPENX_FRANKA


BC_Z_OPENX_GOOGLE_ROBOT = RobotInfo(
    # fps = 10
    # has_eef=False,
    state_meta_source_dict={
        "state_eef_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_eef_map_dict={
            "x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "roll": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "pitch": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "yaw": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
            # TODO(heyuan): dont know what's this
            # "pad": AlignActionDim.LEFT_ALIGN_PAD.value,
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

BERKELEY_AUTOLAB_OPENX_UR5 = RobotInfo(
    # fps = 10
    # has_eef=False,
    state_meta_source_dict={
        "state_eef_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_eef_map_dict={
            "x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "rx": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "ry": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "rz": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
            # TODO(heyuan): need code support ROT_W
            # "rw": AlignActionDim.LEFT_ALIGN_EEF_ROT_W.value,
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


BERKELEY_CABLE_ROUTING_OPENX_FRANKA = RobotInfo(
    # fps = 10
    # has_eef=False,
    state_meta_source_dict={
        "state_joint_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_gripper_map_dict={
            "pad": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
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


BERKELEY_FANUC_MANIPULATION_OPENX_FANUC_MATE = RobotInfo(
    # fps = 10
    # has_eef=False,
    state_meta_source_dict={
        "state_joint_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            # "pad": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
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

BERKELEY_MVP_OPENX_XARM = RobotInfo(
    # fps = 5
    state_meta_source_dict={
        "state_eef_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_joint_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_eef_map_dict={
            "x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "rx": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "ry": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "rz": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
            # TODO(heyuan): need code support ROT_W
            # "rw": AlignActionDim.LEFT_ALIGN_EEF_ROT_W.value,
        },
        state_gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
        action_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        action_gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
    ),
)


BERKELEY_RPT_OPENX_FRANKA = RobotInfo(
    # has_eef=False,
    # fps = 20
    state_meta_source_dict={
        "joint_map_dict": "observation.state",
        "gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "joint_map_dict": "action",
        "gripper_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        joint_map_dict={
            # state
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
    ),
)


BRIDGE_ORIG_OPENX_WINDOWX = RobotInfo(
    # has_eef=False,
    # fps = 5
    state_meta_source_dict={
        "state_eef_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_eef_map_dict": "action",
        "action_gripper_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_eef_map_dict={
            "x": AlignActionDim.LEFT_ALIGN_EEF_POS_X.value,
            "y": AlignActionDim.LEFT_ALIGN_EEF_POS_Y.value,
            "z": AlignActionDim.LEFT_ALIGN_EEF_POS_Z.value,
            "roll": AlignActionDim.LEFT_ALIGN_EEF_ROT_X.value,
            "pitch": AlignActionDim.LEFT_ALIGN_EEF_ROT_Y.value,
            "yaw": AlignActionDim.LEFT_ALIGN_EEF_ROT_Z.value,
            # TODO(heyuan): need code support pad
            # "pad": AlignActionDim.LEFT_ALIGN_PAD.value,
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
            # TODO(heyuan): need code support pad
            # "pad": AlignActionDim.LEFT_ALIGN_PAD.value,
        },
        action_gripper_map_dict={
            "gripper": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
        },
    ),
)

# fps = 5
CMU_PLAY_FUSION_OPENX_FRANKA = AUSTIN_BUDS_OPENX_FRANKA

CMU_STRETCH_OPENX_STRETCH = RobotInfo(
    # fps = 10
    # has_eef=False,
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

DLR_EDAN_SHARED_CONTROL_OPENX = CMU_STRETCH_OPENX_STRETCH
DOBBE_OPENX_STRETCH = CMU_STRETCH_OPENX_STRETCH
FMB_OPENX_FRANKA = CMU_STRETCH_OPENX_STRETCH

FRACTRAL_OPENX_GOOGLE_ROBOT = BERKELEY_AUTOLAB_OPENX_UR5
FURNITURE_BENCH_OPENX_FRANKA = BERKELEY_AUTOLAB_OPENX_UR5


IAMLAB_CMU_PICKUP_INSERT_OPENX_FRANKA = RobotInfo(
    # fps = 10
    # has_eef=False,
    state_meta_source_dict={
        "state_joint_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_gripper_map_dict={
            "motor_7": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
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

JACO_PLAY_OPENX_JACO = CMU_STRETCH_OPENX_STRETCH
KUKA_OPENX = BERKELEY_AUTOLAB_OPENX_UR5
LANGUAGE_TABLE_OPENX = BC_Z_OPENX_GOOGLE_ROBOT

NYU_DOOR_OPEN_OPENX_STRETCH = RobotInfo(
    # fps = 10
    # has_eef=False,
    state_meta_source_dict={
        "state_joint_map_dict": "observation.state",
        "state_gripper_map_dict": "observation.state",
    },
    action_meta_source_dict={
        "action_gripper_map_dict": "action",
        "action_eef_map_dict": "action",
    },
    meta_mapping_dict={
        "action/names/motors": "action/names",
        "observation.state/names/motors": "observation.state/names",
    },
    name_mapping_dict=dict(
        state_joint_map_dict={
            "motor_0": AlignActionDim.LEFT_ALIGN_JOINT_1.value,
            "motor_1": AlignActionDim.LEFT_ALIGN_JOINT_2.value,
            "motor_2": AlignActionDim.LEFT_ALIGN_JOINT_3.value,
            "motor_3": AlignActionDim.LEFT_ALIGN_JOINT_4.value,
            "motor_4": AlignActionDim.LEFT_ALIGN_JOINT_5.value,
            "motor_5": AlignActionDim.LEFT_ALIGN_JOINT_6.value,
            "motor_6": AlignActionDim.LEFT_ALIGN_JOINT_7.value,
        },
        state_gripper_map_dict={
            "motor_7": AlignActionDim.LEFT_ALIGN_GRIPPER.value,
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

NYU_FRANKA_PLAY_OPENX = BRIDGE_ORIG_OPENX_WINDOWX
ROBOTURK_OPENX_FRANKA = NYU_DOOR_OPEN_OPENX_STRETCH

STANFORD_HYDRA_OPENX_FRANKA = BRIDGE_ORIG_OPENX_WINDOWX
TACO_PLAY_OPENX_FRANKA = BRIDGE_ORIG_OPENX_WINDOWX
TOTO_OPENX_FRANKA = BERKELEY_CABLE_ROUTING_OPENX_FRANKA
USCD_KITCHEN_OPENX_XARM = IAMLAB_CMU_PICKUP_INSERT_OPENX_FRANKA
UTAUSTIN_MUTEX_OPENX_FRANKA = AUSTIN_BUDS_OPENX_FRANKA
VIOLA_OPENX_FRANKA = IAMLAB_CMU_PICKUP_INSERT_OPENX_FRANKA
