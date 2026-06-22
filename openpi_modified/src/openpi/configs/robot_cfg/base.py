from flax import struct
from typing import Any, Union
from enum import IntEnum, unique
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class RobotInfo:
    """
    Mapping relationships for various robot parts, totaling 30 dimensions.

    Attributes:
        left_joint_map_dict: Left arm joint mapping, 7 dimensions.
        left_gripper_map_dict: Left gripper mapping, 1 dimension.
        right_joint_map_dict: Right arm joint mapping, 7 dimensions.
        right_gripper_map_dict: Right gripper mapping, 1 dimension.
        left_eef_map_dict: Left end-effector mapping, 6 dimensions.
        right_eef_map_dict: Right end-effector mapping, 6 dimensions.

    """

    # meta mapping dict, exp. "action/names/motors": "action/names",
    meta_mapping_dict: dict[str, str] = struct.field(default_factory=dict)

    # hf_dataset_mapping dict, exp. "actions.joint.position": "action",
    hf_dataset_mapping_dict: dict[str, str] = struct.field(default_factory=dict)

    # get state source which original defined dataset meta/info.json
    state_meta_source_dict: dict[str, str | tuple] = struct.field(default_factory=dict)
    action_meta_source_dict: dict[str, str] = struct.field(default_factory=dict)

    # map dof names in state/action which original defined dataset meta/info.json
    name_mapping_dict: dict[str, dict] = struct.field(default_factory=dict)

    def get_state_name_dict(self) -> dict[str, dict]:
        state_name_dict = {}
        for state_source, state_target in self.state_meta_source_dict.items():
            state_name_dict[state_source] = self.name_mapping_dict[state_source]
        return state_name_dict

    def get_action_name_dict(self) -> dict[str, dict]:
        action_name_dict = {}
        for action_source, action_target in self.action_meta_source_dict.items():
            action_name_dict[action_source] = self.name_mapping_dict[action_source]
        return action_name_dict

    def get_meta_mapping_dict(self) -> dict[str, dict]:
        """
        Get the meta mapping dictionary.

        Returns:
            dict[str, str]: The meta mapping dictionary.
        """
        return self.meta_mapping_dict

    def get_hf_dataset_mapping_dict(self) -> dict[str, dict]:
        """"""
        return self.hf_dataset_mapping_dict

    def __post_init__(self):
        for k in self.state_meta_source_dict.keys():
            if k not in self.name_mapping_dict:
                raise ValueError(
                    f"Key {k} from state_meta_source_dict not found in name_mapping_dict"
                )
        for k in self.action_meta_source_dict.keys():
            if k not in self.name_mapping_dict:
                raise ValueError(
                    f"Key {k} from action_meta_source_dict not found in name_mapping_dict"
                )


@at.typecheck
@struct.dataclass
class RobotAlignInfo:
    """
    Robot alignment information class for storing alignment mapping data of different robots.

    Attributes:
        robot_align_info: A dictionary where keys are robot names and values are corresponding RobotInfo objects.
    """

    robot_align_info: dict[str, RobotInfo] = struct.field(default_factory=dict)


@unique
class AlignActionDim(IntEnum):
    """对齐动作维度枚举"""

    LEFT_ALIGN_JOINT_1 = 0
    LEFT_ALIGN_JOINT_2 = 1
    LEFT_ALIGN_JOINT_3 = 2
    LEFT_ALIGN_JOINT_4 = 3
    LEFT_ALIGN_JOINT_5 = 4
    LEFT_ALIGN_JOINT_6 = 5
    LEFT_ALIGN_JOINT_7 = 6
    LEFT_ALIGN_GRIPPER = 7
    RIGHT_ALIGN_JOINT_1 = 8
    RIGHT_ALIGN_JOINT_2 = 9
    RIGHT_ALIGN_JOINT_3 = 10
    RIGHT_ALIGN_JOINT_4 = 11
    RIGHT_ALIGN_JOINT_5 = 12
    RIGHT_ALIGN_JOINT_6 = 13
    RIGHT_ALIGN_JOINT_7 = 14
    RIGHT_ALIGN_GRIPPER = 15
    LEFT_ALIGN_EEF_POS_X = 16
    LEFT_ALIGN_EEF_POS_Y = 17
    LEFT_ALIGN_EEF_POS_Z = 18
    LEFT_ALIGN_EEF_ROT_X = 19
    LEFT_ALIGN_EEF_ROT_Y = 20
    LEFT_ALIGN_EEF_ROT_Z = 21
    RIGHT_ALIGN_EEF_POS_X = 22
    RIGHT_ALIGN_EEF_POS_Y = 23
    RIGHT_ALIGN_EEF_POS_Z = 24
    RIGHT_ALIGN_EEF_ROT_X = 25
    RIGHT_ALIGN_EEF_ROT_Y = 26
    RIGHT_ALIGN_EEF_ROT_Z = 27
    LEFT_ALIGN_EEF_ROT_W = 28
    RIGHT_ALIGN_EEF_ROT_W = 29

    @classmethod
    def count(cls):
        """返回枚举值的总数"""
        return len(cls)

    @classmethod
    def values(cls):
        """返回所有枚举值列表"""
        return list(cls.__members__.values())

    @classmethod
    def names(cls):
        """返回所有枚举名称列表"""
        return list(cls.__members__.keys())
