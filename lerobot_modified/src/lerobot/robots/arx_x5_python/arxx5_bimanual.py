#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from typing import Any, Dict


from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import Robot, RobotConfig
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from typing import Any, Dict
from lerobot.sensors import SensorConfig
from lerobot.sensors.paxini_tactile_sensor import PaxiniTactileSensorConfig
from lerobot.sensors import make_sensors_from_configs
from lerobot.robots.arx_x5_python.bimanual import BimanualArm, SingleArm

logger = logging.getLogger(__name__)


@RobotConfig.register_subclass("arxx5_bimanual")
@dataclass
class ArxX5BimanualConfig(RobotConfig):
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "cam_1": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=640,
                height=480,
            ),
        }
    )

    left_arm_config: dict[str, Any] = field(
        default_factory=lambda: {
            "can_port": "can1",
            # "type": 0,
        }
        # Add necessary configuration parameters for the left arm
    )
    right_arm_config: dict[str, Any] = field(
        default_factory=lambda: {
            "can_port": "can3",
            # "type": 0,
        }
        # Add necessary configuration parameters for the left arm
    )
    sensors: dict[str, SensorConfig] | None = None
class ArxX5Bimanual(Robot):
    """
    [Bimanual SO-100 Follower Arms](GitHub - TheRobotStudio/SO-ARM100: Standard Open Arm 100) designed by TheRobotStudio
    This bimanual robot can also be easily adapted to use SO-101 follower arms, just replace the SO100Follower class with SO101Follower and SO100FollowerConfig with SO101FollowerConfig.
    """

    config_class = ArxX5BimanualConfig
    name = "arxx5_bimanual"

    def __init__(self, config: ArxX5BimanualConfig):
        super().__init__(config)
        self.config = config

        self.sdk = BimanualArm(config.left_arm_config, config.right_arm_config)
        self.sdk.gravity_compensation()

        self.cameras = make_cameras_from_configs(config.cameras)
        self.wait_for_new_frame: bool = True
        if config.sensors is not None:
            self.sensors = make_sensors_from_configs(config.sensors)
        else:
            self.sensors = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_joint_{i+1}.pos": float for i in range(7)} | {
            f"right_joint_{i+1}.pos": float for i in range(7)
        } | {f"left_joint_{i+1}.vel": float for i in range(7)} | {
            f"right_joint_{i+1}.vel": float for i in range(7)
        } | {f"left_joint_{i+1}.cur": float for i in range(7)} | {
            f"right_joint_{i+1}.cur": float for i in range(7)
        } | {f"left.ee.{i+1}.xyzwxyz": float for i in range(7)} | {
            f"right.ee.{i+1}.xyzwxyz": float for i in range(7)
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }
    @property
    def _sensors_ori_ft(self) -> dict[str, tuple]:
            if not self.sensors:
                return {}
            return {
                sensor: (
                    self.sensors[sensor].num_modules,
                    self.sensors[sensor].points_per_module,
                    3,
                )
                for sensor in self.sensors
            }
    @property
    def _sensor_heat_map_ft(self) -> dict[str, tuple]:
        if not self.sensors:
            return {}
        return {
            f"{sensor}_heat_map": (
                90,  # height
                75*self.sensors[sensor].num_modules,  # width
                3,
            )
            for sensor in self.sensors
        }
    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        if self.sensors:
            return {**self._motors_ft, **self._cameras_ft, **self._sensors_ori_ft, **self._sensor_heat_map_ft}
        else:
            return {**self._motors_ft, **self._cameras_ft}
    @property
    def action_features(self) -> dict:
        return {f"left_joint_{i+1}.pos": float for i in range(7)} | {
            f"right_joint_{i+1}.pos": float for i in range(7)
        }

    @property
    def is_connected(self) -> bool:
        # Assume always connected after SDK init
        return True

    def connect(self, calibrate: bool = True) -> None:
        # Already connected in SDK init
        for cam in self.cameras.values():
            cam.connect()
        if self.sensors:
            for sensor in self.sensors.values():
                sensor.SerialPortConnect()
        self.configure()

    def disconnect(self) -> None:
        for cam in self.cameras.values():
            cam.disconnect()
        if self.sensors:
            for sensor in self.sensors.values():
                sensor.disconnect()

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_joint_positions(self) -> dict[str, float]:
        position_dict = {}
        joint_positions = self.sdk.get_joint_positions("both")
        position_dict.update(
            {f"left_joint_{i+1}.pos": joint_positions["left"][i] for i in range(7)}
        )
        position_dict.update(
            {f"right_joint_{i+1}.pos": joint_positions["right"][i] for i in range(7)}
        )
        return position_dict

    def get_joint_velocities(self) -> dict[str, float]:
        velocity_dict = {}
        joint_velocities = self.sdk.get_joint_velocities("both")
        velocity_dict.update(
            {f"left_joint_{i+1}.vel": joint_velocities["left"][i] for i in range(7)}
        )
        velocity_dict.update(
            {f"right_joint_{i+1}.vel": joint_velocities["right"][i] for i in range(7)}
        )
        return velocity_dict

    def get_joint_currents(self) -> dict[str, float]:
        current_dict = {}
        joint_currents = self.sdk.get_joint_currents("both")
        current_dict.update(
            {f"left_joint_{i+1}.cur": joint_currents["left"][i] for i in range(7)}
        )
        current_dict.update(
            {f"right_joint_{i+1}.cur": joint_currents["right"][i] for i in range(7)}
        )
        return current_dict

    def get_ee_pose(self) -> dict[str, float]:
        pose_dict = {}
        ee_pose = self.sdk.get_ee_pose("both")
        pose_dict.update(
            {f"left.ee.{i+1}.xyzwxyz": ee_pose["left"][i] for i in range(7)}
        )
        pose_dict.update(
            {f"right.ee.{i+1}.xyzwxyz": ee_pose["right"][i] for i in range(7)}
        )
        return pose_dict

    def get_current_vector(self) -> list[float]:
        """返回按维度排序的电流值列表：[left_1, ..., left_7, right_1, ..., right_7]。

        用于碰撞检测等需要按索引访问电流值的场景，避免每次重复扫描和排序 observation keys。
        """
        joint_currents = self.sdk.get_joint_currents("both")
        return list(joint_currents["left"]) + list(joint_currents["right"])

    def get_joint_observations(self) -> dict[str, float]:
        obs_dict = {}
        obs_dict.update(self.get_joint_positions())
        obs_dict.update(self.get_joint_velocities())
        obs_dict.update(self.get_joint_currents())
        obs_dict.update(self.get_ee_pose())
        return obs_dict

    def set_gravity_compensation_mode(self) -> dict[str, bool]:
        self.sdk.gravity_compensation()

    def go_home(self) -> dict[str, bool]:
        return self.sdk.go_home()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = self.get_joint_observations()

        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read(wait_for_new=self.wait_for_new_frame)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")
        if self.sensors:
            for sensor_key, sensor in self.sensors.items():
                obs_dict[sensor_key] = sensor.async_read_ori()
                obs_dict[f"{sensor_key}_heat_map"] = sensor.async_read_heat_map()

        return obs_dict

    def mapping_actions(self, action: dict[str, Any]) -> dict[str, Any]:
        mapped_left_action = {
            key: value.item() if hasattr(value, 'item') and callable(getattr(value, 'item')) else value
            for key, value in action.items() if key.startswith("left_joint")
        }
        mapped_right_action = {
            key: value.item() if hasattr(value, 'item') and callable(getattr(value, 'item')) else value
            for key, value in action.items() if key.startswith("right_joint")
        }
        return mapped_left_action | mapped_right_action

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value.item() if hasattr(value, 'item') and callable(getattr(value, 'item')) else value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value.item() if hasattr(value, 'item') and callable(getattr(value, 'item')) else value for key, value in action.items() if key.startswith("right_")
        }

        joint_names = {
            'left': [f"joint_{i+1}.pos" for i in range(7)],
            'right': [f"joint_{i+1}.pos" for i in range(7)]
        }
        # get left_actions_list and right_actions_list
        left_actions_list = [left_action.get(joint_name, None) for joint_name in joint_names.get('left', [])]
        right_actions_list = [right_action.get(joint_name, None) for joint_name in joint_names.get('right', [])]

        # TBD(heyuan): 夹爪动作处理
        gripper_close = 3.0
        if left_actions_list[-1] > gripper_close:
            left_actions_list[-1] = 3.5
        if right_actions_list[-1] > gripper_close:
            right_actions_list[-1] = 3.5

        poses = {
            'left': left_actions_list,
            'right': right_actions_list
        }

        self.sdk.set_joint_positions(poses)

        return self.mapping_actions(action)


    def disconnect(self):
        # self.left_arm.disconnect()
        # self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()