# Implementation of ArxX5 robot for LeRobot

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


@RobotConfig.register_subclass("arxx5")
@dataclass
class ArxX5Config(RobotConfig):
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

    arm_config: dict[str, Any] = field(
        default_factory=lambda: {
            "can_port": "can1",
            # "type": 0,
        }
        # Add necessary configuration parameters for the left arm
    )
    sensors: dict[str, SensorConfig] | None = None


class ArxX5(Robot):
    config_class = ArxX5Config
    name = "arxx5"

    def __init__(self, config: ArxX5Config):
        super().__init__(config)
        right_arm_configarm_config = config.arm_config
        self.sdk = SingleArm(arm_config)
        self.sdk.gravity_compensation()
        self.cameras = make_cameras_from_configs(config.cameras)
        self.wait_for_new_frame: bool = True
        if config.sensors is not None:
            self.sensors = make_sensors_from_configs(config.sensors)
        else:
            self.sensors = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"joint_{i+1}.pos": float for i in range(7)
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
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

    @property
    def observation_features(self) -> dict:
        if self.sensors:
            return {**self._motors_ft, **self._cameras_ft, **self._sensors_ori_ft, **self._sensor_heat_map_ft}
        else:
            return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict:
        return self._motors_ft

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

    def get_joint_observations(self) -> dict[str, float]:
        positions = self.sdk.get_joint_positions()
        # catch_pos = self.sdk.get_catch_pos()

        print("get_observation: ", positions)
        joint_dict = {f"joint_{i+1}.pos": positions[i] for i in range(7)}
        # joint_dict["joint_7.pos"] = catch_pos
        return joint_dict

    def get_observation(self) -> dict[str, Any]:
        obs_dict = self.get_joint_observations()

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read(wait_for_new=self.wait_for_new_frame)
        if self.sensors:
            for sensor_key, sensor in self.sensors.items():
                obs_dict[sensor_key] = sensor.async_read_ori()
                obs_dict[f"{sensor_key}_heat_map"] = sensor.async_read_heat_map()
        return obs_dict

    def send_ee_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # TODO(heyuan): neet further modification
        # map the action from the leader to joints for the follower
        # import pdb; pdb.set_trace()
        '''
        left_position = np.array([0.0, 0.0, 0.1])  # Example position (x, y, z)
        left_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Example orientation as a quaternion

        right_position = np.array([0.0, 0.0, 0.1])  # Example position (x, y, z)
        right_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Example orientation as a quaternion
        
        poses = {
            'left': (left_position, left_orientation),
            'right': (right_position, right_orientation)
        }
        self.sdk.set_ee_pose(poses)
        '''
        position = np.array([0.0, 0.0, 0.1])  # x, y, z 位置
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # 四元数表示方向
        ee_actions = dict(pos=position, quat=quaternion)
        success = self.sdk.set_ee_pose(**ee_actions)

        return poses

    def send_teleop_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # NOTE(heyuan): map the action from the leader to joints for the follower
        for key in action:
            if key in self.action_features:
                print("send_action: ", key, action[key])
        positions = [ 
            action.get("joint_1.pos"),
            action.get("joint_2.pos"),
            action.get("joint_3.pos"),
            action.get("joint_4.pos"),
            action.get("joint_5.pos"),
            action.get("joint_6.pos"),
            action.get("joint_7.pos"),
        ]
        joint_names = {f"joint{i+1}": positions[i] for i in range(7)}
        joint_actions = dict(positions=positions, joint_names=joint_names)
        # NOTE(heyuan): do nothing when collecting data
        # self.sdk.set_joint_positions(**joint_actions)
        mapped_action = {f"joint_{i+1}.pos": positions[i] for i in range(7)}
        return mapped_action

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # NOTE(heyuan): map the action from the leader to joints for the follower
        for key in action:
            if key in self.action_features:
                print("send_action: ", key, action[key])
        positions = [ 
            action.get("joint_1.pos"),
            action.get("joint_2.pos"),
            action.get("joint_3.pos"),
            action.get("joint_4.pos"),
            action.get("joint_5.pos"),
            action.get("joint_6.pos"),
            action.get("joint_7.pos"),
        ]
        joint_names = {f"joint{i+1}": positions[i] for i in range(7)}
        joint_actions = dict(positions=positions, joint_names=joint_names)
        # NOTE(heyuan): do nothing when collecting data
        self.sdk.set_joint_positions(**joint_actions)
        self.sdk.set_catch_pos(pos=action.get("joint_7.pos"))
        mapped_action = {f"joint_{i+1}.pos": positions[i] for i in range(7)}
        return mapped_action

    def send_action_inference(self, action: dict[str, Any]) -> dict[str, Any]:
        # map the action from the leader to joints for the follower
        # import pdb; pdb.set_trace()
        positions = [action.get(key) for key in self.action_features.keys()]
        self.sdk.set_joint_positions(positions)
        return action
