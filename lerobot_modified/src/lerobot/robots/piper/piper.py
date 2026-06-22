# Implementation of Piper robot for LeRobot

from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.robots import Robot, RobotConfig
from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from typing import Any
from .piper_sdk_interface import PiperSDKInterface
from lerobot.sensors import SensorConfig
from lerobot.sensors.paxini_tactile_sensor import PaxiniTactileSensorConfig
from lerobot.sensors import make_sensors_from_configs


@dataclass
@RobotConfig.register_subclass("piper")
@dataclass
class PiperConfig(RobotConfig):
    port: str
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
    sensors: dict[str, SensorConfig] | None = None
class Piper(Robot):
    config_class = PiperConfig
    name = "piper"

    def __init__(self, config: PiperConfig):
        super().__init__(config)
        self.sdk = PiperSDKInterface(port=config.port)
        self.cameras = make_cameras_from_configs(config.cameras)
        self.wait_for_new_frame: bool = True
        if config.sensors is not None:
            self.sensors = make_sensors_from_configs(config.sensors)
        else:
            self.sensors = {}
    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"joint_{i}.pos": float for i in range(7)
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
        # self.sdk.connect()
        # Already connected in SDK init
        for cam in self.cameras.values():
            cam.connect()
        if self.sensors:
            for sensor in self.sensors.values():
                sensor.SerialPortConnect()
        self.configure()

    def disconnect(self) -> None:
        self.sdk.disconnect()
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

    def get_observation(self) -> dict[str, Any]:
        obs_dict = self.sdk.get_status()

        for cam_key, cam in self.cameras.items():
            obs_dict[cam_key] = cam.async_read(wait_for_new=self.wait_for_new_frame)
        if self.sensors:
            for sensor_key, sensor in self.sensors.items():
                obs_dict[sensor_key] = sensor.async_read_ori()
                obs_dict[f"{sensor_key}_heat_map"] = sensor.async_read_heat_map()
        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # map the action from the leader to joints for the follower
        # import pdb; pdb.set_trace()
        positions = [ 
            # 先检查 "shoulder_pan.pos" 是否存在，存在则用其值（即使为 0），否则用 "joint_0.pos"
            action["shoulder_pan.pos"] if "shoulder_pan.pos" in action else action.get("joint_0.pos"),
            action["shoulder_lift.pos"] if "shoulder_lift.pos" in action else action.get("joint_1.pos"),
            action["elbow_flex.pos"] if "elbow_flex.pos" in action else action.get("joint_2.pos"),
            0 if "joint_3.pos" not in action else action.get("joint_3.pos"),
            action["wrist_flex.pos"] if "wrist_flex.pos" in action else action.get("joint_4.pos"),
            action["wrist_roll.pos"] if "wrist_roll.pos" in action else action.get("joint_5.pos"),
            action["gripper.pos"] if "gripper.pos" in action else action.get("joint_6.pos"),
        ]

        self.sdk.set_joint_positions(positions)
        mapped_action = {f"joint_{i}.pos": positions[i] for i in range(7)}
        return mapped_action

    def send_action_inference(self, action: dict[str, Any]) -> dict[str, Any]:
        # map the action from the leader to joints for the follower
        # import pdb; pdb.set_trace()
        positions = [action.get(key) for key in self.action_features.keys()]
        self.sdk.set_joint_positions(positions)
        return action