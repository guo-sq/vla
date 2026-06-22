#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.piper.piper import PiperConfig
from lerobot.robots.piper.piper import Piper

from ..robot import Robot
from .config_bi_piper_follower import BiPiperFollowerConfig

logger = logging.getLogger(__name__)


class BiPiperFollower(Robot):
    """Bimanual Piper Follower Arms."""

    config_class = BiPiperFollowerConfig
    name = "bi_piper_follower"

    def __init__(self, config: BiPiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self.wait_for_new_frame: bool = True

        self.left_arm = Piper(
            PiperConfig(
                id=f"{config.id}_left" if config.id else None,
                port=config.left_arm_port,
                cameras={},  # Cameras handled at bimanual level
            )
        )

        self.right_arm = Piper(
            PiperConfig(
                id=f"{config.id}_right" if config.id else None,
                port=config.right_arm_port,
                cameras={},  # Cameras handled at bimanual level
            )
        )

        self.cameras = make_cameras_from_configs(config.cameras)

    @cached_property
    def _motors_ft(self) -> dict[str, type]:
        return {f"left_{k}": v for k, v in self.left_arm._motors_ft.items()} | {
            f"right_{k}": v for k, v in self.right_arm._motors_ft.items()
        }

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {cam: (cfg.height, cfg.width, 3) for cam, cfg in self.config.cameras.items()}

    @cached_property
    def observation_features(self) -> dict:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        logger.info(f"Connecting {self}")
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)
        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        logger.info(f"Calibrating {self}")
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_joint_positions(self) -> dict[str, Any]:
        obs_dict = {}

        # Motor observations with prefixes
        left_obs = self.left_arm.get_observation()
        left_motor_obs = {k: v for k, v in left_obs.items() if k in self.left_arm._motors_ft}
        obs_dict.update({f"left_{k}": v for k, v in left_motor_obs.items()})

        right_obs = self.right_arm.get_observation()
        right_motor_obs = {k: v for k, v in right_obs.items() if k in self.right_arm._motors_ft}
        obs_dict.update({f"right_{k}": v for k, v in right_motor_obs.items()})
        return obs_dict


    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Motor observations with prefixes
        left_obs = self.left_arm.get_observation()
        left_motor_obs = {k: v for k, v in left_obs.items() if k in self.left_arm._motors_ft}
        obs_dict.update({f"left_{k}": v for k, v in left_motor_obs.items()})

        right_obs = self.right_arm.get_observation()
        right_motor_obs = {k: v for k, v in right_obs.items() if k in self.right_arm._motors_ft}
        obs_dict.update({f"right_{k}": v for k, v in right_motor_obs.items()})

        # Camera observations
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read(wait_for_new=self.wait_for_new_frame)
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Route to appropriate arms
        left_action = {k[5:]: v for k, v in action.items() if k.startswith("left_")}
        right_action = {k[6:]: v for k, v in action.items() if k.startswith("right_")}

        sent_left = self.left_arm.send_action(left_action) if left_action else {}
        sent_right = self.right_arm.send_action(right_action) if right_action else {}

        # Re-prefix returned actions
        return {f"left_{k}": v for k, v in sent_left.items()} | {
            f"right_{k}": v for k, v in sent_right.items()
        }
    
    def send_action_inference(self, action: dict[str, Any]) -> dict[str, Any]:
        # Route to appropriate arms
        left_action = {k[5:]: v for k, v in action.items() if k.startswith("left_")}
        right_action = {k[6:]: v for k, v in action.items() if k.startswith("right_")}

        sent_left = self.left_arm.send_action(left_action) if left_action else {}
        sent_right = self.right_arm.send_action(right_action) if right_action else {}

        # Re-prefix returned actions
        return {f"left_{k}": v for k, v in sent_left.items()} | {
            f"right_{k}": v for k, v in sent_right.items()
        }

    def disconnect(self) -> None:
        logger.info(f"Disconnecting {self}")
        self.left_arm.disconnect()
        self.right_arm.disconnect()
        for cam in self.cameras.values():
            cam.disconnect()