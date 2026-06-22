#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
import logging
from functools import cached_property

from lerobot.teleoperators.piper_leader.config_piper_leader import PiperLeaderConfig
from lerobot.teleoperators.piper_leader.piper_leader import PiperLeader

from ..teleoperator import Teleoperator
from .config_bi_piper_leader import BiPiperLeaderConfig

logger = logging.getLogger(__name__)


class BiPiperLeader(Teleoperator):
    """Bimanual Piper Leader Arms for teleoperation."""

    config_class = BiPiperLeaderConfig
    name = "bi_piper_leader"

    def __init__(self, config: BiPiperLeaderConfig):
        super().__init__(config)
        self.config = config

        self.left_arm = PiperLeader(
            PiperLeaderConfig(
                id=f"{config.id}_left" if config.id else None,
                calibration_dir=config.calibration_dir,
                port=config.left_arm_port,
            )
        )

        self.right_arm = PiperLeader(
            PiperLeaderConfig(
                id=f"{config.id}_right" if config.id else None,
                calibration_dir=config.calibration_dir,
                port=config.right_arm_port,
            )
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{k}": v for k, v in self.left_arm.action_features.items()} | {
            f"right_{k}": v for k, v in self.right_arm.action_features.items()
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {f"left_{k}": v for k, v in self.left_arm.feedback_features.items()} | {
            f"right_{k}": v for k, v in self.right_arm.feedback_features.items()
        }

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

    @property
    def is_calibrated(self) -> bool:
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, float]:
        action_dict = {}
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{k}": v for k, v in left_action.items()})
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{k}": v for k, v in right_action.items()})
        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        left_fb = {k[5:]: v for k, v in feedback.items() if k.startswith("left_")}
        right_fb = {k[6:]: v for k, v in feedback.items() if k.startswith("right_")}
        if left_fb:
            self.left_arm.send_feedback(left_fb)
        if right_fb:
            self.right_arm.send_feedback(right_fb)

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()