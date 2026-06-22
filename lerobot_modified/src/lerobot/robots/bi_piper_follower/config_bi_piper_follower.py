#!/usr/bin/env python
from dataclasses import dataclass, field
from lerobot.cameras import CameraConfig
from ..config import RobotConfig

@RobotConfig.register_subclass("bi_piper_follower")  # Must match this exact string
@dataclass
class BiPiperFollowerConfig(RobotConfig):
    left_arm_port: str
    right_arm_port: str
    cameras: dict[str, CameraConfig] = field(default_factory=dict)