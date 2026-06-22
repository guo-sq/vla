#!/usr/bin/env python
from dataclasses import dataclass
from ..config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("bi_piper_leader")  # Must match this exact string
@dataclass
class BiPiperLeaderConfig(TeleoperatorConfig):
    left_arm_port: str
    right_arm_port: str