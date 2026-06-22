from dataclasses import dataclass
from pathlib import Path

from ..configs import SensorConfig


@SensorConfig.register_subclass("paxini_tactile_sensor")
@dataclass
class PaxiniTactileSensorConfig(SensorConfig):
    index_or_path: int | Path
    warmup_s: int = 1
    fourcc: str | None = None

    # def __init__(self, name: str, port: str, baudrate: int):
    #     self.name = name
    #     self.port = port
    #     self.baudrate = baudrate
