from pathlib import Path
from typing import TypeAlias

from .sensor import Sensor
from .configs import SensorConfig

IndexOrPath: TypeAlias = int | Path


def make_sensors_from_configs(
    sensor_configs: dict[str, SensorConfig],
) -> dict[str, Sensor]:
    sensors = {}

    for key, cfg in sensor_configs.items():
        if cfg.type == "paxini_tactile_sensor":
            from .paxini_tactile_sensor import PaxiniTactileSensor

            sensors[key] = PaxiniTactileSensor(cfg)

        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return sensors
