import abc
from typing import Any

import numpy as np

from .configs import SensorConfig


class Sensor(abc.ABC):

    def __init__(self, config: SensorConfig):
        self.baudrate: int | None = config.baudrate
        self.com_port: str | None = config.com_port
        self.num_modules: int | None = config.num_modules
        self.points_per_module: int | None = config.points_per_module

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    # @staticmethod
    # @abc.abstractmethod
    # def find_sensors() -> list[dict[str, Any]]:
    #     pass

    @abc.abstractmethod
    def SerialPortConnect(self) -> bool:
        pass

    @abc.abstractmethod
    def read_registers(self):
        pass

    def async_read_ori(self, timeout_ms: float = ...) -> np.ndarray:

        pass
    def async_read_heat_map(self, timeout_ms: float = ...) -> np.ndarray:
        pass
    @abc.abstractmethod
    def disconnect(self) -> None:
        pass
