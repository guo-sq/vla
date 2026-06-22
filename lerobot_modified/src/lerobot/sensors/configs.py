import abc
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import draccus


@dataclass(kw_only=True)
class SensorConfig(draccus.ChoiceRegistry, abc.ABC):
    baudrate: int | None = None
    com_port: str | Path
    num_modules: int | None = None
    points_per_module: int | None = None

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)
