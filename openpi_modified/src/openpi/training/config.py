import difflib

from openpi.training.base_cfg import *
import openpi.training.misc.roboarena_config as roboarena_config

_CONFIGS = [
    # RoboArena configs.
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli(config="") -> tuple[TrainConfig, str | None]:
    """Load config from file and return (config, config_path).

    Returns:
        Tuple of (TrainConfig, config_file_path). config_file_path is None if not loaded from a file.
    """
    config_path = config if config else None
    config_obj = Config.fromfile(config).cfg
    return config_obj, config_path
    # return tyro.extras.overridable_config_cli(
    #     {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    # )


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        if config_name.endswith(".py"):
            # Try to load from a file.
            try:
                config = Config.fromfile(config_name).cfg
                if not isinstance(config, TrainConfig):
                    raise ValueError(f"Config in file '{config_name}' is not a TrainConfig.")
                return config
            except Exception as e:
                raise ValueError(f"Failed to load config from file '{config_name}': {e}") from e
        else:
            closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
            closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
            raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
