import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    LEROBOT = "lerobot"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.LIBERO
    robot_type: str = "bi_piper_follower"

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    host: str = "0.0.0.0"
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Enable the value scoring endpoint for value prediction.
    # Only works with models that have RL value head enabled.
    enable_value_endpoint: bool = True

    # Temperature scaling for distributional value prediction (value_bins > 1).
    # Higher values (> 1.0) produce smoother distributions.
    # Has no effect when value_bins == 1 (scalar value prediction).
    value_temperature: float = 1.0

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.LEROBOT: Checkpoint(
        config="src/openpi/configs/cfg_pi0.5_pour_water.py",
        dir="checkpoints/cfg_pi0.5_pour_water/cfg_pi0.5_pour_water_exp/39999",
    ),
}


def create_default_policy(
    env: EnvMode,
    *,
    default_prompt: str | None = None,
    robot_type: str = "bi_piper_follower",
) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        config = _config.get_config(checkpoint.config)
        return _policy_config.create_trained_policy(
            config,
            checkpoint.dir,
            default_prompt=default_prompt,
            unify_action_mode=config.data.unify_action_space,
            robot_type=robot_type,  # refer src/openpi/training/utils.py
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            config = _config.get_config(args.policy.config)
            return _policy_config.create_trained_policy(
                config,
                args.policy.dir,
                default_prompt=args.default_prompt,
                unify_action_mode=config.data.unify_action_space,
                robot_type=args.robot_type,  # refer src/openpi/training/utils.py
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt, robot_type=args.robot_type)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy_metadata,
        enable_score_endpoint=args.enable_value_endpoint,
        value_temperature=args.value_temperature,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
