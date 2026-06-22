"""Structured task specification loaded from JSON.

A TaskSpec describes *what* a task is: roles, prompts, success criteria,
reset configuration, and safety parameters.  It is robot-agnostic and
environment-agnostic — runtime details (server addresses, dataset paths)
live in the CLI config.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path


# Placeholder syntax supported by ``apply_template``: ``{name}`` or
# ``{nested.path.to.value}``. The session-time substitution pass leaves any
# placeholder it cannot resolve from its context untouched, so per-episode
# placeholders like ``{speed}`` (handled separately in
# ``RoleSpec.get_prompt_for_episode``) survive intact.
_TEMPLATE_RE = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}")


def _substitute_template(template: str, context: dict, *, strict: bool = False) -> str:
    """Replace ``{a.b.c}`` placeholders with ``context['a']['b']['c']`` values.

    Unresolvable placeholders are left as-is by default so that prompts
    mixing session-time tokens (``{socks.color}``) and per-episode tokens
    (``{speed}``) can be substituted in two passes. Pass ``strict=True`` to
    raise ``KeyError`` instead.
    """
    def _resolve(match: "re.Match[str]") -> str:
        path = match.group(1)
        parts = path.split(".")
        obj: object = context
        for p in parts:
            if isinstance(obj, dict) and p in obj:
                obj = obj[p]
            else:
                if strict:
                    raise KeyError(
                        f"prompt template references {{{path}}} but {p!r} is not in context"
                    )
                return match.group(0)
        return str(obj)

    return _TEMPLATE_RE.sub(_resolve, template)


@dataclass
class SuccessCondition:
    at_home: bool = True
    value_gte: float | None = None
    value_lte: float | None = None
    speed_threshold: float | None = None  # if set, speed must be below this for success

    @property
    def needs_value_model(self) -> bool:
        """True if success requires a value model score (has value thresholds)."""
        return self.value_gte is not None or self.value_lte is not None

    @property
    def is_home_only(self) -> bool:
        """True if success is determined solely by reaching home position."""
        return self.at_home and not self.needs_value_model

    @classmethod
    def from_dict(cls, d: dict | None) -> SuccessCondition:
        if not d:
            return cls()
        return cls(
            at_home=d.get("at_home", True),
            value_gte=d.get("value_gte"),
            value_lte=d.get("value_lte"),
            speed_threshold=float(d["speed_threshold"]) if "speed_threshold" in d else None,
        )

    def to_dict(self) -> dict:
        d: dict = {"at_home": self.at_home}
        if self.value_gte is not None:
            d["value_gte"] = self.value_gte
        if self.value_lte is not None:
            d["value_lte"] = self.value_lte
        if self.speed_threshold is not None:
            d["speed_threshold"] = self.speed_threshold
        return d


@dataclass
class PolicyServerConfig:
    """Per-role policy server address. Allows different roles to use different servers."""
    host: str = "localhost"
    port: int = 8001

    @classmethod
    def from_dict(cls, d: dict) -> PolicyServerConfig:
        return cls(host=d.get("host", "localhost"), port=int(d.get("port", 8001)))

    def to_dict(self) -> dict:
        return {"host": self.host, "port": self.port}


@dataclass
class ValueModelConfig:
    """Value model server address (used for self-play success evaluation)."""
    host: str = "localhost"
    port: int = 8002

    @classmethod
    def from_dict(cls, d: dict) -> ValueModelConfig:
        return cls(host=d.get("host", "localhost"), port=int(d.get("port", 8002)))

    def to_dict(self) -> dict:
        return {"host": self.host, "port": self.port}


# Sentinel for "no time limit"
NO_TIME_LIMIT = 999999.0


@dataclass
class SpeedVariation:
    """Cycle through different speed targets in the prompt.

    The prompt must contain a placeholder like "under {speed}s" or "under {speed} seconds".
    Each episode, the speed target cycles: lower → lower+step → ... → upper → lower → ...
    """
    lower_bound_s: float  # e.g. 10
    upper_bound_s: float  # e.g. 30
    step_s: float         # e.g. 5

    @classmethod
    def from_dict(cls, d: dict | None) -> SpeedVariation | None:
        if not d:
            return None
        return cls(
            lower_bound_s=float(d["lower_bound_s"]),
            upper_bound_s=float(d["upper_bound_s"]),
            step_s=float(d["step_s"]),
        )

    def to_dict(self) -> dict:
        return {
            "lower_bound_s": self.lower_bound_s,
            "upper_bound_s": self.upper_bound_s,
            "step_s": self.step_s,
        }

    def generate_values(self) -> list[float]:
        """Generate the list of speed values to cycle through."""
        vals = []
        v = self.lower_bound_s
        while v <= self.upper_bound_s + 1e-9:
            vals.append(round(v, 1))
            v += self.step_s
        return vals if vals else [self.lower_bound_s]


@dataclass
class RoleSpec:
    prompt: str
    max_time_s: float | None = None
    action_mask: str | None = None
    success_when: SuccessCondition = field(default_factory=SuccessCondition)
    policy_server: PolicyServerConfig | None = None
    # What to do when episode times out and robot is not at home:
    #   "recovery_prompt": use N2 prompt to go home
    #   "force_human_takeover": require human to bring robot home
    #   Must be explicitly specified in the task spec for self-play roles.
    timeout_recovery: str | None = None
    # Speed variation: cycle through different target speeds per episode
    speed_variation: SpeedVariation | None = None
    # How long to stay at home before ending the episode even if value model
    # didn't pass the success check.  None = wait indefinitely (only end on
    # value success or episode timeout).  Useful when the value model is
    # unreliable or when you want a finite home-dwell budget per role.
    home_wait_before_end_s: float | None = None
    # Per-role inference parameters (override CLI defaults when set)
    infer_interval: int | None = None
    default_infer_delay: int | None = None

    @property
    def effective_max_time_s(self) -> float:
        """Returns actual max_time_s, using NO_TIME_LIMIT sentinel when None."""
        return self.max_time_s if self.max_time_s is not None else NO_TIME_LIMIT

    def get_prompt_for_episode(self, episode_index: int) -> str:
        """Get the prompt for a given episode, applying speed variation if configured.

        The prompt template should contain '{speed}' placeholder, e.g.:
            "Hang the seatbelt with right hand under {speed} seconds."

        If no speed_variation is set, returns the base prompt unchanged.
        """
        if self.speed_variation is None or "{speed}" not in self.prompt:
            return self.prompt
        values = self.speed_variation.generate_values()
        speed = values[episode_index % len(values)]
        # Format as int if whole number, otherwise float
        speed_str = str(int(speed)) if speed == int(speed) else str(speed)
        return self.prompt.replace("{speed}", speed_str)

    def apply_template(self, context: dict) -> RoleSpec:
        """Substitute ``{a.b.c}`` placeholders in the prompt from ``context``.

        Unresolvable placeholders (notably ``{speed}``, which is handled per
        episode in ``get_prompt_for_episode``) are preserved.
        """
        return replace(self, prompt=_substitute_template(self.prompt, context))

    @classmethod
    def from_dict(cls, d: dict) -> RoleSpec:
        max_time = d.get("max_time_s")
        if max_time is not None:
            max_time = float(max_time)
        policy_server = None
        if "policy_server" in d:
            policy_server = PolicyServerConfig.from_dict(d["policy_server"])
        home_wait_end = d.get("home_wait_before_end_s")
        if home_wait_end is not None:
            home_wait_end = float(home_wait_end)
        infer_interval = d.get("infer_interval")
        if infer_interval is not None:
            infer_interval = int(infer_interval)
        default_infer_delay = d.get("default_infer_delay")
        if default_infer_delay is not None:
            default_infer_delay = int(default_infer_delay)
        return cls(
            prompt=d["prompt"],
            max_time_s=max_time,
            action_mask=d.get("action_mask"),
            success_when=SuccessCondition.from_dict(d.get("success_when")),
            policy_server=policy_server,
            timeout_recovery=d.get("timeout_recovery"),
            speed_variation=SpeedVariation.from_dict(d.get("speed_variation")),
            home_wait_before_end_s=home_wait_end,
            infer_interval=infer_interval,
            default_infer_delay=default_infer_delay,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "prompt": self.prompt,
        }
        if self.max_time_s is not None:
            d["max_time_s"] = self.max_time_s
        if self.action_mask is not None:
            d["action_mask"] = self.action_mask
        if self.success_when != SuccessCondition():
            d["success_when"] = self.success_when.to_dict()
        if self.policy_server is not None:
            d["policy_server"] = self.policy_server.to_dict()
        if self.timeout_recovery is not None:
            d["timeout_recovery"] = self.timeout_recovery
        if self.speed_variation is not None:
            d["speed_variation"] = self.speed_variation.to_dict()
        if self.home_wait_before_end_s is not None:
            d["home_wait_before_end_s"] = self.home_wait_before_end_s
        if self.infer_interval is not None:
            d["infer_interval"] = self.infer_interval
        if self.default_infer_delay is not None:
            d["default_infer_delay"] = self.default_infer_delay
        return d


@dataclass
class ResetConfig:
    reference_pose: list[float]
    threshold: float
    speed_threshold: float
    home_wait_s: float = 2.0

    @classmethod
    def from_dict(cls, d: dict) -> ResetConfig:
        return cls(
            reference_pose=list(d["reference_pose"]),
            threshold=float(d["threshold"]),
            speed_threshold=float(d["speed_threshold"]),
            home_wait_s=float(d.get("home_wait_s", 2.0)),
        )

    def to_dict(self) -> dict:
        return {
            "reference_pose": self.reference_pose,
            "threshold": self.threshold,
            "speed_threshold": self.speed_threshold,
            "home_wait_s": self.home_wait_s,
        }


@dataclass
class RecoveryLevel:
    threshold: int
    prompt: str | None = None
    execute_time_s: float = 5.0     # run recovery prompt for this long before checking stability
    stable_time_s: float = 0.5      # speed must stay low for this long to confirm stable
    speed_threshold: float = 0.1    # speed threshold for stability check during this recovery level
    timeout_s: float | None = None  # hard deadline: escalate to next level after this time
    action: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> RecoveryLevel:
        return cls(
            threshold=int(d["threshold"]),
            prompt=d.get("prompt"),
            execute_time_s=float(d.get("execute_time_s", 5.0)),
            stable_time_s=float(d.get("stable_time_s", 0.5)),
            speed_threshold=float(d.get("speed_threshold", 0.1)),
            timeout_s=float(d["timeout_s"]) if "timeout_s" in d else None,
            action=d.get("action"),
        )

    def to_dict(self) -> dict:
        d: dict = {"threshold": self.threshold}
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.timeout_s is not None:
            d["timeout_s"] = self.timeout_s
        if self.action is not None:
            d["action"] = self.action
        return d


@dataclass
class SafetyConfig:
    collision_current_bounds: dict[str, tuple[float | None, float | None]]
    recovery: dict[str, RecoveryLevel]

    @classmethod
    def from_dict(cls, d: dict) -> SafetyConfig:
        raw_bounds = d.get("collision_current_bounds", {})
        bounds = {}
        for dim_key, pair in raw_bounds.items():
            # Skip inline annotations (e.g. ``_comment_collision_current_bounds``).
            if isinstance(dim_key, str) and dim_key.startswith("_"):
                continue
            lower = float(pair[0]) if pair[0] is not None else None
            upper = float(pair[1]) if pair[1] is not None else None
            bounds[str(dim_key)] = (lower, upper)

        raw_recovery = d.get("recovery", {})
        recovery = {
            name: RecoveryLevel.from_dict(level)
            for name, level in raw_recovery.items()
            if not (isinstance(name, str) and name.startswith("_"))
        }
        return cls(collision_current_bounds=bounds, recovery=recovery)

    def to_dict(self) -> dict:
        bounds = {k: list(v) for k, v in self.collision_current_bounds.items()}
        recovery = {k: v.to_dict() for k, v in self.recovery.items()}
        return {"collision_current_bounds": bounds, "recovery": recovery}


@dataclass
class TaskSpec:
    """Structured task specification.

    Loaded from a JSON file or constructed programmatically.
    Describes roles, prompts, success criteria, reset config, safety,
    and server connection details (policy servers, value model).
    """

    task_id: str
    roles: dict[str, RoleSpec]
    reset: ResetConfig | None = None
    safety: SafetyConfig | None = None
    # Default policy server for all roles (per-role policy_server overrides this)
    policy_server: PolicyServerConfig | None = None
    # Value model server (used for self-play success evaluation)
    value_model: ValueModelConfig | None = None
    # Single-char keyed prompt variants for mid-session Ctrl+<key> switching.
    # Keys must be a single character (letter or digit); the keyboard listener
    # matches via ``key.char.lower()`` so multi-char entries would never fire.
    # See ``PromptSwitcher`` in ``lerobot.utils.control_utils``.
    prompts: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str | Path) -> TaskSpec:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Task spec not found: {path}")
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> TaskSpec:
        if "task_id" not in d:
            raise ValueError("task_id is required")
        if "roles" not in d or not d["roles"]:
            raise ValueError("at least one role is required")

        # Keys starting with "_" inside ``roles`` are inline annotations (e.g.
        # ``_comment_roles``) and not real roles. Skip them so templates can
        # carry per-role documentation without breaking RoleSpec parsing.
        roles = {
            name: RoleSpec.from_dict(role_data)
            for name, role_data in d["roles"].items()
            if not (isinstance(name, str) and name.startswith("_"))
        }
        if not roles:
            raise ValueError("at least one role is required")
        reset = ResetConfig.from_dict(d["reset"]) if d.get("reset") else None
        safety = SafetyConfig.from_dict(d["safety"]) if d.get("safety") else None
        policy_server = PolicyServerConfig.from_dict(d["policy_server"]) if d.get("policy_server") else None
        value_model = ValueModelConfig.from_dict(d["value_model"]) if d.get("value_model") else None

        # ``prompts`` is an optional map of single-char shortcuts → prompt string
        # for live Ctrl+<key> switching during inference. Underscore-prefixed
        # keys are inline annotations (e.g. ``_comment_prompts``) and skipped.
        prompts_raw = d.get("prompts") or {}
        prompts: dict[str, str] = {}
        for k, v in prompts_raw.items():
            sk = str(k)
            if sk.startswith("_"):
                continue
            if len(sk) != 1:
                raise ValueError(
                    f"task_spec.prompts keys must be a single character (used as Ctrl+<key>), "
                    f"got {sk!r}. Use digits or letters like '1','2','a','b'."
                )
            prompts[sk.lower()] = str(v)

        return cls(
            task_id=d["task_id"],
            roles=roles,
            reset=reset,
            safety=safety,
            policy_server=policy_server,
            value_model=value_model,
            prompts=prompts,
        )

    def apply_template(self, context: dict) -> TaskSpec:
        """Return a copy with ``{a.b.c}`` placeholders in role prompts AND
        ``prompts`` variants substituted from ``context`` (typically built
        from ``CollectionInfo`` — see ``recording.record._build_template_context``).

        Unresolvable placeholders are preserved so that per-episode tokens
        like ``{speed}`` survive the session-time pass.
        """
        return replace(
            self,
            roles={name: role.apply_template(context) for name, role in self.roles.items()},
            prompts={k: _substitute_template(v, context) for k, v in self.prompts.items()},
        )

    @classmethod
    def default_single_role(
        cls,
        prompt: str,
        episode_time_s: float = 60,
        reference_pose: list[float] | None = None,
        threshold: float = 0.12,
        speed_threshold: float = 0.01,
    ) -> TaskSpec:
        """Create a minimal single-role task spec for record mode (no server config)."""
        reset = None
        if reference_pose:
            reset = ResetConfig(
                reference_pose=reference_pose,
                threshold=threshold,
                speed_threshold=speed_threshold,
            )
        return cls(
            task_id="default",
            roles={
                "operator": RoleSpec(prompt=prompt, max_time_s=episode_time_s),
            },
            reset=reset,
        )

    def to_dict(self) -> dict:
        d: dict = {
            "task_id": self.task_id,
            "roles": {name: role.to_dict() for name, role in self.roles.items()},
        }
        if self.reset is not None:
            d["reset"] = self.reset.to_dict()
        if self.safety is not None:
            d["safety"] = self.safety.to_dict()
        if self.policy_server is not None:
            d["policy_server"] = self.policy_server.to_dict()
        if self.value_model is not None:
            d["value_model"] = self.value_model.to_dict()
        if self.prompts:
            d["prompts"] = dict(self.prompts)
        return d

    def get_policy_server(self, role_name: str) -> PolicyServerConfig | None:
        """Resolve policy server for a role: per-role config wins over task-level default."""
        role = self.roles.get(role_name)
        if role and role.policy_server:
            return role.policy_server
        return self.policy_server

    @property
    def role_names(self) -> list[str]:
        return list(self.roles.keys())

    @property
    def initial_role(self) -> str:
        return self.role_names[0]

    @property
    def initial_prompt(self) -> str:
        return self.roles[self.initial_role].prompt

    @property
    def is_self_play(self) -> bool:
        return len(self.roles) > 1

    @property
    def has_safety(self) -> bool:
        return self.safety is not None

    @property
    def has_reset(self) -> bool:
        return self.reset is not None
