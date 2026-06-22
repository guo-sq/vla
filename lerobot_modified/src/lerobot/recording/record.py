"""Unified recording entry point.

Single command supporting four modes:
  - record:      Pure teleop recording
  - infer:       Pure policy inference (no dataset)
  - infer_record: Policy inference + recording
  - self_play:   Builder/destroyer self-play with value model

Usage:
    python -m lerobot.recording.record --mode self_play \\
        --task_spec_path=lerobot_example_config_files/task_specs/seatbelt/arxx5_self_play.json \\
        --robot.type=arxx5_bimanual ...
"""

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional, Union

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features, write_json
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    make_robot_from_config,
    arx_x5_python,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    piper,
    so100_follower,
    so101_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.utils.control_utils import (
    PromptSwitcher,
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

from lerobot.recording.runtime.control_loop import ControlLoop, TeleopSource
from lerobot.recording.runtime.intervention import InterventionRuntime
from lerobot.recording.runtime.policy_runtime import PolicyRuntime
from lerobot.recording.runtime.safety_runtime import SafetyRuntime
from lerobot.recording.task.episode_manager import (
    RoleManager,
    SelfPlayStatsAggregator,
)
from lerobot.recording.task.collection_info import CollectionInfo
from lerobot.recording.task.evaluators import HomeEvaluator
from lerobot.recording.task.session_config import SessionConfig
from lerobot.recording.task.state_machine import StateMachine
from lerobot.recording.task.subtask_manager import SubTaskManager
from lerobot.recording.task.task_spec import TaskSpec
from lerobot.recording.utils.logging import ActionChunkLogger, AsyncLogger, SelfPlayLogger, debug_print
from lerobot.recording.utils.tts import log_say, make_log_say


def _wait_until_home_stable(
    robot, home_evaluator, events, cfg, log_say_fn=None, logger=None,
    check_interval_s=0.5, timeout_s=120.0,
):
    """Block until robot is at home and speed-stable. Used between episodes.

    If robot is not at home, announces and waits. If the operator needs to
    move it back manually (gravity comp mode), they should do so.
    """
    import numpy as np

    if home_evaluator.is_home(robot.get_observation()):
        return  # already at home and stable

    if log_say_fn:
        log_say_fn("等待机械臂回到初始位")
    if logger:
        logger.log("Post-episode: waiting for robot to reach home position")

    # Enable gravity comp so operator can manually move robot back
    if hasattr(robot, "set_gravity_compensation_mode"):
        robot.set_gravity_compensation_mode()

    start = time.perf_counter()
    stable_since = None
    stable_time_s = 1.0  # wait 1s of stability at home

    while time.perf_counter() - start < timeout_s:
        if events.get("stop_recording"):
            break
        obs = robot.get_observation()
        at_home = home_evaluator.is_home(obs)
        if at_home:
            if stable_since is None:
                stable_since = time.perf_counter()
            elif time.perf_counter() - stable_since >= stable_time_s:
                if logger:
                    logger.log("Post-episode: home position reached and stable")
                return
        else:
            stable_since = None
        time.sleep(check_interval_s)

    if logger:
        logger.log("Post-episode: home wait timed out, proceeding anyway")

os.environ.setdefault("SDL_AUDIODRIVER", "pulse")

INFO_PATH = "meta/info.json"


# ---------------------------------------------------------------------------
# CLI Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DatasetRecordConfig:
    repo_id: str = ""
    single_task: str = ""
    root: Optional[Union[str, Path]] = None
    fps: int = 30
    episode_time_s: Union[int, float] = 60
    reset_time_s: Union[int, float] = 5
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: Optional[List[str]] = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1


@dataclass
class RecordConfig:
    """Unified config for all recording modes.

    Parameter applicability by mode:
      record:       teleop, dataset.episode_time_s, dataset.reset_time_s, sub_task_durations, auto_success
      infer:        task_spec_path (required), inference params — no dataset, no teleop
      infer_record: task_spec_path (required), inference params, intervention params, dataset, teleop
      self_play:    task_spec_path (required), inference params, intervention params,
                    self_play_max_total_time_s, auto_success
                    (episode time comes from task_spec roles, NOT dataset.episode_time_s)

    Server config (policy_server, value_model) is specified in the task spec JSON,
    not as CLI arguments. See task_spec.py for details.

    Mode source-of-truth: ``collection_info.collection_meta.mode``. There is no
    ``--mode`` CLI flag — pass ``--session_config_path`` (preferred) or the
    legacy ``--collection_info_path`` instead. Only pure ``infer`` mode may
    omit both (in which case mode defaults to "infer").
    """
    robot: RobotConfig
    dataset: DatasetRecordConfig = field(default_factory=DatasetRecordConfig)

    # Path to a single-file session config (preferred). Bundles the operator
    # metadata, the task spec, and every recording knob the wrapper script
    # would otherwise need to splay onto the command line. See
    # ``session_config.py`` for the full schema.
    session_config_path: Optional[str] = None

    # Legacy: path to anyverse_collection_info.json. Kept so existing
    # invocations still work; new sessions should use session_config_path.
    collection_info_path: Optional[str] = None

    # --- Shared across modes ---
    # Cameras are configured via anyverse_collection_info.json's
    # hardware_meta.cameras (name → {type, index_or_path, width, height, fps,
    # ...}). Different tasks may declare different counts and names.
    # Display and logging
    display_data: bool = False
    play_sounds: bool = True
    enable_log_say: bool = True
    # Throttle for the per-tick episode progress line in
    # ``ControlLoop.run_episode``. ``1.0`` (default) prints once per second;
    # ``0`` prints on every tick (the original ~30 Hz behaviour); any other
    # positive value throttles to that interval. The print is also gated on
    # ``play_sounds`` so setting that to False disables it entirely.
    progress_print_interval_s: float = 1.0
    enable_logging: bool = True
    # Enable verbose debug prints (observation keys, buffer state, collision details, etc.)
    debug: bool = False
    # Resume existing dataset (record, infer_record, self_play)
    resume: bool = False
    # Auto-mark all episodes as success (record, infer_record, self_play)
    auto_success: bool = False

    # --- record mode only ---
    teleop: Optional[TeleoperatorConfig] = None
    # Sub-task durations (record mode with sub-tasks)
    sub_task_durations: Optional[List[float]] = None
    # Optional YAML config path overriding sub_task_durations; when set it must be
    # paired with ``record_task`` to pick the task block inside the YAML.
    subtask_config_path: Optional[str] = None
    record_task: Optional[str] = None
    # Populated by the YAML parser (or caller) to carry subtask index sequence.
    sub_task_inds: Optional[List[int]] = None

    # --- Inference: infer, infer_record, self_play ---
    policy: Optional[PreTrainedConfig] = None
    inference_mode: str = "async"
    infer_interval: int = 20
    default_infer_delay: int = 0
    inference_speedup: float = 1.0
    auto_infer_interval: bool = True  # Auto-compute optimal infer_interval from warmup latency
    action_horizon: int = 50
    fusion_type: str = "linear"
    fusion_exp_decay: float = 2.0
    fusion_window: int = 0  # Chunk transition blend steps (0=disable, 5=recommended with speedup)
    smooth_sigma: float = 0.0  # Gaussian smoothing on action chunks (0=off, 1.0=recommended with speedup)
    transition_steps: int = 15
    # Log raw action chunks to JSONL for open-loop evaluation
    log_action_chunks: bool = False
    # Persist a per-inference log (submit/complete timestamps, latency, raw
    # pre-fusion action chunk, prompt, RTC delay) to
    # ``<dataset>/meta/inference_log.parquet``. Off by default. No-op in pure
    # ``infer`` mode (no dataset to colocate with).
    persist_inference_log: bool = False

    # --- Intervention: infer_record, self_play ---
    waiting_intervention_time_s: float = 2.0
    waiting_evacuation_time_s: float = 2.0
    pose_sync_duration_s: float = 3.0
    # Max seconds to wait for the operator to start moving the leader arm
    # after pressing Ctrl+Space. Applies only to infer_record / self_play.
    leader_movement_timeout_s: float = 30.0

    # --- self_play only ---
    task_spec_path: Optional[str] = None
    self_play_max_total_time_s: Optional[float] = None
    # Run self_play loop without saving data (inference only, no dataset)
    self_play_infer_only: bool = False

    # Loaded + validated in __post_init__ from collection_info_path. Not a CLI
    # field. Source of truth for ``mode``.
    collection_info: Optional[CollectionInfo] = field(default=None, init=False, repr=False)

    # Loaded + validated in __post_init__ from session_config_path. Not a CLI
    # field. Stashed into ``meta.info["session_config"]`` for reproducibility.
    session_config: Optional[SessionConfig] = field(default=None, init=False, repr=False)

    @property
    def mode(self) -> str:
        """Recording mode, derived from collection_info or defaulted to 'infer'."""
        if self.collection_info is not None:
            return self.collection_info.collection_meta.mode
        return "infer"

    def __post_init__(self):
        # session_config_path takes precedence and supplies both the
        # collection_info and the task_spec. Cannot be combined with the
        # legacy --collection_info_path / --task_spec_path flags — pick one
        # source of truth or the other.
        if self.session_config_path:
            if self.collection_info_path:
                raise ValueError(
                    "--session_config_path cannot be combined with --collection_info_path; "
                    "the session config supersedes it"
                )
            self.session_config = SessionConfig.from_json(self.session_config_path)
            self.session_config.validate()
            self.collection_info = self.session_config.to_collection_info()
            self._apply_session_overrides(self.session_config)
        elif self.collection_info_path:
            self.collection_info = CollectionInfo.from_json(self.collection_info_path)
            self.collection_info.validate()

        valid_modes = ["record", "infer", "infer_record", "self_play"]
        if self.mode not in valid_modes:
            raise ValueError(f"derived mode={self.mode!r} must be one of {valid_modes}")

        # collection_info is required for every mode that produces a dataset.
        if self.collection_info is None and self.mode != "infer":
            raise ValueError(
                f"--session_config_path (or --collection_info_path) is required for "
                f"mode={self.mode!r}; only pure 'infer' mode may omit it"
            )

        self._validate_mode_params()

        if isinstance(self.sub_task_durations, str):
            self.sub_task_durations = json.loads(self.sub_task_durations)

        # collection_info.hardware_meta.cameras → robot.cameras
        self._setup_cameras_from_collection_info()

        # Policy path handling
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = policy_path

    def _apply_session_overrides(self, session: SessionConfig) -> None:
        """Apply the parts of the session config the shell wrapper can't
        cleanly express as CLI args.

        Flat scalar knobs (``num_episodes``, ``episode_time_s``, … and the
        whole ``recording``/``inference``/``intervention``/``self_play`` /
        ``subtask`` sections) are passed in from ``scripts/run_session.sh``
        as ordinary ``--flag=value`` overrides — that keeps the executed
        command line self-explanatory and lets operators add ad-hoc
        overrides on top.

        What we still have to do here:
          - Promote an *inline* task_spec object to a path the recorder can
            load via ``resolve_task_spec``. We dump it to a stable temp file
            so a single recorder run sees a consistent on-disk artifact.
          - Promote a task_spec specified by *path* in the session JSON to
            ``self.task_spec_path`` when the operator didn't pass one
            explicitly.
        """
        # If the operator explicitly supplied --task_spec_path, leave it
        # alone — explicit CLI wins.
        if self.task_spec_path:
            return

        if session.task_spec_source:
            self.task_spec_path = session.task_spec_source
            return

        if session.task_spec is not None:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".task_spec.json", delete=False,
            )
            json.dump(session.task_spec.to_dict(), tmp)
            tmp.flush()
            tmp.close()
            self.task_spec_path = tmp.name

    def _validate_mode_params(self):
        """Warn about parameters that are irrelevant for the current mode."""
        mode = self.mode
        warnings = []

        # record mode: no inference params needed
        if mode == "record":
            _inference_defaults = {
                "inference_mode": "async", "infer_interval": 20, "default_infer_delay": 0,
                "action_horizon": 50, "fusion_type": "linear", "fusion_exp_decay": 2.0,
                "transition_steps": 15,
            }
            for param, default in _inference_defaults.items():
                if getattr(self, param) != default:
                    warnings.append(f"  '{param}' is set but ignored in record mode")
            if self.policy is not None:
                warnings.append("  'policy' is set but ignored in record mode")
            if self.log_action_chunks:
                warnings.append("  'log_action_chunks' is set but ignored in record mode (no inference)")
            if self.self_play_max_total_time_s is not None:
                warnings.append("  'self_play_max_total_time_s' is set but ignored in record mode")
            if self.self_play_infer_only:
                warnings.append("  'self_play_infer_only' is set but ignored in record mode")

        # infer mode: requires task_spec_path (for policy server config)
        if mode == "infer":
            if not self.task_spec_path:
                raise ValueError("infer mode requires --task_spec_path (policy server is configured in the task spec)")
            if self.teleop is not None:
                warnings.append("  'teleop' is set but ignored in infer mode")
            if self.sub_task_durations is not None:
                warnings.append("  'sub_task_durations' is set but ignored in infer mode")
            if self.subtask_config_path is not None:
                warnings.append("  'subtask_config_path' is set but ignored in infer mode")
            if self.auto_success:
                warnings.append("  'auto_success' is set but ignored in infer mode (no dataset)")
            if self.self_play_max_total_time_s is not None:
                warnings.append("  'self_play_max_total_time_s' is set but ignored in infer mode")
            if self.self_play_infer_only:
                warnings.append("  'self_play_infer_only' is set but ignored in infer mode (already infer-only)")

        # infer_record mode: requires task_spec_path (for policy server config)
        if mode == "infer_record":
            if not self.task_spec_path:
                raise ValueError("infer_record mode requires --task_spec_path (policy server is configured in the task spec)")
            if self.self_play_max_total_time_s is not None:
                warnings.append("  'self_play_max_total_time_s' is set but ignored in infer_record mode")
            if self.sub_task_durations is not None:
                warnings.append("  'sub_task_durations' is set but ignored in infer_record mode")
            if self.subtask_config_path is not None:
                warnings.append("  'subtask_config_path' is set but ignored in infer_record mode")
            if self.self_play_infer_only:
                warnings.append("  'self_play_infer_only' is set but ignored in infer_record mode")

        # self_play mode: episode_time comes from task_spec, not dataset config
        if mode == "self_play":
            if not self.task_spec_path:
                raise ValueError("self_play mode requires --task_spec_path")
            if self.sub_task_durations is not None:
                warnings.append("  'sub_task_durations' is set but ignored in self_play mode")
            if self.subtask_config_path is not None:
                warnings.append("  'subtask_config_path' is set but ignored in self_play mode")
            if self.dataset.episode_time_s != 60:
                warnings.append(
                    "  'dataset.episode_time_s' is set but ignored in self_play mode"
                    " (episode time is defined per-role in the task spec)"
                )

        # record mode: subtask_config_path must pair with record_task
        if mode == "record":
            if self.subtask_config_path is not None and not self.record_task:
                raise ValueError(
                    "record mode: --subtask_config_path requires --record_task "
                    "(e.g. POUR_WATER, PICK_PLACE)"
                )

        if warnings:
            logging.warning(
                f"[RecordConfig] The following parameters are not applicable to '{mode}' mode:\n"
                + "\n".join(warnings)
            )

    def _setup_cameras_from_collection_info(self):
        """Build ``self.robot.cameras`` from
        ``collection_info.hardware_meta.cameras``.

        Each entry is ``{name: {type, index_or_path, width, height, fps, ...}}``.
        Currently only ``type=opencv`` is supported by this helper; for other
        camera backends (realsense etc.) extend the dispatch below or pass
        ``--robot.cameras=...`` directly via CLI YAML.
        """
        if self.collection_info is None:
            return
        cams = self.collection_info.hardware_meta.cameras
        if not cams:
            return
        built: Dict[str, Any] = {}
        for name, spec in cams.items():
            # Underscore-prefixed entries are inline annotations (e.g.
            # ``_comment_head``) — skip them just like the validator does.
            if isinstance(name, str) and name.startswith("_"):
                continue
            cam_type = spec.get("type", "opencv")
            kwargs = {k: v for k, v in spec.items() if k != "type" and not k.startswith("_")}
            # ``index_or_path`` may be int or str; preserve as-is — OpenCV
            # config accepts both.
            if cam_type == "opencv":
                built[name] = OpenCVCameraConfig(**kwargs)
            else:
                raise ValueError(
                    f"hardware_meta.cameras.{name}.type={cam_type!r} not supported by "
                    "_setup_cameras_from_collection_info; pass --robot.cameras=... directly"
                )
        self.robot.cameras = built

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _build_template_context(info: CollectionInfo) -> dict:
    """Build the substitution context for prompt templates.

    Top-level keys are each entry in ``task_meta.objects`` (so a prompt can
    reference ``{socks.color}`` or ``{kettle.material}``), plus a few
    convenience scalars from ``collection_meta`` and ``task_meta``.
    """
    return {
        **info.task_meta.objects,
        "task_name": info.task_meta.task_name,
        "operator": info.collection_meta.operator_name,
        "city": info.collection_meta.city,
        "site": info.collection_meta.site_location,
    }


def resolve_task_spec(cfg: RecordConfig) -> TaskSpec:
    if cfg.task_spec_path:
        spec = TaskSpec.from_json(cfg.task_spec_path)
    else:
        spec = TaskSpec.default_single_role(
            prompt=cfg.dataset.single_task or "",
            episode_time_s=cfg.dataset.episode_time_s,
        )
    if cfg.collection_info is not None:
        spec = spec.apply_template(_build_template_context(cfg.collection_info))
    return spec


def get_effective_fps(cfg) -> int:
    """Return effective control loop FPS, accounting for inference_speedup."""
    base_fps = cfg.dataset.fps
    speedup = getattr(cfg, "inference_speedup", 1.0)
    if speedup > 1.0 and getattr(cfg, "mode", "record") in ("infer", "infer_record", "self_play"):
        return int(base_fps * speedup)
    return base_fps


def enable_inference_speedup(robot, speedup: float) -> None:
    """Enable frame reuse on robot when speedup > 1."""
    if speedup <= 1.0:
        return
    if not hasattr(robot, "wait_for_new_frame"):
        logging.warning(
            f"inference_speedup={speedup} requested but robot {type(robot).__name__} "
            f"does not support wait_for_new_frame. Speedup will NOT reuse camera frames."
        )
        return
    robot.wait_for_new_frame = False


def setup_policy_client(
    robot: Robot,
    cfg: RecordConfig,
    logger: AsyncLogger,
    policy_host: str = "localhost",
    policy_port: int = 8001,
    value_model_host: Optional[str] = None,
    value_model_port: Optional[int] = None,
):
    """Create a remote policy inference client.

    Args:
        policy_host/port: Policy server address (from task spec).
        value_model_host/port: Value model server address (from task spec).
    """
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    host = policy_host
    port = policy_port
    v_host = value_model_host
    v_port = value_model_port

    camera_keys = list(cfg.robot.cameras.keys()) if cfg.robot.cameras else []
    robot_state_keys = [k for k in robot._motors_ft.keys() if "pos" in k and "joint" in k]

    import numpy as np

    class PolicyInferenceClient:
        def __init__(self):
            self.robot_state_keys = robot_state_keys
            self.action_dim = len(robot_state_keys)
            self._client = WebsocketClientPolicy(host=host, port=port)
            self._camera_keys = camera_keys
            self._value_client = None
            if v_host and v_port:
                self._value_client = WebsocketClientPolicy(host=v_host, port=v_port)

        def get_action(self, obs_dict, action_prefix, infer_delay, valid_len, lang_prompt):
            payload = self._build_payload(obs_dict, action_prefix, infer_delay, valid_len, lang_prompt)
            result = self._client.infer(payload)
            if "actions" in result:
                return np.array(result["actions"], dtype=np.float32)
            return result

        def get_value_score(self, obs_dict, lang):
            if self._value_client is None:
                return None
            payload = self._build_payload(obs_dict, None, 0, 0, lang)
            # Try score_observation first (preferred), fall back to infer with score flag
            if hasattr(self._value_client, "score_observation"):
                return self._value_client.score_observation(payload)
            payload["_request_type"] = "score"
            return self._value_client.infer(payload)

        def _build_payload(self, obs_dict, action_prefix, infer_delay, valid_len, lang_prompt):
            import einops

            # Debug: log observation keys on first call
            if not hasattr(self, '_debug_logged'):
                self._debug_logged = True
                obs_keys = list(obs_dict.keys())
                debug_print(f"Observation keys: {obs_keys}")
                debug_print(f"Robot state keys: {robot_state_keys}")
                # Check if individual motor keys exist
                sample_key = robot_state_keys[0] if robot_state_keys else None
                if sample_key:
                    debug_print(f"obs_dict.get('{sample_key}') = {obs_dict.get(sample_key, 'MISSING')}")
                if "observation.state" in obs_dict:
                    obs_state = obs_dict["observation.state"]
                    if hasattr(obs_state, 'shape'):
                        debug_print(f"observation.state shape={obs_state.shape}, values={obs_state[:5]}")

            # Build state vector from robot state keys
            state = np.array(
                [obs_dict.get(k, 0.0) for k in robot_state_keys]
            ).astype(np.float32)

            # Debug: check if state is all zeros
            if not hasattr(self, '_state_debug_logged'):
                self._state_debug_logged = True
                debug_print(f"Built state vector: {state[:5]}... (all_zero={np.allclose(state, 0)})")

            # Build camera observations with openpi key mapping
            # openpi expects: observation/front_image, observation/wrist_image, observation/wrist_image_lf
            # Robot returns images under bare camera names (e.g., "head", "left_wrist")
            # or under "observation.images.{name}" depending on robot type
            cam_obs = {}
            for cam_key in self._camera_keys:
                # Try bare key first (ARX), then prefixed key (lerobot standard)
                img = obs_dict.get(cam_key)
                if img is None:
                    img = obs_dict.get(f"observation.images.{cam_key}")
                if img is not None:
                    if hasattr(img, "numpy"):
                        img = img.numpy()
                    cam_obs[cam_key] = einops.rearrange(img, "h w c -> c h w")

            # Map camera names to openpi observation keys
            camera_key_mapping = {
                "head": "observation/front_image",
                "right_wrist": "observation/wrist_image",
                "left_wrist": "observation/wrist_image_lf",
            }

            payload = {
                "observation/state": state,
                "prompt": lang_prompt,
            }

            for cam_name, obs_key in camera_key_mapping.items():
                if cam_name in cam_obs:
                    payload[obs_key] = cam_obs[cam_name]
                elif self._camera_keys and cam_name == "head" and self._camera_keys[0] in cam_obs:
                    # Fallback: use first camera as front image
                    payload[obs_key] = cam_obs[self._camera_keys[0]]

            if action_prefix is not None and valid_len > 0:
                payload["action"] = action_prefix
                # Only mark actually valid prefix entries in the mask
                action_mask = np.zeros(action_prefix.shape[0], dtype=np.float32)
                action_mask[:valid_len] = 1.0
                payload["action_mask"] = action_mask
                if infer_delay and infer_delay > 0:
                    payload["infer_delay"] = infer_delay

            return payload

    client = PolicyInferenceClient()
    logger.log(f"Policy client: {host}:{port}, cameras={camera_keys}")
    logger.log(f"Robot state keys ({len(robot_state_keys)}): {robot_state_keys}")
    return client


# Seven knobs persisted to ``meta.info["infer_meta"]``. Smaller than the
# full INFERENCE_DEFAULTS set on purpose — the rest (fusion_*,
# transition_steps, log_*, persist_*) are operational/runtime knobs that
# don't describe the trained policy itself. ``model_name`` is a free-form
# label so analyses can group episodes by checkpoint without re-parsing
# logs.
_INFER_META_KEYS = (
    "mode",
    "model_name",
    "action_horizon",
    "infer_interval",
    "default_infer_delay",
    "inference_speedup",
    "smooth_sigma",
)

# Record-mode sentinel: no model inference happened. Empty string for
# enum-like / free-form text fields; -1 for numeric fields (-1.0 for
# floats) so they're trivially distinguishable from any valid setting and
# from the "field missing" case.
_INFER_META_RECORD_SENTINEL = {
    "mode": "",
    "model_name": "",
    "action_horizon": -1,
    "infer_interval": -1,
    "default_infer_delay": -1,
    "inference_speedup": -1.0,
    "smooth_sigma": -1.0,
}


def _build_infer_meta(cfg: RecordConfig) -> dict:
    """Six-key infer_meta dict for ``meta.info["infer_meta"]``.

    Returns record-mode sentinels when no model inference is involved
    (``mode == "record"`` or the legacy ``--collection_info_path`` path was
    used without a session config). Otherwise reads from the session
    config's merged ``inference`` dict, defaulting missing keys to the
    sentinel so the shape is uniform regardless of what the operator wrote.
    """
    mode = ""
    if cfg.collection_info is not None:
        mode = cfg.collection_info.collection_meta.mode
    has_inference = mode in ("infer", "infer_record", "self_play") and cfg.session_config is not None
    if not has_inference:
        return dict(_INFER_META_RECORD_SENTINEL)
    inference = cfg.session_config.inference
    return {k: inference.get(k, _INFER_META_RECORD_SENTINEL[k]) for k in _INFER_META_KEYS}


def setup_dataset(robot: Robot, cfg: RecordConfig, logger: AsyncLogger) -> Optional[LeRobotDataset]:
    if cfg.mode == "infer":
        return None
    if cfg.self_play_infer_only:
        logger.log("self_play_infer_only=True: skipping dataset creation")
        return None

    effective_fps = get_effective_fps(cfg)
    # Dataset is always tagged with base fps because the control loop
    # downsamples writes to base fps regardless of speedup. Tagging with
    # effective_fps here would make playback run at speedup× real time.
    dataset_fps = cfg.dataset.fps
    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    extra_features = {
        "is_human_intervention": {"dtype": "bool", "shape": (1,), "names": None},
    }
    # Record mode with subtasks: add a per-frame subtask index. For YAML-driven
    # subtasks, the field is also added — durations get filled in later by
    # run_recording_session after parsing the YAML.
    if cfg.mode == "record" and (cfg.sub_task_durations or cfg.subtask_config_path):
        extra_features["subtask_index"] = {"dtype": "int64", "shape": (1,), "names": None}
    dataset_features = {**action_features, **obs_features, **extra_features}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id, root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, dataset_fps, dataset_features)
    else:
        # Check if data path already exists
        data_root = Path(cfg.dataset.root) if cfg.dataset.root else None
        if data_root and data_root.exists() and any(data_root.iterdir()):
            log_say("数据路径已存在，请选择操作", play_sounds=True, blocking=True)
            print(f"\n{'='*60}")
            print(f"[WARNING] 数据路径已存在: {data_root}")
            print(f"  1) 覆盖（删除已有数据）")
            print(f"  2) 自动递增batch号")
            print(f"  3) 退出")
            print(f"{'='*60}")
            if not sys.stdin.isatty():
                logging.info("Non-interactive environment detected, auto-incrementing batch number")
                choice = "2"
            else:
                choice = input("请选择 [1/2/3]: ").strip()
            if choice == "1":
                import shutil
                shutil.rmtree(data_root)
                log_say("已删除旧数据", play_sounds=True)
            elif choice == "2":
                # Auto-increment: find next available batch number
                base = str(data_root)
                import re
                m = re.search(r'\.batch\.(\d+)$', base)
                if m:
                    prefix = base[:m.start()] + ".batch."
                    batch_num = int(m.group(1))
                    while True:
                        batch_num += 1
                        candidate = Path(f"{prefix}{batch_num}")
                        if not candidate.exists():
                            data_root = candidate
                            cfg.dataset.root = str(data_root)
                            new_repo_id = re.sub(r'\.batch\.\d+$', f'.batch.{batch_num}', cfg.dataset.repo_id)
                            cfg.dataset.repo_id = new_repo_id
                            break
                else:
                    data_root = Path(f"{base}.1")
                    cfg.dataset.root = str(data_root)
                log_say(f"已切换到新路径", play_sounds=True)
                print(f"  新路径: {data_root}")
            else:
                log_say("已退出", play_sounds=True)
                print("  已退出。")
                return None

        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id, dataset_fps, root=cfg.dataset.root,
            robot_type=robot.name, features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * max(len(robot.cameras), 1),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    info_path = dataset.root / INFO_PATH
    if cfg.collection_info is not None:
        # Flat at the top level so downstream readers don't have to know about
        # the historical ``anyverse_collection_info`` wrapper. Four metas +
        # robot_id + task_description; nothing else from the session config
        # persists into the dataset.
        #
        # start_time/end_time are stamped onto collection_meta by record()
        # before setup_dataset is called.
        ci = cfg.collection_info
        dataset.meta.info["collection_meta"] = ci.collection_meta.to_dict()
        # hardware_meta omits cameras here — they're already on every dataset
        # under ``features.observation.images.*`` (with full resolution /
        # codec / fps metadata). Duplicating them would just be drift bait.
        dataset.meta.info["hardware_meta"] = {
            "end_effector": ci.hardware_meta.end_effector.to_dict(),
        }
        dataset.meta.info["task_meta"] = ci.task_meta.to_dict()
        dataset.meta.info["robot_id"] = ci.robot_id
        dataset.meta.info["task_description"] = ci.task_description
        # infer_meta — pinned to six fields so downstream readers can rely on
        # the shape. In record mode (or when no session_config was provided
        # via legacy --collection_info_path), every field is set to a sentinel
        # value indicating "no inference happened": mode="" and numeric
        # fields = -1 / -1.0. This makes record-vs-inference datasets
        # distinguishable from info.json alone without a separate flag.
        dataset.meta.info["infer_meta"] = _build_infer_meta(cfg)
    if cfg.sub_task_durations:
        dataset.meta.info["sub_task_durations"] = cfg.sub_task_durations
    write_json(dataset.meta.info, info_path)
    logger.log(f"Dataset: {dataset.root}, fps={dataset_fps} (control loop runs at {effective_fps})")
    return dataset


def _announce_reset_before_next_episode(
    log_say_fn,
    *,
    current_episode_index: int,
    num_episodes: int,
    stop_recording: bool,
    rerecord_episode: bool,
    reset_time_s: Union[int, float],
) -> bool:
    """Tell the operator to reset whenever another episode will follow.

    Suppressed entirely when ``reset_time_s == 0`` — there is no reset gap
    to announce. Otherwise the announcement is emitted *blocking* so it
    finishes before the sleep that follows; non-blocking risked the TTS
    voice colliding with the next episode's start prompt.
    """
    if stop_recording:
        return False
    if current_episode_index < num_episodes - 1 or rerecord_episode:
        if reset_time_s == 0:
            return False
        log_say_fn("请重置环境", blocking=True)
        return True
    return False


# ---------------------------------------------------------------------------
# Batch tracking helper
# ---------------------------------------------------------------------------

def _record_batch_info(dataset, cfg, recorded_episodes, session_start_time, logger):
    """Record batch info to the upload system tracker (best-effort)."""
    if dataset is None or recorded_episodes <= 0:
        return
    try:
        from lerobot.common.data_tracker import BatchTracker

        # config.dataset.root structure: /path/to/data_collection/YYYYMMDD/task_name/batch_id
        # Go up 3 levels to reach data_collection directory
        data_root = Path(cfg.dataset.root).parent.parent.parent
        tracker = BatchTracker(data_root)

        batch_info = tracker.record_batch_info(
            robot_type=cfg.robot.type,
            robot_id=cfg.robot.id,
            repo_id=cfg.dataset.repo_id,
            dataset_root=cfg.dataset.root,
            num_episodes=recorded_episodes,
            num_expected_episodes=cfg.dataset.num_episodes,
            session_start_time=session_start_time,
        )
        logger.log(f"Batch recorded: {batch_info.batch_id}")
    except Exception as e:
        logger.log(f"Failed to record batch info: {e}")
        logging.error(f"Failed to record batch info: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Recording session
# ---------------------------------------------------------------------------

def select_action_source(cfg, teleop, robot, policy_runtime, prompt):
    """Pick the per-episode action source for the non-self-play recording flow.

    record:        always teleop.
    infer_record:  policy_runtime is required; reinit clears stale buffers
                   from the previous episode and re-warms up with the current
                   prompt. Without reinit, leftover end-of-episode actions
                   execute at the start of the next episode, causing sudden
                   arm jerks.
    Fallback:      if policy_runtime is unexpectedly None in a non-record mode,
                   fall back to teleop so the loop can still progress.

    Self-play has its own per-tick interleaving inside ControlLoop and does not
    use this helper.
    """
    if cfg.mode == "record":
        return TeleopSource(teleop, robot)
    if policy_runtime is not None:
        policy_runtime.reinit(prompt)
        return policy_runtime
    return TeleopSource(teleop, robot)


def run_recording_session(
    robot: Robot,
    teleop: Optional[Teleoperator],
    policy_runtime: Optional[PolicyRuntime],
    dataset: Optional[LeRobotDataset],
    cfg: RecordConfig,
    task_spec: TaskSpec,
    events: dict,
    logger: AsyncLogger,
    self_play_logger: Optional[SelfPlayLogger] = None,
    per_role_policy_runtimes: Optional[Dict[str, PolicyRuntime]] = None,
    prompt_switcher: Optional[PromptSwitcher] = None,
):
    fps = get_effective_fps(cfg)
    role_manager = RoleManager(task_spec) if task_spec.is_self_play else None
    stats = SelfPlayStatsAggregator()


    intervention = None
    if cfg.mode in ("infer_record", "self_play"):
        intervention = InterventionRuntime(
            robot, teleop,
            pose_sync_duration_s=cfg.pose_sync_duration_s,
            waiting_intervention_time_s=cfg.waiting_intervention_time_s,
            waiting_evacuation_time_s=cfg.waiting_evacuation_time_s,
            leader_movement_timeout_s=cfg.leader_movement_timeout_s,
        )

    safety = None
    if task_spec.has_safety:
        # Sort current keys: left_joint_1..7 (indices 0-6), right_joint_1..7 (indices 7-13)
        # This matches the old code's _extract_current_dim_value ordering
        current_keys = sorted(
            [k for k in robot._motors_ft.keys() if k.endswith(".cur")],
            key=lambda k: (
                0 if k.startswith("left_joint_") else 1 if k.startswith("right_joint_") else 2,
                int(k.split("_")[-1].split(".")[0]) if k.split("_")[-1].split(".")[0].isdigit() else 999,
            ),
        )
        safety = SafetyRuntime(task_spec.safety, robot=robot, current_keys=current_keys)
        logger.log(f"Safety current keys ({len(current_keys)}): {current_keys}")
        debug_print(f"Safety current keys: {current_keys}")
        debug_print(f"Robot has get_current_vector: {hasattr(robot, 'get_current_vector')}")
    state_machine = StateMachine(task_spec) if task_spec.is_self_play else None

    # Home evaluator (self_play only)
    home_evaluator = None
    if task_spec.has_reset and task_spec.is_self_play:
        robot_state_keys = [k for k in robot._motors_ft.keys() if "pos" in k and "joint" in k]
        robot_speed_keys = [k for k in robot._motors_ft.keys() if "vel" in k and "joint" in k]
        home_evaluator = HomeEvaluator(
            task_spec.reset,
            robot_state_keys=robot_state_keys,
            robot_speed_keys=robot_speed_keys if robot_speed_keys else None,
        )

    session_start = time.time()
    recorded = 0

    _log_say = make_log_say(cfg)

    # ---- Optional inference log writer ----
    # When inference.persist_inference_log is set AND we're writing a dataset,
    # every inference call is appended to <dataset>/meta/inference_log.parquet
    # (submit/complete timestamps, latency, raw pre-fusion chunk, prompt, RTC
    # delay). Attached to the policy_runtime(s) so they record straight from
    # the inference path. Discarded per-episode on rerecord, flushed once at
    # session end.
    inference_log_writer = None
    if getattr(cfg, "persist_inference_log", False) and dataset is not None:
        from lerobot.recording.runtime.inference_log_writer import InferenceLogWriter

        action_horizon_for_writer = cfg.action_horizon
        action_dim_for_writer = 0
        for runtime in [policy_runtime] + list((per_role_policy_runtimes or {}).values()):
            if runtime is not None:
                action_dim_for_writer = runtime.action_dim
                break
        if action_dim_for_writer:
            inference_log_writer = InferenceLogWriter(
                dataset_root=dataset.root if dataset is not None else None,
                action_horizon=action_horizon_for_writer,
                action_dim=action_dim_for_writer,
            )
            if policy_runtime is not None:
                policy_runtime.inference_log_writer = inference_log_writer
                if not policy_runtime.role:
                    policy_runtime.role = "operator"
            if per_role_policy_runtimes:
                for role_name, rt in per_role_policy_runtimes.items():
                    rt.inference_log_writer = inference_log_writer
                    if not rt.role:
                        rt.role = role_name
            logger.log(
                "persist_inference_log=True: appending inference rows to "
                f"{dataset.root}/meta/inference_log.parquet"
            )

    # ---- Optional YAML-driven subtask config (record mode only) ----
    # Parses the YAML, overrides cfg.sub_task_durations / cfg.sub_task_inds,
    # copies the subtask annotation JSON into the dataset, and generates the
    # per-episode Chinese/English prompt list.
    generated_prompts: Optional[list] = None
    if cfg.mode == "record" and cfg.subtask_config_path:
        from lerobot.utils.tts_config_parser import parse_record_config
        subtask_cfg = parse_record_config(
            cfg.subtask_config_path,
            cfg.dataset.episode_time_s,
            cfg.record_task,
        )
        cfg.sub_task_inds = list(subtask_cfg["sub_episode_taskinds"])
        cfg.sub_task_durations = list(subtask_cfg["sub_episode_dur_time"])
        logger.log(
            f"Loaded subtask YAML: task={cfg.record_task} "
            f"inds={cfg.sub_task_inds} durations={cfg.sub_task_durations}"
        )

        # setup_dataset already wrote meta.info with durations = None / CLI value.
        # Rewrite now that YAML-derived durations are known.
        if dataset is not None:
            dataset.meta.info["sub_task_durations"] = cfg.sub_task_durations
            write_json(dataset.meta.info, dataset.root / INFO_PATH)

        if dataset is not None and subtask_cfg.get("sub_episode_task_json"):
            annotations_dir = Path(dataset.meta.root) / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                subtask_cfg["sub_episode_task_json"],
                annotations_dir / "subtask_annotations.jsonl",
            )

        # Per-task prompt pools: each entry is {'zh': ..., 'en': ...}
        if cfg.record_task == "PICK_PLACE":
            from lerobot.utils.generate_pptask_prompts import execute_prompt_generation_random
            _, prompt_pools = execute_prompt_generation_random(
                num_prompts=cfg.dataset.num_episodes,
                with_restore=False,
                with_unzip=True,
            )
        elif cfg.record_task == "POUR_WATER":
            from lerobot.utils.generate_pourtask_prompts import execute_prompt_generation_random_pour_water
            _, prompt_pools = execute_prompt_generation_random_pour_water(
                num_prompts=cfg.dataset.num_episodes,
            )
        else:
            raise NotImplementedError(
                f"record_task='{cfg.record_task}' has no prompt generator registered"
            )
        generated_prompts = prompt_pools[0]
        assert len(generated_prompts) > 0 and cfg.dataset.num_episodes >= len(generated_prompts)
        num_loop = cfg.dataset.num_episodes // len(generated_prompts)
        if num_loop > 0:
            generated_prompts = generated_prompts * num_loop
            num_excess = cfg.dataset.num_episodes % len(generated_prompts)
            generated_prompts += generated_prompts[:num_excess]

    class NullCtx:
        def __enter__(self): return self
        def __exit__(self, *a): return False

    ctx_mgr = VideoEncodingManager(dataset) if dataset else NullCtx()

    try:
        with ctx_mgr:
            while recorded < cfg.dataset.num_episodes:
                if cfg.self_play_max_total_time_s is not None:
                    if time.time() - session_start >= cfg.self_play_max_total_time_s:
                        logger.log("Total time limit reached")
                        break
                if events.get("stop_recording"):
                    break

                # Reset transient event flags before starting a new episode.
                # Without this, a Ctrl+Left/Right pressed during the reset
                # interlude (e.g. while time.sleep(reset_time_s) runs in the
                # rerecord branch below) leaves exit_early=True / task_success
                # set, so the next iteration's run_episode exits on its first
                # tick with zero frames and save_episode then crashes with
                # "You must add one or several frames before calling
                # 'add_episode'". stop_recording is left intact — Ctrl+Esc
                # during the interlude is a real shutdown request.
                events["exit_early"] = False
                events["rerecord_episode"] = False
                events["task_success"] = None

                role = role_manager.current_role if role_manager else "operator"
                role_spec = task_spec.roles.get(role)
                if role_spec and role_spec.speed_variation and "{speed}" in role_spec.prompt:
                    prompt = role_spec.get_prompt_for_episode(recorded)
                else:
                    prompt = role_manager.current_prompt if role_manager else (cfg.dataset.single_task or task_spec.initial_prompt)

                # YAML-driven subtask mode: announce Chinese prompt + use English as lang_prompt
                if generated_prompts is not None:
                    _log_say(f"{generated_prompts[recorded]['zh']}", blocking=True)
                    logger.log(f"Episode {recorded}: prompt_zh={generated_prompts[recorded]['zh']}")
                    time.sleep(4.0)
                    prompt = generated_prompts[recorded]['en']
                else:
                    _log_say(f"录制第 {recorded} 条数据", blocking=True)
                logger.log(f"Episode {recorded}: role={role}")

                # Update chunk logger episode index
                if policy_runtime and policy_runtime.chunk_logger:
                    policy_runtime.chunk_logger.set_episode(recorded)
                # Tag inference log rows with the current episode index. The
                # writer is shared across per-role runtimes in self_play; each
                # runtime stamps its own rows with this value at append time.
                for rt in [policy_runtime] + list((per_role_policy_runtimes or {}).values()):
                    if rt is not None:
                        rt.current_episode_index = recorded

                if state_machine:
                    state_machine.reset_for_episode(role, prompt)
                if safety:
                    safety.reset()
                if intervention:
                    intervention.reset()

                role_spec = task_spec.roles.get(role)
                episode_time = role_spec.effective_max_time_s if role_spec and task_spec.is_self_play else cfg.dataset.episode_time_s

                loop = ControlLoop(robot, fps, dataset, logger, display_data=cfg.display_data,
                                   base_fps=cfg.dataset.fps)

                # ----- Self-play mode: full interleaved per-tick logic -----
                if cfg.mode == "self_play" and state_machine is not None and policy_runtime is not None:
                    # Switch to per-role policy runtime if available
                    if per_role_policy_runtimes and role in per_role_policy_runtimes:
                        policy_runtime = per_role_policy_runtimes[role]
                    policy_runtime.reinit(prompt)

                    result = loop.run_self_play_episode(
                        policy_runtime=policy_runtime,
                        intervention=intervention,
                        safety=safety,
                        home_evaluator=home_evaluator,
                        state_machine=state_machine,
                        events=events,
                        control_time_s=episode_time,
                        lang_prompt=prompt,
                        role=role,
                        log_say_fn=_log_say,
                        self_play_logger=self_play_logger,
                        prompt_switcher=prompt_switcher,
                    )
                    task_success = result.task_success
                    end_reason = result.end_reason

                # ----- Record / infer_record: simpler flow -----
                else:
                    action_src = select_action_source(cfg, teleop, robot, policy_runtime, prompt)

                    # Subtask manager (record mode only; disabled when no durations)
                    subtask_manager = None
                    if cfg.mode == "record" and cfg.sub_task_durations:
                        subtask_manager = SubTaskManager(
                            durations=cfg.sub_task_durations,
                            is_inference_mode=False,
                            subtask_ind=cfg.sub_task_inds,
                            play_sounds=cfg.play_sounds,
                            enable_log_say=cfg.enable_log_say,
                        )

                    result = loop.run_episode(
                        action_source=action_src, hooks=[],
                        control_time_s=episode_time, events=events, lang_prompt=prompt,
                        intervention=intervention,
                        log_say_fn=_log_say,
                        subtask_manager=subtask_manager,
                        episode_idx=recorded,
                        play_sounds=cfg.play_sounds,
                        progress_print_interval_s=cfg.progress_print_interval_s,
                        prompt_switcher=prompt_switcher,
                    )
                    task_success = events.get("task_success")
                    end_reason = "completed"

                events["exit_early"] = False

                if events.get("stop_recording"):
                    _log_say("提前退出录制")
                    break

                if events.get("rerecord_episode"):
                    _announce_reset_before_next_episode(
                        _log_say,
                        current_episode_index=recorded,
                        num_episodes=cfg.dataset.num_episodes,
                        stop_recording=events.get("stop_recording", False),
                        rerecord_episode=True,
                        reset_time_s=cfg.dataset.reset_time_s,
                    )
                    _log_say("重新录制当前数据")
                    time.sleep(cfg.dataset.reset_time_s)
                    events["rerecord_episode"] = False
                    events["task_success"] = None
                    # Drop the rerecorded episode's inference rows so the
                    # persisted log doesn't reference frames that were never
                    # saved to the main dataset.
                    if inference_log_writer is not None:
                        inference_log_writer.discard_episode(recorded)
                    if dataset is not None:
                        # Without this guard a clear_episode_buffer race
                        # (image-writer threads still flushing, partial
                        # buffer state, ndarray mutation in flight, etc.)
                        # would propagate to the outer ``except Exception``
                        # in run_recording_session — the session then ends
                        # via the ``finally`` block and the operator sees
                        # something indistinguishable from a Ctrl+C exit.
                        # The operator pressed Ctrl+Left to retry, not to
                        # quit; log the failure but honor the retry.
                        try:
                            dataset.clear_episode_buffer()
                        except Exception as e:
                            logger.log(
                                f"clear_episode_buffer failed during rerecord: {e}"
                            )
                            logging.error(
                                f"clear_episode_buffer failed during rerecord: {e}",
                                exc_info=True,
                            )
                    continue

                try:
                    if task_success is None and cfg.auto_success:
                        task_success = True

                    if task_success is True:
                        _log_say("任务成功", blocking=True)
                    elif task_success is False:
                        _log_say("任务失败", blocking=True)

                    if dataset is not None:
                        _log_say("存储数据中")

                        episode_metadata = {}
                        if task_success is not None:
                            episode_metadata["success"] = task_success

                        # Self-play mode: add role, value, intervention stats etc.
                        if cfg.mode == "self_play":
                            episode_metadata.update(SelfPlayStatsAggregator.build_episode_metadata(
                                episode_role=role,
                                end_reason=end_reason,
                                final_value=result.last_value_score,
                                final_is_home=result.last_is_home,
                                intervention_count=result.intervention_count,
                                intervention_duration_s=result.intervention_duration_s,
                                episode_duration_s=result.duration_s,
                                time_to_home_s=result.first_home_time_s,
                                home_duration_s=result.home_duration_s,
                                collision_count=result.collision_count,
                                collision_max_recovery=result.collision_max_recovery,
                                collision_events=result.collision_events,
                            ))

                        # Prompt timeline (initial + user Ctrl+<key> switches +
                        # safety/recovery overrides). Each entry: step, t_s,
                        # prompt, source.
                        if getattr(result, "prompt_switches", None):
                            episode_metadata["prompt_switches"] = result.prompt_switches

                        dataset.save_episode(episode_metadata=episode_metadata or None)
                        _log_say("存储完毕", blocking=True)

                    stats.update_episode(
                        episode_role=role, task_success=task_success,
                        episode_duration_s=result.duration_s,
                        intervention_duration_s=result.intervention_duration_s,
                        intervention_count=result.intervention_count,
                        time_to_home_s=result.first_home_time_s,
                        end_reason=end_reason,
                        collision_count=safety.collision_count if safety else 0,
                        home_duration_s=result.home_duration_s,
                    )

                    logger.log(f"Episode {recorded} {'saved' if dataset else 'finished'}, success={task_success}, end_reason={end_reason}")

                    # Post-episode: ensure robot is at home before role switch
                    if cfg.mode == "self_play" and home_evaluator is not None:
                        _wait_until_home_stable(
                            robot, home_evaluator, events, cfg,
                            log_say_fn=_log_say, logger=logger,
                        )

                    _announce_reset_before_next_episode(
                        _log_say,
                        current_episode_index=recorded,
                        num_episodes=cfg.dataset.num_episodes,
                        stop_recording=events.get("stop_recording", False),
                        rerecord_episode=False,
                        reset_time_s=cfg.dataset.reset_time_s,
                    )
                    time.sleep(cfg.dataset.reset_time_s)

                    if role_manager:
                        role_manager.advance_role()

                    recorded += 1
                    events["task_success"] = None

                except Exception as e:
                    if dataset is not None:
                        dataset.clear_episode_buffer()
                    logger.log(f"Save error: {e}")
                    logging.error(f"Save failed: {e}", exc_info=True)
                    events["stop_recording"] = True
                    break

    except KeyboardInterrupt:
        logger.log(f"Interrupted after {recorded} episodes")
    except Exception as e:
        logger.log(f"Session error: {e}")
        logging.error(f"Session error: {e}", exc_info=True)
        # Belt-and-suspenders: print the traceback directly to stderr so
        # operators see it even if the root logger's formatter was
        # configured to drop exc_info.
        import traceback as _tb
        _tb.print_exc()
    finally:
        if dataset and recorded > 0:
            # Aggregated session stats (success_count, takeover_rate, etc.)
            # used to be written here under ``meta.info["session_stats"]``
            # but the same numbers are derivable by summing/averaging the
            # per-episode rows already in ``meta/episodes.jsonl``. Skipping
            # the duplicate write keeps info.json lean; the in-memory
            # ``stats`` aggregator stays for log-line reporting during the
            # session.
            # Record batch info for the upload system
            _record_batch_info(dataset, cfg, recorded, session_start, logger)

        # Flush the inference log even on error — partial logs are still
        # useful for diagnosing why the session bailed.
        if inference_log_writer is not None:
            try:
                inference_log_writer.flush()
            except Exception as e:
                logger.log(f"InferenceLogWriter.flush failed: {e}")
                logging.error(f"InferenceLogWriter.flush failed: {e}", exc_info=True)


def run_inference_only(
    robot: Robot,
    policy_runtime: PolicyRuntime,
    cfg: RecordConfig,
    events: dict,
    logger: AsyncLogger,
    prompt_switcher: Optional[PromptSwitcher] = None,
):
    fps = get_effective_fps(cfg)
    loop = ControlLoop(robot, fps, dataset=None, logger=logger, display_data=cfg.display_data,
                       base_fps=cfg.dataset.fps)
    say = make_log_say(cfg)
    say("开始纯推理模式")
    result = loop.run_indefinite(
        policy_runtime, events,
        prompt_switcher=prompt_switcher,
        initial_prompt=policy_runtime.lang_prompt,
    )
    logger.log(f"Inference ended: {result.step_count} steps, {result.duration_s:.1f}s")
    if result.prompt_switches:
        logger.log(f"Prompt timeline: {result.prompt_switches}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@parser.wrap()
def record(cfg: RecordConfig) -> Optional[LeRobotDataset]:
    """Unified recording/inference entry point."""
    init_logging()
    logging.info(pformat(cfg.__dict__))
    say = make_log_say(cfg)

    # Stamp start_time as early as possible so even crash paths capture it.
    # Register an atexit safety net that stamps end_time and persists the info
    # if the dataset has been created — covers SIGINT and most exception paths.
    if cfg.collection_info is not None:
        cfg.collection_info.stamp_start()

        import atexit
        _finalized = {"done": False}
        _dataset_holder: Dict[str, Any] = {"dataset": None}

        def _finalize_collection_info():
            if _finalized["done"]:
                return
            _finalized["done"] = True
            cfg.collection_info.stamp_end()
            ds = _dataset_holder["dataset"]
            if ds is not None:
                try:
                    # Mirror the flat layout from setup_dataset: only
                    # collection_meta carries the freshly stamped end_time,
                    # the other metas are already on disk and unchanged.
                    ds.meta.info["collection_meta"] = cfg.collection_info.collection_meta.to_dict()
                    write_json(ds.meta.info, ds.root / INFO_PATH)
                except Exception as e:
                    logging.warning(f"Could not persist collection_info on shutdown: {e}")

        atexit.register(_finalize_collection_info)
    else:
        _finalized = None
        _dataset_holder = None
        _finalize_collection_info = lambda: None  # noqa: E731

    # Set module-level debug flag so all recording components can check it
    from lerobot.recording.utils.logging import set_debug_enabled
    set_debug_enabled(cfg.debug)

    task_spec = resolve_task_spec(cfg)

    if cfg.mode == "self_play" and not task_spec.is_self_play:
        raise ValueError("self_play mode requires a task spec with multiple roles")

    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_dir = f"./recording_logs/{now}"
    logger = AsyncLogger(log_dir, enabled=cfg.enable_logging)
    self_play_logger = SelfPlayLogger(log_dir, enabled=cfg.enable_logging and cfg.mode == "self_play")
    chunk_logger = ActionChunkLogger(log_dir) if cfg.log_action_chunks and cfg.mode != "record" else None
    logger.log(f"Mode={cfg.mode}, task={task_spec.task_id}, roles={task_spec.role_names}")

    if cfg.display_data:
        _init_rerun(session_name="recording")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.log(f"Robot: {robot.name}")

    if cfg.inference_speedup > 1.0:
        enable_inference_speedup(robot, cfg.inference_speedup)
        speedup_msg = (
            f"inference_speedup={cfg.inference_speedup}: frame reuse enabled, "
            f"effective_fps={get_effective_fps(cfg)}"
        )
        logger.log(speedup_msg)
        logging.info(speedup_msg)

    teleop = None
    if cfg.teleop is not None:
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()
        logger.log("Teleop connected")

    policy_client = None
    policy_runtime = None
    per_role_policy_runtimes = {}  # role_name → PolicyRuntime (for per-role servers)
    if cfg.mode in ("infer", "infer_record", "self_play"):
        # Resolve value model server from task spec
        vm_host = task_spec.value_model.host if task_spec.value_model else None
        vm_port = task_spec.value_model.port if task_spec.value_model else None

        # Check if any role has per-role overrides (policy_server or inference params).
        # Any of these requires separate PolicyRuntime instances per role.
        has_per_role_overrides = any(
            role.policy_server is not None
            or role.infer_interval is not None
            or role.default_infer_delay is not None
            for role in task_spec.roles.values()
        )

        if has_per_role_overrides and cfg.mode == "self_play":
            # Create separate policy runtimes for each role with a policy_server
            for role_name, role_spec in task_spec.roles.items():
                ps = task_spec.get_policy_server(role_name)
                if ps is None:
                    raise ValueError(
                        f"Role '{role_name}' has no policy_server and no task-level default. "
                        "Add policy_server to the role or to the top-level task spec."
                    )
                role_client = setup_policy_client(
                    robot, cfg, logger,
                    policy_host=ps.host, policy_port=ps.port,
                    value_model_host=vm_host, value_model_port=vm_port,
                )
                # Per-role inference params override CLI defaults
                role_infer_interval = role_spec.infer_interval if role_spec.infer_interval is not None else cfg.infer_interval
                role_infer_delay = role_spec.default_infer_delay if role_spec.default_infer_delay is not None else cfg.default_infer_delay
                role_rt = PolicyRuntime(
                    policy_client=role_client, robot=robot,
                    lang_prompt=role_spec.prompt, logger=logger,
                    inference_mode=cfg.inference_mode, action_horizon=cfg.action_horizon,
                    fusion_type=cfg.fusion_type, fusion_exp_decay=cfg.fusion_exp_decay,
                    infer_interval=role_infer_interval, default_infer_delay=role_infer_delay,
                    transition_steps=cfg.transition_steps,
                    effective_fps=get_effective_fps(cfg),
                    auto_infer_interval=(cfg.auto_infer_interval and cfg.inference_speedup > 1.0),
                    smooth_sigma=cfg.smooth_sigma,
                    fusion_window=cfg.fusion_window,
                    chunk_logger=chunk_logger,
                )
                per_role_policy_runtimes[role_name] = role_rt
                if role_spec.infer_interval is not None or role_spec.default_infer_delay is not None:
                    logger.log(f"Role '{role_name}': infer_interval={role_infer_interval}, default_infer_delay={role_infer_delay}")
            # Use first role's runtime as the initial active one
            policy_runtime = per_role_policy_runtimes.get(task_spec.initial_role)
            logger.log(f"Per-role policy runtimes created for roles: {list(per_role_policy_runtimes.keys())}")
        else:
            # Single shared policy client (resolve from initial role or task-level default)
            ps = task_spec.get_policy_server(task_spec.initial_role)
            if ps is None:
                raise ValueError(
                    "No policy_server configured in task spec. "
                    "Add policy_server to the role or to the top-level task spec."
                )
            policy_client = setup_policy_client(
                robot, cfg, logger,
                policy_host=ps.host, policy_port=ps.port,
                value_model_host=vm_host, value_model_port=vm_port,
            )
            prompt = task_spec.initial_prompt or cfg.dataset.single_task or ""
            policy_runtime = PolicyRuntime(
                policy_client=policy_client, robot=robot, lang_prompt=prompt, logger=logger,
                inference_mode=cfg.inference_mode, action_horizon=cfg.action_horizon,
                fusion_type=cfg.fusion_type, fusion_exp_decay=cfg.fusion_exp_decay,
                infer_interval=cfg.infer_interval, default_infer_delay=cfg.default_infer_delay,
                transition_steps=cfg.transition_steps,
                effective_fps=get_effective_fps(cfg),
                auto_infer_interval=(cfg.auto_infer_interval and cfg.inference_speedup > 1.0),
                smooth_sigma=cfg.smooth_sigma,
                fusion_window=cfg.fusion_window,
                chunk_logger=chunk_logger,
            )

    dataset = setup_dataset(robot, cfg, logger)
    if _dataset_holder is not None:
        _dataset_holder["dataset"] = dataset

    # Build the Ctrl+<key> prompt switcher from task_spec.prompts. When the
    # dict is empty, the switcher is disabled and the keyboard listener / loop
    # treat it as a no-op.
    prompt_switcher = PromptSwitcher.from_task_spec(task_spec)
    if prompt_switcher.enabled:
        logger.log(prompt_switcher.describe())

    listener, events = init_keyboard_listener(
        prompt_switcher=prompt_switcher,
        play_sounds=cfg.play_sounds,
        enable_log_say=cfg.enable_log_say,
    )

    try:
        if cfg.mode == "infer":
            if policy_runtime is None:
                raise ValueError("Infer mode requires policy client")
            run_inference_only(
                robot, policy_runtime, cfg, events, logger,
                prompt_switcher=prompt_switcher,
            )
        else:
            if dataset is None and not cfg.self_play_infer_only:
                raise ValueError("Recording modes require dataset")
            run_recording_session(
                robot=robot, teleop=teleop, policy_runtime=policy_runtime,
                dataset=dataset, cfg=cfg, task_spec=task_spec,
                events=events, logger=logger, self_play_logger=self_play_logger,
                per_role_policy_runtimes=per_role_policy_runtimes or None,
                prompt_switcher=prompt_switcher,
            )
    except Exception as e:
        logger.log(f"Error: {e}")
        raise
    finally:
        # --- Safety: never exit while robot is not at home ---
        try:
            home_evaluator = None
            if task_spec.reset is not None:
                state_keys = [k for k in robot._motors_ft.keys() if "pos" in k and "joint" in k]
                vel_keys = [k for k in robot._motors_ft.keys() if "vel" in k and "joint" in k]
                home_evaluator = HomeEvaluator(
                    task_spec.reset, robot_state_keys=state_keys, robot_speed_keys=vel_keys,
                )
            if home_evaluator is not None:
                obs = robot.get_observation()
                if not home_evaluator.is_home_pose(obs):
                    say("机械臂不在初始位，请将机械臂移回初始位后再退出")
                    logger.log("EXIT GUARD: robot not at home, waiting for manual return")
                    # Hold current position first
                    try:
                        hold = robot.get_joint_positions()
                        robot.send_action(hold)
                    except Exception as e:
                        logging.warning(f"EXIT GUARD: failed to hold position: {e}")
                    # Enable gravity comp so operator can move robot
                    if hasattr(robot, "set_gravity_compensation_mode"):
                        robot.set_gravity_compensation_mode()
                    # Block until home
                    last_log = 0.0
                    while True:
                        try:
                            obs = robot.get_observation()
                            if home_evaluator.is_home(obs):
                                logger.log("EXIT GUARD: robot at home, proceeding with shutdown")
                                say("机械臂已回到初始位")
                                break
                            now = time.perf_counter()
                            if now - last_log >= 5.0:
                                logger.log("EXIT GUARD: still waiting for robot to reach home...")
                                last_log = now
                            time.sleep(0.5)
                        except KeyboardInterrupt:
                            logger.log("EXIT GUARD: forced exit by user (double Ctrl+C)")
                            break
        except Exception as e:
            logger.log(f"EXIT GUARD error (non-fatal): {e}")

        say("程序结束")
        # Stamp end_time and persist the final collection_info into meta.info.
        # The atexit handler is the safety net for SIGINT / abrupt exits; the
        # happy-path call here ensures the dataset's info.json is updated before
        # we disconnect the robot and tear down the listener.
        _finalize_collection_info()
        # Cleanup all policy runtimes
        cleaned = set()
        if policy_runtime:
            policy_runtime.cleanup()
            cleaned.add(id(policy_runtime))
        for rt in per_role_policy_runtimes.values():
            if id(rt) not in cleaned:
                rt.cleanup()
                cleaned.add(id(rt))
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()
        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        logger.close()
        self_play_logger.close()
        if chunk_logger is not None:
            chunk_logger.close()

    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
