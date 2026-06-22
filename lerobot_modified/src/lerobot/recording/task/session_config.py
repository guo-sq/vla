"""Single-file session config for a recording job.

A ``SessionConfig`` is the one and only artifact a developer hands to a data
collector. It bundles three things that previously lived in separate files:

  - the operator/hardware/task metadata (formerly ``anyverse_collection_info.json``)
  - the runtime task spec — roles, prompts, success criteria, safety, server
    config (formerly ``task_specs/<task>/<mode>.json``)
  - the recording / inference / intervention / self_play / subtask CLI knobs
    that used to be hard-coded in per-task shell scripts (NUM_EPISODES,
    EPISODE_TIME_S, ACTION_HORIZON, …)

The data collector runs::

    bash scripts/run_session.sh path/to/<task>.session.json

and that's it — no env vars, no per-task shell to edit, no separate
collection_info / task_spec paths to keep in sync.

JSON schema (top level)::

    {
        "schema_version": "1",
        "session_id": "seatbelt.arxx5.record.v1",

        // → CollectionInfo
        "robot":           {"type": "...", "id": "..."},
        "hardware_meta":   {"end_effector": {...}, "cameras": {...}},
        "collection_meta": {"operator_name": "...", "mode": "record", ...},
        "task_meta":       {"task_name": "...", "task_stage": {...}, "objects": {...}},
        "task_description": "...",

        // → TaskSpec (inline OR path string OR null)
        "task_spec": { "task_id": "...", "roles": {...}, ... }
                  |  "lerobot_example_config_files/task_specs/seatbelt/record.json"
                  |  null,

        // → CLI knobs (every key optional; defaults match RecordConfig)
        "recording":    { num_episodes, episode_time_s, reset_time_s, fps, ... },
        "inference":    { mode, infer_interval, action_horizon, ... },
        "intervention": { waiting_intervention_time_s, ... },
        "self_play":    { max_total_time_s, infer_only },
        "subtask":      { config_path, record_task, durations }
    }

Operators may still pass extra ``--flag=value`` overrides on the command
line; the shell wrapper appends them after the session-derived flags so
they win.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from lerobot.recording.task.collection_info import (
    CollectionInfo,
    CollectionInfoError,
    CollectionMeta,
    HardwareMeta,
    TaskMeta,
)
from lerobot.recording.task.task_spec import TaskSpec


SCHEMA_VERSION = "1"


# Recording-section defaults. Mirrored from ``RecordConfig`` /
# ``DatasetRecordConfig`` so a session that omits a key gets the same value
# the recorder would have used. Adding or changing a default here is a
# deliberate, single-place edit.
RECORDING_DEFAULTS: dict[str, Any] = {
    "num_episodes": 50,
    "episode_time_s": 60,
    "reset_time_s": 5,
    "fps": 30,
    "video": True,
    "display_data": False,
    "play_sounds": True,
    "enable_log_say": True,
    "enable_logging": True,
    "progress_print_interval_s": 1.0,
    "auto_success": False,
    "resume": False,
    "debug": False,
    "video_encoding_batch_size": 1,
    "num_image_writer_processes": 0,
    "num_image_writer_threads_per_camera": 4,
    "push_to_hub": False,
    "private": False,
    "data_root": "$HOME/lerobot_data_collection",
}

INFERENCE_DEFAULTS: dict[str, Any] = {
    "mode": "async",
    "infer_interval": 20,
    "default_infer_delay": 0,
    "inference_speedup": 1.0,
    "auto_infer_interval": True,
    "action_horizon": 50,
    "fusion_type": "linear",
    "fusion_exp_decay": 2.0,
    "fusion_window": 0,
    "smooth_sigma": 0.0,
    "transition_steps": 15,
    "log_action_chunks": False,
    # Free-form label identifying which model is being served (e.g. a
    # checkpoint slug like "seatbelt_recap_v1"). Empty string when the
    # session runs no model inference (record mode). Lands in
    # ``meta.info["infer_meta"]["model_name"]`` so downstream analyses can
    # group episodes by model without re-parsing logs.
    "model_name": "",
    # When true and the session writes a dataset, every inference call is
    # appended to ``<dataset>/meta/inference_log.parquet`` with submit/complete
    # timestamps, latency, raw pre-fusion action chunk, prompt, and RTC delay.
    # Off by default — opt in only when you intend to consume the log.
    "persist_inference_log": False,
}

INTERVENTION_DEFAULTS: dict[str, Any] = {
    "waiting_intervention_time_s": 2.0,
    "waiting_evacuation_time_s": 2.0,
    "pose_sync_duration_s": 3.0,
    "leader_movement_timeout_s": 30.0,
}

SELF_PLAY_DEFAULTS: dict[str, Any] = {
    "max_total_time_s": None,
    "infer_only": False,
}

SUBTASK_DEFAULTS: dict[str, Any] = {
    "config_path": None,
    "record_task": None,
    "durations": None,
}


class SessionConfigError(ValueError):
    """Raised when a session config JSON fails validation.

    Lists every problem found, not just the first.
    """

    def __init__(self, violations: list[str], path: str | Path | None = None):
        self.violations = list(violations)
        self.path = str(path) if path else None
        super().__init__(self._format())

    def _format(self) -> str:
        bullet = "\n  - "
        head = (
            f"session config {self.path!r} failed validation"
            if self.path
            else "session config failed validation"
        )
        return f"{head} ({len(self.violations)} problem(s)):" + bullet + bullet.join(
            self.violations
        )


def _merge_with_defaults(section: dict | None, defaults: dict[str, Any]) -> dict[str, Any]:
    """Return ``defaults`` merged with ``section``, accepting unknown keys.

    Unknown keys are kept so we can complain about them in :meth:`validate`
    instead of silently dropping operator-supplied values.
    """
    out = dict(defaults)
    if section:
        out.update(section)
    return out


@dataclass
class SessionConfig:
    """Bundled session config: collection info + task spec + runtime knobs.

    Construct with :meth:`from_json`. Run :meth:`validate` once before use.
    The recorder converts the metadata half to a :class:`CollectionInfo`
    via :meth:`to_collection_info`; the task half is exposed as
    ``task_spec`` (already loaded if it was a path string).
    """

    schema_version: str
    session_id: str

    robot_type: str
    robot_id: str

    hardware_meta: dict
    collection_meta: dict
    task_meta: dict
    task_description: str

    task_spec: Optional[TaskSpec]
    task_spec_source: Optional[str]  # set when task_spec was loaded from a path

    recording: dict[str, Any]
    inference: dict[str, Any]
    intervention: dict[str, Any]
    self_play: dict[str, Any]
    subtask: dict[str, Any]

    # The original raw JSON, for stashing into the dataset's meta.info so
    # the dataset is fully reproducible from a single artifact.
    raw: dict = field(default_factory=dict)
    source_path: Optional[str] = None

    # ---- Loading -------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "SessionConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data, source_path=str(path))

    @classmethod
    def from_dict(cls, d: dict, source_path: Optional[str] = None) -> "SessionConfig":
        robot = d.get("robot") or {}

        # task_spec may be inline (dict), a path (str), or omitted (null).
        task_spec_raw = d.get("task_spec")
        task_spec_obj: Optional[TaskSpec] = None
        task_spec_source: Optional[str] = None
        if isinstance(task_spec_raw, str) and task_spec_raw.strip():
            spec_path = _resolve_path(task_spec_raw, source_path)
            task_spec_source = str(spec_path)
            task_spec_obj = TaskSpec.from_json(spec_path)
        elif isinstance(task_spec_raw, dict):
            task_spec_obj = TaskSpec.from_dict(task_spec_raw)

        return cls(
            schema_version=str(d.get("schema_version", "")),
            session_id=str(d.get("session_id", "")),
            robot_type=str(robot.get("type", "")),
            robot_id=str(robot.get("id", "")),
            hardware_meta=dict(d.get("hardware_meta") or {}),
            collection_meta=dict(d.get("collection_meta") or {}),
            task_meta=dict(d.get("task_meta") or {}),
            task_description=str(d.get("task_description", "")),
            task_spec=task_spec_obj,
            task_spec_source=task_spec_source,
            recording=_merge_with_defaults(d.get("recording"), RECORDING_DEFAULTS),
            inference=_merge_with_defaults(d.get("inference"), INFERENCE_DEFAULTS),
            intervention=_merge_with_defaults(d.get("intervention"), INTERVENTION_DEFAULTS),
            self_play=_merge_with_defaults(d.get("self_play"), SELF_PLAY_DEFAULTS),
            subtask=_merge_with_defaults(d.get("subtask"), SUBTASK_DEFAULTS),
            raw=dict(d),
            source_path=source_path,
        )

    # ---- Conversion ----------------------------------------------------

    def to_collection_info(self) -> CollectionInfo:
        """Build a :class:`CollectionInfo` for the recorder.

        Uses the same dataclass shape the recorder already understands so
        the rest of the pipeline (validation, prompt template context,
        flat meta.info writes for each sub-meta) is unchanged.
        """
        return CollectionInfo(
            hardware_meta=HardwareMeta.from_dict(self.hardware_meta),
            collection_meta=CollectionMeta.from_dict(self.collection_meta),
            task_meta=TaskMeta.from_dict(self.task_meta),
            robot_type=self.robot_type,
            robot_id=self.robot_id,
            task_description=self.task_description,
        )

    @property
    def mode(self) -> str:
        return str(self.collection_meta.get("mode", ""))

    @property
    def task_name(self) -> str:
        return str(self.task_meta.get("task_name", ""))

    @property
    def data_root(self) -> str:
        """Resolved data_root with ``$HOME``/``$VAR`` expansion applied."""
        raw = str(self.recording.get("data_root") or RECORDING_DEFAULTS["data_root"])
        return os.path.expandvars(os.path.expanduser(raw))

    # ---- Validation ----------------------------------------------------

    def validate(self) -> None:
        """Validate the session config and all embedded sections.

        Raises :class:`SessionConfigError` listing every problem found.
        """
        violations: list[str] = []

        if not self.schema_version:
            violations.append("schema_version is required")
        elif self.schema_version != SCHEMA_VERSION:
            violations.append(
                f"schema_version={self.schema_version!r} unsupported "
                f"(this build understands {SCHEMA_VERSION!r})"
            )
        if not self.session_id.strip():
            violations.append("session_id is required and must be non-empty")

        # Embedded CollectionInfo: reuse its validator and prefix every
        # violation so the operator sees `collection_meta.mode is required`
        # vs `recording.num_episodes …` distinctly.
        try:
            self.to_collection_info().validate()
        except CollectionInfoError as e:
            violations.extend(e.violations)

        # Recording knobs — type + range checks (only fields we know about).
        violations.extend(_check_recording(self.recording))
        violations.extend(_check_inference(self.inference))
        violations.extend(_check_intervention(self.intervention))
        violations.extend(_check_self_play(self.self_play))
        violations.extend(_check_subtask(self.subtask))

        # Mode-specific cross-checks
        violations.extend(self._check_mode_specific())

        # Unknown keys at the top level — surface typos instead of silently
        # dropping them.
        known_top_level = {
            "schema_version", "session_id", "robot",
            "hardware_meta", "collection_meta", "task_meta", "task_description",
            "task_spec",
            "recording", "inference", "intervention", "self_play", "subtask",
        }
        for key in self.raw:
            # Keys starting with "_" are reserved for inline annotation
            # (e.g. ``_comment``, ``_comment_session_id``) and ignored.
            if key not in known_top_level and not key.startswith("_"):
                violations.append(f"unknown top-level key: {key!r}")

        if violations:
            raise SessionConfigError(violations, path=self.source_path)

    def _check_mode_specific(self) -> list[str]:
        out: list[str] = []
        mode = self.mode
        if mode in ("infer", "infer_record", "self_play") and self.task_spec is None:
            out.append(
                f"task_spec is required for mode={mode!r} "
                "(policy server config lives in the task spec)"
            )
        if mode == "self_play":
            if self.task_spec is not None and not self.task_spec.is_self_play:
                out.append("self_play mode requires a task_spec with multiple roles")
        if mode == "record":
            sub_path = self.subtask.get("config_path")
            sub_record = self.subtask.get("record_task")
            if bool(sub_path) ^ bool(sub_record):
                out.append(
                    "subtask.config_path and subtask.record_task must be "
                    "specified together (record mode YAML subtask flow)"
                )
        return out


# ---------------------------------------------------------------------------
# Section-level field checks
# ---------------------------------------------------------------------------

def _check_known_keys(section_name: str, section: dict, known: set[str]) -> list[str]:
    # Keys starting with "_" are reserved for inline annotation (e.g.
    # ``_comment_*``) and ignored by the validator + every downstream
    # consumer. Templates use this to embed per-field documentation without
    # an out-of-band README.
    return [
        f"{section_name}.{k} is not a recognized key"
        for k in section
        if k not in known and not k.startswith("_")
    ]


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _check_recording(rec: dict) -> list[str]:
    out = _check_known_keys("recording", rec, set(RECORDING_DEFAULTS) | {"data_root"})
    if not isinstance(rec.get("num_episodes"), int) or rec["num_episodes"] <= 0:
        out.append("recording.num_episodes must be a positive integer")
    for k in ("episode_time_s", "reset_time_s", "progress_print_interval_s"):
        v = rec.get(k)
        if not _is_number(v) or v < 0:
            out.append(f"recording.{k} must be a non-negative number")
    if not isinstance(rec.get("fps"), int) or rec["fps"] <= 0:
        out.append("recording.fps must be a positive integer")
    for k in ("video", "display_data", "play_sounds", "enable_log_say",
              "enable_logging", "auto_success", "resume", "debug",
              "push_to_hub", "private"):
        if not isinstance(rec.get(k), bool):
            out.append(f"recording.{k} must be a boolean")
    return out


def _check_inference(inf: dict) -> list[str]:
    out = _check_known_keys("inference", inf, set(INFERENCE_DEFAULTS))
    if inf.get("mode") not in ("async", "sync"):
        out.append("inference.mode must be 'async' or 'sync'")
    for k in ("infer_interval", "default_infer_delay", "action_horizon",
              "fusion_window", "transition_steps"):
        v = inf.get(k)
        if not isinstance(v, int) or v < 0:
            out.append(f"inference.{k} must be a non-negative integer")
    for k in ("inference_speedup", "fusion_exp_decay", "smooth_sigma"):
        v = inf.get(k)
        if not _is_number(v) or v < 0:
            out.append(f"inference.{k} must be a non-negative number")
    if not isinstance(inf.get("auto_infer_interval"), bool):
        out.append("inference.auto_infer_interval must be a boolean")
    if not isinstance(inf.get("log_action_chunks"), bool):
        out.append("inference.log_action_chunks must be a boolean")
    if not isinstance(inf.get("persist_inference_log"), bool):
        out.append("inference.persist_inference_log must be a boolean")
    if not isinstance(inf.get("fusion_type"), str) or not inf["fusion_type"].strip():
        out.append("inference.fusion_type must be a non-empty string")
    if not isinstance(inf.get("model_name"), str):
        out.append("inference.model_name must be a string (use \"\" when no model is served)")
    return out


def _check_intervention(it: dict) -> list[str]:
    out = _check_known_keys("intervention", it, set(INTERVENTION_DEFAULTS))
    for k in INTERVENTION_DEFAULTS:
        v = it.get(k)
        if not _is_number(v) or v < 0:
            out.append(f"intervention.{k} must be a non-negative number")
    return out


def _check_self_play(sp: dict) -> list[str]:
    out = _check_known_keys("self_play", sp, set(SELF_PLAY_DEFAULTS))
    mt = sp.get("max_total_time_s")
    if mt is not None and (not _is_number(mt) or mt < 0):
        out.append("self_play.max_total_time_s must be null or a non-negative number")
    if not isinstance(sp.get("infer_only"), bool):
        out.append("self_play.infer_only must be a boolean")
    return out


def _check_subtask(st: dict) -> list[str]:
    out = _check_known_keys("subtask", st, set(SUBTASK_DEFAULTS))
    cp = st.get("config_path")
    if cp is not None and not isinstance(cp, str):
        out.append("subtask.config_path must be a string or null")
    rt = st.get("record_task")
    if rt is not None and not isinstance(rt, str):
        out.append("subtask.record_task must be a string or null")
    durations = st.get("durations")
    if durations is not None:
        if not isinstance(durations, list) or not all(_is_number(x) and x >= 0 for x in durations):
            out.append("subtask.durations must be null or a list of non-negative numbers")
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(p: str, source: Optional[str]) -> Path:
    """Resolve a path that may live next to the session JSON or be relative
    to the repo root.

    - Absolute paths are returned unchanged.
    - Paths starting with ``./`` or ``../`` are interpreted relative to the
      session JSON's own directory (sibling-file convention).
    - Anything else is interpreted relative to the current working
      directory — ``scripts/run_session.sh`` always ``cd``s into the repo
      root before invoking Python, so this matches operator expectations
      (``"task_spec": "lerobot_example_config_files/task_specs/..."``
      reads from the repo root, like every other path in the repo).
    """
    expanded = os.path.expandvars(os.path.expanduser(p))
    path = Path(expanded)
    if path.is_absolute():
        return path
    if expanded.startswith("./") or expanded.startswith("../"):
        if source is not None:
            return (Path(source).parent / path).resolve()
        return path.resolve()
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists() or source is None:
        return cwd_candidate
    sibling_candidate = (Path(source).parent / path).resolve()
    if sibling_candidate.exists():
        return sibling_candidate
    return cwd_candidate
