"""Convert a session config JSON to a list of ``--flag=value`` CLI args.

Used by ``scripts/run_session.sh`` to splat the session's recording /
inference / intervention / self_play / subtask sections onto the
``python -m lerobot.recording.record`` command line. The resulting argv
is identical to what a developer would have typed by hand pre-overhaul,
which keeps the executed command self-explanatory in process listings
and recording logs.

Subcommands:

  python -m lerobot.recording.utils.session_to_args args <session.json>
      Print one ``--flag=value`` per line, ready for ``mapfile -t``.

  python -m lerobot.recording.utils.session_to_args field <session.json> <dotted.path>
      Print the resolved scalar value at the given path (e.g.
      ``task_meta.task_name``). For shell-side path defaulting only.

  python -m lerobot.recording.utils.session_to_args video_devs <session.json>
      Print one ``/dev/videoN`` per line for each integer-indexed camera
      so the wrapper can ``chmod 777`` them in a loop.

Exit code 0 on success, 1 on validation failure (errors go to stderr).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

from lerobot.recording.task.session_config import SessionConfig, SessionConfigError


# ---------------------------------------------------------------------------
# args: emit --flag=value list
# ---------------------------------------------------------------------------

def _emit(flag: str, value: Any) -> str:
    """Format a CLI arg as ``--flag=value``.

    - bools render lower-case (``true`` / ``false``) — matches draccus.
    - lists render as JSON — same convention the existing record.py uses
      for ``--sub_task_durations``.
    - None is emitted as the literal string ``null`` so draccus can clear an
      Optional field. (We skip None entirely below; this branch is here for
      completeness.)
    """
    if isinstance(value, bool):
        return f"--{flag}={'true' if value else 'false'}"
    if isinstance(value, (list, dict)):
        return f"--{flag}={json.dumps(value)}"
    if value is None:
        return f"--{flag}=null"
    return f"--{flag}={value}"


def session_to_args(session: SessionConfig) -> list[str]:
    """Build the full ``--flag=value`` list from a validated session.

    Path of ``--session_config_path`` is included so the recorder can stash
    the raw JSON into ``meta.info["session_config"]``.
    """
    args: list[str] = []

    # Session config path itself — always first so it logs at the front.
    if session.source_path:
        args.append(f"--session_config_path={session.source_path}")

    # Robot — flat fields only. Cameras flow through collection_info →
    # _setup_cameras_from_collection_info inside RecordConfig.__post_init__,
    # so we don't enumerate them here.
    if session.robot_type:
        args.append(f"--robot.type={session.robot_type}")
    if session.robot_id:
        args.append(f"--robot.id={session.robot_id}")

    # Recording section → dataset.* + top-level fields.
    rec = session.recording
    args += _emit_recording(rec)
    args += _emit_inference(session.inference, mode=session.mode)
    args += _emit_intervention(session.intervention)
    args += _emit_self_play(session.self_play)
    args += _emit_subtask(session.subtask)

    return args


# Mapping from session-key → record.py CLI flag. Keys are intentionally
# different in the session JSON (operator-readable) vs. in record.py (where
# they grew organically); the table below is the only place that bridge
# lives.

_RECORDING_FLAT = {
    # session key → CLI flag (top-level RecordConfig)
    "display_data": "display_data",
    "play_sounds": "play_sounds",
    "enable_log_say": "enable_log_say",
    "enable_logging": "enable_logging",
    "progress_print_interval_s": "progress_print_interval_s",
    "auto_success": "auto_success",
    "resume": "resume",
    "debug": "debug",
}

_RECORDING_DATASET = {
    "num_episodes": "dataset.num_episodes",
    "episode_time_s": "dataset.episode_time_s",
    "reset_time_s": "dataset.reset_time_s",
    "fps": "dataset.fps",
    "video": "dataset.video",
    "video_encoding_batch_size": "dataset.video_encoding_batch_size",
    "num_image_writer_processes": "dataset.num_image_writer_processes",
    "num_image_writer_threads_per_camera": "dataset.num_image_writer_threads_per_camera",
    "push_to_hub": "dataset.push_to_hub",
    "private": "dataset.private",
}

_INFERENCE_MAP = {
    "mode": "inference_mode",
    "infer_interval": "infer_interval",
    "default_infer_delay": "default_infer_delay",
    "inference_speedup": "inference_speedup",
    "auto_infer_interval": "auto_infer_interval",
    "action_horizon": "action_horizon",
    "fusion_type": "fusion_type",
    "fusion_exp_decay": "fusion_exp_decay",
    "fusion_window": "fusion_window",
    "smooth_sigma": "smooth_sigma",
    "transition_steps": "transition_steps",
    "log_action_chunks": "log_action_chunks",
    "persist_inference_log": "persist_inference_log",
}

_INTERVENTION_MAP = {
    "waiting_intervention_time_s": "waiting_intervention_time_s",
    "waiting_evacuation_time_s": "waiting_evacuation_time_s",
    "pose_sync_duration_s": "pose_sync_duration_s",
    "leader_movement_timeout_s": "leader_movement_timeout_s",
}

_SELF_PLAY_MAP = {
    "max_total_time_s": "self_play_max_total_time_s",
    "infer_only": "self_play_infer_only",
}

_SUBTASK_MAP = {
    "config_path": "subtask_config_path",
    "record_task": "record_task",
    "durations": "sub_task_durations",
}


def _emit_section(section: dict, mapping: dict, *, skip_none: bool = True) -> list[str]:
    out: list[str] = []
    for src_key, cli_flag in mapping.items():
        if src_key not in section:
            continue
        v = section[src_key]
        if skip_none and v is None:
            continue
        out.append(_emit(cli_flag, v))
    return out


def _emit_recording(rec: dict) -> list[str]:
    out: list[str] = []
    out += _emit_section(rec, _RECORDING_FLAT)
    out += _emit_section(rec, _RECORDING_DATASET)
    # data_root is consumed by the shell wrapper to derive
    # dataset.root / dataset.repo_id; not a CLI flag.
    return out


def _emit_inference(inf: dict, *, mode: str) -> list[str]:
    # In record mode the inference section is irrelevant. Skip it
    # entirely so the recorder's own irrelevant-flag warnings don't
    # fire on every run.
    if mode == "record":
        return []
    return _emit_section(inf, _INFERENCE_MAP)


def _emit_intervention(it: dict) -> list[str]:
    return _emit_section(it, _INTERVENTION_MAP)


def _emit_self_play(sp: dict) -> list[str]:
    return _emit_section(sp, _SELF_PLAY_MAP)


def _emit_subtask(st: dict) -> list[str]:
    return _emit_section(st, _SUBTASK_MAP)


# ---------------------------------------------------------------------------
# field: emit a scalar value at a dotted path
# ---------------------------------------------------------------------------

def _resolve_path(d: dict, dotted: str) -> Any:
    obj: Any = d
    for part in dotted.split("."):
        if isinstance(obj, dict) and part in obj:
            obj = obj[part]
        else:
            raise KeyError(dotted)
    return obj


# ---------------------------------------------------------------------------
# video_devs: list /dev/videoN entries from the session
# ---------------------------------------------------------------------------

def _video_devs(session: SessionConfig) -> list[str]:
    out: list[str] = []
    cams = session.hardware_meta.get("cameras") or {}
    for spec in cams.values():
        idx = spec.get("index_or_path") if isinstance(spec, dict) else None
        if isinstance(idx, int):
            out.append(f"/dev/video{idx}")
        elif isinstance(idx, str) and idx.isdigit():
            out.append(f"/dev/video{int(idx)}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_usage() -> None:
    print(
        "Usage:\n"
        "  python -m lerobot.recording.utils.session_to_args args <session.json>\n"
        "  python -m lerobot.recording.utils.session_to_args field <session.json> <dotted.path>\n"
        "  python -m lerobot.recording.utils.session_to_args video_devs <session.json>",
        file=sys.stderr,
    )


def _load_session(path: str) -> SessionConfig:
    p = Path(path)
    if not p.is_file():
        print(f"session config not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    try:
        sess = SessionConfig.from_json(p)
        sess.validate()
    except SessionConfigError as e:
        print(str(e), file=sys.stderr)
        raise SystemExit(1)
    except Exception as e:
        print(f"Failed to load {path}: {e}", file=sys.stderr)
        raise SystemExit(1)
    return sess


def _print_lines(lines: Iterable[str]) -> None:
    for line in lines:
        print(line)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_usage()
        return 2

    cmd = args[0]

    if cmd == "args":
        if len(args) != 2:
            _print_usage()
            return 2
        sess = _load_session(args[1])
        _print_lines(session_to_args(sess))
        return 0

    if cmd == "field":
        if len(args) != 3:
            _print_usage()
            return 2
        sess = _load_session(args[1])
        try:
            v = _resolve_path(sess.raw, args[2])
        except KeyError:
            print(f"path not found in session: {args[2]}", file=sys.stderr)
            return 1
        # Scalar only — composite values are never queried by the wrapper.
        if isinstance(v, (dict, list)):
            print(f"path resolves to a composite value: {args[2]}", file=sys.stderr)
            return 1
        print(v if v is not None else "")
        return 0

    if cmd == "video_devs":
        if len(args) != 2:
            _print_usage()
            return 2
        sess = _load_session(args[1])
        _print_lines(_video_devs(sess))
        return 0

    _print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
