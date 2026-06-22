"""Tests for recording.utils.session_to_args — JSON → CLI args translation."""

import json

import pytest

from lerobot.recording.task.session_config import SCHEMA_VERSION, SessionConfig
from lerobot.recording.utils.session_to_args import (
    _video_devs,
    session_to_args,
)


def _minimal_dict(mode: str = "record") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": "test.v1",
        "robot": {"type": "arxx5_bimanual", "id": "5"},
        "hardware_meta": {
            "end_effector": {
                "left":  {"type": "gripper", "model": "long"},
                "right": {"type": "gripper", "model": "long"},
            },
            "cameras": {
                "head":  {"type": "opencv", "index_or_path": 4,  "width": 640, "height": 480, "fps": 30},
                "left":  {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30},
                "right": {"type": "opencv", "index_or_path": "12", "width": 640, "height": 480, "fps": 30},
            },
        },
        "collection_meta": {
            "operator_name": "alice", "adversary_operator": "",
            "is_adversary": False, "is_self_play": False,
            "site_location": "lab", "city": "beijing", "mode": mode,
        },
        "task_meta": {
            "task_name": "t", "task_stage": {"mode": "full", "stages": ""}, "objects": {},
        },
        "task_description": "",
        "task_spec": {"task_id": "t", "roles": {"a": {"prompt": "p"}, "b": {"prompt": "p2"}}}
            if mode == "self_play"
            else {"task_id": "t", "roles": {"operator": {"prompt": "p"}}},
        "recording": {"num_episodes": 7, "episode_time_s": 99},
    }


class TestSessionToArgs:
    def test_record_emits_no_inference_section(self):
        sess = SessionConfig.from_dict(_minimal_dict("record"))
        sess.validate()
        args = session_to_args(sess)
        joined = " ".join(args)
        assert "--robot.type=arxx5_bimanual" in args
        assert "--robot.id=5" in args
        assert "--dataset.num_episodes=7" in args
        assert "--dataset.episode_time_s=99" in args
        # In record mode the inference section is suppressed (otherwise the
        # recorder warns about every irrelevant flag every run).
        assert "--inference_mode" not in joined
        assert "--action_horizon" not in joined

    def test_self_play_emits_inference_args(self):
        d = _minimal_dict("self_play")
        d["inference"] = {"action_horizon": 30}
        sess = SessionConfig.from_dict(d)
        sess.validate()
        args = session_to_args(sess)
        joined = " ".join(args)
        assert "--inference_mode=async" in joined
        assert "--action_horizon=30" in joined
        assert "--self_play_infer_only=false" in joined

    def test_session_config_path_first_when_present(self, tmp_path):
        d = _minimal_dict()
        p = tmp_path / "s.json"
        p.write_text(json.dumps(d))
        sess = SessionConfig.from_json(p)
        sess.validate()
        args = session_to_args(sess)
        assert args[0] == f"--session_config_path={p}"

    def test_bool_serialization(self):
        d = _minimal_dict()
        d["recording"]["display_data"] = True
        d["recording"]["auto_success"] = False
        sess = SessionConfig.from_dict(d)
        sess.validate()
        args = session_to_args(sess)
        assert "--display_data=true" in args
        assert "--auto_success=false" in args


class TestVideoDevs:
    def test_int_and_string_indices(self):
        sess = SessionConfig.from_dict(_minimal_dict())
        devs = _video_devs(sess)
        # Order follows dict iteration order from the JSON.
        assert devs == ["/dev/video4", "/dev/video10", "/dev/video12"]

    def test_path_device_skipped(self):
        d = _minimal_dict()
        d["hardware_meta"]["cameras"]["realsense"] = {
            "type": "realsense",
            "index_or_path": "/dev/video20",  # non-numeric → skipped
            "width": 640, "height": 480, "fps": 30,
        }
        sess = SessionConfig.from_dict(d)
        devs = _video_devs(sess)
        assert "/dev/video20" not in devs
