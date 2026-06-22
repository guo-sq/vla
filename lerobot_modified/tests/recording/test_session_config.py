"""Tests for recording.task.session_config — schema, validation, conversion."""

import json
from pathlib import Path

import pytest

from lerobot.recording.task.collection_info import CollectionInfo
from lerobot.recording.task.session_config import (
    RECORDING_DEFAULTS,
    SCHEMA_VERSION,
    SessionConfig,
    SessionConfigError,
)
from lerobot.recording.task.task_spec import TaskSpec


def _minimal_dict(mode: str = "record") -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": "test.session.v1",
        "robot": {"type": "arxx5_bimanual", "id": "5"},
        "hardware_meta": {
            "end_effector": {
                "left":  {"type": "gripper", "model": "long"},
                "right": {"type": "gripper", "model": "long"},
            },
            "cameras": {
                "head": {
                    "type": "opencv",
                    "index_or_path": 0,
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
            },
        },
        "collection_meta": {
            "operator_name": "alice",
            "adversary_operator": "",
            "is_adversary": False,
            "is_self_play": False,
            "site_location": "lab",
            "city": "beijing",
            "mode": mode,
        },
        "task_meta": {
            "task_name": "test_task",
            "task_stage": {"mode": "full", "stages": ""},
            "objects": {"socks": {"color": "white"}},
        },
        "task_description": "test",
        "task_spec": {
            "task_id": "test_task",
            "roles": {"operator": {"prompt": "do the {socks.color} thing"}},
        },
    }


class TestSessionConfigBasics:
    def test_minimal_record_validates(self):
        sess = SessionConfig.from_dict(_minimal_dict())
        sess.validate()
        assert sess.mode == "record"
        assert sess.task_name == "test_task"
        assert isinstance(sess.task_spec, TaskSpec)

    def test_recording_defaults_are_filled_in(self):
        sess = SessionConfig.from_dict(_minimal_dict())
        for k, v in RECORDING_DEFAULTS.items():
            assert sess.recording[k] == v

    def test_recording_overrides_apply(self):
        d = _minimal_dict()
        d["recording"] = {"num_episodes": 99, "episode_time_s": 7.5}
        sess = SessionConfig.from_dict(d)
        sess.validate()
        assert sess.recording["num_episodes"] == 99
        assert sess.recording["episode_time_s"] == 7.5
        # untouched defaults still present
        assert sess.recording["fps"] == RECORDING_DEFAULTS["fps"]

    def test_to_collection_info_round_trips(self):
        sess = SessionConfig.from_dict(_minimal_dict())
        info = sess.to_collection_info()
        assert isinstance(info, CollectionInfo)
        assert info.collection_meta.mode == "record"
        assert info.task_meta.task_name == "test_task"
        info.validate()  # delegated structural validation


class TestSessionConfigValidation:
    def test_missing_schema_version_fails(self):
        d = _minimal_dict()
        del d["schema_version"]
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("schema_version" in v for v in ei.value.violations)

    def test_unsupported_schema_version_fails(self):
        d = _minimal_dict()
        d["schema_version"] = "9999"
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError):
            sess.validate()

    def test_self_play_requires_task_spec(self):
        d = _minimal_dict("self_play")
        d.pop("task_spec")
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("task_spec" in v and "self_play" in v for v in ei.value.violations)

    def test_self_play_requires_multiple_roles(self):
        d = _minimal_dict("self_play")
        # task_spec already only has one role — invalid for self_play
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("multiple roles" in v for v in ei.value.violations)

    def test_unknown_top_level_key_flagged(self):
        d = _minimal_dict()
        d["bogus_field"] = 42
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("bogus_field" in v for v in ei.value.violations)

    def test_negative_num_episodes_fails(self):
        d = _minimal_dict()
        d["recording"] = {"num_episodes": 0}
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("num_episodes" in v for v in ei.value.violations)

    def test_unknown_recording_key_flagged(self):
        d = _minimal_dict()
        d["recording"] = {"frames_per_second": 99}  # typo for fps
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("frames_per_second" in v for v in ei.value.violations)

    def test_subtask_path_without_record_task_flagged(self):
        d = _minimal_dict()
        d["subtask"] = {"config_path": "foo.yaml"}
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("subtask.config_path" in v and "record_task" in v for v in ei.value.violations)

    def test_collection_info_violations_propagate(self):
        d = _minimal_dict()
        d["collection_meta"]["operator_name"] = ""
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("operator_name" in v for v in ei.value.violations)

    def test_inference_model_name_defaults_to_empty_string(self):
        d = _minimal_dict()
        sess = SessionConfig.from_dict(d)
        sess.validate()
        assert sess.inference["model_name"] == ""

    def test_inference_model_name_round_trips(self):
        d = _minimal_dict()
        d["inference"] = {"model_name": "seatbelt_recap_v1"}
        sess = SessionConfig.from_dict(d)
        sess.validate()
        assert sess.inference["model_name"] == "seatbelt_recap_v1"

    def test_inference_model_name_non_string_rejected(self):
        d = _minimal_dict()
        d["inference"] = {"model_name": 42}
        sess = SessionConfig.from_dict(d)
        with pytest.raises(SessionConfigError) as ei:
            sess.validate()
        assert any("model_name" in v for v in ei.value.violations)


class TestTaskSpecLoading:
    def test_inline_task_spec(self):
        sess = SessionConfig.from_dict(_minimal_dict())
        assert sess.task_spec is not None
        assert sess.task_spec.task_id == "test_task"
        assert sess.task_spec_source is None

    def test_task_spec_from_path(self, tmp_path):
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps({
            "task_id": "external_spec",
            "roles": {"operator": {"prompt": "external"}},
        }))
        d = _minimal_dict()
        d["task_spec"] = str(spec_path)
        sess = SessionConfig.from_dict(d, source_path=str(tmp_path / "session.json"))
        sess.validate()
        assert sess.task_spec is not None
        assert sess.task_spec.task_id == "external_spec"
        assert sess.task_spec_source == str(spec_path)

    def test_task_spec_relative_path_from_session(self, tmp_path):
        # ./spec.json is interpreted relative to the session file's parent.
        spec_path = tmp_path / "specs" / "spec.json"
        spec_path.parent.mkdir(parents=True)
        spec_path.write_text(json.dumps({
            "task_id": "sibling_spec",
            "roles": {"operator": {"prompt": "sibling"}},
        }))
        d = _minimal_dict()
        d["task_spec"] = "./specs/spec.json"
        sess = SessionConfig.from_dict(d, source_path=str(tmp_path / "session.json"))
        sess.validate()
        assert sess.task_spec.task_id == "sibling_spec"


class TestInlineComments:
    """``_``-prefixed keys are reserved for inline annotation (e.g.
    ``_comment_session_id`` next to ``session_id``) and must not raise
    validation errors. This is the mechanism templates rely on for
    self-documenting JSON without a separate README."""

    def test_underscore_keys_pass_at_every_level(self):
        d = _minimal_dict()
        d["recording"] = {"_comment_episode_time_s": "per-episode timeout"}
        d["inference"] = {"_comment_infer_interval": "control ticks between calls"}
        d["intervention"] = {"_doc": "intervention defaults"}
        d["self_play"] = {"_doc": "self-play knobs"}
        d["subtask"] = {"_doc": "subtask schedule"}
        d["_doc"] = "top-level orientation comment"
        d["_comment_session_id"] = "convention: <task>.<mode>.v<n>"
        d["robot"]["_comment_type"] = "options: arxx5_bimanual / bi_piper_follower"
        d["hardware_meta"]["cameras"]["_comment_head"] = (
            "index_or_path: integer → /dev/videoN, string → arbitrary path"
        )
        d["collection_meta"]["_comment_operator_name"] = "who is running this run"
        d["task_meta"]["_comment_task_stage"] = "set mode='full' for an end-to-end run"
        sess = SessionConfig.from_dict(d)
        sess.validate()
        # Comment keys survive in raw so the on-disk JSON is preserved verbatim.
        assert sess.raw["_doc"].startswith("top-level")

    def test_underscore_top_level_key_does_not_raise_unknown_key(self):
        d = _minimal_dict()
        d["_lint_marker"] = "ignored"
        SessionConfig.from_dict(d).validate()  # no SessionConfigError

    def test_underscore_camera_name_ignored(self):
        d = _minimal_dict()
        d["hardware_meta"]["cameras"]["_comment_layout"] = (
            "3-camera layout: head / left_wrist / right_wrist"
        )
        SessionConfig.from_dict(d).validate()


class TestFromJson:
    def test_round_trip_through_disk(self, tmp_path):
        d = _minimal_dict()
        p = tmp_path / "session.json"
        p.write_text(json.dumps(d))
        sess = SessionConfig.from_json(p)
        sess.validate()
        assert sess.source_path == str(p)
        assert sess.raw == d
