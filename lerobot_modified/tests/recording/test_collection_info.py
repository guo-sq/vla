"""Tests for recording.task.collection_info — schema, validation, time stamping."""

import json
import re
from pathlib import Path

import pytest

from lerobot.recording.task.collection_info import (
    CollectionInfo,
    CollectionInfoError,
    VALID_MODES,
    VALID_TASK_STAGE_MODES,
)


def _valid_dict() -> dict:
    return {
        "hardware_meta": {
            "end_effector": {
                "left":  {"type": "gripper", "model": "long"},
                "right": {"type": "gripper", "model": "long"},
            },
            "cameras": {
                "head": {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
                "left_wrist": {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30},
                "right_wrist": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
            },
        },
        "collection_meta": {
            "operator_name": "baichenglong",
            "adversary_operator": "",
            "is_adversary": False,
            "is_self_play": False,
            "site_location": "beijing_6F",
            "city": "beijing",
            "mode": "record",
        },
        "task_meta": {
            "task_name": "pack_socks",
            "task_stage": {"mode": "full", "stages": ""},
            "objects": {"socks": {"color": "gray", "size": "M"}},
        },
        "robot_type": "arxx5_bimanual",
        "robot_id": "4",
        "task_description": "正常灯光环境",
    }


class TestRoundTrip:
    def test_from_dict_to_dict_idempotent_keys(self):
        info = CollectionInfo.from_dict(_valid_dict())
        out = info.to_dict()
        assert set(out) == {
            "hardware_meta", "collection_meta", "task_meta",
            "robot_type", "robot_id", "task_description",
        }
        assert out["robot_type"] == "arxx5_bimanual"
        assert out["task_meta"]["objects"]["socks"]["color"] == "gray"

    def test_from_json(self, tmp_path: Path):
        path = tmp_path / "info.json"
        path.write_text(json.dumps(_valid_dict()), encoding="utf-8")
        info = CollectionInfo.from_json(path)
        info.validate()  # should not raise
        assert info.collection_meta.mode == "record"


class TestObjectsDefault:
    def test_objects_omitted_defaults_to_empty_dict(self):
        d = _valid_dict()
        del d["task_meta"]["objects"]
        info = CollectionInfo.from_dict(d)
        info.validate()
        assert info.task_meta.objects == {}

    def test_objects_null_treated_as_empty_dict(self):
        d = _valid_dict()
        d["task_meta"]["objects"] = None
        info = CollectionInfo.from_dict(d)
        info.validate()
        assert info.task_meta.objects == {}

    def test_objects_arbitrary_nesting_allowed(self):
        d = _valid_dict()
        d["task_meta"]["objects"] = {"a": {"b": {"c": [1, 2, 3]}}}
        info = CollectionInfo.from_dict(d)
        info.validate()


class TestTaskDescriptionOptional:
    def test_task_description_omitted_defaults_empty(self):
        d = _valid_dict()
        del d["task_description"]
        info = CollectionInfo.from_dict(d)
        info.validate()
        assert info.task_description == ""


class TestRequiredFields:
    @pytest.mark.parametrize("path", [
        ("hardware_meta", "end_effector", "left", "type"),
        ("hardware_meta", "end_effector", "left", "model"),
        ("hardware_meta", "end_effector", "right", "type"),
        ("hardware_meta", "end_effector", "right", "model"),
        ("collection_meta", "operator_name"),
        ("collection_meta", "site_location"),
        ("collection_meta", "city"),
        ("collection_meta", "mode"),
        ("task_meta", "task_name"),
        ("task_meta", "task_stage", "mode"),
        ("robot_type",),
        ("robot_id",),
    ])
    def test_missing_required_field_fails(self, path):
        d = _valid_dict()
        cur = d
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = ""  # empty string ⇒ violation
        info = CollectionInfo.from_dict(d)
        with pytest.raises(CollectionInfoError) as excinfo:
            info.validate()
        joined = ".".join(path)
        # The violation message references the dotted path
        assert any(joined in v for v in excinfo.value.violations)


class TestAllViolationsListed:
    """Validation must report every problem at once, not just the first."""

    def test_multiple_violations_collected(self):
        d = _valid_dict()
        d["collection_meta"]["operator_name"] = ""
        d["robot_id"] = ""
        d["task_meta"]["task_stage"]["mode"] = "partial"
        d["task_meta"]["task_stage"]["stages"] = ""  # mode != full ⇒ stages required
        info = CollectionInfo.from_dict(d)
        with pytest.raises(CollectionInfoError) as excinfo:
            info.validate()
        violations = excinfo.value.violations
        # Expect at least three distinct problems surfaced together.
        assert len(violations) >= 3
        assert any("operator_name" in v for v in violations)
        assert any("robot_id" in v for v in violations)
        assert any("task_stage.stages" in v for v in violations)


class TestTaskStageRule:
    def test_full_mode_stages_may_be_empty(self):
        d = _valid_dict()
        d["task_meta"]["task_stage"] = {"mode": "full", "stages": ""}
        CollectionInfo.from_dict(d).validate()

    def test_partial_mode_requires_stages(self):
        d = _valid_dict()
        d["task_meta"]["task_stage"] = {"mode": "partial", "stages": ""}
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("stages is required" in v for v in excinfo.value.violations)

    def test_recovery_mode_with_stages_passes(self):
        d = _valid_dict()
        d["task_meta"]["task_stage"] = {"mode": "recovery", "stages": "stage_1,stage_2"}
        CollectionInfo.from_dict(d).validate()

    def test_invalid_task_stage_mode_rejected(self):
        d = _valid_dict()
        d["task_meta"]["task_stage"]["mode"] = "garbage"
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("task_stage.mode" in v for v in excinfo.value.violations)


class TestModeRule:
    def test_invalid_mode_rejected(self):
        d = _valid_dict()
        d["collection_meta"]["mode"] = "no_such_mode"
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("collection_meta.mode" in v for v in excinfo.value.violations)

    @pytest.mark.parametrize("mode", VALID_MODES)
    def test_all_valid_modes_accepted(self, mode):
        d = _valid_dict()
        d["collection_meta"]["mode"] = mode
        CollectionInfo.from_dict(d).validate()


class TestAdversaryRule:
    """``adversary_operator`` is informational only — empty string is valid
    even when ``is_adversary=true`` (one human alternating both roles)."""

    def test_is_adversary_with_empty_operator_allowed(self):
        d = _valid_dict()
        d["collection_meta"]["is_adversary"] = True
        d["collection_meta"]["adversary_operator"] = ""
        CollectionInfo.from_dict(d).validate()  # should not raise

    def test_is_adversary_with_named_operator_allowed(self):
        d = _valid_dict()
        d["collection_meta"]["is_adversary"] = True
        d["collection_meta"]["adversary_operator"] = "alice"
        CollectionInfo.from_dict(d).validate()

    def test_legacy_null_coerced_to_empty_string(self):
        # JSON ``null`` should round-trip through from_dict as "".
        d = _valid_dict()
        d["collection_meta"]["adversary_operator"] = None
        info = CollectionInfo.from_dict(d)
        info.validate()
        assert info.collection_meta.adversary_operator == ""
        assert info.to_dict()["collection_meta"]["adversary_operator"] == ""


class TestActingArms:
    """``collection_meta.acting_arms`` describes which arm(s) are actively
    driven during the session — used downstream to filter single-arm vs
    bi-manual data."""

    def test_defaults_to_both_arms_when_omitted(self):
        d = _valid_dict()
        d["collection_meta"].pop("acting_arms", None)
        info = CollectionInfo.from_dict(d)
        info.validate()
        assert info.collection_meta.acting_arms == ["left", "right"]

    @pytest.mark.parametrize(
        "value, expected",
        [
            (["left", "right"], ["left", "right"]),
            (["left"],          ["left"]),
            (["right"],         ["right"]),
            ("left",            ["left"]),   # bare string accepted for convenience
        ],
    )
    def test_valid_shapes_accepted(self, value, expected):
        d = _valid_dict()
        d["collection_meta"]["acting_arms"] = value
        info = CollectionInfo.from_dict(d)
        info.validate()
        assert info.collection_meta.acting_arms == expected

    def test_empty_list_rejected(self):
        d = _valid_dict()
        d["collection_meta"]["acting_arms"] = []
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("acting_arms must be a non-empty list" in v for v in excinfo.value.violations)

    def test_unknown_arm_rejected(self):
        d = _valid_dict()
        d["collection_meta"]["acting_arms"] = ["left", "middle"]
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("entries must be in" in v for v in excinfo.value.violations)

    def test_duplicate_arms_rejected(self):
        d = _valid_dict()
        d["collection_meta"]["acting_arms"] = ["left", "left"]
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("must not contain duplicates" in v for v in excinfo.value.violations)

    def test_to_dict_round_trips(self):
        d = _valid_dict()
        d["collection_meta"]["acting_arms"] = ["right"]
        info = CollectionInfo.from_dict(d)
        info.validate()
        assert info.to_dict()["collection_meta"]["acting_arms"] == ["right"]


class TestEndEffectorPerArm:
    """``hardware_meta.end_effector`` is split per-arm so bimanual rigs can
    declare mismatched grippers (e.g. ``long`` left, ``short`` right). A flat
    legacy shape stays accepted and is fanned out to both arms."""

    def test_mixed_left_right_models_round_trip(self):
        d = _valid_dict()
        d["hardware_meta"]["end_effector"] = {
            "left":  {"type": "gripper", "model": "long"},
            "right": {"type": "gripper", "model": "short"},
        }
        info = CollectionInfo.from_dict(d)
        info.validate()
        out = info.to_dict()["hardware_meta"]["end_effector"]
        assert out == {
            "left":  {"type": "gripper", "model": "long"},
            "right": {"type": "gripper", "model": "short"},
        }

    def test_flat_shape_back_compat_fans_out_to_both_arms(self):
        d = _valid_dict()
        d["hardware_meta"]["end_effector"] = {"type": "gripper", "model": "long"}
        info = CollectionInfo.from_dict(d)
        info.validate()  # both arms inherit the single config
        assert info.hardware_meta.end_effector.left.model == "long"
        assert info.hardware_meta.end_effector.right.model == "long"

    def test_missing_left_arm_rejected(self):
        d = _valid_dict()
        d["hardware_meta"]["end_effector"] = {
            "right": {"type": "gripper", "model": "long"},
        }
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("end_effector.left.type" in v for v in excinfo.value.violations)

    def test_missing_right_arm_rejected(self):
        d = _valid_dict()
        d["hardware_meta"]["end_effector"] = {
            "left": {"type": "gripper", "model": "long"},
        }
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("end_effector.right.type" in v for v in excinfo.value.violations)


class TestCameras:
    def test_three_cameras_round_trip(self):
        info = CollectionInfo.from_dict(_valid_dict())
        info.validate()
        assert set(info.hardware_meta.cameras) == {"head", "left_wrist", "right_wrist"}

    def test_arbitrary_camera_count_and_names_accepted(self):
        d = _valid_dict()
        d["hardware_meta"]["cameras"] = {
            "head": {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
            "table": {"type": "opencv", "index_or_path": 8, "width": 1280, "height": 720, "fps": 60},
            "kettle_close_up": {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30},
            "side": {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
            "ceiling": {"type": "opencv", "index_or_path": 14, "width": 640, "height": 480, "fps": 30},
        }
        info = CollectionInfo.from_dict(d)
        info.validate()
        assert len(info.hardware_meta.cameras) == 5

    def test_record_mode_requires_at_least_one_camera(self):
        d = _valid_dict()
        d["hardware_meta"]["cameras"] = {}
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("cameras" in v and "record" in v for v in excinfo.value.violations)

    def test_pure_infer_mode_allows_no_cameras(self):
        d = _valid_dict()
        d["collection_meta"]["mode"] = "infer"
        d["hardware_meta"]["cameras"] = {}
        CollectionInfo.from_dict(d).validate()  # should not raise

    @pytest.mark.parametrize("missing", ["type", "index_or_path", "width", "height", "fps"])
    def test_camera_missing_required_field_fails(self, missing):
        d = _valid_dict()
        del d["hardware_meta"]["cameras"]["head"][missing]
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any(f"cameras.head.{missing}" in v for v in excinfo.value.violations)

    def test_non_dict_cameras_rejected(self):
        d = _valid_dict()
        d["hardware_meta"]["cameras"] = ["head", "left_wrist"]  # wrong type
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("must be a dict" in v for v in excinfo.value.violations)

    def test_non_dict_camera_spec_rejected(self):
        d = _valid_dict()
        d["hardware_meta"]["cameras"]["head"] = "/dev/video12"
        with pytest.raises(CollectionInfoError) as excinfo:
            CollectionInfo.from_dict(d).validate()
        assert any("cameras.head" in v and "must be a dict" in v for v in excinfo.value.violations)


class TestTimeStamping:
    def test_stamp_start_then_end_iso_format(self):
        info = CollectionInfo.from_dict(_valid_dict())
        info.stamp_start()
        info.stamp_end()
        ct = info.collection_meta.collection_time
        pat = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$")
        assert pat.match(ct.start_time or "")
        assert pat.match(ct.end_time or "")

    def test_to_dict_includes_stamped_times(self):
        info = CollectionInfo.from_dict(_valid_dict())
        info.stamp_start()
        info.stamp_end()
        d = info.to_dict()
        ct = d["collection_meta"]["collection_time"]
        assert ct["start_time"] is not None
        assert ct["end_time"] is not None

    def test_unstamped_times_remain_none(self):
        info = CollectionInfo.from_dict(_valid_dict())
        d = info.to_dict()
        assert d["collection_meta"]["collection_time"] == {"start_time": None, "end_time": None}
