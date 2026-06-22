"""Tests for recording.utils.preflight — the operator confirmation pipeline."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lerobot.recording.utils.preflight import (
    _HL_CLOSE,
    _HL_OPEN,
    _capture_camera_frame,
    _format_collection_info,
    cameras_check_with_preview,
    collection_info_check,
    main,
)
from lerobot.recording.task.collection_info import CollectionInfo


def _plain(body: str) -> str:
    """Strip highlight sentinels so substring assertions test rendered content."""
    return body.replace(_HL_OPEN, "").replace(_HL_CLOSE, "")


def _valid_dict() -> dict:
    return {
        "hardware_meta": {
            "end_effector": {
                "left":  {"type": "gripper", "model": "long"},
                "right": {"type": "gripper", "model": "long"},
            },
            "cameras": {
                "head": {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
            },
        },
        "collection_meta": {
            "operator_name": "alice",
            "is_adversary": False,
            "is_self_play": False,
            "site_location": "shanghai_lab",
            "city": "shanghai",
            "mode": "record",
        },
        "task_meta": {
            "task_name": "fold_cloth",
            "task_stage": {"mode": "full", "stages": ""},
            "objects": {"cloth": {"color": "blue"}},
        },
        "robot_type": "arxx5_bimanual",
        "robot_id": "7",
        "task_description": "fold cloth in two halves",
    }


def _write_info(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def auto_confirm():
    """Make confirm() always return True for these tests; the UI itself isn't
    under test, only the load + validate + format pipeline."""
    with patch("lerobot.recording.utils.preflight.confirm", return_value=True):
        yield


class TestFormat:
    def test_includes_task_name_and_objects(self):
        info = CollectionInfo.from_dict(_valid_dict())
        body = _format_collection_info(info)
        assert "fold_cloth" in body
        assert "cloth" in body
        assert "blue" in body
        assert "shanghai" in body

    def test_omits_objects_section_when_empty(self):
        d = _valid_dict()
        d["task_meta"]["objects"] = {}
        info = CollectionInfo.from_dict(d)
        body = _format_collection_info(info)
        assert "(none)" in body

    def test_omits_description_when_blank(self):
        d = _valid_dict()
        d["task_description"] = ""
        info = CollectionInfo.from_dict(d)
        body = _format_collection_info(info)
        assert "Task description" not in body

    def test_includes_task_spec_when_passed(self):
        from lerobot.recording.task.task_spec import TaskSpec, RoleSpec
        info = CollectionInfo.from_dict(_valid_dict())
        spec = TaskSpec(
            task_id="fold_cloth",
            roles={
                "operator": RoleSpec(
                    prompt="Fold the {cloth.color} cloth.",
                    max_time_s=60,
                ),
            },
        )
        # Apply the same template substitution the popup pipeline runs.
        from lerobot.recording.utils.preflight import _build_template_context
        spec = spec.apply_template(_build_template_context(info))
        body = _plain(_format_collection_info(info, task_spec=spec))
        assert "Task spec id" in body
        assert "fold_cloth" in body
        assert "Prompts" in body
        # {cloth.color} → "blue" from objects.cloth.color
        assert "[operator] Fold the blue cloth." in body

    def test_omits_task_spec_when_not_passed(self):
        info = CollectionInfo.from_dict(_valid_dict())
        body = _format_collection_info(info)  # no task_spec arg
        assert "Task spec id" not in body
        assert "Prompts" not in body

    def test_end_effector_collapses_when_left_right_match(self):
        info = CollectionInfo.from_dict(_valid_dict())  # both arms gripper/long
        body = _plain(_format_collection_info(info))
        assert "gripper / long" in body
        assert "left=" not in body and "right=" not in body

    def test_end_effector_shows_both_when_mismatched(self):
        d = _valid_dict()
        d["hardware_meta"]["end_effector"] = {
            "left":  {"type": "gripper", "model": "long"},
            "right": {"type": "gripper", "model": "short"},
        }
        info = CollectionInfo.from_dict(d)
        body = _plain(_format_collection_info(info))
        assert "left=gripper/long" in body
        assert "right=gripper/short" in body


class TestCollectionInfoCheck:
    def test_valid_returns_zero(self, tmp_path):
        p = _write_info(tmp_path / "info.json", _valid_dict())
        assert collection_info_check(str(p)) == 0

    def test_missing_file_returns_one(self, tmp_path):
        assert collection_info_check(str(tmp_path / "nope.json")) == 1

    def test_invalid_returns_one_with_violations_to_stderr(self, tmp_path, capsys):
        d = _valid_dict()
        d["collection_meta"]["operator_name"] = ""
        d["robot_id"] = ""
        p = _write_info(tmp_path / "bad.json", d)
        rc = collection_info_check(str(p))
        assert rc == 1
        err = capsys.readouterr().err
        # All violations must be reported, not just the first.
        assert "operator_name" in err
        assert "robot_id" in err

    def test_rejected_by_operator_returns_one(self, tmp_path):
        p = _write_info(tmp_path / "info.json", _valid_dict())
        with patch("lerobot.recording.utils.preflight.confirm", return_value=False):
            assert collection_info_check(str(p)) == 1


class TestCli:
    def test_no_args_returns_two(self, capsys):
        assert main([]) == 2
        assert "Usage" in capsys.readouterr().err

    def test_unknown_command_returns_two(self, capsys):
        assert main(["whatever"]) == 2

    def test_collection_info_without_path_returns_two(self, capsys):
        assert main(["collection_info"]) == 2

    def test_collection_info_with_valid_path_returns_zero(self, tmp_path):
        p = _write_info(tmp_path / "info.json", _valid_dict())
        assert main(["collection_info", str(p)]) == 0

    def test_collection_info_with_task_spec_path_loads_both(self, tmp_path):
        ci_path = _write_info(tmp_path / "info.json", _valid_dict())
        spec_path = tmp_path / "spec.json"
        spec_path.write_text(json.dumps({
            "task_id": "fold_cloth",
            "roles": {"operator": {"prompt": "Fold the {cloth.color} cloth."}},
        }), encoding="utf-8")
        # Patch confirm to capture the body that gets shown.
        captured = {}
        def _capture_confirm(title, body):
            captured["body"] = body
            return True
        with patch("lerobot.recording.utils.preflight.confirm", side_effect=_capture_confirm):
            assert main(["collection_info", str(ci_path), str(spec_path)]) == 0
        # Resolved prompt + task_id surfaced for the operator.
        body = _plain(captured["body"])
        assert "Task spec id" in body
        assert "fold_cloth" in body
        assert "[operator] Fold the blue cloth." in body

    def test_collection_info_with_invalid_task_spec_continues(self, tmp_path, capsys):
        # Bad task_spec path → preflight should warn and still confirm the
        # collection_info (rather than failing the whole preflight).
        ci_path = _write_info(tmp_path / "info.json", _valid_dict())
        rc = main(["collection_info", str(ci_path), "/does/not/exist.json"])
        assert rc == 0
        err = capsys.readouterr().err
        assert "warning" in err.lower() and "task_spec" in err.lower()

    def test_session_subcommand_loads_session_config(self, tmp_path):
        # Session config bundles the same metadata + an inline task_spec;
        # the popup body should include the session id and template-resolved
        # prompts pulled from collection_meta + task_meta.objects.
        from lerobot.recording.task.session_config import SCHEMA_VERSION
        session_dict = {
            "schema_version": SCHEMA_VERSION,
            "session_id": "fold_cloth.test.v1",
            "robot": _valid_dict()["robot_type"] and {"type": "arxx5_bimanual", "id": "7"},
            "hardware_meta": _valid_dict()["hardware_meta"],
            "collection_meta": _valid_dict()["collection_meta"],
            "task_meta": _valid_dict()["task_meta"],
            "task_description": "fold the cloth",
            "task_spec": {
                "task_id": "fold_cloth",
                "roles": {"operator": {"prompt": "Fold the {cloth.color} cloth."}},
            },
        }
        sess_path = tmp_path / "session.json"
        sess_path.write_text(json.dumps(session_dict), encoding="utf-8")

        captured = {}
        def _capture(title, body):
            captured["title"] = title
            captured["body"] = body
            return True
        with patch("lerobot.recording.utils.preflight.confirm", side_effect=_capture):
            assert main(["session", str(sess_path)]) == 0
        assert captured["title"] == "Confirm session config"
        body = _plain(captured["body"])
        assert "fold_cloth.test.v1" in body
        assert "[operator] Fold the blue cloth." in body


class TestCameraCapture:
    """``_capture_camera_frame`` opens cv2.VideoCapture and reads one frame."""

    def _mock_cv2_module(self, opened: bool = True, frame_shape=(480, 640, 3)):
        cv2 = MagicMock()
        cv2.CAP_PROP_FRAME_WIDTH = 3
        cv2.CAP_PROP_FRAME_HEIGHT = 4
        cap = MagicMock()
        cap.isOpened.return_value = opened
        # Return a deterministic small BGR frame.
        cap.read.return_value = (True, np.full(frame_shape, 128, dtype=np.uint8))
        cv2.VideoCapture.return_value = cap
        return cv2, cap

    def test_capture_succeeds(self):
        cv2, cap = self._mock_cv2_module(opened=True)
        with patch.dict("sys.modules", {"cv2": cv2}):
            frame, err = _capture_camera_frame(
                {"index_or_path": 4, "width": 640, "height": 480, "fps": 30}
            )
        assert err is None
        assert frame is not None
        assert frame.shape == (480, 640, 3)
        cap.release.assert_called_once()

    def test_capture_returns_error_when_not_opened(self):
        cv2, cap = self._mock_cv2_module(opened=False)
        with patch.dict("sys.modules", {"cv2": cv2}):
            frame, err = _capture_camera_frame({"index_or_path": 99})
        assert frame is None
        assert err is not None and "could not open" in err

    def test_capture_returns_error_on_missing_index(self):
        frame, err = _capture_camera_frame({"width": 640})
        assert frame is None
        assert "index_or_path" in (err or "")


class TestCamerasCheckWithPreview:
    """End-to-end: load JSON, capture frames per camera, present grid."""

    def _patch_cv2(self):
        cv2 = MagicMock()
        cv2.CAP_PROP_FRAME_WIDTH = 3
        cv2.CAP_PROP_FRAME_HEIGHT = 4
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, np.full((480, 640, 3), 64, dtype=np.uint8))
        cv2.VideoCapture.return_value = cap
        return cv2

    def test_missing_file_returns_one(self, tmp_path):
        assert cameras_check_with_preview(str(tmp_path / "nope.json")) == 1

    def test_empty_cameras_returns_one(self, tmp_path):
        d = _valid_dict()
        d["hardware_meta"]["cameras"] = {}
        # cameras must be empty AND mode must allow that — use 'infer'.
        d["collection_meta"]["mode"] = "infer"
        p = _write_info(tmp_path / "info.json", d)
        assert cameras_check_with_preview(str(p)) == 1

    def test_captures_each_declared_camera_then_confirms(self, tmp_path):
        d = _valid_dict()
        d["hardware_meta"]["cameras"] = {
            "head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
            "left_wrist": {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30},
            "right_wrist": {"type": "opencv", "index_or_path": 16, "width": 640, "height": 480, "fps": 30},
        }
        p = _write_info(tmp_path / "info.json", d)
        cv2 = self._patch_cv2()
        # Force the headless terminal fallback path so we don't try to open Tk.
        with (
            patch.dict("sys.modules", {"cv2": cv2}),
            patch("lerobot.recording.utils.preflight._has_display", return_value=False),
        ):
            rc = cameras_check_with_preview(str(p))
        assert rc == 0
        # cv2.VideoCapture must be called once per declared camera.
        indices_seen = sorted(
            call.args[0] for call in cv2.VideoCapture.call_args_list
        )
        assert indices_seen == [4, 10, 16]

    def test_camera_open_failure_renders_in_grid(self, tmp_path):
        d = _valid_dict()
        d["hardware_meta"]["cameras"] = {
            "head": {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
        }
        p = _write_info(tmp_path / "info.json", d)
        cv2 = MagicMock()
        cap = MagicMock()
        cap.isOpened.return_value = False
        cv2.VideoCapture.return_value = cap
        # Failure still returns 0 because confirm() is auto-True; the operator
        # would normally see "FAILED" in the grid and hit Cancel. We test that
        # the pipeline reaches the confirmation step instead of crashing.
        with (
            patch.dict("sys.modules", {"cv2": cv2}),
            patch("lerobot.recording.utils.preflight._has_display", return_value=False),
        ):
            rc = cameras_check_with_preview(str(p))
        assert rc == 0


class TestCliCameraPreview:
    def test_cameras_with_path_dispatches_to_preview(self, tmp_path):
        p = _write_info(tmp_path / "info.json", _valid_dict())
        with patch(
            "lerobot.recording.utils.preflight.cameras_check_with_preview",
            return_value=0,
        ) as mock_preview:
            assert main(["cameras", str(p)]) == 0
        mock_preview.assert_called_once_with(str(p))

    def test_cameras_without_path_dispatches_to_legacy(self):
        with patch(
            "lerobot.recording.utils.preflight.cameras_check_legacy",
            return_value=0,
        ) as mock_legacy:
            assert main(["cameras"]) == 0
        mock_legacy.assert_called_once()
