"""Operator-supplied collection metadata loaded from JSON.

A ``CollectionInfo`` describes *who*, *where*, *when*, and *what* about a
recording session — independent of the runtime task definition (``TaskSpec``):

  - ``hardware_meta``:    end-effector type/model
  - ``collection_meta``:  operator, location, mode, and program-stamped times
  - ``task_meta``:        task name, stage info, free-form ``objects`` dict
  - ``robot_type`` / ``robot_id``
  - ``task_description``: optional free text (zh/en)

The recording program loads this from a JSON file path and writes the
fully validated dict (with start/end timestamps filled in) flat onto
``meta.info`` — one key per sub-meta (``collection_meta``,
``hardware_meta``, ``task_meta``, plus ``robot_id`` / ``task_description``).

The CLI ``--mode`` flag is intentionally absent — ``collection_meta.mode``
is the single source of truth for the recording mode.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


VALID_MODES = ("record", "infer", "infer_record", "self_play")
VALID_TASK_STAGE_MODES = ("full", "partial", "recovery")
# Modes that require at least one camera to be configured.
MODES_REQUIRING_CAMERAS = ("record", "infer_record", "self_play")
# Required keys on every entry of hardware_meta.cameras.
REQUIRED_CAMERA_FIELDS = ("type", "index_or_path", "width", "height", "fps")


class CollectionInfoError(ValueError):
    """Raised when an anyverse_collection_info.json file fails validation.

    ``violations`` lists every problem found, not just the first.
    """

    def __init__(self, violations: list[str]):
        self.violations = list(violations)
        super().__init__(self._format())

    def _format(self) -> str:
        bullet = "\n  - "
        return (
            f"anyverse_collection_info.json failed validation "
            f"({len(self.violations)} problem(s)):" + bullet + bullet.join(self.violations)
        )


@dataclass
class EndEffector:
    """End-effector configuration for ONE arm."""
    type: str
    model: str

    @classmethod
    def from_dict(cls, d: dict | None) -> EndEffector:
        d = d or {}
        return cls(type=str(d.get("type", "")), model=str(d.get("model", "")))

    def to_dict(self) -> dict:
        return {"type": self.type, "model": self.model}


@dataclass
class EndEffectorSet:
    """Per-arm end-effector configuration. Two slots — ``left`` and
    ``right`` — each an :class:`EndEffector`. Bimanual rigs frequently mix
    end-effector models (different gripper variants per side); the per-arm
    shape makes that explicit instead of forcing operators to encode it in a
    side comment.

    Back-compat: ``from_dict`` accepts both shapes:
      - new: ``{"left": {"type": ..., "model": ...}, "right": {...}}``
      - old: ``{"type": ..., "model": ...}`` — interpreted as the same
        end-effector on both arms.
    """
    left: EndEffector
    right: EndEffector

    @classmethod
    def from_dict(cls, d: dict | None) -> EndEffectorSet:
        d = d or {}
        if isinstance(d, dict) and ("left" in d or "right" in d):
            return cls(
                left=EndEffector.from_dict(d.get("left")),
                right=EndEffector.from_dict(d.get("right")),
            )
        # Flat shape — same end-effector on both arms.
        single = EndEffector.from_dict(d if isinstance(d, dict) else None)
        return cls(
            left=EndEffector(type=single.type, model=single.model),
            right=EndEffector(type=single.type, model=single.model),
        )

    def to_dict(self) -> dict:
        return {"left": self.left.to_dict(), "right": self.right.to_dict()}


@dataclass
class HardwareMeta:
    end_effector: EndEffectorSet
    # Camera set for the rig. Free-form: each key is the camera name (e.g.
    # "head", "left_wrist", "chest", "top_down"); each value is a dict with
    # at minimum {type, index_or_path, width, height, fps}. Cameras need not
    # be limited to head/left_wrist/right_wrist — different tasks may carry
    # arbitrary names and counts.
    cameras: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> HardwareMeta:
        d = d or {}
        cameras = d.get("cameras", {})
        if cameras is None:
            cameras = {}
        # Pass through anything that isn't a dict so validate() can flag it
        # with a precise error instead of crashing in the constructor.
        if isinstance(cameras, dict):
            cameras = dict(cameras)
        return cls(
            end_effector=EndEffectorSet.from_dict(d.get("end_effector")),
            cameras=cameras,
        )

    def to_dict(self) -> dict:
        return {
            "end_effector": self.end_effector.to_dict(),
            "cameras": self.cameras,
        }


@dataclass
class CollectionTime:
    start_time: str | None = None
    end_time: str | None = None

    @classmethod
    def from_dict(cls, d: dict | None) -> CollectionTime:
        d = d or {}
        return cls(start_time=d.get("start_time"), end_time=d.get("end_time"))

    def to_dict(self) -> dict:
        return {"start_time": self.start_time, "end_time": self.end_time}


VALID_ACTING_ARMS = ("left", "right")


@dataclass
class CollectionMeta:
    operator_name: str
    site_location: str
    city: str
    mode: str  # one of VALID_MODES — replaces both --mode CLI and operation_mode
    is_self_play: bool = False
    is_adversary: bool = False
    # Empty string ("") when not applicable (record / infer / infer_record)
    # or when a single human alternates roles during self-play. Purely
    # informational — only set when a distinct second operator is present.
    adversary_operator: str = ""
    # Which arm(s) the policy / operator is actively driving. Lets downstream
    # training pipelines filter / weight single-arm vs bi-manual data.
    # Order is preserved on disk but not semantically meaningful. Defaults to
    # both arms so existing configs keep working.
    acting_arms: list[str] = field(default_factory=lambda: ["left", "right"])
    collection_time: CollectionTime = field(default_factory=CollectionTime)

    @classmethod
    def from_dict(cls, d: dict | None) -> CollectionMeta:
        d = d or {}
        raw_arms = d.get("acting_arms")
        if raw_arms is None:
            acting_arms = ["left", "right"]
        elif isinstance(raw_arms, str):
            # Accept a bare string for convenience ("left" → ["left"]).
            acting_arms = [raw_arms]
        else:
            try:
                acting_arms = list(raw_arms)
            except TypeError:
                # Pass through unchanged so validate() can flag the wrong type
                # with a precise message rather than crashing in the ctor.
                acting_arms = raw_arms  # type: ignore[assignment]
        return cls(
            operator_name=str(d.get("operator_name", "")),
            site_location=str(d.get("site_location", "")),
            city=str(d.get("city", "")),
            mode=str(d.get("mode", "")),
            is_self_play=bool(d.get("is_self_play", False)),
            is_adversary=bool(d.get("is_adversary", False)),
            # Accept legacy ``null`` and coerce to "" so on-disk reads of
            # older configs keep working.
            adversary_operator=str(d.get("adversary_operator") or ""),
            acting_arms=acting_arms,
            collection_time=CollectionTime.from_dict(d.get("collection_time")),
        )

    def to_dict(self) -> dict:
        return {
            "operator_name": self.operator_name,
            "adversary_operator": self.adversary_operator,
            "is_adversary": self.is_adversary,
            "is_self_play": self.is_self_play,
            "site_location": self.site_location,
            "city": self.city,
            "mode": self.mode,
            "acting_arms": list(self.acting_arms),
            "collection_time": self.collection_time.to_dict(),
        }


@dataclass
class TaskStage:
    mode: str  # one of VALID_TASK_STAGE_MODES
    stages: str = ""  # required non-empty when mode != "full"

    @classmethod
    def from_dict(cls, d: dict | None) -> TaskStage:
        d = d or {}
        return cls(mode=str(d.get("mode", "")), stages=str(d.get("stages", "")))

    def to_dict(self) -> dict:
        return {"mode": self.mode, "stages": self.stages}


@dataclass
class TaskMeta:
    task_name: str
    task_stage: TaskStage
    objects: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict | None) -> TaskMeta:
        d = d or {}
        objects = d.get("objects", {})
        if objects is None:
            objects = {}
        return cls(
            task_name=str(d.get("task_name", "")),
            task_stage=TaskStage.from_dict(d.get("task_stage")),
            objects=dict(objects),
        )

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "task_stage": self.task_stage.to_dict(),
            "objects": self.objects,
        }


@dataclass
class CollectionInfo:
    hardware_meta: HardwareMeta
    collection_meta: CollectionMeta
    task_meta: TaskMeta
    robot_type: str
    robot_id: str
    task_description: str = ""

    # ---- Loading & validation ----

    @classmethod
    def from_json(cls, path: str | Path) -> CollectionInfo:
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: dict) -> CollectionInfo:
        return cls(
            hardware_meta=HardwareMeta.from_dict(d.get("hardware_meta")),
            collection_meta=CollectionMeta.from_dict(d.get("collection_meta")),
            task_meta=TaskMeta.from_dict(d.get("task_meta")),
            robot_type=str(d.get("robot_type", "")),
            robot_id=str(d.get("robot_id", "")),
            task_description=str(d.get("task_description", "")),
        )

    def validate(self) -> None:
        """Raise CollectionInfoError listing every problem found.

        Required scalar fields must be non-empty after strip(). Conditional
        rules (e.g. task_stage.stages required when mode != full) are also
        checked. Collected times are program-managed and not validated here.
        """
        violations: list[str] = []

        def _require_nonempty(value: Any, path: str) -> None:
            if value is None or (isinstance(value, str) and not value.strip()):
                violations.append(f"{path} is required and must be non-empty")

        # hardware_meta
        for _side in ("left", "right"):
            _ee = getattr(self.hardware_meta.end_effector, _side)
            _require_nonempty(_ee.type, f"hardware_meta.end_effector.{_side}.type")
            _require_nonempty(_ee.model, f"hardware_meta.end_effector.{_side}.model")

        cams = self.hardware_meta.cameras
        if not isinstance(cams, dict):
            violations.append("hardware_meta.cameras must be a dict (name → camera config)")
        else:
            mode = self.collection_meta.mode
            if mode in MODES_REQUIRING_CAMERAS and not cams:
                violations.append(
                    f"hardware_meta.cameras must declare at least one camera "
                    f"when collection_meta.mode={mode!r}"
                )
            for name, spec in cams.items():
                # Keys starting with "_" are reserved for inline annotation
                # (e.g. "_comment_head") and ignored by both the validator
                # and the recorder.
                if isinstance(name, str) and name.startswith("_"):
                    continue
                if not isinstance(name, str) or not name.strip():
                    violations.append(
                        f"hardware_meta.cameras: camera names must be non-empty strings (got {name!r})"
                    )
                    continue
                if not isinstance(spec, dict):
                    violations.append(
                        f"hardware_meta.cameras.{name} must be a dict, got {type(spec).__name__}"
                    )
                    continue
                for required in REQUIRED_CAMERA_FIELDS:
                    if required not in spec or spec[required] in (None, ""):
                        violations.append(
                            f"hardware_meta.cameras.{name}.{required} is required and must be non-empty"
                        )

        # collection_meta
        _require_nonempty(self.collection_meta.operator_name, "collection_meta.operator_name")
        _require_nonempty(self.collection_meta.site_location, "collection_meta.site_location")
        _require_nonempty(self.collection_meta.city, "collection_meta.city")
        _require_nonempty(self.collection_meta.mode, "collection_meta.mode")
        if self.collection_meta.mode and self.collection_meta.mode not in VALID_MODES:
            violations.append(
                f"collection_meta.mode={self.collection_meta.mode!r} is not one of {list(VALID_MODES)}"
            )
        # adversary_operator is informational only: name of a second human
        # who alternates roles. Empty string is valid even when
        # is_adversary=true (one person wearing both hats during self-play).
        arms = self.collection_meta.acting_arms
        if not isinstance(arms, list) or len(arms) == 0:
            violations.append(
                "collection_meta.acting_arms must be a non-empty list, "
                f"e.g. ['left', 'right'] or ['left'] (got {arms!r})"
            )
        else:
            unknown = [a for a in arms if a not in VALID_ACTING_ARMS]
            if unknown:
                violations.append(
                    "collection_meta.acting_arms entries must be in "
                    f"{list(VALID_ACTING_ARMS)} (got {unknown!r})"
                )
            if len(set(arms)) != len(arms):
                violations.append(
                    f"collection_meta.acting_arms must not contain duplicates (got {arms!r})"
                )

        # task_meta
        _require_nonempty(self.task_meta.task_name, "task_meta.task_name")
        _require_nonempty(self.task_meta.task_stage.mode, "task_meta.task_stage.mode")
        if (
            self.task_meta.task_stage.mode
            and self.task_meta.task_stage.mode not in VALID_TASK_STAGE_MODES
        ):
            violations.append(
                f"task_meta.task_stage.mode={self.task_meta.task_stage.mode!r} is not one of "
                f"{list(VALID_TASK_STAGE_MODES)}"
            )
        if (
            self.task_meta.task_stage.mode
            and self.task_meta.task_stage.mode != "full"
            and not self.task_meta.task_stage.stages.strip()
        ):
            violations.append(
                "task_meta.task_stage.stages is required when task_stage.mode != 'full'"
            )
        if not isinstance(self.task_meta.objects, dict):
            violations.append("task_meta.objects must be a dict (may be empty)")

        # robot
        _require_nonempty(self.robot_type, "robot_type")
        _require_nonempty(self.robot_id, "robot_id")

        # task_description is optional; no validation needed.

        if violations:
            raise CollectionInfoError(violations)

    def to_dict(self) -> dict:
        return {
            "hardware_meta": self.hardware_meta.to_dict(),
            "collection_meta": self.collection_meta.to_dict(),
            "task_meta": self.task_meta.to_dict(),
            "robot_type": self.robot_type,
            "robot_id": self.robot_id,
            "task_description": self.task_description,
        }

    # ---- Time stamping (program-managed) ----

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def stamp_start(self) -> None:
        self.collection_meta.collection_time.start_time = self._now_iso()

    def stamp_end(self) -> None:
        self.collection_meta.collection_time.end_time = self._now_iso()
