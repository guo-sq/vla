#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import platform
import time
from collections import deque
from pathlib import Path
from typing import TypeAlias

from .camera import Camera
from .configs import CameraConfig, Cv2Rotation

IndexOrPath: TypeAlias = int | Path

logger = logging.getLogger(__name__)


class FrameRateMonitor:
    """Tracks camera frame arrivals and warns on sustained fps drops or stale reads.

    The monitor is single-producer (the camera read thread calls
    :meth:`record_arrival`); the consumer thread calls :meth:`check_stale` on
    the cached frame timestamp it just read. Warnings are throttled so a
    persistent fps drop does not flood the log.
    """

    def __init__(
        self,
        name: str,
        configured_fps: float | None,
        window: int = 30,
        drop_ratio: float = 0.8,
        warning_throttle_s: float = 2.0,
    ):
        self._name = name
        self._fps = float(configured_fps) if configured_fps else None
        self._window = window
        self._drop_ratio = drop_ratio
        self._warning_throttle_s = warning_throttle_s
        self._intervals: deque[float] = deque(maxlen=window)
        self._last_arrival: float | None = None
        self._last_drop_warning: float = 0.0
        self._last_stale_warning: float = 0.0

    def record_arrival(self, now: float | None = None) -> None:
        if now is None:
            now = time.perf_counter()
        if self._last_arrival is not None:
            self._intervals.append(now - self._last_arrival)
        self._last_arrival = now
        self._maybe_warn_drop(now)

    def _maybe_warn_drop(self, now: float) -> None:
        if self._fps is None or len(self._intervals) < self._window:
            return
        sorted_iv = sorted(self._intervals)
        median = sorted_iv[len(sorted_iv) // 2]
        if median <= 0:
            return
        observed_fps = 1.0 / median
        if observed_fps < self._drop_ratio * self._fps:
            if now - self._last_drop_warning >= self._warning_throttle_s:
                logger.warning(
                    f"{self._name}: camera fps dropped to {observed_fps:.1f} "
                    f"(configured {self._fps:.1f}, median over last {self._window} frames). "
                    f"VLA observations will be stale; check USB bandwidth/exposure/thermal throttle."
                )
                self._last_drop_warning = now

    def check_stale(self, frame_ts: float | None, now: float | None = None) -> bool:
        """Return True if the cached frame is older than 2/fps. Throttled-warns."""
        if self._fps is None or frame_ts is None:
            return False
        if now is None:
            now = time.perf_counter()
        threshold = 2.0 / self._fps
        age = now - frame_ts
        if age > threshold:
            if now - self._last_stale_warning >= self._warning_throttle_s:
                logger.warning(
                    f"{self._name}: returning stale frame "
                    f"(age={age * 1e3:.0f}ms, threshold={threshold * 1e3:.0f}ms). "
                    f"Policy is acting on outdated visual evidence."
                )
                self._last_stale_warning = now
            return True
        return False


def make_cameras_from_configs(camera_configs: dict[str, CameraConfig]) -> dict[str, Camera]:
    cameras = {}

    for key, cfg in camera_configs.items():
        if cfg.type == "opencv":
            from .opencv import OpenCVCamera

            cameras[key] = OpenCVCamera(cfg)

        elif cfg.type == "intelrealsense":
            from .realsense.camera_realsense import RealSenseCamera

            cameras[key] = RealSenseCamera(cfg)
        else:
            raise ValueError(f"The motor type '{cfg.type}' is not valid.")

    return cameras


def get_cv2_rotation(rotation: Cv2Rotation) -> int | None:
    import cv2

    if rotation == Cv2Rotation.ROTATE_90:
        return cv2.ROTATE_90_CLOCKWISE
    elif rotation == Cv2Rotation.ROTATE_180:
        return cv2.ROTATE_180
    elif rotation == Cv2Rotation.ROTATE_270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE
    else:
        return None


def get_cv2_backend() -> int:
    import cv2

    if platform.system() == "Windows":
        return cv2.CAP_MSMF  # Use MSMF for Windows instead of AVFOUNDATION
    # elif platform.system() == "Darwin":  # macOS
    #     return cv2.CAP_AVFOUNDATION
    else:  # Linux and others
        return cv2.CAP_ANY
