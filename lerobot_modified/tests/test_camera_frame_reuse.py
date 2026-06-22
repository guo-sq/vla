"""Tests for camera async_read with wait_for_new parameter (frame reuse)."""

import threading
import time
from unittest.mock import patch, PropertyMock

import numpy as np
import pytest

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera, OpenCVCameraConfig


def _make_camera():
    """Create an OpenCVCamera with fake internals (no real hardware)."""
    config = OpenCVCameraConfig(index_or_path=0, fps=30, width=640, height=480)
    cam = OpenCVCamera(config)
    # Bypass real hardware: set up frame buffer as if background thread is running
    cam.latest_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cam.new_frame_event.set()
    cam.thread = threading.Thread()  # non-None so _start_read_thread isn't called
    return cam


class TestAsyncReadFrameReuse:
    @patch.object(OpenCVCamera, "is_connected", new_callable=PropertyMock, return_value=True)
    def test_async_read_no_wait_returns_cached_frame(self, _mock_connected):
        """wait_for_new=False with a cached frame should return immediately without blocking."""
        cam = _make_camera()

        # Clear the event to simulate "no NEW frame since last read"
        cam.new_frame_event.clear()

        start = time.perf_counter()
        frame = cam.async_read(timeout_ms=200, wait_for_new=False)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert frame is not None
        assert frame.shape == (480, 640, 3)
        assert elapsed_ms < 50, f"Expected immediate return, took {elapsed_ms:.1f}ms"

    @patch.object(OpenCVCamera, "is_connected", new_callable=PropertyMock, return_value=True)
    def test_async_read_no_wait_first_frame_fallback(self, _mock_connected):
        """wait_for_new=False with no cached frame should wait for the first frame."""
        cam = _make_camera()
        cam.latest_frame = None
        cam.new_frame_event.clear()

        # Simulate background thread delivering the first frame after 50ms
        def deliver_frame():
            time.sleep(0.05)
            with cam.frame_lock:
                cam.latest_frame = np.ones((480, 640, 3), dtype=np.uint8)
            cam.new_frame_event.set()

        t = threading.Thread(target=deliver_frame)
        t.start()

        frame = cam.async_read(timeout_ms=500, wait_for_new=False)
        t.join()

        assert frame is not None
        assert np.all(frame == 1)

    @patch.object(OpenCVCamera, "is_connected", new_callable=PropertyMock, return_value=True)
    def test_async_read_default_waits_for_new(self, _mock_connected):
        """Default wait_for_new=True should block until new frame (original behavior)."""
        cam = _make_camera()

        # First read consumes the initial frame and clears the event
        frame1 = cam.async_read(timeout_ms=200)
        assert frame1 is not None

        # Event is now cleared. Second read with default should timeout.
        with pytest.raises(TimeoutError):
            cam.async_read(timeout_ms=50)
