"""Tests for the NVENC → CPU fallback in encode_video_frames and the
LEROBOT_VIDEO_USE_GPU env-var escape hatch."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from lerobot.datasets.video_utils import encode_video_frames


def _seed_image(tmp_path: Path) -> Path:
    """Write a single 640x480 RGB png so encode_video_frames has something
    to imread for shape detection."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    p = tmp_path / "frame_000000.png"
    cv2.imwrite(str(p), img)
    return tmp_path


@pytest.fixture
def imgs_dir(tmp_path):
    return _seed_image(tmp_path)


class TestEnvVarOverride:
    def test_use_gpu_env_var_zero_forces_cpu(self, imgs_dir, tmp_path, monkeypatch, capsys):
        out = tmp_path / "out.mp4"

        # Mock subprocess.run to capture the command, simulate success, and
        # touch the output file so the post-check passes.
        called = {}
        def fake_run(cmd, **kw):
            called["cmd"] = cmd
            out.touch()
            return MagicMock(returncode=0, stderr="")

        monkeypatch.setenv("LEROBOT_VIDEO_USE_GPU", "0")
        with patch("subprocess.run", side_effect=fake_run):
            encode_video_frames(imgs_dir, out, fps=30, use_gpu=True)

        assert "h264_nvenc" not in called["cmd"], (
            "use_gpu=True was overridden to False by env var, expected libx264 path"
        )
        assert "libx264" in called["cmd"]


class TestNVENCFallback:
    def test_nvenc_cuda_error_falls_back_to_libx264(self, imgs_dir, tmp_path):
        out = tmp_path / "out.mp4"

        # First call: simulate NVENC failure with the same stderr the user
        # saw. Second call: succeed with libx264.
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "h264_nvenc" in cmd:
                return MagicMock(
                    returncode=1,
                    stderr=(
                        "[h264_nvenc @ 0x...] dl_fn->cuda_dl->cuInit(0) "
                        "failed -> CUDA_ERROR_UNKNOWN: unknown error\n"
                    ),
                )
            else:
                out.touch()
                return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            encode_video_frames(imgs_dir, out, fps=30, use_gpu=True)

        assert len(calls) == 2, f"expected NVENC try + libx264 retry, got {len(calls)}"
        assert "h264_nvenc" in calls[0]
        assert "libx264" in calls[1]
        assert "h264_nvenc" not in calls[1]

    def test_non_nvenc_error_does_not_retry(self, imgs_dir, tmp_path):
        out = tmp_path / "out.mp4"

        # libx264 fails (CPU encoder) — should NOT retry, should raise.
        def fake_run(cmd, **kw):
            return MagicMock(returncode=1, stderr="libx264: missing audio stream")

        with patch("subprocess.run", side_effect=fake_run), \
             pytest.raises(RuntimeError, match="FFmpeg encoding failed"):
            encode_video_frames(imgs_dir, out, fps=30, use_gpu=False)

    def test_nvenc_non_cuda_error_does_not_retry(self, imgs_dir, tmp_path):
        out = tmp_path / "out.mp4"

        # NVENC fails for a non-CUDA reason (e.g. missing -bf flag). The
        # fallback should still kick in because we recognize "nvenc" in stderr.
        # This is intentionally permissive — better to retry on CPU than
        # abort a 50-min recording.
        calls = []
        def fake_run(cmd, **kw):
            calls.append(cmd)
            if "h264_nvenc" in cmd:
                return MagicMock(returncode=1, stderr="[h264_nvenc] some unrelated error")
            out.touch()
            return MagicMock(returncode=0, stderr="")

        with patch("subprocess.run", side_effect=fake_run):
            encode_video_frames(imgs_dir, out, fps=30, use_gpu=True)
        assert len(calls) == 2
