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
import glob
import importlib
import logging
import os
import shutil
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import av
import pyarrow as pa
import torch
import torchvision
from datasets.features.features import register_feature
from PIL import Image
import cv2
import os
import subprocess




def get_safe_default_codec():
    if importlib.util.find_spec("torchcodec"):
        return "torchcodec"
    else:
        logging.warning(
            "'torchcodec' is not available in your platform, falling back to 'pyav' as a default decoder"
        )
        return "pyav"


def decode_video_frames(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str | None = None,
) -> torch.Tensor:
    """
    Decodes video frames using the specified backend.

    Args:
        video_path (Path): Path to the video file.
        timestamps (list[float]): List of timestamps to extract frames.
        tolerance_s (float): Allowed deviation in seconds for frame retrieval.
        backend (str, optional): Backend to use for decoding. Defaults to "torchcodec" when available in the platform; otherwise, defaults to "pyav"..

    Returns:
        torch.Tensor: Decoded frames.

    Currently supports torchcodec on cpu and pyav.
    """
    if backend is None:
        backend = get_safe_default_codec()
    if backend == "torchcodec":
        return decode_video_frames_torchcodec(video_path, timestamps, tolerance_s)
    elif backend in ["pyav", "video_reader"]:
        return decode_video_frames_torchvision(video_path, timestamps, tolerance_s, backend)
    else:
        raise ValueError(f"Unsupported video backend: {backend}")


def decode_video_frames_torchvision(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    backend: str = "pyav",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated to the requested timestamps of a video

    The backend can be either "pyav" (default) or "video_reader".
    "video_reader" requires installing torchvision from source, see:
    https://github.com/pytorch/vision/blob/main/torchvision/csrc/io/decoder/gpu/README.rst
    (note that you need to compile against ffmpeg<4.3)

    While both use cpu, "video_reader" is supposedly faster than "pyav" but requires additional setup.
    For more info on video decoding, see `benchmark/video/README.md`

    See torchvision doc for more info on these two backends:
    https://pytorch.org/vision/0.18/index.html?highlight=backend#torchvision.set_video_backend

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """
    video_path = str(video_path)

    # set backend
    keyframes_only = False
    torchvision.set_video_backend(backend)
    if backend == "pyav":
        keyframes_only = True  # pyav doesn't support accurate seek

    # set a video stream reader
    # TODO(rcadene): also load audio stream at the same time
    reader = torchvision.io.VideoReader(video_path, "video")

    # set the first and last requested timestamps
    # Note: previous timestamps are usually loaded, since we need to access the previous key frame
    first_ts = min(timestamps)
    last_ts = max(timestamps)

    # access closest key frame of the first requested frame
    # Note: closest key frame timestamp is usually smaller than `first_ts` (e.g. key frame can be the first frame of the video)
    # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
    reader.seek(first_ts, keyframes_only=keyframes_only)

    # load all frames until last requested frame
    loaded_frames = []
    loaded_ts = []
    for frame in reader:
        current_ts = frame["pts"]
        if log_loaded_timestamps:
            logging.info(f"frame loaded at timestamp={current_ts:.4f}")
        loaded_frames.append(frame["data"])
        loaded_ts.append(current_ts)
        if current_ts >= last_ts:
            break

    if backend == "pyav":
        reader.container.close()

    reader = None

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and timestamps of all loaded frames
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
        f"\nbackend: {backend}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to the pytorch format which is float32 in [0,1] range (and channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames


def decode_video_frames_torchcodec(
    video_path: Path | str,
    timestamps: list[float],
    tolerance_s: float,
    device: str = "cpu",
    log_loaded_timestamps: bool = False,
) -> torch.Tensor:
    """Loads frames associated with the requested timestamps of a video using torchcodec.

    Note: Setting device="cuda" outside the main process, e.g. in data loader workers, will lead to CUDA initialization errors.

    Note: Video benefits from inter-frame compression. Instead of storing every frame individually,
    the encoder stores a reference frame (or a key frame) and subsequent frames as differences relative to
    that key frame. As a consequence, to access a requested frame, we need to load the preceding key frame,
    and all subsequent frames until reaching the requested frame. The number of key frames in a video
    can be adjusted during encoding to take into account decoding time and video size in bytes.
    """

    if importlib.util.find_spec("torchcodec"):
        from torchcodec.decoders import VideoDecoder
    else:
        raise ImportError("torchcodec is required but not available.")

    # initialize video decoder
    decoder = VideoDecoder(video_path, device=device, seek_mode="approximate")
    loaded_frames = []
    loaded_ts = []
    # get metadata for frame information
    metadata = decoder.metadata
    average_fps = metadata.average_fps

    # convert timestamps to frame indices
    frame_indices = [round(ts * average_fps) for ts in timestamps]

    # retrieve frames based on indices
    frames_batch = decoder.get_frames_at(indices=frame_indices)

    for frame, pts in zip(frames_batch.data, frames_batch.pts_seconds, strict=False):
        loaded_frames.append(frame)
        loaded_ts.append(pts.item())
        if log_loaded_timestamps:
            logging.info(f"Frame loaded at timestamp={pts:.4f}")

    query_ts = torch.tensor(timestamps)
    loaded_ts = torch.tensor(loaded_ts)

    # compute distances between each query timestamp and loaded timestamps
    dist = torch.cdist(query_ts[:, None], loaded_ts[:, None], p=1)
    min_, argmin_ = dist.min(1)

    is_within_tol = min_ < tolerance_s
    assert is_within_tol.all(), (
        f"One or several query timestamps unexpectedly violate the tolerance ({min_[~is_within_tol]} > {tolerance_s=})."
        "It means that the closest frame that can be loaded from the video is too far away in time."
        "This might be due to synchronization issues with timestamps during data collection."
        "To be safe, we advise to ignore this item during training."
        f"\nqueried timestamps: {query_ts}"
        f"\nloaded timestamps: {loaded_ts}"
        f"\nvideo: {video_path}"
    )

    # get closest frames to the query timestamps
    closest_frames = torch.stack([loaded_frames[idx] for idx in argmin_])
    closest_ts = loaded_ts[argmin_]

    if log_loaded_timestamps:
        logging.info(f"{closest_ts=}")

    # convert to float32 in [0,1] range (channel first)
    closest_frames = closest_frames.type(torch.float32) / 255

    assert len(timestamps) == len(closest_frames)
    return closest_frames



def encode_video_frames(
    imgs_dir: Path | str,
    video_path: Path | str,
    fps: int,
    vcodec: str = "libsvtav1",
    pix_fmt: str = "yuv420p",  
    g: int | None = 2,
    crf: int | None = 30,
    fast_decode: int = 0,
    log_level: int | None = None,
    overwrite: bool = True,  # 默认为True,直接兼容LerobotDataset
    use_gpu: bool = True,  # 是否使用 GPU进行视频编码（小尺寸帧会自动降级为CPU）
    use_no_render: bool = True,  # 是否使用无渲染模式进行编码
) -> None:
    video_path = Path(video_path)
    imgs_dir = Path(imgs_dir)
    video_path.parent.mkdir(parents=True, exist_ok=True)

    sample = next(imgs_dir.glob("*.png"), None)
    if not sample:
        raise FileNotFoundError(f"No images found in {imgs_dir}.")

    img = cv2.imread(str(sample))
    if img is None:
        raise RuntimeError(f"无法读取示例图片：{sample}")
    original_h, original_w = img.shape[:2]  
    # print(
    #     f"检测到原始帧尺寸：宽={original_w}，高={original_h}（{original_w}×{original_h}）"
    # )

    nvenc_unsupported = original_w < 128 or original_h < 128

    # Env-var escape hatch: ``LEROBOT_VIDEO_USE_GPU=0`` forces CPU encoding
    # regardless of the call-site default. Useful when the operator's
    # NVIDIA driver / CUDA stack is broken and they want to keep collecting
    # data without code edits.
    _gpu_env = os.environ.get("LEROBOT_VIDEO_USE_GPU")
    if _gpu_env == "0":
        if use_gpu:
            print("LEROBOT_VIDEO_USE_GPU=0 — disabling GPU encoder")
        use_gpu = False

    if use_gpu and nvenc_unsupported:
        print(f"小尺寸帧{original_w}×{original_h}不支持GPU编码，自动降级为CPU编码")
        use_gpu = False
    elif use_gpu:
        print("使用GPU进行视频编码...")
    else:
        print("使用CPU进行视频编码...")

    encode_w = original_w if original_w % 2 == 0 else original_w + 1
    encode_h = original_h if original_h % 2 == 0 else original_h + 1
    filter_params = []

    if encode_w != original_w or encode_h != original_h:
        print(
            f"H.264强制要求偶数尺寸，补边到：{encode_w}×{encode_h}（仅多1像素透明边，无颜色影响）"
        )
        filter_params = ["-vf", f"pad={encode_w}:{encode_h}:0:0:none"]

    if "frame_" in sample.name:
        input_pattern = str(imgs_dir / "frame_%06d.png")
    else:
        input_pattern = str(imgs_dir / "%06d.png")

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-loglevel",
        log_level if log_level is not None else "error",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
    ]

    if filter_params:
        cmd += filter_params

    if use_gpu:
        cmd += [
            "-c:v",
            "h264_nvenc",
            "-pix_fmt",
            pix_fmt,
            "-preset",
            "p4",
            "-tune",
            "hq",
            "-b:v",
            "6M",
            "-bf",
            "0",
        ]
        if g is not None:
            cmd += ["-g", str(g)]
        if crf is not None:
            cmd += ["-rc", "constqp", "-qp", str(crf)]
    else:
        if use_no_render:
            cmd += [
                "-c:v",
                "libx264",
                "-pix_fmt",
                "rgb24",
                "-crf",
                "0",
                "-preset",
                "ultrafast",
                "-sws_flags",
                "neighbor",
                "-vf",
                "format=rgb24",
                "-avoid_negative_ts",
                "make_zero",
            ]
        else:
            # 原始 CPU 编码逻辑
            cmd += ["-c:v", vcodec, "-pix_fmt", pix_fmt]

        if g is not None:
            cmd += ["-g", str(g)]
        if vcodec == "libsvtav1" and fast_decode:
            cmd += ["-svtav1-params", f"fast-decode={fast_decode}"]
        elif fast_decode:
            cmd += ["-tune", "fastdecode"]

    cmd.append(str(video_path))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            stderr = result.stderr or ""
            # Automatic NVENC → CPU fallback. If the GPU encoder failed
            # because CUDA can't initialize (driver issue, no GPU
            # available, another process holding it), retry the entire
            # encode with libx264 instead of letting the error propagate
            # and abort the recording session. The operator's data is
            # still saved — just on CPU.
            if use_gpu and ("nvenc" in stderr.lower() or "cuda" in stderr.lower()):
                logging.warning(
                    "h264_nvenc failed (likely CUDA/driver issue); falling "
                    "back to libx264 for this video. Set LEROBOT_VIDEO_USE_GPU=0 "
                    "to skip the GPU attempt entirely. Original error:\n%s",
                    stderr,
                )
                # Force the lossy libx264 + yuv420p path on fallback. The
                # ``use_no_render=True`` default is *intentionally lossless*
                # (rgb24 / crf=0 / preset=ultrafast) for callers that need
                # bit-exact decoding for training; if we inherit that on a
                # GPU→CPU fallback, file sizes balloon 10–50× because the
                # operator was asking for "fast lossy NVENC", not "lossless
                # RGB". Match NVENC's quality intent instead.
                return encode_video_frames(
                    imgs_dir=imgs_dir, video_path=video_path, fps=fps,
                    vcodec="libx264", pix_fmt="yuv420p",
                    g=g, crf=crf,
                    fast_decode=fast_decode, log_level=log_level,
                    overwrite=overwrite, use_gpu=False, use_no_render=False,
                )
            raise RuntimeError(f"FFmpeg encoding failed: {stderr}")
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to start FFmpeg: {e}")

    if not video_path.exists():
        raise OSError(f"Video file was not created: {video_path}")

@dataclass
class VideoFrame:
    # TODO(rcadene, lhoestq): move to Hugging Face `datasets` repo
    """
    Provides a type for a dataset containing video frames.

    Example:

    ```python
    data_dict = [{"image": {"path": "videos/episode_0.mp4", "timestamp": 0.3}}]
    features = {"image": VideoFrame()}
    Dataset.from_dict(data_dict, features=Features(features))
    ```
    """

    pa_type: ClassVar[Any] = pa.struct({"path": pa.string(), "timestamp": pa.float32()})
    _type: str = field(default="VideoFrame", init=False, repr=False)

    def __call__(self):
        return self.pa_type


with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        "'register_feature' is experimental and might be subject to breaking changes in the future.",
        category=UserWarning,
    )
    # to make VideoFrame available in HuggingFace `datasets`
    register_feature(VideoFrame, "VideoFrame")


def get_audio_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting audio stream information
    audio_info = {}
    with av.open(str(video_path), "r") as audio_file:
        try:
            audio_stream = audio_file.streams.audio[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {"has_audio": False}

        audio_info["audio.channels"] = audio_stream.channels
        audio_info["audio.codec"] = audio_stream.codec.canonical_name
        # In an ideal loseless case : bit depth x sample rate x channels = bit rate.
        # In an actual compressed case, the bit rate is set according to the compression level : the lower the bit rate, the more compression is applied.
        audio_info["audio.bit_rate"] = audio_stream.bit_rate
        audio_info["audio.sample_rate"] = audio_stream.sample_rate  # Number of samples per second
        # In an ideal loseless case : fixed number of bits per sample.
        # In an actual compressed case : variable number of bits per sample (often reduced to match a given depth rate).
        audio_info["audio.bit_depth"] = audio_stream.format.bits
        audio_info["audio.channel_layout"] = audio_stream.layout.name
        audio_info["has_audio"] = True

    # Reset logging level
    av.logging.restore_default_callback()

    return audio_info


def get_video_info(video_path: Path | str) -> dict:
    # Set logging level
    logging.getLogger("libav").setLevel(av.logging.ERROR)

    # Getting video stream information
    video_info = {}
    with av.open(str(video_path), "r") as video_file:
        try:
            video_stream = video_file.streams.video[0]
        except IndexError:
            # Reset logging level
            av.logging.restore_default_callback()
            return {}

        video_info["video.height"] = video_stream.height
        video_info["video.width"] = video_stream.width
        video_info["video.codec"] = video_stream.codec.canonical_name
        video_info["video.pix_fmt"] = video_stream.pix_fmt
        video_info["video.is_depth_map"] = False

        # Calculate fps from r_frame_rate
        video_info["video.fps"] = int(video_stream.base_rate)

        pixel_channels = get_video_pixel_channels(video_stream.pix_fmt)
        video_info["video.channels"] = pixel_channels

    # Reset logging level
    av.logging.restore_default_callback()

    # Adding audio stream information
    video_info.update(**get_audio_info(video_path))

    return video_info


def get_video_pixel_channels(pix_fmt: str) -> int:
    if "gray" in pix_fmt or "depth" in pix_fmt or "monochrome" in pix_fmt:
        return 1
    elif "rgba" in pix_fmt or "yuva" in pix_fmt:
        return 4
    elif "rgb" in pix_fmt or "yuv" in pix_fmt:
        return 3
    else:
        raise ValueError("Unknown format")


def get_image_pixel_channels(image: Image):
    if image.mode == "L":
        return 1  # Grayscale
    elif image.mode == "LA":
        return 2  # Grayscale + Alpha
    elif image.mode == "RGB":
        return 3  # RGB
    elif image.mode == "RGBA":
        return 4  # RGBA
    else:
        raise ValueError("Unknown format")


class VideoEncodingManager:
    """
    Context manager that ensures proper video encoding and data cleanup even if exceptions occur.

    This manager handles:
    - Batch encoding for any remaining episodes when recording interrupted
    - Cleaning up temporary image files from interrupted episodes
    - Removing empty image directories

    Args:
        dataset: The LeRobotDataset instance
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Handle any remaining episodes that haven't been batch encoded
        if self.dataset.episodes_since_last_encoding > 0:
            if exc_type is not None:
                logging.info("Exception occurred. Encoding remaining episodes before exit...")
            else:
                logging.info("Recording stopped. Encoding remaining episodes...")

            start_ep = self.dataset.num_episodes - self.dataset.episodes_since_last_encoding
            end_ep = self.dataset.num_episodes
            logging.info(
                f"Encoding remaining {self.dataset.episodes_since_last_encoding} episodes, "
                f"from episode {start_ep} to {end_ep - 1}"
            )
            self.dataset.batch_encode_videos(start_ep, end_ep)

        # Clean up episode images if recording was interrupted
        if exc_type is not None:
            interrupted_episode_index = self.dataset.num_episodes
            for key in self.dataset.meta.video_keys:
                img_dir = self.dataset._get_image_file_path(
                    episode_index=interrupted_episode_index, image_key=key, frame_index=0
                ).parent
                if img_dir.exists():
                    logging.debug(
                        f"Cleaning up interrupted episode images for episode {interrupted_episode_index}, camera {key}"
                    )
                    shutil.rmtree(img_dir)

        # Clean up any remaining images directory if it's empty
        img_dir = self.dataset.root / "images"
        # Check for any remaining PNG files
        png_files = list(img_dir.rglob("*.png"))
        if len(png_files) == 0:
            # Only remove the images directory if no PNG files remain
            if img_dir.exists():
                shutil.rmtree(img_dir)
                logging.debug("Cleaned up empty images directory")
        else:
            logging.debug(f"Images directory is not empty, containing {len(png_files)} PNG files")

        return False  # Don't suppress the original exception
