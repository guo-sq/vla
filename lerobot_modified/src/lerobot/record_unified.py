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

"""
Unified Recording and Inference Framework for LeRobot.

Supports:
- Pure recording mode (teleop only)
- Pure inference mode (no dataset recording, runs indefinitely until stop_recording event)
- Inference with recording mode
- Synchronous inference (blocking, alternating between inference and execution)
- Asynchronous inference (parallel inference and execution via threading)
- Human intervention during inference with pose synchronization
- Sub-task timing and tracking
- Simplified camera configuration (just pass camera device IDs)

Usage Examples:

1. Pure Recording (Teleop) with simplified camera config:
```shell
python record_unified.py \\
    --robot.type=arxx5_bimanual \\
    --robot.id=arxx5_bimanual \\
    --mode=record \\
    --left_camera=4 \\
    --head_camera=10 \\
    --right_camera=16 \\
    --dataset.root=/path/to/data \\
    --dataset.repo_id=my_dataset \\
    --dataset.single_task="Pick and place task"
```

2. Inference + Recording (Async) with sub-tasks:
```shell
python record_unified.py \\
    --robot.type=arxx5_bimanual \\
    --robot.id=arxx5_bimanual \\
    --mode=infer_record \\
    --inference_mode=async \\
    --head_camera=10 \\
    --left_camera=4 \\
    --right_camera=16 \\
    --sub_task_durations="[5, 20, 10, 15]" \\
    --policy_host=localhost \\
    --policy_port=8000 \\
    --dataset.root=/path/to/data \\
    --dataset.repo_id=my_dataset \\
    --dataset.single_task="Pick and place task"
```

3. Pure Inference (No Recording, runs indefinitely):
```shell
python record_unified.py \\
    --robot.type=arxx5_bimanual \\
    --mode=infer \\
    --inference_mode=async \\
    --head_camera=10 \\
    --left_camera=4 \\
    --right_camera=16 \\
    --policy_host=localhost \\
    --policy_port=8000 \\
    --dataset.single_task="Pick and place task"
```

4. Custom camera resolution and fps:
```shell
python record_unified.py \\
    --robot.type=arxx5_bimanual \\
    --head_camera=10 \\
    --left_camera=4 \\
    --right_camera=16 \\
    --camera_width=1280 \\
    --camera_height=720 \\
    --camera_fps=60 \\
    --dataset.repo_id=my_dataset \\
    --dataset.single_task="High-res recording"
```

Note: Camera naming convention:
- head_camera -> "head"
- left_camera -> "left_wrist"
- right_camera -> "right_wrist"
Set camera ID to -1 to disable that camera.
"""

import io
import ast
import json
import logging
import os
import queue
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from pprint import pformat
from typing import Any

import pygame
import einops
import numpy as np
import requests

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import (
    INFO_PATH,
    build_dataset_frame,
    hw_to_dataset_features,
    write_json,
)
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so100_follower,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    piper,
    so100_follower,
    so101_follower,
)

# 条件导入 arx_x5_python（仅在环境中存在时）
try:
    from lerobot.robots import arx_x5_python  # noqa: F401
except ImportError:
    arx_x5_python = None

from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so100_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    so100_leader,
    so101_leader,
)
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop
from lerobot.utils.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging, say
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# External policy client
from openpi_client import websocket_client_policy as _websocket_client_policy

import os
os.environ["SDL_AUDIODRIVER"] = "pulse"


################################################################################
# TTS Utilities
################################################################################

class TTSService:
    """
    TTS 服务封装类。

    初始化时自动从候选 URL 列表中查找可用的外置 TTS 服务。
    如果找到可用服务，则使用外置 TTS；否则回退到系统 say 函数。

    Usage:
        tts = TTSService()  # 自动检测可用服务
        tts.log_say("你好")  # 播放语音
    """

    # 候选 TTS 服务 URL 列表
    DEFAULT_URLS = [
        "http://127.0.0.1:8080/v1/tts",
        "http://192.168.110.132:8080/v1/tts",
        "http://192.168.110.194:8080/v1/tts",
    ]

    def __init__(self, candidate_urls: list[str] | None = None, timeout: float = 2.0):
        """
        初始化 TTS 服务。

        Args:
            candidate_urls: 候选 TTS 服务 URL 列表，默认使用 DEFAULT_URLS
            timeout: 检测服务可用性的超时时间（秒）
        """
        self._service_url: str | None = None
        self._use_external_tts: bool = False
        self._mixer_initialized: bool = False

        urls = candidate_urls or self.DEFAULT_URLS
        self._service_url = self._find_available_service(urls, timeout)
        self._use_external_tts = self._service_url is not None

        if self._use_external_tts:
            self._ensure_mixer_initialized()
            logging.info(f"TTSService: 使用外置 TTS 服务 {self._service_url}")
        else:
            logging.info("TTSService: 未找到可用的外置 TTS 服务，将使用系统 say 函数")

    def _find_available_service(self, urls: list[str], timeout: float) -> str | None:
        """
        从候选 URL 列表中查找第一个可用的 TTS 服务。

        Args:
            urls: 候选 URL 列表
            timeout: 每个 URL 的检测超时时间

        Returns:
            可用的服务 URL，如果都不可用则返回 None
        """
        for url in urls:
            try:
                response = requests.post(
                    url,
                    json={"text": "测试", "format": "wav", "reference_id": "taozi"},
                    timeout=timeout,
                )
                if response.status_code == 200:
                    return url
            except Exception:
                continue
        return None

    def _ensure_mixer_initialized(self):
        """确保 pygame mixer 已初始化（懒加载，只初始化一次）。"""
        if self._mixer_initialized:
            return
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=512)
            self._mixer_initialized = True
            logging.info("TTSService: pygame mixer 初始化完成")
        except Exception as e:
            logging.warning(f"TTSService: pygame mixer 初始化失败: {e}")

    def _play_audio_bytes(self, wav_bytes: bytes):
        """播放音频字节数据。"""
        self._ensure_mixer_initialized()
        try:
            sound = pygame.mixer.Sound(io.BytesIO(wav_bytes))
            sound.set_volume(10.0)
            sound.play()
            time.sleep(sound.get_length())
        except Exception as e:
            logging.warning(f"TTSService: 播放音频失败: {e}")

    def _fetch_tts_audio(self, text: str, reference_id: str = "taozi") -> bytes:
        """从 TTS 服务获取音频数据。"""
        payload = {"text": text, "format": "wav", "reference_id": reference_id}
        return requests.post(self._service_url, json=payload, timeout=10.0).content

    def _play_tts_sound(self, text: str, reference_id: str = "taozi"):
        """使用外置 TTS 服务播放语音（获取音频 + 播放）。"""
        wav_bytes = self._fetch_tts_audio(text, reference_id)
        self._play_audio_bytes(wav_bytes)

    def log_say(self, text: str, play_sounds: bool = True, blocking: bool = True):
        """
        记录日志并可选地播放语音。

        Args:
            text: 要播放的文本
            play_sounds: 是否播放语音
            blocking: 是否阻塞等待播放完成
        """
        if not play_sounds:
            return

        logging.info(text)

        if self._use_external_tts:
            if blocking:
                # 阻塞模式：在子线程中获取音频，然后播放
                audio_result = [None]

                def fetch_audio():
                    try:
                        audio_result[0] = self._fetch_tts_audio(text)
                    except Exception as e:
                        logging.warning(f"TTSService: 获取音频失败: {e}")

                fetch_thread = threading.Thread(target=fetch_audio, daemon=True)
                fetch_thread.start()
                fetch_thread.join(timeout=10.0)

                if audio_result[0] is not None:
                    self._play_audio_bytes(audio_result[0])
            else:
                # 非阻塞模式：在后台线程中播放
                threading.Thread(
                    target=self._play_tts_sound, args=(text,), daemon=True
                ).start()
        else:
            # 回退到系统 say 函数
            say(text, blocking=blocking)

    @property
    def is_external_tts_available(self) -> bool:
        """返回外置 TTS 服务是否可用。"""
        return self._use_external_tts

    @property
    def service_url(self) -> str | None:
        """返回当前使用的 TTS 服务 URL。"""
        return self._service_url


# 全局 TTS 服务实例（延迟初始化）
_tts_service: TTSService | None = None

def init_tts_service() -> TTSService:
    """
    初始化全局 TTS 服务实例。

    Returns:
        TTSService 实例
    """
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService()
    return _tts_service


def log_say(text: str, play_sounds: bool = True, blocking: bool = True):
    """
    全局 log_say 函数，使用全局 TTS 服务实例。

    Args:
        text: 要播放的文本
        play_sounds: 是否播放语音
        blocking: 是否阻塞等待播放完成
    """
    global _tts_service
    if _tts_service is None:
        # 如果还未初始化，先初始化
        init_tts_service()
    _tts_service.log_say(text, play_sounds, blocking)


################################################################################
# Configuration（配置）
################################################################################

@dataclass
class DatasetRecordConfig:
    """数据集录制配置。"""

    repo_id: str
    single_task: str
    root: str | Path | None = None
    fps: int = 30
    episode_time_s: int | float = 60
    reset_time_s: int | float = 5
    num_episodes: int = 50
    video: bool = True
    push_to_hub: bool = False
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("single_task is required")


@dataclass
class RecordConfig:
    """
    统一录制/推理主配置。

    mode 说明：
    - "record": 纯录制模式（遥操作）
    - "infer": 纯推理模式（不录制数据，无限执行直到 stop_recording）
    - "infer_record": 推理+录制模式
    """

    robot: RobotConfig
    dataset: DatasetRecordConfig

    # 运行模式: "record", "infer", "infer_record"
    mode: str = "record"

    # 推理模式: "sync" (同步阻塞) 或 "async" (异步并行)
    inference_mode: str = "async"

    # 遥操作器配置
    teleop: TeleoperatorConfig | None = None

    # 策略配置
    policy: PreTrainedConfig | None = None

    # 远程策略服务器
    policy_host: str = "localhost"
    policy_port: int = 8000

    # 显示和日志
    display_data: bool = False
    play_sounds: bool = True
    enable_logging: bool = True

    # 是否恢复已有数据集
    resume: bool = False

    # 异步推理参数
    infer_interval: int = 20
    default_infer_delay: int = 0

    # 是否自动标记所有 episode 为成功
    auto_success: bool = False
    # 动作 horizon（未来动作帧数）
    action_horizon: int = 30

    # 动作融合类型
    fusion_type: str = "linear"  # "linear" or "exponential"
    fusion_exp_decay: float = 2.0

    # 人工接管参数
    waiting_intervention_time_s: float = 2.0
    waiting_evacuation_time_s: float = 2.0
    pose_sync_duration_s: float = 3.0
    # 自动避碰：仅在 infer_record 模式中生效
    avoid_collision: bool = False
    # 多维 current 边界：{dim_idx: [lower_bound, upper_bound]}
    # 支持 NaN/None 表示单边约束，例如 {1:[nan,10], 3:[-3,None]}
    collision_current_bounds: dict[int, list[float]] | str | None = field(default_factory=dict)

    # 动作平滑
    transition_steps: int = 15

    # 子任务时长
    sub_task_durations: list[float] | None = None

    # 简化相机配置（设备 ID，-1 表示禁用）
    head_camera: int = -1
    left_camera: int = -1
    right_camera: int = -1

    # 相机参数
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30

    def __post_init__(self):
        # 验证模式
        valid_modes = ["record", "infer", "infer_record"]
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {self.mode}")

        valid_infer_modes = ["sync", "async"]
        if self.inference_mode not in valid_infer_modes:
            raise ValueError(f"inference_mode must be one of {valid_infer_modes}")

        if not isinstance(self.action_horizon, int) or self.action_horizon <= 0:
            raise ValueError(
                f"action_horizon must be positive integer, got {self.action_horizon}"
            )
        self.collision_current_bounds = _parse_collision_current_bounds(
            self.collision_current_bounds
        )
        if self.avoid_collision and self.robot.type != "arxx5_bimanual":
            logging.warning(
                "avoid_collision only supports arxx5_bimanual current keys; "
                "disabling avoid_collision for robot.type=%s",
                self.robot.type,
            )
            self.avoid_collision = False

        valid_fusion_types = ["linear", "exponential"]
        if self.fusion_type not in valid_fusion_types:
            raise ValueError(f"fusion_type must be one of {valid_fusion_types}")

        # 纯推理模式禁用语音（可选）
        if self.mode == "infer":
            self.play_sounds = False

        # 解析子任务时长
        if isinstance(self.sub_task_durations, str):
            self.sub_task_durations = json.loads(self.sub_task_durations)

        # 设置相机配置
        self._setup_cameras_from_ids()

        # 加载策略配置
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(
                policy_path, cli_overrides=cli_overrides
            )
            self.policy.pretrained_path = policy_path

    def _setup_cameras_from_ids(self):
        """根据简化相机 ID 构建相机配置。"""
        cameras_to_add = {}

        if self.head_camera >= 0:
            cameras_to_add["head"] = OpenCVCameraConfig(
                index_or_path=self.head_camera,
                width=self.camera_width,
                height=self.camera_height,
                fps=self.camera_fps,
            )
        if self.left_camera >= 0:
            cameras_to_add["left_wrist"] = OpenCVCameraConfig(
                index_or_path=self.left_camera,
                width=self.camera_width,
                height=self.camera_height,
                fps=self.camera_fps,
            )
        if self.right_camera >= 0:
            cameras_to_add["right_wrist"] = OpenCVCameraConfig(
                index_or_path=self.right_camera,
                width=self.camera_width,
                height=self.camera_height,
                fps=self.camera_fps,
            )

        if cameras_to_add:
            self.robot.cameras = cameras_to_add
            logging.info(
                f"Camera configuration from IDs: {list(cameras_to_add.keys())}"
            )

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


################################################################################
# Logging Utilities
################################################################################

class AsyncLogger:
    """异步文件日志器，使用后台线程写入。"""

    _MAX_QUEUE_SIZE = 10_000  # 防止队列无界增长导致内存泄漏

    def __init__(self, output_dir: str, enabled: bool = True):
        self.enabled = enabled
        self.output_dir = output_dir
        self._queue = queue.Queue(maxsize=self._MAX_QUEUE_SIZE)
        self._thread = None
        self._file = None

        if self.enabled:
            os.makedirs(output_dir, exist_ok=True)
            self._filepath = os.path.join(output_dir, "unified_record.log")
            self._file = open(self._filepath, "a", encoding="utf-8")
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    def _worker(self):
        while True:
            msg = self._queue.get()
            if msg is None:
                break
            try:
                print(msg, file=self._file)
                self._file.flush()
            except Exception:
                pass

    def log(self, msg: str):
        if not self.enabled:
            return
        try:
            # 队列满时丢弃最新消息而非阻塞/无限堆积
            self._queue.put_nowait(msg)
        except queue.Full:
            pass

    def close(self):
        if not self.enabled:
            return
        # 先标记停止，防止 close 后继续写入
        self.enabled = False
        if self._thread and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=5.0)
            self._thread = None
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None


################################################################################
# Policy Inference Client
################################################################################

class PolicyInferenceClient:
    """远程策略推理客户端（通过 WebSocket）。"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8000,
        camera_keys: list[str] = None,
        robot_state_keys: list[str] = None,
        action_dim: int = 14,
    ):
        self.policy = _websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.camera_keys = camera_keys or []
        self.robot_state_keys = robot_state_keys or []
        self.action_dim = action_dim
    
    def get_action(
        self,
        observation_dict: dict,
        action_prefix: np.ndarray = None,
        infer_delay: int = 0,
        lang: str = "",
    ) -> np.ndarray:
        """从策略服务器获取动作序列。"""
        obs_dict = {}
        for key in self.camera_keys:
            if key in observation_dict:
                img = observation_dict[key]
                obs_dict[key] = einops.rearrange(img, "h w c -> c h w")
        
        state = np.array(
            [observation_dict.get(k, 0.0) for k in self.robot_state_keys]
        ).astype(np.float32)
        
        observation = {
            "observation/front_image": obs_dict.get("head", obs_dict.get(self.camera_keys[0] if self.camera_keys else "head")),
            "observation/wrist_image": obs_dict.get("right_wrist"),
            "observation/wrist_image_lf": obs_dict.get("left_wrist"),
            "observation/state": state,
            "prompt": lang,
        }
        
        if action_prefix is not None:
            observation["action"] = action_prefix
            observation["action_mask"] = np.ones(action_prefix.shape[0])
            observation["infer_delay"] = infer_delay
        
        result = self.policy.infer(observation)
        return result["actions"]


################################################################################
# Action Buffer
################################################################################

class ActionBuffer:
    """时间对齐的动作缓冲区，用于异步推理。"""
    
    def __init__(
        self,
        robot_state_keys: list[str],
        action_dim: int = 14,
        action_horizon: int = 30,
        fusion_type: str = "linear",
        fusion_start_weight: float = 0.9,
        fusion_end_weight: float = 0.1,
        fusion_exp_decay: float = 2.0,
    ):
        self.robot_state_keys = robot_state_keys
        self.action_dim = action_dim
        self.default_action_horizon = action_horizon
        self.fusion_type = fusion_type
        self.fusion_start_weight = fusion_start_weight
        self.fusion_end_weight = fusion_end_weight
        self.fusion_exp_decay = fusion_exp_decay
        
        self.buffer = None
        self.available_cnt = -1
        self.action_horizon = 0
    
    def is_empty(self) -> bool:
        return self.available_cnt <= 0
    
    def is_initialized(self) -> bool:
        return self.available_cnt >= 0
    
    def init_buffer(self, initial_chunk: np.ndarray):
        self.action_horizon = initial_chunk.shape[0]
        self.buffer = np.copy(initial_chunk)
        self.available_cnt = len(initial_chunk)
    
    def get_next_action(self) -> dict[str, float] | None:
        if self.is_empty():
            return None
        
        action = {}
        for i, key in enumerate(self.robot_state_keys):
            action[key] = float(self.buffer[0, i])
        
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = 0.0
        self.available_cnt -= 1
        return action
    
    def get_future_actions(self, infer_delay: int) -> tuple[np.ndarray, int]:
        if self.available_cnt <= 0:
            return np.zeros((self.default_action_horizon, self.action_dim), dtype=np.float32), 0
        actual_delay = min(infer_delay, self.available_cnt)
        return np.copy(self.buffer), actual_delay
    
    def _compute_fusion_weights(self, num_weights: int) -> np.ndarray:
        if self.fusion_type == "exponential":
            indices = np.arange(num_weights)
            weights = np.exp(-(indices + 0.5) / self.fusion_exp_decay)
        else:
            weights = np.linspace(self.fusion_start_weight, self.fusion_end_weight, num=num_weights)
        return weights
    
    def update_from_inference(
        self,
        new_chunk: np.ndarray,
        start_step_id: int,
        current_step_id: int,
        enable_fusion: bool = True,
    ):
        if not self.is_initialized():
            self.init_buffer(new_chunk)
            return
        
        latency_steps = current_step_id - start_step_id
        if latency_steps >= len(new_chunk):
            return
        
        valid_new_actions = new_chunk[latency_steps:]
        
        if enable_fusion and self.available_cnt > 0:
            num_to_fuse = min(self.available_cnt, len(valid_new_actions))
            weights = self._compute_fusion_weights(num_to_fuse)
            
            for i, weight in enumerate(weights):
                if i < len(valid_new_actions):
                    self.buffer[i] = self.buffer[i] * weight + valid_new_actions[i] * (1 - weight)
            
            if len(valid_new_actions) > self.available_cnt:
                self.buffer[self.available_cnt:len(valid_new_actions)] = valid_new_actions[self.available_cnt:]
        else:
            self.buffer = np.zeros_like(self.buffer)
            self.buffer[:len(valid_new_actions)] = valid_new_actions
        
        self.available_cnt = len(valid_new_actions)
    
    def clear(self):
        """清空缓冲区。"""
        if self.buffer is not None:
            self.buffer.fill(0.0)
        self.available_cnt = -1


################################################################################
# Sub-task Manager
################################################################################

class SubTaskManager:
    """
    子任务管理器，管理子任务计时、播报和索引跟踪。
    
    示例：durations = [10, 5, 20] 表示 3 个子任务，时长分别为 10s, 5s, 20s。
    """

    def __init__(self, durations: list[float] | None, is_inference_mode: bool, play_sounds: bool = True):
        self.durations = durations or []
        self.play_sounds = play_sounds

        # 纯推理模式或无子任务时禁用
        if is_inference_mode or not self.durations:
            self.enabled = False
        else:
            self.enabled = all(d > 0 for d in self.durations)

        if self.enabled:
            self.timestamps = []
            cumsum = 0.0
            for d in self.durations:
                cumsum += d
                self.timestamps.append(cumsum)
            self.total_duration = self.timestamps[-1]
            self.current_index = -1
            self.announced = [False] * len(self.durations)
            self.finished_announced = False
        else:
            self.timestamps = []
            self.total_duration = 0.0
            self.current_index = -1
            self.announced = []
            self.finished_announced = False

    def update(self, timestamp: float) -> int:
        if not self.enabled:
            return -1

        if timestamp >= self.total_duration:
            if not self.finished_announced:
                self.finished_announced = True
                self.current_index = -1
                log_say("结束所有步骤", self.play_sounds, blocking=True)
            return -1

        new_index = 0
        for i, ts in enumerate(self.timestamps):
            if timestamp < ts:
                new_index = i
                break

        if not self.announced[new_index]:
            self.announced[new_index] = True
            self.current_index = new_index
            log_say(f"开始第{new_index + 1}个步骤", self.play_sounds, blocking=False)

        return self.current_index

    def get_current_index(self) -> int:
        return self.current_index if self.enabled else -1

    def is_finished(self) -> bool:
        return self.finished_announced


################################################################################
# Inference Worker (后台推理线程)
################################################################################

def inference_worker(
    policy_client: PolicyInferenceClient,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    logger: AsyncLogger,
):
    """后台推理工作线程。"""
    while not stop_event.is_set():
        try:
            item = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break

        lang_prompt, obs_dict, action_prefix, infer_delay, step_id = item
        logger.log(
            f"inference_worker: Running inference at step_id={step_id} with lang_prompt={lang_prompt}"
        )
        t0 = time.perf_counter()

        try:
            action_chunk = policy_client.get_action(
                obs_dict, action_prefix, infer_delay, lang_prompt
            )
            inference_time = time.perf_counter() - t0
            logger.log(f"inference_worker: Completed at step_id={step_id}, time={inference_time*1e3:.1f}ms")
            output_queue.put((action_chunk, step_id))
        except Exception as e:
            logger.log(f"inference_worker: Error at step_id={step_id}: {e}")


################################################################################
# Inference State（推理状态管理）
################################################################################

class InferenceState:
    """
    推理状态管理类，封装推理相关的所有资源和状态。
    
    包括：
    - 动作缓冲区
    - 推理队列和工作线程
    - 人工接管状态
    - 步数计数器
    """

    def __init__(
        self,
        config: "RecordConfig",
        policy_client: PolicyInferenceClient,
        robot: Robot,
        lang_prompt: str,
        logger: AsyncLogger,
    ):
        self.config = config
        self.policy_client = policy_client
        self.robot = robot
        self.logger = logger

        # 步数和时间
        self.global_step_id: int = 0
        self.timestamp: float = 0.0
        self.start_time: float = 0.0

        # 人工接管状态
        self.is_human_intervention: bool = False
        self.transition_weight: float = 0.9

        # 推理资源
        self.is_async = config.inference_mode == "async"
        self.input_queue = queue.Queue(maxsize=1 if self.is_async else 0)
        self.output_queue = queue.Queue()
        self.stop_event = threading.Event()

        # 动作缓冲区
        self.action_buffer = ActionBuffer(
            robot_state_keys=policy_client.robot_state_keys,
            action_dim=len(policy_client.robot_state_keys),
            action_horizon=config.action_horizon,
            fusion_type=config.fusion_type,
            fusion_exp_decay=config.fusion_exp_decay,
        )

        # 启动推理线程
        self.inference_thread = threading.Thread(
            target=inference_worker,
            args=(
                policy_client,
                self.input_queue,
                self.output_queue,
                self.stop_event,
                logger,
            ),
            daemon=True,
        )
        self.inference_thread.start()

        # 预热推理
        self._warmup_inference(lang_prompt)

    def _warmup_inference(self, lang_prompt: str):
        """执行预热推理，初始化动作缓冲区。"""
        self.logger.log("Performing warmup inference...")
        zero_prefix = np.zeros(
            (self.config.action_horizon, len(self.policy_client.robot_state_keys)), 
            dtype=np.float32
        )
        warmup_obs = self.robot.get_observation()
        warmup_chunk = self.policy_client.get_action(
            warmup_obs, zero_prefix, 0, lang_prompt
        )
        self.action_buffer.init_buffer(warmup_chunk)
        self.logger.log(f"Warmup complete, action horizon: {len(warmup_chunk)}")

    def reset_for_resume(self):
        """从人工接管恢复后重置状态。注意：不重置 global_step_id，保持单调递增。"""
        self.is_human_intervention = False
        self.transition_weight = 0.9

    def clear_queues(self):
        """清空推理队列。"""
        for q in [self.input_queue, self.output_queue]:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def reinit_action_buffer(self, lang_prompt: str):
        """重新初始化动作缓冲区（人工接管结束后）。"""
        zero_prefix = np.zeros(
            (self.config.action_horizon, len(self.policy_client.robot_state_keys)), 
            dtype=np.float32
        )
        obs = self.robot.get_observation()
        new_chunk = self.policy_client.get_action(obs, zero_prefix, 0, lang_prompt)
        self.action_buffer.init_buffer(new_chunk)
        self.clear_queues()

    def cleanup(self):
        """清理推理资源。"""
        self.stop_event.set()
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)

def resolve_episode_success(
    task_success: bool | None,
    auto_success: bool,
) -> tuple[bool | None, dict | None]:
    """Resolve the final task_success value and build episode_metadata.

    Args:
        task_success: Operator-set success tag (None if not set).
        auto_success: If True, default untagged episodes to success.

    Returns:
        (resolved_task_success, episode_metadata) where episode_metadata is
        ``{"success": <bool>}`` or ``None`` if no tag applies.
    """
    if task_success is None and auto_success:
        task_success = True
    episode_metadata = {"success": task_success} if task_success is not None else None
    return task_success, episode_metadata


def _is_finite_number(value: Any) -> bool:
    """Return True when value can be converted to a finite float."""
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _parse_bound_value(value: Any) -> float:
    """Parse one bound endpoint; NaN/None means open bound."""
    if value is None:
        return float("nan")
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "null", "nan"}:
            return float("nan")
    return float(value)


def _parse_collision_current_bounds(
    raw: dict[int, list[float]] | dict[str, list[float]] | str | None,
) -> dict[int, tuple[float, float]]:
    """
    Parse collision current bounds from dict or string.

    Expected format:
      {1: [-10, 10], 3: [-3, 5]}
    Supports open bound by using NaN/None, e.g.:
      {1: [nan, 10], 3: [-3, null]}
    """
    if raw is None:
        return {}

    parsed_obj: Any = raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            parsed_obj = json.loads(text)
        except Exception:
            # Support CLI literals like {1:[nan,10], 3:[-3,None]}.
            normalized_text = re.sub(
                r"(?<![A-Za-z0-9_])(nan|NaN|null|NULL|Null)(?![A-Za-z0-9_])",
                "None",
                text,
            )
            parsed_obj = ast.literal_eval(normalized_text)

    if not isinstance(parsed_obj, dict):
        raise ValueError(
            "collision_current_bounds must be a dict like {1:[-10,10], 3:[-3,5]}"
        )

    parsed: dict[int, tuple[float, float]] = {}
    for key, bounds in parsed_obj.items():
        try:
            dim_idx = int(key)
        except Exception as e:
            raise ValueError(f"Invalid collision dimension key: {key!r}") from e
        if dim_idx < 0:
            raise ValueError(f"Collision dimension index must be >= 0, got {dim_idx}")

        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"Bounds for dim {dim_idx} must be [lower, upper], got {bounds!r}"
            )

        lower = _parse_bound_value(bounds[0])
        upper = _parse_bound_value(bounds[1])
        if _is_finite_number(lower) and _is_finite_number(upper) and lower > upper:
            raise ValueError(
                f"Bounds for dim {dim_idx} are invalid: lower ({lower}) > upper ({upper})"
            )
        parsed[dim_idx] = (lower, upper)

    return parsed


################################################################################
# Pose Synchronization（姿态同步）
################################################################################

def pose_sync_loop(
    robot: Robot,
    teleop: Teleoperator,
    events: dict,
    sync_duration_s: float,
    fps: int,
    logger: AsyncLogger = None,
) -> bool:
    """
    姿态同步：平滑地将从臂移动到主臂的姿态。
    
    Returns:
        True 如果成功完成，False 如果被中断
    """
    if logger:
        logger.log(f"Starting pose synchronization for {sync_duration_s}s at {fps}fps...")

    if events.get("exit_early") or events.get("stop_recording"):
        return False

    teleop_action = teleop.get_action()
    target_pose = {key: teleop_action[key] for key in robot.action_features if key in teleop_action}

    current_observation = robot.get_joint_positions()
    start_pose = {key: current_observation[key] for key in robot.action_features if key in current_observation}

    num_steps = int(sync_duration_s * fps)
    step_duration = 1.0 / fps

    for step in range(num_steps):
        if events.get("exit_early") or events.get("stop_recording"):
            return False
        start_time = time.perf_counter()
        t = step / max(num_steps - 1, 1)

        current_pose = {}
        for key in robot.action_features:
            if key in start_pose and key in target_pose:
                current_pose[key] = start_pose[key] + t * (target_pose[key] - start_pose[key])
            else:
                current_pose[key] = start_pose.get(key, 0.0)

        robot.send_action(current_pose)

        elapsed = time.perf_counter() - start_time
        if elapsed < step_duration:
            time.sleep(step_duration - elapsed)
        print(f"Follower pose synchronizing [{step}/{num_steps}]: timing: {elapsed:.2f}s")

    if logger:
        logger.log("Pose synchronization completed.")
    return True


################################################################################
# 队列辅助函数
################################################################################

def queue_put_replace_oldest(q: queue.Queue, item: Any, logger: AsyncLogger = None) -> bool:
    """
    向队列中添加元素。如果队列已满，则移除最旧的元素后再添加。
    
    Args:
        q: 目标队列
        item: 要添加的元素
        logger: 日志记录器
        
    Returns:
        True 如果替换了旧元素，False 如果直接添加成功
    """
    try:
        q.put_nowait(item)
        return False
    except queue.Full:
        # 队列满，移除最旧的元素
        try:
            old_item = q.get_nowait()
            if logger:
                logger.log(f"Queue full, removed old item (step_id={old_item[3] if len(old_item) > 3 else 'unknown'})")
                print(f"Inference is too frequent !!!")
        except queue.Empty:
            pass
        # 再次尝试添加
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            # 如果还是满的，说明有竞争条件，记录警告
            if logger:
                logger.log("Warning: queue still full after removing old item")
            return False


################################################################################
# 纯推理模式循环（无数据集、无时间限制、无人工接管）
################################################################################

def run_inference_only_loop(
    robot: Robot,
    policy_client: PolicyInferenceClient,
    config: RecordConfig,
    events: dict,
    logger: AsyncLogger,
) -> None:
    """纯推理模式：无数据集、无时间限制、无人工接管，只有 stop_recording 才退出。"""
    fps = config.dataset.fps
    lang_prompt = config.dataset.single_task
    is_async = config.inference_mode == "async"

    state = InferenceState(config, policy_client, robot, lang_prompt, logger)
    logger.log(f"Starting inference-only loop: inference_mode={config.inference_mode}")
    state.start_time = time.perf_counter()

    while True:
        loop_start_t = time.perf_counter()

        if events.get("stop_recording", False):
            logger.log("stop_recording triggered, exiting inference loop")
            break

        if events.get("rerecord_episode", False):
            state.start_time = time.perf_counter()
            events["rerecord_episode"] = False
            print("Reset state timestamp.")

        observation = robot.get_observation()
        action = None

        if is_async:
            if state.global_step_id % config.infer_interval == 0:
                action_prefix, infer_delay = state.action_buffer.get_future_actions(config.default_infer_delay)
                queue_put_replace_oldest(
                    state.input_queue,
                    (
                        lang_prompt,
                        observation,
                        action_prefix,
                        infer_delay,
                        state.global_step_id,
                    ),
                    logger,
                )

            if not state.output_queue.empty():
                new_chunk, start_step = state.output_queue.get_nowait()
                without_action_prefix = infer_delay == 0
                state.action_buffer.update_from_inference(
                    new_chunk,
                    start_step,
                    state.global_step_id,
                    enable_fusion=without_action_prefix,
                )

            wait_start = time.perf_counter()
            while state.action_buffer.is_empty():
                time.sleep(0.001)
                if time.perf_counter() - wait_start > 3.0:
                    logger.log(f"Warning: Inference timeout at step {state.global_step_id}")
                    break

            if not state.action_buffer.is_empty():
                action = state.action_buffer.get_next_action()
        else:
            if state.action_buffer.is_empty():
                zero_prefix = np.zeros((config.action_horizon, len(policy_client.robot_state_keys)), dtype=np.float32)
                state.input_queue.put(
                    (lang_prompt, observation, zero_prefix, 0, state.global_step_id)
                )
                new_chunk, _ = state.output_queue.get()
                state.action_buffer.init_buffer(new_chunk)
            action = state.action_buffer.get_next_action()

        if action is not None:
            robot.send_action(action)

        if config.display_data and action is not None:
            log_rerun_data(observation, action)

        state.global_step_id += 1

        dt_s = time.perf_counter() - loop_start_t
        wait_time = 1.0 / fps - dt_s

        if wait_time > 0:
            busy_wait(wait_time)

        state.timestamp = time.perf_counter() - state.start_time

        if state.global_step_id % fps == 0:
            print(f"Inference | {config.inference_mode} | {state.timestamp:.1f}s | Step: {state.global_step_id}")

    state.cleanup()
    logger.log(f"Inference loop ended: {state.global_step_id} steps, {state.timestamp:.1f}s")


################################################################################
# 录制模式循环（record 和 infer_record）
################################################################################

@safe_stop_image_writer
def run_record_loop(
    robot: Robot,
    events: dict,
    config: RecordConfig,
    episode_idx: int,
    policy_client: PolicyInferenceClient | None,
    dataset: LeRobotDataset,
    teleop: Teleoperator | None,
    logger: AsyncLogger,
) -> None:
    """录制模式：支持纯录制（遥操作）和推理+录制，有时间限制，存储数据到 dataset。"""
    fps = config.dataset.fps
    control_time_s = config.dataset.episode_time_s
    lang_prompt = config.dataset.single_task
    is_inference_mode = config.mode == "infer_record"
    is_async = config.inference_mode == "async"

    state: InferenceState | None = None
    if is_inference_mode and policy_client is not None:
        state = InferenceState(config, policy_client, robot, lang_prompt, logger)
        logger.log(f"Starting infer_record loop: inference_mode={config.inference_mode}")
    inference_paused_after_collision = False

    sub_task_manager = SubTaskManager(config.sub_task_durations, is_inference_mode, config.play_sounds)
    start_time = time.perf_counter()
    timestamp = 0.0

    while timestamp < control_time_s:
        loop_start_t = time.perf_counter()

        if events.get("stop_recording", False):
            break

        if events.get("exit_early", False):
            events["exit_early"] = False
            break

        # 人工接管切换（仅 infer_record 模式）
        if events.get("switch_infer_mode", False) and state is not None:
            events["switch_infer_mode"] = False
            if not state.is_human_intervention:
                log_say("停止模型推理", config.play_sounds)
                time.sleep(config.waiting_intervention_time_s)
                state.is_human_intervention = True
                # 用户主动接管后，清除碰撞暂停状态。
                inference_paused_after_collision = False

                if teleop is not None and config.pose_sync_duration_s > 0:
                    log_say("开始姿态同步", config.play_sounds)
                    pose_sync_loop(robot, teleop, events, config.pose_sync_duration_s, fps, logger)

                if hasattr(robot, "set_gravity_compensation_mode"):
                    robot.set_gravity_compensation_mode()

                log_say("开始接管", config.play_sounds)
                logger.log("Entered human intervention mode")
            else:
                action = robot.get_joint_positions()
                action = {k: action[k] for k in robot.action_features}
                robot.send_action(action)

                log_say("请立即撤离", config.play_sounds, blocking=True)
                time.sleep(config.waiting_evacuation_time_s)
                state.reinit_action_buffer(lang_prompt)
                state.reset_for_resume()
                log_say("恢复模型推理", config.play_sounds)
                logger.log("Exited human intervention mode")
            continue

        # 碰撞后手动恢复推理（不进入人工接管）
        if events.get("resume_inference", False) and state is not None:
            events["resume_inference"] = False
            if inference_paused_after_collision and not state.is_human_intervention:
                state.reinit_action_buffer(lang_prompt)
                state.reset_for_resume()
                inference_paused_after_collision = False
                log_say("恢复模型推理", config.play_sounds)
                logger.log("Inference resumed manually after collision pause")
            continue

        observation = robot.get_observation()
        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        action = None
        current_intervention = state.is_human_intervention if state else False

        # 自动避碰（infer_record）：检测到碰撞电流后暂停推理并保持当前姿态，操作员可通过 Ctrl+Enter 恢复推理或 Ctrl+Space 进入人工接管
        if (
            is_inference_mode
            and state is not None
            and config.avoid_collision
            and config.robot.type == "arxx5_bimanual"
            and not inference_paused_after_collision
            and not current_intervention
            and not events.get("switch_infer_mode", False)
            and config.collision_current_bounds
        ):
            triggered_msg = None
            current_vector = robot.get_current_vector()
            for dim_idx, (lower, upper) in config.collision_current_bounds.items():
                if dim_idx >= len(current_vector):
                    continue
                current_dim_val = current_vector[dim_idx]
                if not np.isfinite(current_dim_val):
                    continue

                lower_hit = _is_finite_number(lower) and current_dim_val < float(lower)
                upper_hit = _is_finite_number(upper) and current_dim_val > float(upper)
                if lower_hit or upper_hit:
                    triggered_msg = (
                        f"dim={dim_idx}, value={current_dim_val:.4f}, "
                        f"bounds=[{lower}, {upper}]"
                    )
                    break

            if triggered_msg is not None:
                hold_action = robot.get_joint_positions()
                hold_action = {
                    k: hold_action[k] for k in robot.action_features if k in hold_action
                }
                if hold_action:
                    robot.send_action(hold_action)
                inference_paused_after_collision = True
                state.clear_queues()
                state.action_buffer.clear()
                log_say("检测到碰撞，已暂停模型推理", config.play_sounds)
                print("Press Ctrl+Enter to resume inference, or Ctrl+Space to enter human takeover.")
                logger.log(
                    "Inference paused by collision current: "
                    + triggered_msg
                )
                continue

        if inference_paused_after_collision and not current_intervention:
            hold_action = robot.get_joint_positions()
            hold_action = {
                k: hold_action[k] for k in robot.action_features if k in hold_action
            }
            if hold_action:
                action = robot.send_action(hold_action)
            else:
                action = robot.get_joint_positions()
        elif current_intervention:
            if teleop is not None:
                tele_action = teleop.get_action()
                action = robot.send_action(tele_action)
            else:
                action = robot.get_joint_positions()
        elif state is not None and policy_client is not None:
            # infer_record 模式
            if is_async:
                if state.global_step_id % config.infer_interval == 0:
                    action_prefix, infer_delay = state.action_buffer.get_future_actions(config.default_infer_delay)
                    queue_put_replace_oldest(
                        state.input_queue,
                        (
                            lang_prompt,
                            observation,
                            action_prefix,
                            infer_delay,
                            state.global_step_id,
                        ),
                        logger,
                    )

                if not state.output_queue.empty():
                    new_chunk, start_step = state.output_queue.get_nowait()
                    without_action_prefix = infer_delay == 0
                    state.action_buffer.update_from_inference(
                        new_chunk,
                        start_step,
                        state.global_step_id,
                        enable_fusion=without_action_prefix,
                    )

                wait_start = time.perf_counter()
                while state.action_buffer.is_empty():
                    time.sleep(0.001)
                    if time.perf_counter() - wait_start > 3.0:
                        events["switch_infer_mode"] = True
                        log_say("推理超时，自动切换至人工接管", config.play_sounds)
                        break

                if not state.action_buffer.is_empty():
                    action = state.action_buffer.get_next_action()
            else:
                if state.action_buffer.is_empty():
                    zero_prefix = np.zeros((config.action_horizon, len(policy_client.robot_state_keys)), dtype=np.float32)
                    state.input_queue.put(
                        (lang_prompt, observation, zero_prefix, 0, state.global_step_id)
                    )
                    new_chunk, _ = state.output_queue.get()
                    state.action_buffer.init_buffer(new_chunk)
                action = state.action_buffer.get_next_action()

            if action is not None:
                if state.transition_weight > 0:
                    joint_obs = robot.get_joint_positions()
                    for key in robot.action_features:
                        if key in action and key in joint_obs:
                            action[key] = (1 - state.transition_weight) * action[key] + state.transition_weight * joint_obs[key]
                    state.transition_weight = max(0, state.transition_weight - 1.0 / config.transition_steps)
                action = robot.send_action(action)
        else:
            # 纯录制模式（遥操作）
            if teleop is not None:
                tele_action = teleop.get_action()
                action = robot.send_action(tele_action)
            else:
                action = robot.get_joint_positions()
        sub_task_index = sub_task_manager.update(timestamp)

        if action is not None:
            action_frame = build_dataset_frame(
                dataset.features, action, prefix="action"
            )
            frame = {**observation_frame, **action_frame}
            frame["is_human_intervention"] = np.array(current_intervention).reshape(1)
            if sub_task_manager.enabled:
                frame["sub_task_index"] = np.array(sub_task_index).reshape(1)
            dataset.add_frame(frame, task=lang_prompt)

        if config.display_data and action is not None:
            log_rerun_data(observation, action)

        if state:
            state.global_step_id += 1

        dt_s = time.perf_counter() - loop_start_t
        wait_time = 1.0 / fps - dt_s
        if wait_time > 0:
            busy_wait(wait_time)
        else:
            print(f"Warning: Control loop is lagging by {-wait_time*1e3:.1f}ms")

        timestamp = time.perf_counter() - start_time
        if state:
            state.timestamp = timestamp

        if config.play_sounds:
            print(f"Episode {episode_idx} | {timestamp:.1f}s / {control_time_s}s | "
                  f"Intervention: {current_intervention} | SubTask: {sub_task_manager.get_current_index()}")

    if state:
        state.cleanup()
        logger.log(f"Record loop ended: step={state.global_step_id}, time={timestamp:.1f}s")


################################################################################
# Setup Functions（设置函数）
################################################################################

def setup_policy_client(
    robot: Robot,
    config: RecordConfig,
    logger: AsyncLogger,
) -> PolicyInferenceClient | None:
    """设置策略推理客户端。"""
    if config.mode not in ["infer", "infer_record"]:
        return None
    
    camera_keys = list(config.robot.cameras.keys())
    robot_state_keys = [k for k in robot._motors_ft.keys() if "pos" in k and "joint" in k]
    
    policy_client = PolicyInferenceClient(
        host=config.policy_host,
        port=config.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        action_dim=len(robot_state_keys),
    )
    
    logger.log(f"Policy client created: {config.policy_host}:{config.policy_port}")
    logger.log(f"Camera keys: {camera_keys}")
    logger.log(f"Robot state keys: {robot_state_keys}")
    
    return policy_client


def setup_dataset(
    robot: Robot,
    config: RecordConfig,
    logger: AsyncLogger,
) -> LeRobotDataset | None:
    """
    设置数据集。
    
    注意：纯推理模式 (mode="infer") 不创建数据集，返回 None。
    """
    # 纯推理模式不需要数据集
    if config.mode == "infer":
        logger.log("Pure inference mode: no dataset created")
        return None
    
    # 只有 record 和 infer_record 模式需要数据集
    if config.mode not in ["record", "infer_record"]:
        return None
    
    # 创建数据集特征
    action_features = hw_to_dataset_features(robot.action_features, "action", config.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", config.dataset.video)
    
    extra_features = {
        "is_human_intervention": {
            "dtype": "bool",
            "shape": (1,),
            "names": None,
        }
    }
    
    # 录制模式下添加子任务索引（如果配置了子任务）
    if config.sub_task_durations and config.mode == "record":
        extra_features["sub_task_index"] = {
            "dtype": "int64",
            "shape": (1,),
            "names": None,
        }
    
    dataset_features = {**action_features, **obs_features, **extra_features}
    
    # 创建或恢复数据集
    if config.resume:
        dataset = LeRobotDataset(
            config.dataset.repo_id,
            root=config.dataset.root,
            batch_encoding_size=config.dataset.video_encoding_batch_size,
        )
        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=config.dataset.num_image_writer_processes,
                num_threads=config.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, config.dataset.fps, dataset_features)
    else:
        dataset = LeRobotDataset.create(
            config.dataset.repo_id,
            config.dataset.fps,
            root=config.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=config.dataset.video,
            image_writer_processes=config.dataset.num_image_writer_processes,
            image_writer_threads=config.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=config.dataset.video_encoding_batch_size,
        )
    
    # 添加元数据
    is_policy_infer = config.mode == "infer_record"
    info_path = dataset.root / INFO_PATH
    dataset.meta.info["is_policy_infer"] = is_policy_infer
    if config.sub_task_durations:
        dataset.meta.info["sub_task_durations"] = config.sub_task_durations
    write_json(dataset.meta.info, info_path)
    
    logger.log(f"Dataset created: {dataset.root}")
    logger.log(f"is_policy_infer: {is_policy_infer}")
    
    return dataset


################################################################################
# 录制会话运行器
################################################################################

def run_recording_session(
    robot: Robot,
    teleop: Teleoperator | None,
    policy_client: PolicyInferenceClient | None,
    dataset: LeRobotDataset,
    config: RecordConfig,
    events: dict,
    logger: AsyncLogger,
):
    """运行录制会话（多个 episode）。"""

    # 记录会话开始时间（用于批次追踪）
    session_start_time = time.time()

    # 标记用于异常处理
    recorded_episodes = 0
    batch_recorded = False

    class NullContextManager:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False

    context_manager = VideoEncodingManager(dataset) if dataset is not None else NullContextManager()

    try:
        with context_manager:
            while recorded_episodes < config.dataset.num_episodes:
                if events.get("stop_recording", False):
                    logger.log("Stop recording triggered")
                    break

                # 播报并记录
                log_say(f"录制第 {recorded_episodes} 条数据", config.play_sounds, blocking=True)
                logger.log(f"Starting episode {recorded_episodes}")

                # 运行单个 episode
                run_record_loop(
                    robot=robot,
                    events=events,
                    config=config,
                    episode_idx=recorded_episodes,
                    policy_client=policy_client,
                    dataset=dataset,
                    teleop=teleop,
                    logger=logger,
                )

                # 检查是否提前停止
                if events.get("stop_recording", False):
                    log_say("提前退出录制", config.play_sounds, blocking=True)
                    time.sleep(2)
                    break

                # 检查是否需要重录
                if events.get("rerecord_episode", False):
                    log_say("重新录制当前数据", config.play_sounds, blocking=True)
                    log_say("请重置环境", config.play_sounds)
                    time.sleep(config.dataset.reset_time_s)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    events["task_success"] = None
                    dataset.clear_episode_buffer()
                    logger.log("Episode cleared for rerecording")
                    continue

                try:
                    task_success, episode_metadata = resolve_episode_success(
                        events.get("task_success"), config.auto_success
                    )

                    if task_success is True:
                        log_say("任务成功", config.play_sounds, blocking=True)
                    elif task_success is False:
                        log_say("任务失败", config.play_sounds, blocking=True)

                    # 保存 episode
                    log_say("存储数据中，请重置环境", config.play_sounds)
                    dataset.save_episode(episode_metadata=episode_metadata)
                    time.sleep(config.dataset.reset_time_s)
                    log_say("存储完毕", config.play_sounds, blocking=True)
                    success_log = f" (success={task_success})" if task_success is not None else ""
                    logger.log(f"Episode {recorded_episodes} saved{success_log}")
                    events["task_success"] = None

                    recorded_episodes += 1
                except Exception as e:
                    dataset.clear_episode_buffer()
                    logger.log(f"Error saving episode: {e}")
                    log_say("存储数据出错，请重新录制", config.play_sounds)

    except KeyboardInterrupt:
        # 捕获Ctrl+C中断，尝试保存已完成的数据
        logger.log(f"KeyboardInterrupt detected! Recorded {recorded_episodes} episodes before interruption.")
        logging.warning(f"Recording interrupted by user. Saving partial batch with {recorded_episodes} episodes.")
        log_say(f"录制中断，已保存{recorded_episodes}条数据", config.play_sounds, blocking=True)

    except Exception as e:
        # 捕获其他异常
        logger.log(f"Unexpected error during recording: {e}")
        logging.error(f"Recording failed with exception: {e}", exc_info=True)
        log_say("录制过程出现异常", config.play_sounds, blocking=True)

    finally:
        # 无论如何都要尝试记录batch信息（如果有已完成的episodes）
        _record_batch_info(dataset, config, recorded_episodes, session_start_time, batch_recorded, logger)


def _record_batch_info(dataset, config, recorded_episodes, session_start_time, batch_recorded, logger):
    """Record batch info to the upload system tracker."""
    if dataset is None or recorded_episodes <= 0 or batch_recorded:
        return
    try:
        from lerobot.common.data_tracker import BatchTracker

        # config.dataset.root 结构: /path/to/data_collection/YYYYMMDD/task_name/batch_id
        # 需要往上3级到达 data_collection 目录
        data_root = Path(config.dataset.root).parent.parent.parent
        tracker = BatchTracker(data_root)

        batch_info = tracker.record_batch_info(
            robot_type=config.robot.type,
            robot_id=config.robot.id,
            repo_id=config.dataset.repo_id,
            dataset_root=config.dataset.root,
            num_episodes=recorded_episodes,
            num_expected_episodes=config.dataset.num_episodes,
            session_start_time=session_start_time,
        )
        logger.log(f"Batch recorded: {batch_info.batch_id}")
    except Exception as e:
        logger.log(f"Failed to record batch info: {e}")
        logging.error(f"Failed to record batch info: {e}", exc_info=True)


################################################################################
# Main Entry Point（主入口）
################################################################################

@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset | None:
    """
    统一录制/推理主入口。
    
    根据 mode 参数：
    - "record": 纯录制模式，使用遥操作收集数据
    - "infer": 纯推理模式，无数据集，无限执行直到 stop_recording
    - "infer_record": 推理+录制模式，同时运行推理和数据收集
    """

    init_logging()
    logging.info(pformat(asdict(cfg)))

    # 初始化 TTS 服务（只检查一次）
    init_tts_service()

    # 设置日志
    now = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    log_dir = f"./unified_record_logs/{now}"
    logger = AsyncLogger(log_dir, enabled=cfg.enable_logging)
    logger.log(f"Starting unified record with config:\n{pformat(asdict(cfg))}")

    # 可视化
    if cfg.display_data:
        _init_rerun(session_name="unified_record")

    # 设置机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.log(f"Robot connected: {robot.name}")

    # 设置遥操作器（可选）
    teleop = None
    if cfg.teleop is not None:
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()
        logger.log("Teleoperator connected")

    # 设置策略客户端（仅推理模式）
    policy_client = setup_policy_client(robot, cfg, logger)

    # 设置数据集（纯推理模式不需要）
    dataset = setup_dataset(robot, cfg, logger)

    # 设置键盘监听
    listener, events = init_keyboard_listener()

    try:
        if cfg.mode == "infer":
            if policy_client is None:
                raise ValueError("Inference mode requires policy client")

            log_say("开始纯推理模式", cfg.play_sounds, blocking=True)
            run_inference_only_loop(
                robot=robot,
                policy_client=policy_client,
                config=cfg,
                events=events,
                logger=logger,
            )
        else:
            if dataset is None:
                raise ValueError("Recording mode requires dataset")
            run_recording_session(
                robot=robot,
                teleop=teleop,
                policy_client=policy_client,
                dataset=dataset,
                config=cfg,
                events=events,
                logger=logger,
            )
    except Exception as e:
        logger.log(f"Error during execution: {e}")
        raise
    finally:
        log_say("程序结束", cfg.play_sounds)
        robot.disconnect()
        if teleop is not None:
            teleop.disconnect()
        if not is_headless() and listener is not None:
            listener.stop()
        if dataset is not None and cfg.dataset.push_to_hub:
            dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
        logger.close()

    return dataset


def main():
    record()


if __name__ == "__main__":
    main()
