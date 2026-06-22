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
记录数据集，支持策略推理失败后切换到遥操录制模式。

工作流程：
1. 首先让模型进行推理控制机械臂，此阶段不记录数据
2. 当模型推理到失败状态时，按"r"键切换到遥操模式
3. 遥操模式下由人控制机械臂，此阶段的所有数据都会被录制

Example:

```shell
lerobot-record-policy-scale \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{laptop: {type: opencv, camera_index: 0, width: 640, height: 480}}" \
    --robot.id=black \
    --dataset.repo_id=aliberts/record-test \
    --dataset.num_episodes=2 \
    --dataset.single_task="Grab the cube" \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --teleop.id=blue \
    --policy_host=localhost \
    --policy_port=8000


python -m lerobot.record_policy_scale   \  
--robot.type=bi_piper_follower   \  
--robot.left_arm_port=can3   \  
--robot.right_arm_port=can0   \  
--robot.cameras="{head: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30},left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30},right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}"  \   
--robot.id=my_awesome_bi_piper_follower_arm3  \   
--teleop.type=bi_piper_leader  \   
--teleop.left_arm_port=can2   \  
--teleop.right_arm_port=can1   \  
--teleop.id=my_awesome_bi_piper_leader_arm3  \   
--dataset.repo_id=1210_bi/record.clothes.bipiper.v0112.policy.1  \   
--dataset.num_episodes=5   \  
--dataset.episode_time_s=100   \  
--dataset.reset_time_s=10   \  
--dataset.single_task="fold clothes。"  \   
--policy_host=localhost  \   
--policy_port=8001   \  
--play_sounds=true 
```
"""

import logging

import time
import threading
from copy import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import requests
import io
import pygame


from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
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
    so100_follower,
    so101_follower,
    piper,
)
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
    is_headless,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.utils.visualization_utils import _init_rerun, log_rerun_data

# pi0
from openpi_client import websocket_client_policy as _websocket_client_policy
import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
import numpy as np
import torch

import os
os.environ["SDL_AUDIODRIVER"] = "dummy"


def play_sound_seq(text: str = "你好啊，我是三零九零。", reference_id: str = "taozi"):
    """播放TTS音频序列"""
    url = "http://192.168.110.132:8080/v1/tts"
    payload = {"text": text,
               "format": "wav",
               "reference_id": reference_id}
    wav_bytes = requests.post(url, json=payload).content

    pygame.mixer.init()
    sound = pygame.mixer.Sound(io.BytesIO(wav_bytes))

    sound.set_volume(4200.0)   # ← 这里调音量，1.0 是原始，2.0 放大一倍
    sound.play()

    time.sleep(sound.get_length() + 0.5)  # 多给 0.5 s 缓冲
    pygame.mixer.quit()


def play_sound(text: str = "你好啊，我是三零九零。", reference_id: str = "taozi"):
    """在单独线程中播放TTS音频"""
    thread = threading.Thread(target=play_sound_seq, args=(text, reference_id))
    thread.start()


class Gr00tRobotInferenceClient:
    """The exact keys used is defined in modality.json

    This currently only supports so100_follower, so101_follower
    modify this code to support other robots with other keys based on modality.json
    """

    def __init__(
        self,
        host="localhost",
        port=8000,
        camera_keys=[],
        robot_state_keys=[],
        show_images=True,
    ):
        self.policy = _websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port,
            )
        self.camera_keys = camera_keys
        self.robot_state_keys = robot_state_keys
        self.show_images = show_images

    def get_action(self, observation_dict, lang: str):
        # first add the images
        obs_dict = {key: observation_dict["observation.images."+key] for key in self.camera_keys}

        # show images
        # if self.show_images:
        #     view_img(obs_dict)

        for k in list(obs_dict.keys()):
            if "_depth" in k:
                del obs_dict[k]

        for cam_name in obs_dict:
            img = obs_dict[cam_name].squeeze(0)
            obs_dict[cam_name] = einops.rearrange(img, "h w c -> c h w")

        state = observation_dict['observation.state']

        observation = {
            "observation/front_image": obs_dict["head"],
            "observation/wrist_image": obs_dict["right_wrist"],
            "observation/wrist_image_lf": obs_dict["left_wrist"],
            "observation/state": state,
            "prompt": lang,
        }

        observation_native = observation
        for k, v in observation_native.items():
            observation_native[k] = v.cpu().detach().numpy() if hasattr(v, "cpu") else v
            if "image" in k:
                observation_native[k] = observation_native[k].transpose(2, 0, 1)  # HWC to CHW
            if "state" in k:
                observation_native[k] = observation_native[k].squeeze(axis=0)  # remove extra dims

        action = self.policy.infer(observation_native)

        # convert the action chunk to lerobot_modified/lerobot/1210_bi/record.clothes.bipiper.v0109.policy.1/metaa list of dict[str, float]
        lerobot_actions = []
        action_horizon = action["actions"].shape[0]
        for idx in range(action_horizon):
            action_dict = self._convert_to_lerobot_action(action["actions"], idx)
            lerobot_actions.append(action_dict)
        return lerobot_actions

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """
        This is a magic function that converts the action chunk to a dict[str, float]
        This is because the action chunk is a dict[str, np.array]
        and we want to convert it to a dict[str, float]
        so that we can send it to the robot
        """
        assert action_chunk.shape[-1] == len(self.robot_state_keys), "this should be size 6"
        # convert the action to dict[str, float]
        action_dict = {key: action_chunk[idx, i] for i, key in enumerate(self.robot_state_keys)}
        return action_dict


@dataclass
class DatasetRecordConfig:
    # Dataset identifier. By convention it should match '{hf_username}/{dataset_name}' (e.g. `lerobot/test`).
    repo_id: str
    # A short but accurate description of the task performed during the recording (e.g. "Pick the Lego block and drop it in the box on the right.")
    single_task: str
    # Root directory where the dataset will be stored (e.g. 'dataset/path').
    root: str | Path | None = None
    # Limit the frames per second.
    fps: int = 30
    # Number of seconds for data recording for each episode.
    episode_time_s: int | float = 60
    # Number of seconds for resetting the environment after each episode.
    reset_time_s: int | float = 60
    # Number of episodes to record.
    num_episodes: int = 50
    # Encode frames in the dataset into video
    video: bool = True
    # Upload dataset to Hugging Face hub.
    push_to_hub: bool = True
    # Upload on private repository on the Hugging Face hub.
    private: bool = False
    # Add tags to your dataset on the hub.
    tags: list[str] | None = None
    # Number of subprocesses handling the saving of frames as PNG. Set to 0 to use threads only;
    # set to ≥1 to use subprocesses, each using threads to write images. The best number of processes
    # and threads depends on your system. We recommend 4 threads per camera with 0 processes.
    # If fps is unstable, adjust the thread count. If still unstable, try using 1 or more subprocesses.
    num_image_writer_processes: int = 0
    # Number of threads writing the frames as png images on disk, per camera.
    # Too many threads might cause unstable teleoperation fps due to main thread being blocked.
    # Not enough threads might cause low camera fps.
    num_image_writer_threads_per_camera: int = 4
    # Number of episodes to record before batch encoding videos
    # Set to 1 for immediate encoding (default behavior), or higher for batched encoding
    video_encoding_batch_size: int = 1

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task as argument in `single_task`.")


def predict_action(
    observation: dict[str, np.ndarray],
    policy: Gr00tRobotInferenceClient,
    device: torch.device,
    use_amp: bool,
    task: str | None = None,
    robot_type: str | None = None,
):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda"),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        for name in observation:
            observation[name] = torch.from_numpy(observation[name])
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        observation["task"] = task if task else ""
        observation["robot_type"] = robot_type if robot_type else ""

        # Compute the next actiosynn with the policy
        action = policy.get_action(observation, task)

    return action


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Whether to control the robot with a teleoperator
    teleop: TeleoperatorConfig | None = None
    # Whether to control the robot with a policy
    policy: PreTrainedConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    policy_host: str = "localhost"
    policy_port: int = 8000

    # 姿态同步参数
    sync_duration_s: float = 5.0  # 姿态同步持续时间（秒）
    sync_fps: int = 30  # 同步阶段的帧率

    def __post_init__(self):
        # HACK: We parse again the cli args here to get the pretrained path if there was one.
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


def init_keyboard_listener_with_rescue():
    """
    初始化键盘监听器，添加"r"键用于切换到遥操录制模式。
    
    返回:
        listener: 键盘监听器对象
        events: 事件字典，包含:
            - exit_early: 提前退出循环
            - rerecord_episode: 重新录制当前episode
            - stop_recording: 停止录制
            - switch_to_teleop: 切换到遥操模式（按"r"键触发）
    """
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False
    events["switch_to_teleop"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
            elif hasattr(key, "char") and key.char == "r":
                print("Key 'r' pressed. Switching to teleoperation recording mode...")
                events["switch_to_teleop"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


@safe_stop_image_writer
def policy_inference_loop(
    robot: Robot,
    events: dict,
    fps: int,
    policy: Gr00tRobotInferenceClient,
    obs_features: dict,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    chunk_exe: int = 40,
):
    """lerobot_modified/lerobot/1210_bi/record.clothes.bipiper.v0109.policy.1/meta
    策略推理循环：只进行模型推理控制机械臂，不记录数据。
    当按下"r"键时，会设置events["switch_to_teleop"]为True，退出循环。
    
    Args:
        robot: 机器人对象
        events: 事件字典
        fps: 帧率
        policy: 策略推理客户端
        obs_features: 观察特征字典（用于构建观察帧）
        control_time_s: 控制时间（秒），None表示无限运行直到按"r"键
        single_task: 任务描述
        display_datlerobot_modified/lerobot/1210_bi/record.clothes.bipiper.v0109.policy.1/metaa: 是否显示数据
        chunk_exe: 动作块执行大小
    """
    timestamp = 0
    start_episode_t = time.perf_counter()
    action_values = None
    count_frames = 0
    
    logging.info("Policy inference mode: Running policy without recording. Press 'r' to switch to teleoperation recording.")
    
    while True:
        start_loop_t = time.perf_counter()

        if events["exit_early"] or events["stop_recording"]:
            events["exit_early"] = False
            break
        
        # 检查是否切换到遥操模式
        if events["switch_to_teleop"]:
            logging.info("Switching to teleoperation recording mode...")
            events["switch_to_teleop"] = False
            break

        # 如果设置了控制时间，检查是否超时
        if control_time_s is not None and timestamp >= control_time_s:
            logging.info("Policy inference time limit reached.")
            break

        observation = robot.get_observation()

        # 构建观察帧（用于策略推理，但不保存到数据集）
        observation_frame = build_dataset_frame(
            obs_features,
            observation,
            prefix="observation"
        )

        # 策略推理
        if count_frames % chunk_exe == 0:
            action_values = predict_action(
                observation_frame,
                policy,
                device="cuda",
                use_amp=True,
                task=single_task,
                robot_type=robot.robot_type,
            )
            count_frames = 0
        
        cur_action_idx = int(count_frames % chunk_exe)
        cur_action = action_values[cur_action_idx]
        action = {key: cur_action[key] for i, key in enumerate(robot.action_features)}

        # 发送动作到机器人
        sent_action = robot.send_action(action)

        if display_data:
            log_rerun_data(observation, action)

        dt_s = time.perf_counter() - start_loop_t
        # busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t
        count_frames += 1


@safe_stop_image_writer
def pose_sync_loop(
    robot: Robot,
    teleop: Teleoperator,
    events: dict,
    sync_duration_s: float,
    sync_fps: int,
    display_data: bool = False,
):
    """
    姿态同步循环：让从臂逐步移动到teleop臂的姿态，确保安全切换。

    Args:
        robot: 从臂机器人对象
        teleop: teleop设备
        events: 事件字典
        sync_duration_s: 同步持续时间（秒）
        sync_fps: 同步帧率
        display_data: 是否显示数据
    """
    logging.info(f"Starting pose synchronization for {sync_duration_s} seconds...")

    if events["exit_early"] or events["stop_recording"]:
        return

    # 获取teleop臂的初始姿态作为目标姿态
    teleop_action = teleop.get_action()
    target_pose = {key: teleop_action[key] for key in robot.action_features if key in teleop_action}

    # 获取从臂的当前姿态作为起始姿态
    current_observation = robot.get_observation()
    start_pose = {key: current_observation[key] for key in robot.action_features if key in current_observation}

    logging.info(f"Target pose: {target_pose}")
    logging.info(f"Start pose: {start_pose}")

    # 计算同步步数
    num_steps = int(sync_duration_s * sync_fps)
    step_duration = 1.0 / sync_fps

    for step in range(num_steps):
        if events["exit_early"] or events["stop_recording"]:
            break

        start_time = time.perf_counter()

        # 计算当前步的插值姿态
        t = step / (num_steps - 1)  # 0到1的插值参数

        # 线性插值
        current_pose = {}
        for key in robot.action_features:
            if key in start_pose and key in target_pose:
                start_val = start_pose[key]
                target_val = target_pose[key]
                current_pose[key] = start_val + t * (target_val - start_val)
            else:
                # 如果某个关节没有在teleop中，直接使用当前值
                current_pose[key] = start_pose.get(key, 0.0)

        # 发送插值姿态到从臂
        sent_action = robot.send_action(current_pose)

        if display_data:
            log_rerun_data(current_observation, current_pose)

        # 控制循环频率
        elapsed = time.perf_counter() - start_time
        if elapsed < step_duration:
            time.sleep(step_duration - elapsed)

        logging.debug(f"Pose sync step {step+1}/{num_steps}, t={t:.3f}")

    logging.info("Pose synchronization completed.")


@safe_stop_image_writer
def teleop_record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    dataset: LeRobotDataset,
    teleop: Teleoperator,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    """
    遥操录制循环：由人通过遥操设备控制机械臂，并记录所有数据。
    
    Args:
        robot: 机器人对象
        events: 事件字典
        fps: 帧率
        dataset: 数据集对象
        teleop: 遥操设备
        control_time_s: 控制时间（秒）
        single_task: 任务描述
        display_data: 是否显示数据
    """
    if dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    timestamp = 0
    start_episode_t = time.perf_counter()
    frame_index = 0

    logging.info("Teleoperation recording mode: All data will be recorded. Press right arrow to exit early.")

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        observation = robot.get_observation()

        observation_frame = build_dataset_frame(dataset.features, observation, prefix="observation")

        # 从遥操设备获取动作
        action = teleop.get_action()

        # 发送动作到机器人
        sent_action = robot.send_action(action)

        # 记录数据到数据集
        action_frame = build_dataset_frame(dataset.features, sent_action, prefix="action")
        frame = {**observation_frame, **action_frame}

        try:
            dataset.add_frame(frame, task=single_task)
            logging.debug(f"Frame {frame_index} added successfully")
        except Exception as e:
            logging.error(f"Failed to add frame {frame_index}: {e}")
            logging.error(f"Frame keys: {list(frame.keys())}")
            logging.error(f"Dataset features keys: {list(dataset.features.keys())}")
            raise

        if display_data:
            log_rerun_data(observation, action)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t
        frame_index += 1


# record入口
@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:

    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="recording")

    # 被操控的机器人
    robot = make_robot_from_config(cfg.robot)
    # 遥操设备（必须提供）
    if cfg.teleop is None:
        raise ValueError("Teleoperator must be provided for policy-scale recording mode.")
    teleop = make_teleoperator_from_config(cfg.teleop)

    action_features = hw_to_dataset_features(robot.action_features, "action", cfg.dataset.video)
    obs_features = hw_to_dataset_features(robot.observation_features, "observation", cfg.dataset.video)
    dataset_features = {**action_features, **obs_features}

    if cfg.resume:
        dataset = LeRobotDataset(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

        if hasattr(robot, "cameras") and len(robot.cameras) > 0:
            dataset.start_image_writer(
                num_processes=cfg.dataset.num_image_writer_processes,
                num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            )
        sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
    else:
        # Create empty dataset or load existing saved episodes
        sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            cfg.dataset.fps,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        )

    robot.connect()
    teleop.connect()

    # 连接到策略推理服务器
    camera_keys = list(cfg.robot.cameras.keys())
    init_obs = robot.get_observation()
    joint_keys = [k for k in init_obs.keys() if k not in camera_keys and '.pos' in k]
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=joint_keys,
    )

    listener, events = init_keyboard_listener_with_rescue()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            # 使用play_sound替代log_say
            print(f"Recording episode {recorded_episodes + 1}/{cfg.dataset.num_episodes}")
            if cfg.play_sounds:
                play_sound(f"请开始第{recorded_episodes + 1}个事件录制。")
            
            # 重置事件标志，确保每个episode从干净的状态开始
            events["switch_to_teleop"] = False
            events["exit_early"] = False
            
            # 阶段1: 策略推理（不记录数据）
            logging.info("=" * 60)
            logging.info(f"Episode {recorded_episodes + 1} - Phase 1: Policy inference (no recording)")
            logging.info("Press 'r' key when policy fails to switch to teleoperation recording")
            logging.info("=" * 60)
            if cfg.play_sounds:
                play_sound("策略推理模式，按r键切换到遥操录制。")
            
            policy_inference_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                policy=policy,
                obs_features=obs_features,
                control_time_s=None,  # 无限运行直到按"r"键
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            # 检查是否要停止录制
            if events["stop_recording"]:
                break

            # 阶段1.5: 姿态同步（安全过渡）
            logging.info("=" * 60)
            logging.info(f"Episode {recorded_episodes + 1} - Phase 1.5: Pose synchronization")
            logging.info("=" * 60)
            if cfg.play_sounds:
                play_sound(f"开始姿态同步，将在{cfg.sync_duration_s}秒内同步到遥操姿态。")

            pose_sync_loop(
                robot=robot,
                teleop=teleop,
                events=events,
                sync_duration_s=cfg.sync_duration_s,
                sync_fps=cfg.sync_fps,
                display_data=cfg.display_data,
            )

            # 阶段2: 遥操录制（记录所有数据）
            logging.info("=" * 60)
            logging.info(f"Episode {recorded_episodes + 1} - Phase 2: Teleoperation recording (all data will be recorded)")
            logging.info("=" * 60)
            if cfg.play_sounds:
                play_sound("进入遥操录制模式，所有数据将被记录。")

            # 确保图像写入器正在运行
            if dataset.image_writer is None:
                logging.warning("Image writer is not initialized, starting...")
                dataset.start_image_writer(
                    num_processes=cfg.dataset.num_image_writer_processes,
                    num_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                )

            teleop_record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                dataset=dataset,
                teleop=teleop,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data,
            )

            # Execute a few seconds without recording to give time to manually reset the environment
            # Skip reset for the last episode to be recorded
            if not events["stop_recording"] and (
                (recorded_episodes < cfg.dataset.num_episodes - 1) or events["rerecord_episode"]
            ):
                if cfg.play_sounds:
                    play_sound("请重置环境。")

            if events["rerecord_episode"]:
                if cfg.play_sounds:
                    play_sound("请重新录制。")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                events["switch_to_teleop"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1
            logging.info(f"Episode {recorded_episodes} saved successfully.")

    if cfg.play_sounds:
        play_sound("录制完成, 谢谢。")

    robot.disconnect()
    teleop.disconnect()

    if not is_headless() and listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    if cfg.play_sounds:
        play_sound("程序退出，谢谢。")
    return dataset


def main():
    record()


if __name__ == "__main__":
    main()


"""
使用说明
========

record_policy_scale.py 是一个支持策略推理失败后切换到遥操录制模式的数据采集工具。

工作流程：
---------
1. 阶段1 - 策略推理模式（不记录数据）：
   - 模型自动控制机械臂执行任务
   - 此阶段的所有数据都不会被记录到数据集中
   - 当模型推理失败或需要人工干预时，按 "r" 键切换到姿态同步模式

2. 阶段1.5 - 姿态同步模式（安全过渡）：
   - 从臂逐步移动到与teleop臂相同的姿态
   - 持续时间可配置（默认3秒），确保安全切换
   - 此阶段不记录数据

3. 阶段2 - 遥操录制模式（记录所有数据）：
   - 由操作员通过遥操设备控制机械臂
   - 此阶段的所有数据都会被完整记录到数据集中
   - 录制完成后自动保存episode并进入下一个episode

4. 下一个episode：
   - 自动回到阶段1，重复上述流程

键盘快捷键：
-----------
- 'r' 键：从策略推理模式切换到遥操录制模式
- 右箭头键：提前退出当前循环
- 左箭头键：重新录制当前episode
- ESC 键：停止整个录制过程

命令行使用示例：
---------------
```shell
# 基本用法

# 禁用语音提示
python -m lerobot.record_policy_scale \
    --robot.type=bi_piper_follower \
    --robot.left_arm_port=can3 \
    --robot.right_arm_port=can0 \
    --robot.cameras="{head: {type: opencv, index_or_path: 12, width: 640, height: 480, fps: 30},left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30},right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}" \
    --robot.id=my_awesome_bi_piper_follower_arm3 \
    --teleop.type=bi_piper_leader \
    --teleop.left_arm_port=can2 \
    --teleop.right_arm_port=can1 \
    --teleop.id=my_awesome_bi_piper_leader_arm3 \
    --dataset.repo_id=1210_bi/record.clothes.bipiper.v0109.policy.1 \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=100 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="fold clothes。" \
    --policy_host=localhost \
    --policy_port=8001 \
    --play_sounds=true
```

必需参数：
---------
- --robot.type: 机器人类型（如 so100_follower, so101_follower 等）
- --robot.port: 机器人串口设备路径
- --robot.cameras: 相机配置（JSON格式）
- --robot.id: 机器人ID
- --dataset.repo_id: 数据集仓库ID（格式：username/dataset_name）
- --dataset.num_episodes: 要录制的episode数量
- --dataset.single_task: 任务描述
- --teleop.type: 遥操设备类型（如 so100_leader）
- --teleop.port: 遥操设备串口路径
- --teleop.id: 遥操设备ID
- --policy_host: 策略推理服务器地址（默认：localhost）
- --policy_port: 策略推理服务器端口（默认：8000）

可选参数：
---------
- --play_sounds: 是否启用语音提示（默认：true）
- --display_data: 是否显示数据可视化（默认：false）
- --dataset.fps: 录制帧率（默认：30）
- --dataset.episode_time_s: 每个episode的录制时长（秒，默认：60）
- --dataset.reset_time_s: 环境重置时间（秒，默认：60）
- --dataset.video: 是否编码为视频（默认：true）
- --dataset.push_to_hub: 是否上传到Hugging Face Hub（默认：true）
- --sync_duration_s: 姿态同步持续时间（秒，默认：3.0）
- --sync_fps: 姿态同步帧率（默认：30）

注意事项：
---------
1. 确保策略推理服务器（policy_host:policy_port）正在运行
2. 确保遥操设备已正确连接并配置
3. 确保TTS服务运行在 http://127.0.0.1:8080/v1/tts（如果使用play_sounds）
4. 每个episode都会先运行策略推理，按"r"键后会进行安全的姿态同步
5. 姿态同步阶段会让从臂逐步移动到teleop臂的姿态，确保切换安全
6. 只有遥操录制模式下的数据会被记录到数据集中
7. 数据集会自动保存到本地，并可选择上传到Hugging Face Hub

数据格式：
---------
数据集包含以下信息：
- observation: 观察数据（图像、关节状态等）
- action: 动作数据（遥操模式下的动作）
- task: 任务描述
- episode_index: episode索引
- frame_index: 帧索引
- timestamp: 时间戳

应用场景：
---------
- 收集模型失败后的人工干预数据
- 训练模型学习从失败状态恢复
- 收集高质量的人工演示数据
- 评估模型性能并收集改进数据
"""
