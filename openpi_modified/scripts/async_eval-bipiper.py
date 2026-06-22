# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import queue
import threading

from PIL import Image

import draccus
import matplotlib.pyplot as plt
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import (  # noqa: F401
    OpenCVCameraConfig,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
    piper,
    # arx_x5_python,
    bi_piper_follower,
)
from lerobot.utils.utils import (
    init_logging,
    log_say,
)

from openpi_client import websocket_client_policy as _websocket_client_policy
import einops
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

# NOTE:
# Sometimes we would like to abstract different env, or run this on a separate machine
# User can just move this single python class method gr00t/eval/service.py
# to their code or do the following line below
import sys
import os

sys.path.append(os.path.expanduser("/home/anyverse/playground/groot_modifed/"))
# from service import ExternalRobotInferenceClient

# from gr00t.eval.service import ExternalRobotInferenceClient

from datetime import datetime


enable_save_images = False
enable_save_actions = False
enable_save_logs = True

now = datetime.now()
now = now.strftime("%Y-%m-%d-%H:%M:%S")
output_dir = f"./async_inference_logs/{now}"
os.makedirs(output_dir, exist_ok=True)

if enable_save_images:
    image_output_dir = f"{output_dir}/images/"
    os.makedirs(image_output_dir, exist_ok=True)

log_queue = queue.Queue()


def logger_worker(filepath):
    while enable_save_logs:
        msg = log_queue.get()
        if msg is None:  # 哨兵信号
            break
        with open(filepath, "a", encoding="utf-8") as f:
            print(msg, file=f)


log_filepath = f"{output_dir}/async_inference.log"
threading.Thread(
    target=logger_worker,
    args=(log_filepath,),
    daemon=True,
).start()


def print_to_file(msg):
    log_queue.put(msg)


#################################################################################


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
        obs_dict = {key: observation_dict[key] for key in self.camera_keys}

        # show images
        if self.show_images:
            view_img(obs_dict)

        for k in list(obs_dict.keys()):
            if "_depth" in k:
                del obs_dict[k]

        for cam_name in obs_dict:
            img = obs_dict[cam_name]
            obs_dict[cam_name] = einops.rearrange(img, "h w c -> c h w")

        state = np.array([observation_dict[k] for k in self.robot_state_keys]).astype(
            np.float32
        )

        observation = {
            "observation/front_image": obs_dict["head"],
            "observation/wrist_image": obs_dict["right_wrist"],
            "observation/wrist_image_lf": obs_dict["left_wrist"],
            "observation/state": state,
            "prompt": lang,
        }

        action = self.policy.infer(observation)

        return action["actions"]  # [horizon, action_dim]

    def _convert_to_lerobot_action(
        self, action_chunk: dict[str, np.array], idx: int
    ) -> dict[str, float]:
        """
        This is a magic function that converts the action chunk to a dict[str, float]
        This is because the action chunk is a dict[str, np.array]
        and we want to convert it to a dict[str, float]
        so that we can send it to the robot
        """
        assert action_chunk.shape[-1] == len(
            self.robot_state_keys
        ), "this should be size 6"
        # convert the action to dict[str, float]
        action_dict = {
            key: action_chunk[idx, i] for i, key in enumerate(self.robot_state_keys)
        }
        return action_dict


#################################################################################


def view_img(img, overlay_img=None):
    """
    This is a matplotlib viewer since cv2.imshow can be flaky in lerobot env
    """
    if isinstance(img, dict):
        # stack the images horizontally
        img = np.concatenate([img[k] for k in img], axis=1)

    plt.imshow(img)
    plt.title("Camera View")
    plt.axis("off")
    plt.pause(0.01)  # Non-blocking show
    plt.clf()  # Clear the figure for the next frame


from dataclasses import field


@dataclass
class EvalConfig:
    robot: RobotConfig = field(
        default_factory=lambda: bi_piper_follower.BiPiperFollowerConfig(
            left_arm_port="can2",  # 默认端口
            right_arm_port="can3",
            cameras={  # 默认相机配置（根据实际需求调整）
                "head": OpenCVCameraConfig(
                    index_or_path=12, width=640, height=480, fps=30
                ),
                "right_wrist": OpenCVCameraConfig(
                    index_or_path=4, width=640, height=480, fps=30
                ),
                "left_wrist": OpenCVCameraConfig(
                    index_or_path=10, width=640, height=480, fps=30
                ),
            },
        )
    )
    policy_host: str = "127.0.0.1"  # host of the gr00t server
    policy_port: int = 8100  # port of the gr00t server
    action_horizon: int = 40  # number of actions to execute from the action chunk
    # lang_instruction: str = "Place the transparent cup on the tray and pour the water from the Baisuishan mineral water into the transparent cup."
    lang_instruction: str = (
        "Place the ceramic cup on the tray and pour the water from the Oriental Leaves into the ceramic cup.."
    )
    # lang_instruction: str = "Place the goblet on the tray and pour the water from the green soda bottle into the goblet.."
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 20  # timeout in seconds
    show_images: bool = True  # whether to show images
    time_interval: float = 0.033  # whether to slow down actions
    debug: bool = False  # whether to show debug info
    snapshot_dir: str = "/home/heyuan/work/openpi/scripts/eval_real"


class TimeAlignedActionBuffer:
    def __init__(
        self, robot_state_keys, action_dim=14, start_weight=0.9, end_weight=0.1
    ):
        self.robot_state_keys = robot_state_keys
        self.action_dim = action_dim
        self.start_weight = start_weight
        self.end_weight = end_weight
        self.available_action_cnt = 0
        if enable_save_actions:
            self.output_dir = f"{output_dir}/actions/"
            os.makedirs(self.output_dir, exist_ok=True)

    def empty(self):
        return self.available_action_cnt == 0

    def init_action_buffer(self, initial_chunk):
        self.buffer = np.copy(initial_chunk)
        self.available_action_cnt = len(initial_chunk)

        if enable_save_actions:
            output_path = f"{self.output_dir}/chunk_0.npy"
            np.save(output_path, self.buffer)
            print_to_file(f"Save init chunk to {output_path}")

            output_path = f"{self.output_dir}/buffer_0.npy"
            np.save(output_path, self.buffer)
            print_to_file(f"Save init buffer to {output_path}")
        print_to_file(f"init_action_buffer: {self.available_action_cnt} actions")

    def get_next_action(self):
        """
        取出当前步要执行的动作，并将 Buffer 整体左移一格
        """
        action = {}
        for i, key in enumerate(self.robot_state_keys):
            action[key] = self.buffer[0, i]

        # 滚动 Buffer：丢弃第0个，后面补0
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = 0.0  # 末尾补零

        self.available_action_cnt -= 1

        return action

    def update_from_inference(self, new_chunk, start_step_id, current_step_id):
        if enable_save_actions:
            output_path = f"{self.output_dir}/chunk_{start_step_id}.npy"
            np.save(output_path, new_chunk)
            print_to_file(f"Save new_chunk to {output_path}")

        latency_steps = current_step_id - start_step_id

        print_to_file(
            f"Updating Action Buffer: current_step_id={current_step_id}, start_step_id={start_step_id}, latency_steps={latency_steps}"
        )

        if latency_steps >= len(new_chunk):
            print_to_file(
                f"Inference too slow! Latency {latency_steps} > Chunk {len(new_chunk)}"
            )
            return

        valid_new_actions = new_chunk[latency_steps:]

        weights = np.linspace(
            self.start_weight, self.end_weight, num=self.available_action_cnt
        )
        for i, weight in enumerate(weights):
            print_to_file(f"Buffer action {i} weight: {weight:.3f}")
            self.buffer[i] = self.buffer[i] * weight + valid_new_actions[i] * (
                1 - weight
            )

        self.buffer[self.available_action_cnt : len(valid_new_actions)] = (
            valid_new_actions[self.available_action_cnt :]
        )
        self.available_action_cnt = len(valid_new_actions)

        if enable_save_actions:
            output_path = f"{self.output_dir}/buffer_{current_step_id}.npy"
            np.save(output_path, self.buffer[: self.available_action_cnt])
            print_to_file(f"Save buffer to {output_path}")


def inference_worker(policy, lang_instruction, input_queue, output_queue):
    """
    input_queue: (observation_dict, step_id)
    output_queue: (action_chunk, step_id)
    """
    while True:
        # 阻塞等待输入
        print_to_file(f"inference_worker: Waiting input queue...")
        obs_dict, step_id = input_queue.get()

        print_to_file(f"Running inference at step_id={step_id}...")
        t0 = time.perf_counter()
        action_chunk = policy.get_action(obs_dict, lang_instruction)
        t1 = time.perf_counter()
        inference_time = t1 - t0

        print_to_file(
            f"Populate output queue at step_id={step_id} with inference time = {inference_time*1e3:.2f}ms"
        )

        output_queue.put((action_chunk, step_id))


def image_save_worker(camera_keys, input_queue):
    while True and enable_save_images:
        print_to_file(f"image_save_worker: Waiting input queue...")
        obs_dict, step_id = input_queue.get()
        t1 = time.perf_counter()
        image = np.concatenate([obs_dict[k] for k in camera_keys], axis=1)
        image = Image.fromarray(image)
        output_path = f"{image_output_dir}/image_{step_id}.png"
        image.save(output_path)
        duration = time.perf_counter() - t1
        print_to_file(
            f"Saved image at step_id={step_id} to {output_path} in {duration*1e3:.1f}ms"
        )


def filter_observation(observation: dict) -> dict:
    camera_keys = ["head", "right_wrist", "left_wrist"]
    filtered_observation = {}
    for key, value in observation.items():
        if key in camera_keys or ("joint" in key and "pos" in key):
            filtered_observation[key] = value
    return filtered_observation


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()

    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    init_obs = robot.get_observation()
    print("Initial observation keys:", list(init_obs.keys()))
    camera_keys = list(cfg.robot.cameras.keys())

    robot_state_keys = [
        k for k in init_obs.keys() if k not in camera_keys and ".pos" in k
    ]
    print("robot_state_keys: ", robot_state_keys)

    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
        show_images=False,
    )

    # 通信队列
    # maxsize=1 保证推理线程不会积压过时的观测，只处理最新的
    inference_input_queue = queue.Queue(maxsize=1)
    inference_images_queue = queue.Queue(maxsize=1)
    inference_output_queue = queue.Queue()

    # 启动推理线程
    t_infer = threading.Thread(
        target=inference_worker,
        args=(
            policy,
            cfg.lang_instruction,
            inference_input_queue,
            inference_output_queue,
        ),
        daemon=True,
    )
    t_infer.start()

    t_image = threading.Thread(
        target=image_save_worker,
        args=(camera_keys, inference_images_queue),
        daemon=True,
    )
    t_image.start()

    # Action Buffer
    action_buffer = TimeAlignedActionBuffer(robot_state_keys=policy.robot_state_keys)

    # 预热：先手动跑一次推理，填满 Buffer，防止第一帧没动作
    logging.info("Warming up policy...")
    warmup_action_array = policy.get_action(
        robot.get_observation(), cfg.lang_instruction
    )

    action_buffer.init_action_buffer(warmup_action_array)

    logging.info("Starting Control Loop...")

    global_step_id = 0
    target_dt = cfg.time_interval * 0.9
    next_cycle_time = time.perf_counter()

    infer_interval = 30
    while True:
        print_to_file(f"\nNew loop: global_step_id: {global_step_id}")
        if global_step_id % infer_interval == 0:
            t0 = time.perf_counter()
            obs_dict = robot.get_observation()
            t1 = time.perf_counter()
            get_obs_time = t1 - t0
            print_to_file(f"get_obs_time: {get_obs_time*1e3:.2f}ms")

            print_to_file(f"Populate input queue at step_id={global_step_id}...")
            inference_input_queue.put_nowait((obs_dict, global_step_id))
            if enable_save_images:
                inference_images_queue.put_nowait((obs_dict, global_step_id))
        else:
            print_to_file(
                f"Skipping inference at step_id={global_step_id} since available_action_cnt = {action_buffer.available_action_cnt}."
            )

        if not inference_output_queue.empty():
            new_chunk, start_step = inference_output_queue.get_nowait()
            print_to_file(
                f"get new_chunk, start_step: {start_step}, global_step_id: {global_step_id}, inference_output_queue size: {inference_output_queue.qsize()}"
            )

            action_buffer.update_from_inference(
                new_chunk, start_step_id=start_step, current_step_id=global_step_id
            )

        if action_buffer.available_action_cnt > 0:
            action_to_execute = action_buffer.get_next_action()
        else:
            time.sleep(0.01)
            print_to_file(
                f"No available action to execute at step_id={global_step_id}, skipping this step."
            )
            continue
        robot.send_action_inference(action_to_execute)
        print(action_to_execute)
        print_to_file(f"Executing action at step_id={global_step_id}")

        global_step_id += 1
        next_cycle_time += target_dt
        sleep_time = next_cycle_time - time.perf_counter()

        print_to_file(f"sleep_time: {sleep_time*1e3:.2f}ms")
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            print_to_file(f"Control loop loop overrun: {-sleep_time*1000:.2f}ms")
            next_cycle_time = time.perf_counter()


if __name__ == "__main__":
    eval()
