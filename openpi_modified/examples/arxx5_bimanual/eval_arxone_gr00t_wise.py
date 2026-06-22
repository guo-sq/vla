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

"""
This is the new Gr00T policy eval script with so100, so101 robot arm. Based on:
https://github.com/huggingface/lerobot/pull/777

Example command:

```shell

python eval_gr00t_so100.py \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM01 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ wrist: {type: opencv, index_or_path: 9, width: 640, height: 480, fps: 30}, front: {type: opencv, index_or_path: 15, width: 640, height: 480, fps: 30}}" \
    --lang_instruction="Grab markers and place into pen holder."
```


First replay to ensure the robot is working:
```shell
python -m lerobot.replay \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_awesome_follower_arm \
    --dataset.repo_id=youliangtan/so101-table-cleanup \
    --dataset.episode=2
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

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
    arx_x5_python,
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
        # assert (
        #     len(robot_state_keys) == 6
        # ), f"robot_state_keys should be size 6, but got {len(robot_state_keys)} "
        # self.modality_keys = ["left_arm", "left_gripper", "right_arm", "right_gripper"]
        # self.modality_keys = ["actions"]

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
        """
        print(f"---observation keys and shape---:")
        [print(k, value.shape, value.dtype, value.min(), value.max(), value.mean()) for k, value in observation.items() if isinstance(value, np.ndarray)]
        import pdb; pdb.set_trace() # exit()
        test_image = observation['observation/front_image'].transpose([1,2,0])[:,:,::-1]
        import cv2; cv2.imwrite("front_test.png", test_image)
        print("action keys(): ", action.keys())
        print("action/action.shape: ", action["actions"].shape)
        # print min/max and mean of each action dimension
        for i in range(action["actions"].shape[1]):
            print(f"action dimension {i}: min {np.min(action['actions'][:,i])}, max {np.max(action['actions'][:,i])}, mean {np.mean(action['actions'][:,i])}")

        """

        # convert the action chunk to a list of dict[str, float]
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


def print_yellow(text):
    print("\033[93m {}\033[00m".format(text))


@dataclass
class EvalConfig:
    robot: RobotConfig  # the robot to use
    policy_host: str = "localhost"  # host of the gr00t server
    policy_port: int = 8000  # port of the gr00t server
    action_horizon: int = 50  # number of actions to execute from the action chunk
    lang_instruction: str = "Grab the toy duck and pub it into white paper box."
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 20  # timeout in seconds
    show_images: bool = True  # whether to show images
    time_interval: float = 0.033  # whether to slow down actions
    debug: bool = False  # whether to show debug info


def filter_observation(observation: dict) -> dict:
    camera_keys = ["head", "right_wrist", "left_wrist"]
    filtered_observation = {}
    for key, value in observation.items():
        if key in camera_keys or ("joint" in key and "pos" in key):
            filtered_observation[key] = value
    return filtered_observation


def get_state_camera_keys(raw_keys):
    filtered_keys = []
    camera_keys = ["head", "right_wrist", "left_wrist"]
    for key in raw_keys:
        if key in camera_keys or ("joint" in key and "pos" in key):
            filtered_keys.append(key)
    return filtered_keys


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: Initialize the robot
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    # get camera keys from RobotConfig
    camera_keys = list(cfg.robot.cameras.keys())
    print("camera_keys: ", camera_keys)

    log_say("Initializing robot", cfg.play_sounds, blocking=True)

    language_instruction = cfg.lang_instruction

    # NOTE: for so100/so101, this should be:
    # ['shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos', 'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos']
    robot_state_keys = list(robot._motors_ft.keys())
    robot_state_keys = [
        key for key in robot_state_keys if "pos" in key and "joint" in key
    ]
    print("robot_state_keys: ", robot_state_keys)

    # Step 2: Initialize the policy
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=robot_state_keys,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 3: Run the Eval Loop
    while True:
        observation_dict = robot.get_observation()
        observation_dict = filter_observation(observation_dict)
        print(
            f"language_instruction: {language_instruction}, observation keys: {observation_dict.keys()}"
        )

        action_chunk = policy.get_action(observation_dict, language_instruction)

        execute_steps = len(action_chunk)
        for i in range(execute_steps):
            start_t = time.perf_counter()

            action_dict = action_chunk[i]
            if hasattr(robot, "send_action_inference"):
                robot.send_action_inference(action_dict)
            else:
                robot.send_action(action_dict)

            send_action_time = time.perf_counter() - start_t
            sleep_time = cfg.time_interval - send_action_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                logging.warning(f"Control loop loop overrun: {-sleep_time*1000:.2f}ms")


if __name__ == "__main__":
    eval()
