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
import draccus
from datetime import datetime

# from service import ExternalRobotInferenceClient

# from gr00t.eval.service import ExternalRobotInferenceClient


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

    def get_action(self, observation_dict, lang: str, robot_type: str):
        # first add the images
        obs_dict = {key: observation_dict[key] for key in self.camera_keys}

        # show images
        # if self.show_images:
        #     view_img(obs_dict)

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
            "robot_type": robot_type,
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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from typing import Dict, List, Any
import cv2


class StableRealtimeVisualizer:
    def __init__(self, joint_keys: List[str], camera_keys: List[str]):
        """
        稳定的实时可视化工具
        """
        self.joint_keys = joint_keys
        self.camera_keys = camera_keys
        self.time_steps = []

        # 存储历史数据
        self.obs_joint_data = {k: [] for k in joint_keys}
        self.act_joint_data = {k: [] for k in joint_keys}

        # 当前图像数据
        self.current_images = {k: None for k in camera_keys}

        # 创建画布和子图
        self.fig = plt.figure(figsize=(20, 10))

        # 创建网格布局：2行，第一行关节图，第二行两个相机图
        gs = plt.GridSpec(2, 2, figure=self.fig, height_ratios=[2, 1])
        self.ax_joint = self.fig.add_subplot(gs[0, :])  # 关节图占据第一行
        self.ax_head = self.fig.add_subplot(gs[1, 0])  # 头部相机在第二行左边
        self.ax_wrist = self.fig.add_subplot(gs[1, 1])  # 手腕相机在第二行右边

        plt.ion()
        plt.tight_layout(pad=3.0)

        # 初始化关节趋势图
        self.obs_lines = {}
        self.act_lines = {}
        colors = plt.cm.tab10(np.linspace(0, 1, len(joint_keys)))

        for idx, key in enumerate(joint_keys):
            # 观测数据线
            (obs_line,) = self.ax_joint.plot(
                [], [], color=colors[idx], linewidth=2.5, label=f"Obs: {key}", alpha=0.9
            )
            # 动作数据线
            (act_line,) = self.ax_joint.plot(
                [],
                [],
                color=colors[idx],
                linewidth=2,
                linestyle="--",
                label=f"Act: {key}",
                alpha=0.7,
            )
            self.obs_lines[key] = obs_line
            self.act_lines[key] = act_line

        # 关节图配置
        self.ax_joint.set_xlabel("Time Step", fontsize=12)
        self.ax_joint.set_ylabel("Joint Value", fontsize=12)
        self.ax_joint.set_title(
            "Real-Time Joint Observation vs Action", fontsize=14, fontweight="bold"
        )
        self.ax_joint.legend(loc="upper right", fontsize=8)
        self.ax_joint.grid(True, alpha=0.3)
        self.ax_joint.set_xlim(0, 50)
        self.ax_joint.set_ylim(-150, 150)

        # 初始化图像显示
        self.head_img_display = self._init_image_ax(self.ax_head, "Head Camera")
        self.wrist_img_display = self._init_image_ax(
            self.ax_wrist, "Right Wrist Camera"
        )

        # 强制初始显示
        self.fig.canvas.draw()
        plt.pause(1)
        print("Visualizer initialized successfully")

    def _init_image_ax(self, ax, title: str):
        """初始化图像子图"""
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        # 创建初始图像 - 使用彩色提示
        init_img = np.ones((240, 320, 3), dtype=np.uint8) * 128  # 灰色背景
        cv2.putText(
            init_img,
            "Waiting...",
            (80, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        img_display = ax.imshow(init_img)
        return img_display

    def _safe_process_image(self, img: Any, cam_key: str) -> np.ndarray:
        """安全处理图像数据"""
        if img is None:
            # 创建无数据图像
            no_data_img = np.ones((240, 320, 3), dtype=np.uint8) * 128
            cv2.putText(
                no_data_img,
                f"No {cam_key} Data",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            return no_data_img

        try:
            # 确保是numpy数组
            if not isinstance(img, np.ndarray):
                img = np.array(img)

            # 处理不同格式的图像
            if len(img.shape) == 2:  # 灰度图
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # 假设是BGR，转换为RGB
                pass
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 调整图像大小以提高性能
            if img.shape[0] > 480 or img.shape[1] > 640:
                img = cv2.resize(img, (320, 240))

            # 确保数据类型
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            return img

        except Exception as e:
            print(f"Error processing {cam_key} image: {e}")
            error_img = np.ones((240, 320, 3), dtype=np.uint8) * 128
            cv2.putText(
                error_img,
                f"Error: {cam_key}",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            return error_img

    def update_data(
        self, obs_dict: Dict[str, Any], act_dict: Dict[str, float], time_step: int
    ):
        """更新数据"""
        # 更新时间步
        self.time_steps.append(time_step)

        # 更新关节数据
        for key in self.joint_keys:
            # 观测数据
            obs_val = obs_dict.get(key, 0.0)
            if obs_val is not None:
                self.obs_joint_data[key].append(float(obs_val))
            else:
                self.obs_joint_data[key].append(0.0)

            # 动作数据
            act_val = act_dict.get(key, 0.0)
            if act_val is not None:
                self.act_joint_data[key].append(float(act_val))
            else:
                self.act_joint_data[key].append(0.0)

        # 更新图像数据
        for cam_key in self.camera_keys:
            img_data = obs_dict.get(cam_key)
            self.current_images[cam_key] = self._safe_process_image(img_data, cam_key)

    def refresh_plot(self):
        """刷新图表 - 使用更稳定的更新方法"""
        try:
            current_time_steps = len(self.time_steps)

            if current_time_steps > 0:
                # 更新关节数据线条
                for key in self.joint_keys:
                    # 确保数据长度匹配
                    obs_data = self.obs_joint_data[key][:current_time_steps]
                    act_data = self.act_joint_data[key][:current_time_steps]

                    self.obs_lines[key].set_data(
                        self.time_steps[:current_time_steps], obs_data
                    )
                    self.act_lines[key].set_data(
                        self.time_steps[:current_time_steps], act_data
                    )

                # 自动调整坐标轴
                max_time = max(self.time_steps) if self.time_steps else 50
                self.ax_joint.set_xlim(0, max(50, max_time + 5))

                # 计算y轴范围
                all_values = []
                for key in self.joint_keys:
                    all_values.extend(self.obs_joint_data[key][:current_time_steps])
                    all_values.extend(self.act_joint_data[key][:current_time_steps])

                if all_values:
                    y_min, y_max = min(all_values), max(all_values)
                    y_range = max(y_max - y_min, 10)  # 最小范围10
                    self.ax_joint.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

            # 更新图像显示 - 使用set_array而不是set_data
            if (
                "head" in self.current_images
                and self.current_images["head"] is not None
            ):
                self.head_img_display.set_array(self.current_images["head"])
                self.ax_head.set_title(
                    "Head Camera - Live", fontweight="bold", color="green"
                )

            if (
                "right_wrist" in self.current_images
                and self.current_images["right_wrist"] is not None
            ):
                self.wrist_img_display.set_array(self.current_images["right_wrist"])
                self.ax_wrist.set_title(
                    "Right Wrist Camera - Live", fontweight="bold", color="green"
                )

            # 使用更稳定的刷新方式
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(0.001)  # 极短的暂停

        except Exception as e:
            print(f"Refresh error: {e}")

    def close(self):
        """关闭可视化"""
        plt.ioff()
        plt.close(self.fig)


class SnapshotVisualizer:
    def __init__(
        self, joint_keys: List[str], camera_keys: List[str], save_dir: str = "snapshots"
    ):
        """
        快照可视化工具 - 每3秒保存一张图
        """
        self.joint_keys = joint_keys
        self.camera_keys = camera_keys
        self.save_dir = save_dir
        self.time_steps = []

        # 存储历史数据
        self.obs_joint_data = {k: [] for k in joint_keys}
        self.act_joint_data = {k: [] for k in joint_keys}

        # 当前图像数据
        self.current_images = {k: None for k in camera_keys}

        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)

        # 时间跟踪
        self.last_save_time = time.time()
        self.snapshot_count = 0

        print(
            f"Snapshot visualizer initialized. Snapshots will be saved to: {save_dir}"
        )

    def _safe_process_image(self, img: Any, cam_key: str) -> np.ndarray:
        """安全处理图像数据"""
        if img is None:
            # 创建无数据图像
            no_data_img = np.ones((240, 320, 3), dtype=np.uint8) * 128
            cv2.putText(
                no_data_img,
                f"No {cam_key} Data",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            return no_data_img

        try:
            # 确保是numpy数组
            if not isinstance(img, np.ndarray):
                img = np.array(img)

            # 处理不同格式的图像
            if len(img.shape) == 2:  # 灰度图
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                # 假设是BGR，转换为RGB
                pass
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 调整图像大小以提高性能
            if img.shape[0] > 480 or img.shape[1] > 640:
                img = cv2.resize(img, (320, 240))

            # 确保数据类型
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            return img

        except Exception as e:
            print(f"Error processing {cam_key} image: {e}")
            error_img = np.ones((240, 320, 3), dtype=np.uint8) * 128
            cv2.putText(
                error_img,
                f"Error: {cam_key}",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )
            return error_img

    def update_data(
        self, obs_dict: Dict[str, Any], act_dict: Dict[str, float], time_step: int
    ):
        """更新数据"""
        # 更新时间步
        self.time_steps.append(time_step)

        # 更新关节数据
        for key in self.joint_keys:
            # 观测数据
            obs_val = obs_dict.get(key, 0.0)
            if obs_val is not None:
                self.obs_joint_data[key].append(float(obs_val))
            else:
                self.obs_joint_data[key].append(0.0)

            # 动作数据
            act_val = act_dict.get(key, 0.0)
            if act_val is not None:
                self.act_joint_data[key].append(float(act_val))
            else:
                self.act_joint_data[key].append(0.0)

        # 更新图像数据
        for cam_key in self.camera_keys:
            img_data = obs_dict.get(cam_key)
            self.current_images[cam_key] = self._safe_process_image(img_data, cam_key)

    def check_and_save_snapshot(self):
        """检查是否需要保存快照（每3秒）"""
        current_time = time.time()
        if current_time - self.last_save_time >= 5.0:  # 每3秒保存一次
            self.save_snapshot()
            self.last_save_time = current_time
            return True
        return False

    def save_snapshot(self):
        """保存当前状态为图片"""
        try:
            self.snapshot_count += 1
            timestamp = datetime.now().strftime("%H%M%S")
            # filename = f"snapshot_{self.snapshot_count:04d}_{timestamp}.png"
            filename = "sanpshot.png"
            filepath = os.path.join(self.save_dir, filename)

            # 创建图形
            fig = plt.figure(figsize=(20, 12))

            # 创建网格布局
            gs = plt.GridSpec(2, 2, figure=fig, height_ratios=[2, 1])
            ax_joint = fig.add_subplot(gs[0, :])  # 关节图占据第一行
            ax_head = fig.add_subplot(gs[1, 0])  # 头部相机
            ax_wrist = fig.add_subplot(gs[1, 1])  # 手腕相机

            # 绘制关节趋势图
            current_time_steps = len(self.time_steps)
            if current_time_steps > 0:
                colors = plt.cm.tab10(np.linspace(0, 1, len(self.joint_keys)))

                for idx, key in enumerate(self.joint_keys):
                    # 确保数据长度匹配
                    obs_data = self.obs_joint_data[key][:current_time_steps]
                    act_data = self.act_joint_data[key][:current_time_steps]

                    time_points = self.time_steps[:current_time_steps]

                    # 观测数据线
                    ax_joint.plot(
                        time_points,
                        obs_data,
                        color=colors[idx],
                        linewidth=2.5,
                        label=f"Obs: {key}",
                        alpha=0.9,
                    )
                    # 动作数据线
                    ax_joint.plot(
                        time_points,
                        act_data,
                        color=colors[idx],
                        linewidth=2,
                        linestyle="--",
                        label=f"Act: {key}",
                        alpha=0.7,
                    )

            # 关节图配置
            ax_joint.set_xlabel("Time Step", fontsize=12)
            ax_joint.set_ylabel("Joint Value", fontsize=12)
            ax_joint.set_title(
                f"Joint Observation vs Action (Snapshot {self.snapshot_count}, Time: {current_time_steps} steps)",
                fontsize=14,
                fontweight="bold",
            )
            ax_joint.legend(loc="upper right", fontsize=8)
            ax_joint.grid(True, alpha=0.3)

            # 自动调整坐标轴
            if current_time_steps > 0:
                max_time = max(self.time_steps)
                ax_joint.set_xlim(0, max(50, max_time + 5))

                # 计算y轴范围
                all_values = []
                for key in self.joint_keys:
                    all_values.extend(self.obs_joint_data[key][:current_time_steps])
                    all_values.extend(self.act_joint_data[key][:current_time_steps])

                if all_values:
                    y_min, y_max = min(all_values), max(all_values)
                    y_range = max(y_max - y_min, 10)
                    ax_joint.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.1)

            # 绘制图像
            if (
                "head" in self.current_images
                and self.current_images["head"] is not None
            ):
                ax_head.imshow(self.current_images["head"])
                ax_head.set_title("Head Camera", fontsize=12, fontweight="bold")
            ax_head.axis("off")

            if (
                "right_wrist" in self.current_images
                and self.current_images["right_wrist"] is not None
            ):
                ax_wrist.imshow(self.current_images["right_wrist"])
                ax_wrist.set_title("Right Wrist Camera", fontsize=12, fontweight="bold")
            ax_wrist.axis("off")

            # 调整布局并保存
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches="tight")
            plt.close(fig)  # 关闭图形释放内存

            print(f"Saved snapshot: {filename}")

        except Exception as e:
            print(f"Error saving snapshot: {e}")

    def close(self):
        """保存最终快照并清理"""
        print("Saving final snapshot...")
        self.save_snapshot()
        print(f"Total snapshots saved: {self.snapshot_count}")


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
    action_horizon: int = 50  # number of actions to execute from the action chunk
    # lang_instruction: str = "Place the transparent cup on the tray and pour the water from the Baisuishan mineral water into the transparent cup."
    # lang_instruction: str = "Place the ceramic cup on the tray and pour the water from the Oriental Leaves into the ceramic cup.."
    lang_instruction: str = (
        "Place the goblet on the tray and pour the water from the green soda bottle into the goblet.."
    )
    robot_type: str = "bi_piper_follower"
    play_sounds: bool = False  # whether to play sounds
    timeout: int = 400  # timeout in seconds
    show_images: bool = True  # whether to show images
    time_interval: float = 0.033  # whether to slow down actions
    debug: bool = False  # whether to show debug info
    snapshot_dir: str = "/home/heyuan/work/openpi_modified/scripts/eval_real"


@draccus.wrap()
def eval(cfg: EvalConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    # Step 1: 初始化机器人
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    camera_keys = list(cfg.robot.cameras.keys())
    print("Camera keys:", camera_keys)

    # Step 2: 提取关节键
    init_obs = robot.get_observation()
    print("Initial observation keys:", list(init_obs.keys()))

    joint_keys = [k for k in init_obs.keys() if k not in camera_keys and ".pos" in k]
    print("Joint keys:", joint_keys)

    # Step 3: 初始化快照可视化工具
    print("Initializing snapshot visualizer...")
    # visualizer = SnapshotVisualizer(
    #     joint_keys=joint_keys,
    #     camera_keys=camera_keys,
    #     save_dir=cfg.snapshot_dir
    # )
    time_step = 0

    log_say("Initializing robot", cfg.play_sounds, blocking=True)
    language_instruction = cfg.lang_instruction
    robot_type = cfg.robot_type

    # Step 4: 初始化policy
    policy = Gr00tRobotInferenceClient(
        host=cfg.policy_host,
        port=cfg.policy_port,
        camera_keys=camera_keys,
        robot_state_keys=joint_keys,
    )
    log_say(
        "Initializing policy client with language instruction: " + language_instruction,
        cfg.play_sounds,
        blocking=True,
    )

    # Step 5: 运行主循环
    try:
        print("Starting main evaluation loop...")
        print("Snapshots will be saved every 3 seconds")
        start_time = time.time()

        while True:
            # 获取观测数据
            observation_dict = robot.get_observation()

            for k, v in observation_dict.items():
                print(f"Observation key: {k}, type: {type(v)}")

            # 获取动作数据
            action_chunk = policy.get_action(
                observation_dict, language_instruction, robot_type
            )

            # 执行动作序列
            for i in range(min(cfg.action_horizon, len(action_chunk))):
                action_dict = action_chunk[i]
                # print(f"observation keys: {list(observation_dict.keys())} action keys: {action_dict}")

                # 更新可视化数据
                # visualizer.update_data(
                #     obs_dict=observation_dict,
                #     act_dict=action_dict,
                #     time_step=time_step
                # )

                # 检查并保存快照（每3秒）
                # if visualizer.check_and_save_snapshot():
                #     elapsed_time = time.time() - start_time
                #     print(f"Progress: {time_step} steps, {elapsed_time:.1f} seconds elapsed")

                # 执行动作（按需启用）
                robot.send_action_inference(action_dict)
                print("----------------------------------------")
                print(action_dict)

                time_step += 1
                time.sleep(cfg.time_interval)

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("Cleaning up...")
        robot.disconnect()
        # visualizer.close()

        # 打印总结信息
        total_time = time.time() - start_time if "start_time" in locals() else 0
        print(f"\n=== Evaluation Summary ===")
        print(f"Total time steps: {time_step}")
        print(f"Total time: {total_time:.1f} seconds")
        # print(f"Total snapshots: {visualizer.snapshot_count}")
        print(f"Snapshots saved to: {cfg.snapshot_dir}")
        print("Done.")


if __name__ == "__main__":
    eval()
