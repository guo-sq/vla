import dataclasses
import enum
import logging
import pathlib
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

logger = logging.getLogger(__name__)


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "0.0.0.0"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # Number of steps to run the policy for.
    num_steps: int = 60
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    env: EnvMode = EnvMode.ALOHA_SIM


def plot_14d_timeseries(
    data, save_path="./14d_timeseries_plot.png", title="14-dimensional Time Series"
):
    """
    Plots 14-dimensional time series data and saves the image

    Parameters:
        data: numpy array with shape (50, 14), each row is a timestamp, each column is a dimension
        save_path: path to save the image (including filename and extension)
        title: plot title
    """
    # Check input data shape
    if data.shape != (50, 14):
        raise ValueError(f"Input data must have shape (50, 14), but got {data.shape}")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 6))

    # Generate 14 distinct colors using colormap
    colors = cm.rainbow(np.linspace(0, 1, 14))

    # Plot each dimension's curve
    time_steps = np.arange(50)  # Timestamps (0 to 49)
    for i in range(14):
        ax.plot(time_steps, data[:, i], color=colors[i], label=f"Dimension {i+1}")

    # Add title and labels (in English)
    ax.set_title(title, fontsize=15)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.7)

    # Add legend (placed outside the plot to avoid overlapping)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10)

    # Adjust layout to ensure all elements are visible
    plt.tight_layout()

    # Save the image
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Close figure to free memory

    print(f"Plot saved to: {save_path}")


def plot_14d_vertical_layout(
    pred_data,
    gt_data,
    save_path="./14d_vertical_comparison.png",
    title="14-dimensional Time Series: Prediction vs Ground Truth",
):
    """
    Plots 14 subplots in vertical layout (1 per row) to compare prediction and ground truth
    """
    # 验证输入数据
    # if pred_data.shape != (50, 14) or gt_data.shape != (50, 14):
    #     raise ValueError("Both pred_data and gt_data must have shape (50, 14)")

    # 创建14行1列的子图布局
    fig, axes = plt.subplots(
        nrows=14,
        ncols=1,
        figsize=(10, 35),  # 宽度10，高度35（每行约2.5单位高度）
        gridspec_kw={"hspace": 0.8},  # 行间距
    )
    fig.suptitle(title, fontsize=20, y=0.91)  # 大图标题

    # 时间步和样式配置
    time_steps = np.arange(300)
    gt_color = "#2E86AB"  # 真实值：深蓝色实线
    pred_color = "#E74C3C"  # 预测值：红色虚线
    line_width = 1.2

    # 为每个维度绘制子图
    for dim in range(14):
        ax = axes[dim]  # 当前维度的子图

        # 绘制真实值和预测值
        ax.plot(
            time_steps,
            gt_data[:, dim],
            color=gt_color,
            linewidth=line_width,
            label="Ground Truth",
        )
        ax.plot(
            time_steps,
            pred_data[:, dim],
            color=pred_color,
            linewidth=line_width,
            linestyle="--",
            label="Prediction",
        )

        # 子图配置
        ax.set_title(f"Dimension {dim + 1}", fontsize=14, pad=10)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=10, loc="upper right")
        ax.tick_params(axis="both", labelsize=10)

    # 调整布局
    # plt.tight_layout(rect=[0, 0.01, 1, 0.17])  # 预留顶部标题空间

    # 保存图像（高分辨率确保细节清晰）
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Vertical layout plot saved to: {save_path}")


def main(args: Args) -> None:
    # 返回一个随机的observation
    # 不同环境返回的结果的结构不一样

    """
    ALOHA/ALOHA_SIM的observation的格式
    {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }
    """

    dataset_root = "/media/xuwenda/4456BE5656BE4886/heyuan/playground/Isaac-GR00T/demo_data/record.arxx5_bimanual.right_arm_grab_toy_duck"
    dataset = LeRobotDataset(
        repo_id="my_local_robot_dataset1010",  # 自定义名称，必须非空且符合格式
        root=dataset_root,
        force_cache_sync=False,
    )
    print(dataset[0].keys())
    # print(dataset[0])

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    print(f"--Server metadata--: {policy.get_server_metadata()}")

    # Send a few observations to make sure the model is loaded.
    # for _ in range(2):
    #     observation = obs_fn()
    #     policy.infer(observation)

    # timing_recorder = TimingRecorder()
    chunk_size = 50

    pred_lst = []
    gt_lst = []
    for id in tqdm.trange(args.num_steps, desc="Running policy"):
        id_start = id * chunk_size
        state = dataset[id_start]["action"]
        print(f"---type state---:{type(state)}")
        obs_img_head = dataset[id_start]["observation.images.head"]
        obs_img_rt_wrist = dataset[id_start]["observation.images.right_wrist"]
        obs_img_lf_wrist = dataset[id_start]["observation.images.left_wrist"]
        prompt = dataset[id_start]["task"]

        observation = {
            "state": state.numpy(),
            "images": {
                "cam_high": obs_img_head.numpy().astype(np.uint8),
                "cam_low": np.zeros((3, 480, 640), dtype=np.uint8),
                "cam_left_wrist": obs_img_lf_wrist.numpy().astype(np.uint8),
                "cam_right_wrist": obs_img_rt_wrist.numpy().astype(np.uint8),
            },
            "prompt": prompt,
        }

        inference_start = time.time()
        # observation = obs_fn()
        # print(observation)
        action = policy.infer(observation)
        # action['actions'] shape [10, 7]
        # print(f"action:{action['actions'].shape}")
        # print(action)

        pred_lst.append(action["actions"])
        # ---action chunk gt---
        actions_list = []
        for i in range(chunk_size):
            action_tensor = dataset[id_start + i]["action"]
            # 转换为NumPy数组并添加到列表
            actions_list.append(action_tensor.numpy())
        actions_array = np.stack(actions_list, axis=0)
        gt_lst.append(actions_array)

        if id > 0 and (id + 1) % 6 == 0:

            pred_final = np.concatenate(pred_lst, axis=0)
            gt_final = np.concatenate(gt_lst, axis=0)
            print(f"pred_final:{pred_final.shape} gt_final:{gt_final.shape}")
            plot_14d_vertical_layout(
                pred_final,
                gt_final,
                save_path=f"/media/xuwenda/4456BE5656BE4886/heyuan/playground/up_to_date/openpi/examples/simple_client/actions_and_gt_{(id+1)//6}.png",
                title="14-dimensional Time Series: Prediction vs Ground Truth",
            )
            pred_lst = []
            gt_lst = []

        # save_path = "/media/xuwenda/4456BE5656BE4886/heyuan/playground/up_to_date/openpi/examples/simple_client/actions.png"
        # plot_14d_timeseries(action['actions'], save_path)
        # save_path_gt = "/media/xuwenda/4456BE5656BE4886/heyuan/playground/up_to_date/openpi/examples/simple_client/actions_gt.png"
        # plot_14d_timeseries(actions_array, save_path_gt)
        # timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))

    #     for key, value in action.get("server_timing", {}).items():
    #         timing_recorder.record(f"server_{key}", value)
    #     for key, value in action.get("policy_timing", {}).items():
    #         timing_recorder.record(f"policy_{key}", value)

    # timing_recorder.print_all_stats()

    # if args.timing_file is not None:
    #     timing_recorder.write_parquet(args.timing_file)


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(
                256, size=(3, 224, 224), dtype=np.uint8
            ),
            "cam_right_wrist": np.random.randint(
                256, size=(3, 224, 224), dtype=np.uint8
            ),
        },
        "prompt": "do something",
    }


def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image_left": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
