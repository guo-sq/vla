import dataclasses
import logging
import os
import pathlib

import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import nnx_utils
from openpi.training import checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

"""

uv run scripts/policy_inference.py \
--action_horizon 30 \
--robot_type arxx5_bimanual \
--policy_dir checkpoints/pi05_base_pack_socks_data_0106_0224_h30_trtc8_filter_inter_static_ft_on_0225_0228_long_wo_jinlong_recover/pi05_base_pack_socks_data_0106_0224_h30_trtc8_filter_inter_static_ft_on_0225_0228_long_wo_jinlong_recover_exp_0228/29999 \
--repo_id pack_socks.purple.M.pair_s0s1s2.200s.20260212.batch.18

"""


logger = logging.getLogger(__name__)


@dataclasses.dataclass
class Args:

    exp_name: str | None = None
    policy_config: str | None = None
    policy_dir: str | None = None

    robot_type: str = "arxx5_bimanual"

    action_dim: int = 14
    fps: int = 30
    action_horizon: int = 30
    save_path: str | None = None
    dataset_root: str = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/pack_socks/"
    repo_id: str | None = None

    plot_mode: bool = True
    plot_episode_num: int = 2

    def __post_init__(self):
        """Check for required parameters after initialization."""
        missing_params = []
        if self.policy_dir is None:
            missing_params.append("policy_dir")
        if self.repo_id is None:
            missing_params.append("repo_id")

        if missing_params:
            # 给出明确的英文提示信息
            error_msg = (
                f"\n[Missing Required Arguments]: {', '.join(missing_params)}\n"
                f"You must provide values for both 'policy_dir' and 'repo_id' to run the script.\n"
                f"Example: python ./scripts/policy_inference.py --policy-dir checkpoints/pi05_base_finetune_data_1201_1215_delta_action/pi05_base_finetune_data_1201_1215_delta_action.exp.1215/59999 --repo-id grab_and_attach_tube.full_tray_52_tube.40s.1219.batch.1"
            )
            raise ValueError(error_msg)


def plot_14d_vertical_layout(
    pred_data,
    gt_data,
    loss,
    save_path="./14d_vertical_comparison.png",
    title="14-dimensional Time Series: Prediction vs Ground Truth",
    args=None,
):
    """
    Plots 14 subplots in vertical layout (1 per row) to compare prediction and ground truth
    """
    # 验证输入数据
    # if pred_data.shape != (50, 14) or gt_data.shape != (50, 14):
    #     raise ValueError("Both pred_data and gt_data must have shape (50, 14)")

    # 创建14行1列的子图布局
    fig, axes = plt.subplots(
        nrows=2,
        ncols=7,
        figsize=(35, 10),  # 宽度10，高度35（每行约2.5单位高度）
        # gridspec_kw={'hspace': 0.8}  # 行间距
        dpi=300,
    )
    fig.suptitle(title, fontsize=20, y=0.95)  # 大图标题

    # 时间步和样式配置
    time_steps = np.arange(pred_data.shape[0])
    gt_color = "#2E86AB"  # 真实值：深蓝色实线
    pred_color = "#E74C3C"  # 预测值：红色虚线
    line_width = 1.2

    # 为每个维度绘制子图
    for dim in range(args.action_dim):
        row, col = dim // 7, dim % 7
        ax = axes[row, col]  # 当前维度的子图

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
        ax.set_title(f"Dimension {dim + 1}, loss: {loss[dim].item():.4f}", fontsize=14, pad=10)
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


def plot_async(pred_lst, gt_lst, infer_steps, save_path):
    """
    绘制模型预测动作与Ground Truth的对比图。

    参数:
        pred_lst (list of np.ndarray): 预测动作列表，每个元素 shape 为 (H, 14)。
        gt_lst (list of np.ndarray): 真实动作列表，每个元素 shape 为 (H, 14)。
        infer_steps (list of int): 每次推理对应的起始时间步。
        save_path (str): 图片保存路径。
    """

    # 1. 基础设置
    D = 14  # 动作维度
    # 创建 7行 2列 的画布，增加高度以容纳多行
    fig, axs = plt.subplots(7, 2, figsize=(18, 24), sharex=True)

    # 定义预测曲线的交替颜色，以便区分相邻的推理步
    pred_colors = ["tab:red", "tab:blue"]
    # 定义GT的颜色 (使用灰色或绿色作为背景参考)
    gt_color = "gray"

    print(f"正在绘图，共 {len(pred_lst)} 个推理片段...")

    # 2. 遍历每一个维度进行绘图 (0~13)
    for dim in range(D):
        # 确定子图位置：第1列对应0-6，第2列对应7-13
        if dim <= 6:
            row = dim
            col = 0
        else:
            row = dim - 7
            col = 1

        ax = axs[row, col]

        # 3. 遍历列表中的每一次推理 (Chunk)
        for i, (pred, gt, start_t) in enumerate(zip(pred_lst, gt_lst, infer_steps)):
            # 确保数据是numpy array且为float类型
            pred = np.array(pred, dtype=float)
            gt = np.array(gt, dtype=float)

            H = pred.shape[0]  # 获取未来预测帧数

            # 生成横轴坐标：从 infer_step 开始
            x_vals = np.arange(start_t, start_t + H)

            # 获取当前维度的数据
            # 加上安全检查，防止维度不足
            if dim >= pred.shape[1] or dim >= gt.shape[1]:
                continue

            pred_vals = pred[:, dim]
            gt_vals = gt[:, dim]

            # --- 绘图操作 ---

            # A. 画 Ground Truth
            # 线宽 3.0，颜色统一 (alpha=0.4 防止太抢眼且允许重叠显示)
            ax.plot(
                x_vals,
                gt_vals,
                color=gt_color,
                linewidth=3.5,
                linestyle="-",
                marker=".",
                markersize=4,
                alpha=0.3,
                label="Ground Truth",
            )  # 仅在第一个添加图例标签

            # B. 画 Prediction
            # 线宽 1.5，颜色交替，alpha=0.9 保证清晰
            p_color = pred_colors[i % 2]
            ax.plot(
                x_vals,
                pred_vals,
                color=p_color,
                linewidth=1.5,
                linestyle="-",
                marker=".",
                markersize=2,
                alpha=1.0,
                label="Prediction",
            )

        # 4. 子图修饰
        ax.set_title(f"Dimension {dim}")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_ylabel("Value")

        # 仅在每个维度的第一次绘制时添加图例，防止图例重复
        # 这里我们在 row=0 的时候统一加一次图例即可
        if row == 0:
            # 重新获取句柄以去除重复标签
            handles, labels = ax.get_legend_handles_labels()
            # 简单去重逻辑：只取前两个（GT 和 Pred）
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize="small")

    # 5. 全局修饰
    # 设置底部的横轴标签
    axs[6, 0].set_xlabel("Frame Index (Time Step)")
    axs[6, 1].set_xlabel("Frame Index (Time Step)")

    plt.suptitle(
        f"Action Trajectories: Prediction (Red/Blue) vs Ground Truth ({gt_color})",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.98])  # 留出顶部标题空间

    # 6. 保存图片
    # 确保目录存在
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.savefig(save_path, dpi=200)
    plt.close(fig)  # 关闭画布释放内存
    print(f"绘图完成！图片已保存至: {save_path}")


def get_observation(dataset, idx):
    obs_img_head = dataset[idx]["observation.images.head"]
    obs_img_rt_wrist = dataset[idx]["observation.images.right_wrist"]
    obs_img_lf_wrist = dataset[idx]["observation.images.left_wrist"]
    prompt = dataset[idx]["task"]
    prompt = "Left hand grabs tube"
    state = dataset[idx]["observation.state"].numpy()

    return {
        "observation/state": state,
        "observation/front_image": (255 * obs_img_head.numpy()).astype(np.uint8),
        "observation/wrist_image": (255 * obs_img_rt_wrist.numpy()).astype(np.uint8),
        "observation/wrist_image_lf": (255 * obs_img_lf_wrist.numpy()).astype(np.uint8),
        "prompt": prompt,
    }


def create_policy(args: Args) -> _policy.Policy:
    """直接从检查点创建策略（原server功能）"""
    try:
        # 使用policy_config模块创建训练好的策略
        policy = _policy_config.create_trained_policy(
            _config.get_config(args.policy_config),
            args.policy_dir,
            sample_kwargs={"num_steps": 10},
            default_prompt=None,  # 可以根据需要添加default_prompt参数
            unify_action_mode=False,
            robot_type=args.robot_type,  # refer src/openpi/training/utils.py
        )
        logger.info(f"Successfully loaded policy from {args.policy_dir}")
        return policy
    except Exception as e:
        logger.error(f"Failed to create policy: {e}")
        raise


class PolicyInfer:
    def __init__(
        self,
        model: _model.BaseModel,
    ):
        self._model = model
        self._sample_actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("run_subtask_inference",))
        self._rng = jax.random.key(0)

    def infer(self, observation, num_steps=10) -> dict:
        # inputs = jax.tree.map(lambda x: x, obs)
        # inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)

        # observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        outputs = {
            # "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, num_steps=5),
        }
        model_time = time.monotonic() - start_time
        outputs["infer_timing_ms"] = model_time * 1000
        return outputs


def create_policy_simple(train_config, checkpoint_dir) -> _policy.Policy:
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
    # Check if this is a PyTorch model by looking for model.safetensors
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    return PolicyInfer(model)


def create_output_transforms(config, checkpoint_dir):
    if isinstance(checkpoint_dir, str):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
    data_config = config.data.create(config.assets_dirs, config.model)
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
    output_transforms = [
        *data_config.model_transforms.outputs,
        _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
        *data_config.repack_transforms.outputs,
    ]
    return _transforms.compose(output_transforms)


def infer_on_dataset(args):
    policy = create_policy(args)

    data_dir = os.path.join(args.dataset_root, args.repo_id)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=data_dir,
        delta_timestamps={"action": [t / args.fps for t in range(args.action_horizon)]},
    )
    print(f"repo_id: {args.repo_id}, data length: {len(dataset)}")
    print(f"data keys: {dataset[0].keys()}, prompt: {dataset[0]['task']}")

    pred_final, gt_final = [], []
    pred_lst = []
    gt_lst = []
    last_episode_idx = -1
    eval_steps = len(dataset) // args.action_horizon
    for idx in tqdm.trange(eval_steps, desc="Running policy"):
        data_idx = idx * args.action_horizon
        episode_index = dataset[data_idx]["episode_index"].item()

        observation = get_observation(dataset, data_idx)

        outputs = policy.infer(observation)

        pred_lst.append(outputs["actions"])
        gt_lst.append(dataset[data_idx]["action"])

        if last_episode_idx != episode_index:
            if last_episode_idx >= 0:
                pred_final.extend(pred_lst)
                gt_final.extend(gt_lst)

                pred_lst = np.concatenate(pred_lst, axis=0)
                gt_lst = np.concatenate(gt_lst, axis=0)

                output_dir = f"eval_results/{args.save_path}/{args.repo_id}"
                os.makedirs(output_dir, exist_ok=True)

                loss = np.mean(np.abs(pred_lst - gt_lst), axis=0)
                plot_14d_vertical_layout(
                    pred_lst,
                    gt_lst,
                    loss,
                    save_path=f"{output_dir}/episode_{last_episode_idx}_position.png",
                    title="Joint Position: Prediction vs Ground Truth",
                    args=args,
                )
                plot_14d_vertical_layout(
                    np.gradient(pred_lst, 1 / args.fps, axis=0),
                    np.gradient(gt_lst, 1 / args.fps, axis=0),
                    loss,
                    save_path=f"{output_dir}/episode_{last_episode_idx}_velocity.png",
                    title="Joint Velocity: Prediction vs Ground Truth",
                    args=args,
                )
                pred_lst = []
                gt_lst = []

            last_episode_idx = episode_index
            print(f"new episode index: {last_episode_idx}")
            if last_episode_idx >= args.plot_episode_num:
                break

    # 处理最后一个episode
    if pred_lst:
        pred_final.extend(pred_lst)
        gt_final.extend(gt_lst)

    if pred_final:
        pred_final = np.concatenate(pred_final, axis=0)
        gt_final = np.concatenate(gt_final, axis=0)
        loss = np.mean(np.abs(pred_final - gt_final), axis=0)
        print(f"eval loss: {loss}")
    print()


def infer_on_dataset_async(args):
    policy = create_policy(args)

    data_dir = os.path.join(args.dataset_root, args.repo_id)
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=data_dir,
        delta_timestamps={"action": [t / args.fps for t in range(args.action_horizon)]},
    )

    print(f"repo_id: {args.repo_id}, data length: {len(dataset)}")
    print(f"data keys: {dataset[0].keys()}, prompt: {dataset[0]['task']}")

    pred_final, gt_final = [], []
    pred_lst = []
    gt_lst = []
    last_episode_idx = -1
    eval_steps = len(dataset) // args.action_horizon
    infer_interval = 25
    infer_delay = 5
    infer_steps = []
    for idx in tqdm.trange(eval_steps, desc="Running policy"):
        data_idx = idx * infer_interval
        episode_index = dataset[data_idx]["episode_index"].item()

        action_prefix = np.zeros((args.action_horizon, args.action_dim))
        if len(pred_lst) > 0 and infer_delay > 0:
            prefix_len = args.action_horizon - infer_interval
            action_prefix[:prefix_len, :] = pred_lst[-1][infer_interval:]

        observation = get_observation(dataset, data_idx)
        observation.update(
            {
                "action": action_prefix,
                "action_mask": np.ones(args.action_horizon),
                "infer_delay": infer_delay,
                "robot_type": args.robot_type,
            }
        )

        outputs = policy.infer(observation)

        infer_steps.append(data_idx)
        pred_lst.append(outputs["actions"])
        gt_lst.append(dataset[data_idx]["action"])

        if last_episode_idx != episode_index:
            if last_episode_idx >= 0:
                output_dir = f"eval_results/{args.save_path}/{args.repo_id}/infer_interval_{infer_interval}_infer_delay_{infer_delay}"
                os.makedirs(output_dir, exist_ok=True)

                plot_async(
                    pred_lst,
                    gt_lst,
                    infer_steps,
                    save_path=f"{output_dir}/episode_{last_episode_idx}_position.png",
                )

                pred_final.extend(pred_lst)
                gt_final.extend(gt_lst)

                pred_lst = np.concatenate(pred_lst, axis=0)
                gt_lst = np.concatenate(gt_lst, axis=0)

                loss = np.mean(np.abs(pred_lst - gt_lst), axis=0)
                # plot_14d_vertical_layout(
                #     pred_lst,
                #     gt_lst,
                #     loss,
                #     save_path=f"{output_dir}/episode_{last_episode_idx}_position.png",
                #     title="Joint Position: Prediction vs Ground Truth",
                #     args=args,
                # )
                # plot_14d_vertical_layout(
                #     np.gradient(pred_lst, 1 / args.fps, axis=0),
                #     np.gradient(gt_lst, 1 / args.fps, axis=0),
                #     loss,
                #     save_path=f"{output_dir}/episode_{last_episode_idx}_velocity.png",
                #     title="Joint Velocity: Prediction vs Ground Truth",
                #     args=args,
                # )
                pred_lst = []
                gt_lst = []

            last_episode_idx = episode_index
            print(f"new episode index: {last_episode_idx}")
            if last_episode_idx >= args.plot_episode_num:
                break

    # 处理最后一个episode
    if pred_lst:
        pred_final.extend(pred_lst)
        gt_final.extend(gt_lst)

    if pred_final:
        pred_final = np.concatenate(pred_final, axis=0)
        gt_final = np.concatenate(gt_final, axis=0)
        loss = np.mean(np.abs(pred_final - gt_final), axis=0)
        print(f"eval loss: {loss}")
    print()


def infer_on_dataloader(args):
    config = _config.get_config(args.policy_config)
    data_cfg_factory = dataclasses.replace(config.data, repo_id=args.repo_id)
    config = dataclasses.replace(config, data=data_cfg_factory)
    checkpoint_dir = args.policy_dir

    policy = create_policy_simple(config, checkpoint_dir)
    data_loader = _data_loader.create_data_loader(
        config,
    )
    output_transforms = create_output_transforms(config, checkpoint_dir)

    pred_lst = []
    gt_lst = []
    num_batches = len(data_loader._data_loader._data_loader)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    for i in tqdm.tqdm(range(num_batches), desc="Processing Batches"):
        observation, actions_gt, actions_mask = batch
        outputs = policy.infer(observation)

        inputs_dict = observation.to_dict()
        inputs_dict["actions"] = actions_gt
        inputs_dict["actions_mask"] = actions_mask

        outputs["state"] = inputs_dict["state"]

        pred = output_transforms(outputs)
        gt = output_transforms(inputs_dict)

        pred_lst.append(pred)
        gt_lst.append(gt)
        batch = next(data_iter)

        # print(f"step: {step}, infer_timing: {outputs['infer_timing_ms']:.1f}ms")

    pred_final = np.concatenate(pred_lst, axis=0)
    gt_final = np.concatenate(gt_lst, axis=0)
    loss = np.mean(np.abs(pred_final - gt_final), axis=0)
    print(f"eval loss: {loss}")


def args_preprocess(args: Args):
    items = args.policy_dir.split("/")
    if "cfg" in items[1]:
        args.policy_config = f"src/openpi/configs/{items[1]}.py"
    else:
        args.policy_config = f"src/openpi/configs/cfg_{items[1]}.py"
    args.exp_name = f"{items[2]}/{items[3]}"
    args.save_path = args.exp_name
    print("\n--Args--")
    for field_name, value in dataclasses.asdict(args).items():
        print(f"{field_name}: {value}")
    print()


def main(args: Args) -> None:
    args_preprocess(args)
    if args.plot_mode:
        infer_on_dataset_async(args)
    else:
        infer_on_dataloader(args)


if __name__ == "__main__":
    import time

    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
