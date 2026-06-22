import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def load_norm_stats(json_path):
    """加载归一化统计信息"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["norm_stats"]


def plot_robot_comparison(norm_stats, stat_type="mean", feature_type="state"):
    """
    比较不同机器人的指定统计量

    Args:
        norm_stats: 归一化统计字典
        stat_type: 统计类型 ('mean', 'std', 'q01', 'q99')
        feature_type: 特征类型 ('state', 'actions')
    """
    robots = list(norm_stats.keys())
    robot_data = []

    # 获取所有机器人的指定统计数据
    for robot in robots:
        robot_data.append(norm_stats[robot][feature_type][stat_type])

    robot_data = np.array(robot_data)

    # 创建热力图
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        robot_data,
        xticklabels=[f"Dim {i}" for i in range(robot_data.shape[1])],
        yticklabels=robots,
        annot=True,
        fmt=".3f",
        cmap="viridis",
    )
    plt.title(
        f"{feature_type.capitalize()} {stat_type.upper()} Comparison Across Robots"
    )
    plt.tight_layout()
    plt.savefig(f"./outputs/{feature_type}_{stat_type}_heatmap.png")


def plot_feature_distribution(norm_stats, feature_type="state"):
    """
    绘制特征分布的箱线图
    """
    robots = list(norm_stats.keys())
    stats_types = ["mean", "std", "q01", "q99"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, stat_type in enumerate(stats_types):
        # 准备数据用于箱线图
        all_robot_data = []
        labels = []

        for robot in robots:
            data = norm_stats[robot][feature_type][stat_type]
            all_robot_data.extend(data)
            labels.extend([robot] * len(data))

        # 创建DataFrame用于seaborn绘图
        import pandas as pd

        df = pd.DataFrame(
            {
                "Feature_Dimension": list(range(len(all_robot_data))),
                "Value": all_robot_data,
                "Robot": labels,
            }
        )

        sns.boxplot(data=df, x="Robot", y="Value", ax=axes[idx])
        axes[idx].set_title(
            f"{feature_type.capitalize()} {stat_type.upper()} Distribution"
        )
        axes[idx].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"./outputs/{feature_type}_feature_distribution.png")


def plot_robot_scatter_comparison(
    norm_stats, feature_type="state", stat_x="mean", stat_y="std"
):
    """
    散点图比较两个统计量之间的关系
    """
    robots = list(norm_stats.keys())
    fig, ax = plt.subplots(figsize=(12, 8))

    for robot in robots:
        x_vals = norm_stats[robot][feature_type][stat_x]
        y_vals = norm_stats[robot][feature_type][stat_y]
        ax.scatter(x_vals, y_vals, label=robot, alpha=0.7, s=60)

        # 添加标签
        for i in range(min(len(x_vals), 5)):  # 只标记前5个维度避免过于拥挤
            ax.annotate(f"Dim{i}", (x_vals[i], y_vals[i]), fontsize=8)

    ax.set_xlabel(stat_x.upper())
    ax.set_ylabel(stat_y.upper())
    ax.set_title(
        f"{feature_type.capitalize()} {stat_x.upper()} vs {stat_y.upper()} Comparison"
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # 修复路径拼写错误并确保目录存在
    save_path = Path(f"./outputs/{feature_type}_{stat_x}_{stat_y}_comparison.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path)


def calculate_stat_ranges(norm_stats, feature_type="state"):
    """计算每个机器人的数据范围"""
    robots = list(norm_stats.keys())
    ranges = {}

    for robot in robots:
        q01 = np.array(norm_stats[robot][feature_type]["q01"])
        q99 = np.array(norm_stats[robot][feature_type]["q99"])
        ranges[robot] = q99 - q01  # 数据范围

    return ranges


def plot_stat_ranges(norm_stats):
    """绘制各机器人数据范围的比较"""
    state_ranges = calculate_stat_ranges(norm_stats, "state")
    action_ranges = calculate_stat_ranges(norm_stats, "actions")

    robots = list(state_ranges.keys())
    n_dims_state = len(list(state_ranges.values())[0])
    n_dims_action = len(list(action_ranges.values())[0])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # State ranges
    state_data = np.array(list(state_ranges.values()))
    im1 = ax1.imshow(state_data, cmap="YlOrRd", aspect="auto")
    ax1.set_xticks(range(n_dims_state))
    ax1.set_yticks(range(len(robots)))
    ax1.set_xticklabels([f"Dim {i}" for i in range(n_dims_state)], rotation=45)
    ax1.set_yticklabels(robots)
    ax1.set_title("State Feature Ranges (q99 - q01)")

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label("Range")

    # Action ranges
    action_data = np.array(list(action_ranges.values()))
    im2 = ax2.imshow(action_data, cmap="YlOrRd", aspect="auto")
    ax2.set_xticks(range(n_dims_action))
    ax2.set_yticks(range(len(robots)))
    ax2.set_xticklabels([f"Dim {i}" for i in range(n_dims_action)], rotation=45)
    ax2.set_yticklabels(robots)
    ax2.set_title("Action Feature Ranges (q99 - q01)")

    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label("Range")

    plt.tight_layout()
    plt.savefig("./outputs/action_feature_ranges.png")


def plot_detailed_comparison(norm_stats):
    """绘制详细的统计比较图"""
    robots = list(norm_stats.keys())
    n_robots = len(robots)

    # 为每个特征维度绘制柱状图，比较不同机器人的统计值
    # 只绘制前几个维度以避免图表过长
    n_dims_to_plot = min(10, len(norm_stats[robots[0]]["state"]["mean"]))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    stats_types = ["mean", "std", "q01", "q99"]

    for idx, stat_type in enumerate(stats_types):
        # 为前n_dims_to_plot个维度创建比较图
        data_to_plot = []
        for dim_idx in range(n_dims_to_plot):
            dim_data = [
                norm_stats[robot]["state"][stat_type][dim_idx] for robot in robots
            ]
            data_to_plot.append(dim_data)

        # 转置以使机器人成为x轴
        data_to_plot = np.array(data_to_plot).T

        x = np.arange(len(robots))  # 机器人
        width = 0.8 / n_dims_to_plot  # 柱宽

        for dim_idx in range(n_dims_to_plot):
            offset = (dim_idx - n_dims_to_plot / 2) * width
            axes[idx].bar(
                x + offset, data_to_plot[:, dim_idx], width, label=f"Dim {dim_idx}"
            )

        axes[idx].set_xlabel("Robots")
        axes[idx].set_ylabel(stat_type.upper())
        axes[idx].set_title(f"State {stat_type.upper()} Comparison by Dimension")
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(robots, rotation=45)
        axes[idx].legend()

    plt.tight_layout()
    plt.savefig("./outputs/detailed_comparison.png")


def analyze_robot_statistics(json_path):
    """主分析函数"""
    norm_stats = load_norm_stats(json_path)

    print("开始分析机器人统计数据分布差异...")
    print(f"共发现 {len(norm_stats)} 个机器人数据集: {list(norm_stats.keys())}")

    # 输出各机器人统计数据的形状，确认是否相同
    for robot in norm_stats.keys():
        state_shape = np.array(norm_stats[robot]["state"]["mean"]).shape
        action_shape = np.array(norm_stats[robot]["actions"]["mean"]).shape
        print(f"{robot}: State shape: {state_shape}, Action shape: {action_shape}")

    # 分析状态特征
    print("\n=== 状态特征分析 ===")
    plot_feature_distribution(norm_stats, "state")

    # 分析动作特征
    print("\n=== 动作特征分析 ===")
    plot_feature_distribution(norm_stats, "actions")

    # 绘制统计量热力图
    print("\n=== 状态均值热力图 ===")
    plot_robot_comparison(norm_stats, "mean", "state")

    print("\n=== 动作均值热力图 ===")
    plot_robot_comparison(norm_stats, "mean", "actions")

    # 绘制散点图比较
    print("\n=== 状态特征散点图比较 ===")
    plot_robot_scatter_comparison(norm_stats, "state", "mean", "std")

    print("\n=== 动作特征散点图比较 ===")
    plot_robot_scatter_comparison(norm_stats, "actions", "mean", "std")

    # 绘制数据范围比较
    print("\n=== 数据范围比较 ===")
    plot_stat_ranges(norm_stats)

    # 绘制详细比较图
    print("\n=== 详细比较 ===")
    plot_detailed_comparison(norm_stats)

    print("\n分析完成!")


if __name__ == "__main__":
    # 使用提供的JSON文件路径
    json_file_path = "/mnt/workspace/jy/temp/openpi_modified/assets/all_valid_public_dataset_pretrain/20260127/norm_stats.json"

    if Path(json_file_path).exists():
        analyze_robot_statistics(json_file_path)
    else:
        print(f"错误: 文件 {json_file_path} 不存在")
