import os
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pdb
import matplotlib.pyplot as plt
import cv2

def read_data_infos(input_parquet_path: str):
    import pandas as pd
    df = pd.read_parquet(input_parquet_path)
    is_human_intervention = df["is_human_intervention"].to_numpy()
    action_np = np.stack(df["action"].to_numpy())
    return action_np, is_human_intervention

def vis_intervention_flag(
    input_video_path: list[tuple[str, str]],
    input_parquet_path: str,
    output_video_path: str,
    episode_idx: int,
):
    action_np, is_human_intervention_np = read_data_infos(input_parquet_path)
    add_border_to_video_with_intervention_flag(
        input_video_path,
        input_parquet_path,
        f"{output_video_path}/episode_{episode_idx:06d}.mp4",
        is_human_intervention_np,
    )
    visualize_action_with_intervention_flag(
        action_np,
        is_human_intervention_np,
        f"{output_video_path}/episode_{episode_idx:06d}.png",
    )
    
def visualize_action_with_intervention_flag(gt_seq, is_human_intervention, filename: str):
    """Visualize prediction vs GT for 14 dimensions over time.

    Args:
        pred_seq: array-like shape (T, action_dim) predicted sequence for one example.
        gt_seq: array-like shape (T, action_dim) ground-truth sequence for one example.
        filename: optional filename to save the figure. If None, uses
            `vis_pred_vs_gt_{timestamp}.png`.
    """
    gt = np.asarray(gt_seq)
    if gt.ndim != 2:
        raise ValueError("gt_seq must have shape (T, action_dim)")

    T = gt.shape[0]
    n = gt.shape[1]
    rows = 2
    cols = n // rows
    
    changes = np.where(np.diff(is_human_intervention.astype(int)) != 0)[0] + 1
    indices = np.concatenate(([0], changes, [T]))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    x = np.arange(T) / 30.0
    split_frame = 10 * 30
    for i in range(n):
        ax = axes[i]
        
        # # 前 20s (0 到 split_frame) 涂蓝色
        # ax.axvspan(0, min(split_frame, T), color='green', alpha=0.1, label='First 20s' if i == 0 else "")
        
        # # 20s 之后 (split_frame 到 T) 涂红色
        # ax.axvspan(split_frame, T, color='red', alpha=0.1, label='After 20s' if i == 0 else "")
        
        # 绘制背景色块
        for start_idx, end_idx in zip(indices[:-1], indices[1:]):
            state = is_human_intervention[start_idx]
            color = 'red' if state else 'green'
            # 将索引转换为对应的 x 轴坐标
            ax.axvspan(x[start_idx], x[min(end_idx, T-1)], color=color, alpha=0.1)
            
        ax.plot(x, gt[:, i], label="State", color="#1f77b4", linewidth=1.2)
        ax.set_title(f"Dim {i}")
        ax.set_xlabel("t/s")
        ax.grid(True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    plt.close(fig)

def add_border_to_video_with_intervention_flag(
    input_video_paths: list[str], # 修改为路径列表
    input_parquet_path: str,
    output_video_path: str,
    is_human_intervention: np.ndarray,
    border_size: int = 4,
) -> str:
    green_bgr = (0, 128, 0) # 深绿色
    red_bgr = (0, 0, 255)   # 红色
    
    # 1. 打开所有视频捕获对象
    caps = [cv2.VideoCapture(Path(p)) for n, p in input_video_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {input_video_paths[i]}")
    
    # 2. 校验所有视频帧数是否一致，并与 parquet 对齐
    video_total_frames = int(caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
    assert len(is_human_intervention) == video_total_frames, \
        f"Frame count mismatch: video has {video_total_frames}, parquet has {len(is_human_intervention)}"
    
    fps = caps[0].get(cv2.CAP_PROP_FPS)

    # 3. 预读第一帧以计算拼接后的总宽度和高度
    rets, first_frames = zip(*[cap.read() for cap in caps])
    if not all(rets):
        raise RuntimeError("Failed to read the first frame from one or more videos.")

    # 以第一个视频的高度为基准进行对齐
    base_h = first_frames[0].shape[0]
    
    def get_combined_dim(frames):
        total_w = 0
        for f in frames:
            h, w = f.shape[:2]
            scaled_w = int(w * (base_h / h))
            total_w += scaled_w
        return total_w + border_size * 2, base_h + border_size * 2

    out_width, out_height = get_combined_dim(first_frames)

    # 4. 初始化写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))
    
    def process_and_concat(frames):
        # 调整所有帧到相同高度并拼接
        resized = []
        for f in frames:
            h, w = f.shape[:2]
            if h != base_h:
                new_w = int(w * (base_h / h))
                f = cv2.resize(f, (new_w, base_h), interpolation=cv2.INTER_LINEAR)
            resized.append(f)
        return cv2.hconcat(resized)

    # 5. 逐帧处理
    frame_count = 0
    # 由于前面 read 了第一帧，这里需要用 do-while 逻辑或者重置索引
    # 为了简单，我们直接处理第一帧，然后进入循环
    current_frames = first_frames
    
    while True:
        # 拼接当前帧
        combined = process_and_concat(current_frames)
        
        # 添加边框
        bg_color = red_bgr if is_human_intervention[frame_count] else green_bgr
        bordered = cv2.copyMakeBorder(
            combined, border_size, border_size, border_size, border_size,
            borderType=cv2.BORDER_CONSTANT, value=bg_color
        )

        # 绘制标签文本
        text = "human intervention" if is_human_intervention[frame_count] else "autonomous"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7 if len(input_video_paths) == 3 else 0.6 # 拼接后画面变宽，字号稍微加大
        thickness = 1 # should be interger
        (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)

        x = border_size + 10 if len(input_video_paths) == 3 else border_size + 5
        y = border_size + text_h + 10 if len(input_video_paths) == 3 else border_size + text_h + 5
        
        pad = 10 if len(input_video_paths) == 3 else 5
        # 绘制实心背景
        cv2.rectangle(bordered, (x-pad, y-text_h-pad), (x+text_w+pad, y+baseline+pad), bg_color, -1)
        # 绘制白色文字
        cv2.putText(bordered, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        writer.write(bordered)
        frame_count += 1

        # 读取下一帧
        rets, current_frames = zip(*[cap.read() for cap in caps])
        if not all(rets) or frame_count >= video_total_frames:
            break

    # 6. 释放资源
    for cap in caps:
        cap.release()
    writer.release()

    return output_video_path

if __name__ == "__main__":
    # parser
    import argparse
    parser = argparse.ArgumentParser(description="Visualize action and intervention flags from parquet and video.")
    parser.add_argument("--root_path", type=str, default="/mnt/shared/datasets/anyverse_human_data_record_raw/arxx5_bimanual/fold_box/")
    parser.add_argument("--batch_name", type=str, default="infer_pi05_base_finetune_fold_new_box_qirecap_0129.20260130.batch_3")
    parser.add_argument("--output_path", type=str, default="./vis_intervention")
    parser.add_argument("--head_path", type=str, default="observation.images.head")
    parser.add_argument("--left_path", type=str, default="observation.images.left_wrist")
    parser.add_argument("--right_path", type=str, default="observation.images.right_wrist")
    parser.add_argument("--vis_all_views", action="store_true", help="Whether to visualize all camera views. otherwise only head view is visualized.")
    args = parser.parse_args()
    
    output_path = args.output_path + "/" + args.batch_name
    if os.path.exists(output_path) is False:
        os.makedirs(output_path)

    for episode_idx in range(100):
        head_video_path = f"/videos/chunk-000/{args.head_path}/episode_{episode_idx:06d}.mp4"
        left_video_path = f"/videos/chunk-000/{args.left_path}/episode_{episode_idx:06d}.mp4"
        right_video_path = f"/videos/chunk-000/{args.right_path}/episode_{episode_idx:06d}.mp4"
        
        video_path = [("head", args.root_path + args.batch_name + head_video_path)]
        if args.vis_all_views:
            video_path = [
                ("left_wrist", args.root_path + args.batch_name + left_video_path),
                ("head", args.root_path + args.batch_name + head_video_path),
                ("right_wrist", args.root_path + args.batch_name + right_video_path),
            ]
            
        parquet_path = f"/data/chunk-000/episode_{episode_idx:06d}.parquet"
        full_head_video_path = args.root_path + args.batch_name + head_video_path
        full_parquet_path = args.root_path + args.batch_name + parquet_path

        if not os.path.exists(full_head_video_path):
            print(f"Video path not exists: {full_head_video_path}, break.")
            break
        
        vis_intervention_flag(video_path, full_parquet_path, f"./{output_path}/", episode_idx)
        print(f"Saved episode {episode_idx} visualization.")