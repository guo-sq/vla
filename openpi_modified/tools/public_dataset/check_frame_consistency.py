import os
import glob
import pandas as pd
import cv2
import json


BASE_DIR = "/mnt/oss_data/"
SUB_DIR = "RoboChallenge/train_data_lerobot/ur5_single_arm/"
DATA_DIR = os.path.join(BASE_DIR, SUB_DIR)

dataset_key = "_".join(SUB_DIR.split("/"))


def get_repo_list(data_dir):
    """获取数据目录下的所有 repo_id"""
    repos = []
    for repo_id in os.listdir(data_dir):
        repo_path = os.path.join(data_dir, repo_id)
        if os.path.isdir(repo_path):
            repos.append(repo_id)
    return repos

def check_episode_consistency(repo_path, default_camera="cam_high_rgb"):
    """
    检查单个 Repo 中每个 Episode 的一致性：
    1. Meta (episodes.jsonl) vs Parquet rows
    2. Meta (episodes.jsonl) vs Video frames
    3. Meta (info.json total_frames) vs Sum of Meta (episodes.jsonl length)
    4. Meta (info.json total_frames) vs Sum of Parquet rows
    """
    meta_path = os.path.join(repo_path, "meta/episodes.jsonl")
    info_path = os.path.join(repo_path, "meta/info.json")

    if not os.path.exists(meta_path):
        print(f"Warning: {repo_path} 缺少 meta/episodes.jsonl，无法进行逐 Episode 检查")
        return []

    print(f"正在检查: {os.path.basename(repo_path)} ...")

    # 1. 读取 Meta 信息
    try:
        episodes_meta = pd.read_json(meta_path, lines=True)
    except ValueError:
         print(f"Warning: {repo_path} meta/episodes.jsonl 格式错误")
         return []
    
    # 获取期望帧数: 字典映射 episode_index -> length
    if "episode_index" in episodes_meta.columns:
        meta_lengths = episodes_meta.set_index("episode_index")["length"].to_dict()
    else:
        # 假设行号即索引
        meta_lengths = episodes_meta["length"].to_dict()

    inconsistent_records = []

    # [新增] 检查 info.json 的 total_frames vs episodes.jsonl 的总和
    total_frames_info = None
    if os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                info_data = json.load(f)
            total_frames_info = info_data.get("total_frames")
            
            total_episodes_length_sum = sum(meta_lengths.values())
            
            if total_frames_info is not None and total_frames_info != total_episodes_length_sum:
                 inconsistent_records.append({
                     "repo_id": os.path.basename(repo_path),
                     "episode": "GLOBAL",
                     "type": "info_total_vs_episodes_sum",
                     "meta": total_frames_info,
                     "actual": total_episodes_length_sum,
                     "severity": "ERROR"
                 })
        except Exception as e:
            print(f"Warning: 读取 {info_path} 失败: {e}")

    # 2. 统计 Parquet 帧数 (按 episode_index 分组)
    parquet_files = glob.glob(os.path.join(repo_path, "data/**/*.parquet"), recursive=True)
    parquet_counts = {}
    
    if parquet_files:
        for pf in parquet_files:
            try:
                # 只读取 episode_index 列以节省内存并加速
                df = pd.read_parquet(pf, columns=["episode_index"])
                counts = df["episode_index"].value_counts().to_dict()
                for ep_idx, count in counts.items():
                    parquet_counts[ep_idx] = parquet_counts.get(ep_idx, 0) + count
            except Exception as e:
                print(f"读取 {pf} 失败: {e}")

    # [新增] 检查 Parquet 总数 vs Info Total
    if total_frames_info is not None:
        total_parquet_sum = sum(parquet_counts.values())
        if total_frames_info != total_parquet_sum:
             inconsistent_records.append({
                 "repo_id": os.path.basename(repo_path),
                 "episode": "GLOBAL",
                 "type": "info_total_vs_parquet_sum",
                 "meta": total_frames_info,
                 "actual": total_parquet_sum,
                 "severity": "ERROR"
             })

    # 3. 统计 Video 帧数
    # 查找特定相机的视频文件 pattern
    video_pattern = os.path.join(repo_path, f"videos/**/{default_camera}/*.mp4")
    video_files = glob.glob(video_pattern, recursive=True)
    video_counts = {}

    for vf in video_files:
        filename = os.path.basename(vf)
        try:
            # 文件名通常为 episode_XXXXXX.mp4
            name_part = filename.split('.')[0]
            if name_part.startswith("episode_"):
                ep_idx = int(name_part.split('_')[1])
            elif name_part.isdigit():
                ep_idx = int(name_part)
            else:
                continue
            
            cap = cv2.VideoCapture(vf)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                video_counts[ep_idx] = frame_count
                cap.release()
        except Exception:
            pass

    # 4. 逐个 Episode 对比
    for ep_idx, meta_len in meta_lengths.items():
        p_len = parquet_counts.get(ep_idx, 0)
        
        # 检查 Parquet
        if p_len != meta_len:
             inconsistent_records.append({
                 "repo_id": os.path.basename(repo_path),
                 "episode": ep_idx,
                 "type": "parquet_mismatch",
                 "meta": meta_len,
                 "actual": p_len,
                 "severity": "ERROR"
             })

        # 检查 Video (仅当视频文件存在时)
        if ep_idx in video_counts:
             v_len = video_counts[ep_idx]
             if v_len != meta_len:
                # 逻辑判断：Video > Parquet (Meta) 通常无害，只是多余尾部；Video < Parquet 是严重的，会导致读取越界
                if v_len > meta_len:
                    severity = "WARNING (Safe redundancy)"
                else:
                    severity = "CRITICAL (Missing frames)"
                    inconsistent_records.append({
                        "repo_id": os.path.basename(repo_path),
                        "episode": ep_idx,
                        "type": "video_mismatch",
                        "meta": meta_len,
                        "actual": v_len,
                        "severity": severity
                    })
    
    return inconsistent_records

def main():
    repos = get_repo_list(DATA_DIR)
    all_inconsistent = []

    print(f"开始检查 {len(repos)} 个数据集的每集帧数一致性...")

    for repo_id in repos:
        repo_path = os.path.join(DATA_DIR, repo_id)
        # 可以根据需要修改 default_camera
        issues = check_episode_consistency(repo_path, default_camera="cam_high_rgb")
        if issues:
            print(f"Found {len(issues)} issues in {repo_id}")
            all_inconsistent.extend(issues)
        else:
            print(f"{repo_id}: 检查通过")

    if not all_inconsistent:
        print("\n所有数据集的所有 Episode 帧数一致!")
    else:
        print("\n发现以下不一致:")
        for issue in all_inconsistent:
            print(f"Repo: {issue['repo_id']}, Episode: {issue['episode']}, Type: {issue['type']}, Severity: {issue['severity']}, Meta: {issue['meta']}, Actual: {issue['actual']}")


    # 可以选择将结果保存到文件
    import json
    with open(f"{dataset_key}_frame_consistency_report.json", "w") as f:
        json.dump(all_inconsistent, f, indent=4)

if __name__ == "__main__":
    main()
