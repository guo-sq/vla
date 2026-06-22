import os
import json
import jsonlines
import numpy as np
import random
import time
import matplotlib.pyplot as plt

from datasets import load_dataset
from pathlib import Path
from lerobot.common.datasets.utils import (
    load_episodes,
)


def load_json_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonlines(fpath):
    with jsonlines.open(fpath, "r") as reader:
        return list(reader)


def compute_state_static_mask(
    state,
    threshold=0.15,
    buffer=15,
    compute_start=True,
    compute_end=False,
    start_frame_ratio=0.75,
    end_frame_ratio=0.0,
    smooth_window=5,  # moving average window (frames of diffs)
    consec_frames=3,  # require this many consecutive diff-frames above threshold
):
    num_frames = int(state.shape[0])
    if num_frames <= 1:
        return {
            "left_start_frame": 0,
            "right_start_frame": 0,
            "left_end_frame": 0,
            "right_end_frame": 0,
            "start_frame": 0,
            "end_frame": max(0, num_frames - 1),
        }

    # split arms (handle variable channel counts safely)
    C = state.shape[1] if state.ndim == 2 else 0
    left_state = state[:, : min(7, C)]
    right_state = state[:, 7 : min(14, C)] if C > 7 else np.zeros((num_frames, 0))

    # per-diff absolute changes, shape (T-1, dof)
    left_diffs = (
        np.abs(left_state[1:, :] - left_state[:-1, :])
        if left_state.size
        else np.zeros((num_frames - 1, 0))
    )
    right_diffs = (
        np.abs(right_state[1:, :] - right_state[:-1, :])
        if right_state.size
        else np.zeros((num_frames - 1, 0))
    )

    # summary per-diff: use max across DOFs (detect any DOF movement), then smooth
    def smooth_summary(diffs):
        if diffs.size == 0:
            return np.zeros((num_frames - 1,), dtype=float)
        summary = np.max(diffs, axis=1)
        if smooth_window > 1 and len(summary) >= 1:
            kernel = np.ones(smooth_window, dtype=float) / float(smooth_window)
            # 'same' convolution to keep length
            summary = np.convolve(summary, kernel, mode="same")
        return summary

    left_summary = smooth_summary(left_diffs)
    right_summary = smooth_summary(right_diffs)

    # movement mask where smoothed summary >= threshold
    left_move = left_summary >= threshold
    right_move = right_summary >= threshold

    def find_first_sustained(move_mask):
        L = move_mask.shape[0]
        if L == 0:
            return 0
        # scan forward for sustained window
        for i in range(consec_frames - 1, L):
            if move_mask[i - consec_frames + 1 : i + 1].all():
                # movement window starts at s = i-consec_frames+1; motion corresponds to change between frame s and s+1
                return i - consec_frames + 1
        return L - 1

    def find_last_sustained_from_end(move_mask):
        L = move_mask.shape[0]
        if L == 0:
            return 0
        for i in range(L - 1, consec_frames - 2, -1):
            if move_mask[i - consec_frames + 1 : i + 1].all():
                return i - consec_frames + 1
        return 0

    left_start_idx = find_first_sustained(left_move)
    right_start_idx = find_first_sustained(right_move)

    left_end_idx = find_last_sustained_from_end(left_move)
    right_end_idx = find_last_sustained_from_end(right_move)

    # map diffs-index to frame-index: a diff at index i corresponds to change between frames i and i+1.
    # We keep the same convention as original: treat these indices as the last static diff-frame index.
    left_start_frame = int(left_start_idx)
    right_start_frame = int(right_start_idx)
    left_end_frame = int(left_end_idx)
    right_end_frame = int(right_end_idx)

    # compute combined start/end frames
    if compute_start:
        start_frame = int(start_frame_ratio * min(left_start_frame, right_start_frame))
    else:
        start_frame = 0

    if compute_end:
        end_frame = max(left_end_frame, right_end_frame)
        # extend end_frame towards the true end by end_frame_ratio of remaining frames
        end_frame = min(
            num_frames - 1, int((num_frames - end_frame) * end_frame_ratio) + end_frame
        )
    else:
        end_frame = num_frames - 1

    # enforce buffer: if detected start is very small (<buffer) treat as 0
    if start_frame < buffer:
        start_frame = 0

    # sanity fallback for weird episodes
    if (
        start_frame >= end_frame
        or start_frame / 30.0 > 4.0
        or (num_frames - end_frame) / 30.0 > 4.0
    ):
        start_frame = 0
        end_frame = num_frames - 1

    return {
        "left_start_frame": left_start_frame,
        "right_start_frame": right_start_frame,
        "left_end_frame": left_end_frame,
        "right_end_frame": right_end_frame,
        "start_frame": start_frame,
        "end_frame": end_frame,
    }



def extract_states_from_df(states):
    vals = states.tolist()
    stacked = np.stack([np.asarray(v) for v in vals], axis=0)
    return stacked


def update_meta_parquet_by_anno(anno_path: str, record_path: str):
    frame_state_json_path = os.path.join(anno_path, "frame_sub_task_state.jsonl")
    frame_state_annotations = load_jsonlines(frame_state_json_path)
    frame_subtask_path = os.path.join(anno_path, "frame_sub_task.jsonl")
    frame_subtask_annotations = load_jsonlines(frame_subtask_path)

    episode_frame_states = {
        item["episode_index"]: item["frame_states"]
        for item in sorted(frame_state_annotations, key=lambda x: x["episode_index"])
    }
    episode_frame_subtask = {
        item["episode_index"]: item["sub_tasks"]
        for item in sorted(frame_subtask_annotations, key=lambda x: x["episode_index"])
    }
    with open(os.path.join(record_path, "meta/info.json"), "r") as f:
        info_data = json.load(f)

    chunks_size = info_data["chunks_size"]
    for episode_num, frame_states in episode_frame_states.items():
        frame_subtask = episode_frame_subtask[episode_num]
        episode_chunk_index = int(episode_num) // chunks_size
        chunk_dir = f"chunk-{episode_chunk_index:03d}"
        parquet_path = os.path.join(
            record_path, f"data/{chunk_dir}/episode_{episode_num:06d}.parquet"
        )
        if not os.path.exists(parquet_path):
            print(f"parquet path {parquet_path} not exist!!!")
            return

        ds = load_dataset("parquet", data_files=parquet_path, split="train")

        # 2. 转成 pandas 方便修改（也可直接在 Dataset 上 map）
        df = ds.to_pandas()
        df["frame_state"] = 3  # 新增一列 frame_state,默认为success
        df["subtask_index"] = -1  # 新增一列 subtask_index,默认为0
        has_valid_key = "is_valid" in df.keys()
        if not has_valid_key:
            df["is_valid"] = True
        df_states = extract_states_from_df(df["observation.state"])
        state_static_info = compute_state_static_mask(df_states)
        for ind, frame_index in enumerate(df["frame_index"]):
            state_start_frame = state_static_info["start_frame"]
            state_end_frame = state_static_info["end_frame"]
            frame_state = frame_states[frame_index]
            sub_task_index = frame_subtask[frame_index]
            df.loc[ind, "subtask_index"] = np.int64(sub_task_index)
            df.loc[ind, "frame_state"] = np.int64(frame_state)
            if has_valid_key:
                is_valid = (
                    df["is_valid"][ind]
                    & (frame_state > 0)
                    & (state_start_frame <= ind <= state_end_frame)
                )
            else:
                is_valid = (
                    (frame_state > 0)
                    and (sub_task_index >= 0)
                    and (state_start_frame <= ind <= state_end_frame)
                )
            df.loc[ind, "is_valid"] = is_valid
        new_parquet_path = os.path.join(
            record_path, f"data_refractor/{chunk_dir}/episode_{episode_num:06d}.parquet"
        )
        if not os.path.exists(os.path.dirname(new_parquet_path)):
            os.makedirs(os.path.dirname(new_parquet_path), exist_ok=True)
        df.to_parquet(new_parquet_path, engine="pyarrow", index=False)
