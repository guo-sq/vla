"""
Compute normalization statistics for a config.

GPU-accelerated + deterministic (repeatable) version.
- Results are strictly repeatable across runs on the same machine/software stack/GPU.
- Not required to match CPU exactly (as requested).
"""

import os
import sys
import json
import time
import tempfile
import dataclasses
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import torch
import tyro
from tqdm import tqdm

import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.transforms as _transforms
import openpi.training.utils as _utils
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata


# ---------------------------
# Determinism utilities
# ---------------------------
def set_deterministic(seed: int = 0):
    """
    Make CUDA ops deterministic (repeatable).
    Call this BEFORE creating CUDA tensors.
    """
    # cuBLAS determinism (needs to be set before CUDA context is initialized)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make algorithms deterministic
    torch.use_deterministic_algorithms(True)

    # Disable TF32 (can introduce small numeric differences)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # cuDNN settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_episodes_length(fpath):
    episodes_list = list(jsonlines.open(fpath, "r"))
    return [
        item["length"]
        for item in sorted(episodes_list, key=lambda x: x["episode_index"])
    ]


def calculate_stats_numpy(data: np.ndarray, is_valid_dim: np.ndarray):
    """
    Original CPU stats (exact semantics):
      mean/std(ddof=1)/q01/q99 by dim, using np.quantile(method="linear")
    """
    if data.ndim > 2:
        data = data.reshape(-1, data.shape[-1])
    assert is_valid_dim.shape == data.shape, "is_valid_dim shape must match data shape"

    means = np.zeros(data.shape[-1], dtype=np.float64)
    stds = np.zeros(data.shape[-1], dtype=np.float64)
    q01 = np.zeros(data.shape[-1], dtype=np.float64)
    q99 = np.zeros(data.shape[-1], dtype=np.float64)

    for dim in range(data.shape[-1]):
        valid_samples = data[is_valid_dim[:, dim], dim]
        if len(valid_samples) > 0:
            means[dim] = np.mean(valid_samples)
            stds[dim] = np.std(valid_samples, ddof=1)
            q01[dim], q99[dim] = np.quantile(
                valid_samples, [0.01, 0.99], method="linear"
            )
        else:
            means[dim] = 0.0
            stds[dim] = 0.0
            q01[dim] = 0.0
            q99[dim] = 0.0

    return {
        "mean": means.tolist(),
        "std": stds.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
    }


def get_parquet_paths(repo_id, parquet_dir=None):
    if parquet_dir is not None and parquet_dir != "data":
        refractor_path = f"{repo_id}/{parquet_dir}"
        base_dir = refractor_path if os.path.exists(refractor_path) else f"{repo_id}/data/"
    else:
        base_dir = f"{repo_id}/data/"
    chunk_parquet_files = []
    root_parquet_files = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            for sub_item in os.listdir(item_path):
                if sub_item.endswith(".parquet"):
                    chunk_parquet_files.append(os.path.join(item_path, sub_item))
        elif os.path.isfile(item_path) and item.endswith(".parquet"):
            root_parquet_files.append(item_path)
    return sorted(chunk_parquet_files + root_parquet_files)


class ParquetDataset(torch.utils.data.Dataset):
    def __init__(self, config, device: str = "cuda"):
        repo_ids = (
            config.data.repo_id
            if isinstance(config.data.repo_id, list)
            else [config.data.repo_id]
        )
        self.repo_ids = [
            os.path.join(config.data.root_dir, repo_id) for repo_id in repo_ids
        ]
        self.parquet_dir = config.data.base_config.parquet_dir
        self.use_delta_action = config.data.use_delta_joint_actions
        self.horizon = int(config.model.action_horizon)
        self.dim = int(config.data.align_dim)
        self.unify_action_space = config.data.unify_action_space
        self.robot_align_info = _utils.ROBOT_ALIGN_INFO
        self.delta_action_mask_indices = config.data.delta_action_mask_indices

        # device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARN] CUDA not available, fallback to CPU.")
            device = "cpu"
        self.device = torch.device(device)

        print("Start loading parquet files...")
        self.load_parquets()

    def load_parquets(self):
        total_frames = 0
        for repo_id in self.repo_ids:
            info_path = f"{repo_id}/meta/info.json"
            with open(info_path, "r") as f:
                info = json.load(f)
            total_frames += info["total_frames"]

        self.is_valid = np.zeros((total_frames), dtype=np.bool_)
        self.actions = np.zeros((total_frames, self.dim), dtype=np.float64)
        self.states = np.zeros((total_frames, self.dim), dtype=np.float64)
        self.is_valid_dim = np.zeros((total_frames, self.dim), dtype=np.bool_)

        ptr = 0
        self.episodes_length = []
        self.cum_episodes_length = []

        for repo_id in tqdm(self.repo_ids, desc="Loading parquet files..."):
            ds_meta = LeRobotDatasetMetadata(os.path.basename(repo_id), Path(repo_id))
            parqet_paths = get_parquet_paths(repo_id, self.parquet_dir)
            print(f"Loading {repo_id}: parquet num: {len(parqet_paths)}")
            repo_id_frames = 0
            for filepath in parqet_paths:
                df = pd.read_parquet(filepath)
                n = len(df)

                if "is_valid" in df:
                    self.is_valid[ptr : ptr + n] = np.stack(
                        df["is_valid"].values
                    ).astype(np.bool_, copy=False)
                else:
                    self.is_valid[ptr : ptr + n] = np.ones(n, dtype=np.bool_)

                df, ds_meta = self.update_df_columns_name(df, ds_meta)

                if self.unify_action_space:
                    assert (
                        ds_meta.robot_type in self.robot_align_info
                    ), f"robot type {ds_meta.robot_type} not in robot_align_info"
                    robot_align_info = self.robot_align_info[ds_meta.robot_type]

                    meta_mapping = robot_align_info.get_meta_mapping_dict()
                    if len(meta_mapping):
                        features_dict = _transforms.flatten_dict(ds_meta.features)
                        features_dict = {
                            meta_mapping.get(k, k): v for k, v in features_dict.items()
                        }
                        ds_meta.info["features"] = _transforms.unflatten_dict(
                            features_dict
                        )

                    self.align_action_space_dim(
                        df, robot_align_info, ds_meta.features, ptr, n
                    )
                else:
                    raw_action = np.stack(df["action"].values).astype(
                        np.float64, copy=False
                    )
                    raw_state = np.stack(df["observation.state"].values).astype(
                        np.float64, copy=False
                    )
                    self.actions[ptr : ptr + n] = raw_action
                    self.states[ptr : ptr + n] = raw_state
                    self.is_valid_dim[ptr : ptr + n] = np.ones(
                        (n, self.dim), dtype=np.bool_
                    )

                ptr += n
                repo_id_frames += n
                self.episodes_length.append(n)
                self.cum_episodes_length.append(ptr)
                del df
            assert (
                repo_id_frames == ds_meta.info["total_frames"]
            ), f"{repo_id} total frames {ds_meta.info['total_frames']} does not match loaded frames {repo_id_frames}"

        assert (
            ptr == total_frames
        ), f"total frames {total_frames} does not match loaded frames {ptr}"
        self.episodes_length = np.asarray(self.episodes_length, dtype=np.int64)
        self.cum_episodes_length = np.asarray(self.cum_episodes_length, dtype=np.int64)

        # base-valid indices in original frame space
        self.valid_base_indices = np.flatnonzero(self.is_valid).astype(np.int64)

        # IMPORTANT: keep the same behavior as your code (states filtered, actions not)
        self.states = self.states[self.is_valid]

        print("Loading done.")

    def update_df_columns_name(self, df, ds_meta):
        if self.unify_action_space:
            camera_mapping = self.robot_align_info[
                ds_meta.robot_type
            ].get_hf_dataset_mapping_dict()
            if len(camera_mapping):
                for key, value in camera_mapping.items():
                    df = df.rename(columns={key: value})
                ds_meta.info["features"] = {
                    camera_mapping.get(k, k): v
                    for k, v in ds_meta.info["features"].items()
                }
        return df, ds_meta

    def align_action_space_dim(
        self, df, robot_align_info, feature_names, start, interval
    ):
        state_key_prior = list(robot_align_info.state_meta_source_dict.values())[0]
        if isinstance(state_key_prior, tuple):
            state_key_prior, _ = state_key_prior
        state_frame_num = df[state_key_prior].values.shape[0]

        action_key_prior = list(robot_align_info.action_meta_source_dict.values())[0]
        action_frame_num = df[action_key_prior].values.shape[0]
        assert state_frame_num == action_frame_num

        for (
            robot_part_name,
            target_dof,
        ) in robot_align_info.get_state_name_dict().items():
            for src_dof, tgt_dof in target_dof.items():
                state_meta_key = robot_align_info.state_meta_source_dict[
                    robot_part_name
                ]
                if isinstance(state_meta_key, tuple):
                    state_meta_key, _frame_offset = state_meta_key
                assert src_dof in feature_names[state_meta_key]["names"]
                state_index = feature_names[state_meta_key]["names"].index(src_dof)

                current_state = np.stack(df[state_meta_key].values).reshape(
                    state_frame_num, -1
                )
                self.states[start : start + interval, tgt_dof] = current_state[
                    :, state_index
                ].astype(np.float64, copy=False)

        for (
            robot_part_name,
            target_dof,
        ) in robot_align_info.get_action_name_dict().items():
            for src_dof, tgt_dof in target_dof.items():
                action_meta_key = robot_align_info.action_meta_source_dict[
                    robot_part_name
                ]
                assert src_dof in feature_names[action_meta_key]["names"]
                action_index = feature_names[action_meta_key]["names"].index(src_dof)

                current_action = np.stack(df[action_meta_key].values).reshape(
                    action_frame_num, -1
                )
                self.actions[start : start + interval, tgt_dof] = current_action[
                    :, action_index
                ].astype(np.float64, copy=False)

                self.is_valid_dim[start : start + interval, tgt_dof] = np.ones(
                    (interval), dtype=np.bool_
                )

    # ---------------------------
    # GPU-accelerated exact delta+horizon stats (q01/q99 exact via numpy memmap)
    # ---------------------------
    def _delta_horizon_action_stats_exact_gpu(self, chunk_bases: int = 200000):
        """
        Same semantics as your _delta_horizon_action_stats_exact,
        but accelerate heavy masking/gather/delta subtraction on GPU deterministically.

        Final mean/std/q01/q99 computed on CPU from memmap with numpy (exact quantile method="linear").
        """
        H = int(self.horizon)
        D = int(self.dim)

        # Reproduce your original cum_length semantics
        cum_length = np.repeat(
            np.cumsum(self.cum_episodes_length), self.episodes_length
        ).astype(np.int64)

        frame_num = int(self.is_valid.shape[0])
        base_indices = self.valid_base_indices  # [M]
        M = int(base_indices.shape[0])

        # delta mask (same as to_delta_actions)
        mask_np = np.asarray(
            _transforms.make_bool_mask(*self.delta_action_mask_indices)
        )
        mask_np = mask_np.astype(np.bool_, copy=False)
        dims = int(mask_np.shape[-1])

        # ---- move big arrays to GPU (float64 + bool) ----
        # actions is in original frame space
        actions_t = torch.from_numpy(self.actions).to(self.device, dtype=torch.float64)
        is_valid_t = torch.from_numpy(self.is_valid).to(self.device, dtype=torch.bool)
        is_valid_dim_t = torch.from_numpy(self.is_valid_dim).to(
            self.device, dtype=torch.bool
        )

        # states already filtered by base-valid, aligned with base_indices order
        states_valid_t = torch.from_numpy(self.states).to(
            self.device, dtype=torch.float64
        )

        cum_length_t = torch.from_numpy(cum_length).to(self.device, dtype=torch.int64)
        base_indices_t = torch.from_numpy(base_indices).to(
            self.device, dtype=torch.int64
        )
        offsets_t = torch.arange(H, device=self.device, dtype=torch.int64)[
            None, :
        ]  # [1,H]
        mask_t = torch.from_numpy(mask_np).to(self.device, dtype=torch.bool)

        # ----------------
        # Pass1: count per dim on GPU
        # ----------------
        cnt = np.zeros(D, dtype=np.int64)

        for s in tqdm(range(0, M, chunk_bases), desc="GPU counting (delta+horizon)..."):
            b = base_indices_t[s : s + chunk_bases]  # [B]
            B = int(b.numel())
            if B == 0:
                continue

            idx = b[:, None] + offsets_t  # [B,H]
            episode_end_mask = idx < cum_length_t[b][:, None]  # [B,H]

            # clipping (same)
            idx = torch.where(
                idx >= frame_num, torch.full_like(idx, frame_num - 1), idx
            )

            horizon_valid = is_valid_t[idx] & episode_end_mask  # [B,H]
            vdim = is_valid_dim_t[idx]  # [B,H,D]
            v = vdim & horizon_valid[:, :, None]  # [B,H,D]

            # reduce on GPU, then bring to CPU
            cnt_chunk = v.sum(dim=(0, 1)).to("cpu", dtype=torch.int64).numpy()
            cnt += cnt_chunk

        offsets_1d = np.zeros(D + 1, dtype=np.int64)
        np.cumsum(cnt, out=offsets_1d[1:])
        total = int(offsets_1d[-1])

        # ----------------
        # Pass2: fill memmap (values computed on GPU, written to CPU memmap)
        # ----------------
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "delta_horizon_action_vals.dat")
            mem = np.memmap(path, dtype=np.float64, mode="w+", shape=(total,))
            write_ptr = np.zeros(D, dtype=np.int64)

            for s in tqdm(
                range(0, M, chunk_bases), desc="GPU filling memmap (delta+horizon)..."
            ):
                b = base_indices_t[s : s + chunk_bases]  # [B]
                B = int(b.numel())
                if B == 0:
                    continue

                idx = b[:, None] + offsets_t  # [B,H]
                episode_end_mask = idx < cum_length_t[b][:, None]
                idx = torch.where(
                    idx >= frame_num, torch.full_like(idx, frame_num - 1), idx
                )

                horizon_valid = is_valid_t[idx] & episode_end_mask  # [B,H]
                vdim = is_valid_dim_t[idx]  # [B,H,D]
                v = vdim & horizon_valid[:, :, None]  # [B,H,D]

                ah = actions_t[idx]  # [B,H,D] float64 on GPU

                # delta subtraction
                if dims > 0:
                    sb = states_valid_t[s : s + B, :dims]  # [B,dims]
                    sb = sb[:, None, :]  # [B,1,dims]
                    # subtract only where mask is True, else subtract 0
                    ah_part = ah[..., :dims]
                    ah_part = ah_part - torch.where(
                        mask_t, sb, torch.zeros_like(sb, dtype=torch.float64)
                    )
                    ah = (
                        torch.cat([ah_part, ah[..., dims:]], dim=-1)
                        if dims < D
                        else ah_part
                    )

                # write per dim (same layout as your memmap method)
                for d in range(D):
                    start_off = offsets_1d[d]
                    end_off = offsets_1d[d + 1]
                    if end_off == start_off:
                        continue

                    vals = ah[..., d][v[..., d]]  # 1D tensor on GPU
                    if vals.numel() == 0:
                        continue

                    vals_cpu = vals.detach().to("cpu").numpy()

                    w0 = start_off + write_ptr[d]
                    w1 = w0 + vals_cpu.size
                    mem[w0:w1] = vals_cpu
                    write_ptr[d] += vals_cpu.size

            # sanity
            for d in range(D):
                expected = offsets_1d[d + 1] - offsets_1d[d]
                if write_ptr[d] != expected:
                    raise RuntimeError(
                        f"delta-horizon action dim {d} fill mismatch: {write_ptr[d]} vs {expected}"
                    )

            # ----------------
            # Compute exact stats per dim from memmap slices (CPU numpy, exact quantile)
            # ----------------
            means = np.zeros(D, dtype=np.float64)
            stds = np.zeros(D, dtype=np.float64)
            q01 = np.zeros(D, dtype=np.float64)
            q99 = np.zeros(D, dtype=np.float64)

            for d in range(D):
                a = int(offsets_1d[d])
                b2 = int(offsets_1d[d + 1])
                vals = mem[a:b2]
                if vals.size > 0:
                    means[d] = np.mean(vals)
                    stds[d] = np.std(vals, ddof=1)  # size==1 -> nan (same as original)
                    q01[d], q99[d] = np.quantile(vals, [0.01, 0.99], method="linear")
                else:
                    means[d] = 0.0
                    stds[d] = 0.0
                    q01[d] = 0.0
                    q99[d] = 0.0

            del mem  # flush

        return {
            "mean": means.tolist(),
            "std": stds.tolist(),
            "q01": q01.tolist(),
            "q99": q99.tolist(),
        }

    def get_norm_stats(self, *, use_gpu_for_delta: bool = True):
        print("self.states", self.states.shape)

        # state stats: keep exact numpy semantics (and usually states stats isn't the bottleneck)
        state_stats = calculate_stats_numpy(
            self.states, self.is_valid_dim[self.is_valid]
        )

        # actions stats
        if self.use_delta_action:
            if use_gpu_for_delta and self.device.type == "cuda":
                action_stats = self._delta_horizon_action_stats_exact_gpu()
            else:
                # fallback (your original CPU exact implementation can be kept if you want)
                action_stats = (
                    self._delta_horizon_action_stats_exact_gpu()
                )  # still works on CPU device
        else:
            action_stats = calculate_stats_numpy(
                self.actions[self.is_valid], self.is_valid_dim[self.is_valid]
            )

        return {"state": state_stats, "actions": action_stats}

    def __getitem__(self, idx):
        return {
            "state": self.states[idx],
            "actions": self.actions[self.valid_base_indices[idx]],
        }

    def __len__(self):
        return self.states.shape[0]


def main(
    config_name: str = "",
    print_output_path: bool = False,
    test_load_norm_stats: bool = False,
    device: str = "cuda",
    seed: int = 0,
    use_gpu_for_delta: bool = True,
):
    config = _config.get_config(config_name)
    output_dir = config.assets_dirs / config.data.assets.asset_id
    output_path = str(output_dir / "norm_stats.json")
    if print_output_path:
        print(output_path)
        return

    # deterministic must be set ASAP
    set_deterministic(seed)
    t0 = time.time()

    # 优先从 reuse_norm_stats_path 直接读取，不复制；否则从 output_path 读取
    norm_stats_source_path = config.reuse_norm_stats_path
    load_source_path = None
    if norm_stats_source_path and os.path.exists(norm_stats_source_path):
        load_source_path = norm_stats_source_path
        print(
            f"Will load norm_stats directly from: {norm_stats_source_path} (no copy)"
        )

    if config.data.unify_action_space:
        repo_id_list = config.data.repo_id
        repo_id_dict = {}
        for repo_id in repo_id_list:
            ds_meta = LeRobotDatasetMetadata(
                repo_id, Path(config.data.root_dir) / repo_id
            )
            robot_type = ds_meta.robot_type
            repo_id_dict.setdefault(robot_type, []).append(repo_id)

        output_dict = {"norm_stats": {}, "repo_id": {}}
        robots_to_compute = []
        if load_source_path:
            print(f"Loading norm_stats from {load_source_path}...")
            with open(load_source_path, "r") as f:
                existing_dict = json.load(f)
            # Copy existing data
            if "norm_stats" in existing_dict:
                output_dict["norm_stats"] = existing_dict["norm_stats"].copy()
            if "repo_id" in existing_dict:
                output_dict["repo_id"] = existing_dict["repo_id"].copy()

            # Determine which robots need to be computed
            for robot_type in repo_id_dict.keys():
                if robot_type in output_dict["norm_stats"]:
                    print(
                        f"Robot Type '{robot_type}' already exists in norm_stats.json, skipping computation."
                    )
                else:
                    robots_to_compute.append(robot_type)
        else:
            robots_to_compute = list(repo_id_dict.keys())

        for robot_type in tqdm(robots_to_compute, desc="Processing repo lists"):
            print("Robot Type:", robot_type)
            t1 = time.time()

            updated_config = dataclasses.replace(
                config,
                data=dataclasses.replace(config.data, repo_id=repo_id_dict[robot_type]),
            )
            dataset = ParquetDataset(updated_config, device=device)

            t2 = time.time()
            print(f"Loading dataset took: {t2 - t1:.2f}s")

            output_dict["norm_stats"][robot_type] = dataset.get_norm_stats(
                use_gpu_for_delta=use_gpu_for_delta
            )
            output_dict["repo_id"][robot_type] = repo_id_dict[robot_type]

            t3 = time.time()
            print(f"Process {robot_type} done! Calculating stats took: {t3 - t2:.2f}s")
    else:
        # 非 unify_action_space
        if load_source_path:
            # 直接读取 reuse 文件，无需计算
            print(f"Reusing norm_stats from {load_source_path}, skipping computation.")
            with open(load_source_path, "r") as f:
                output_dict = json.load(f)
        else:
            t1 = time.time()
            dataset = ParquetDataset(config, device=device)
            t2 = time.time()
            print(f"Loading dataset took: {t2 - t1:.2f}s")

            output_dict = {
                "norm_stats": dataset.get_norm_stats(use_gpu_for_delta=use_gpu_for_delta),
                "repo_id": config.data.repo_id,
            }

            t3 = time.time()
            print(f"Calculating stats took: {t3 - t2:.2f}s")

    t4 = time.time()
    print(f"Total time: {t4 - t0:.2f}s")

    os.makedirs(output_dir, exist_ok=True)
    # Atomic write: write to temp file then rename, so other ranks never see partial content
    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=str(output_dir), text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(output_dict, f, indent=2)
        os.replace(tmp_path, output_path)
        print(f"Writing stats to: {output_path}")
    except Exception:
        os.unlink(tmp_path)
        raise

    if test_load_norm_stats:
        print("Test loading norm stats")
        loaded_norm_stats = _normalize.load(config.assets_dirs)
        print(f"Loaded norm stats: {loaded_norm_stats}")


if __name__ == "__main__":
    tyro.cli(main)
