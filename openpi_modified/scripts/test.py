"""Minimal JAX-based evaluation script inspired by `scripts/train.py`.

Behaviors:
- Runs model parallel inference offline on user-provided dataset.

Usage (example):
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
    uv run scripts/test.py \
        --ckpt_dir checkpoints/pour_water_14_dim_0310/pour_water_14_dim_0310_exp/1000 \
        --dataset_root /mnt/ \
        --config_name src/openpi/configs/cfg_pi0.5_pour_water_14_dim.py \
        --num_batches 10 \
        --batch_size 128 \
        --sample_steps 10 \
        --repo_id oss_data/anyverse_pour_water_record/record.pourwater.bipiper.0228.5
"""

import dataclasses
import logging
import os
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import matplotlib.pyplot as plt
import numpy as np

from openpi.models import model as _model
from openpi.shared import nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.frame_attributes_preprocessors import TemporalWeightProcessor
import openpi.training.sharding as sharding
import openpi.transforms as transforms

DEFAULT_FPS = 30.0

ENABLE_BIMANUAL = True
try:
    import bimanual
except ImportError:
    ENABLE_BIMANUAL = False
    print("bimanual module not found; kinematic functions will be disabled.")


def init_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def vis_arm_states(
    left_states,
    right_states,
    vis_dir: str = "./open_loop_vis_rl",
    filename: str | None = None,
    subtitle_suffix: str | None = None,
):
    """Visualize left and right arm states.

    Plots 2D projections of the trajectories (XY, XZ, YZ) and saves an image.
    Left arm is drawn in red, right arm in blue.

    Args:
        left_states: array-like shape (T, 3) or (T, >=3). XYZ in first 3 dims.
        right_states: array-like shape (T, 3) or (T, >=3). XYZ in first 3 dims.
        vis_dir: output directory to save the image.
        filename: optional filename; if None, uses timestamped name.
    """
    # Convert to numpy
    left_arr = np.asarray(left_states)
    right_arr = np.asarray(right_states)

    # Ensure two-dimensional arrays
    if left_arr.ndim == 1:
        left_arr = left_arr.reshape(-1, left_arr.shape[0]) if left_arr.size % 3 != 0 else left_arr.reshape(-1, 3)
    if right_arr.ndim == 1:
        right_arr = right_arr.reshape(-1, right_arr.shape[0]) if right_arr.size % 3 != 0 else right_arr.reshape(-1, 3)

    # If arrays have more than 3 dims per timestep, take first 3 as XYZ
    if left_arr.ndim == 2 and left_arr.shape[1] >= 3:
        left_xyz = left_arr[:, :3]
    else:
        raise ValueError("left_states must be shape (T,>=3) or (T,3)")
    if right_arr.ndim == 2 and right_arr.shape[1] >= 3:
        right_xyz = right_arr[:, :3]
    else:
        raise ValueError("right_states must be shape (T,>=3) or (T,3)")

    # Create a 3D plot of the trajectories and mark start/end
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectories
    if left_xyz.shape[0] > 0:
        ax.plot(
            left_xyz[:, 0],
            left_xyz[:, 1],
            left_xyz[:, 2],
            color="red",
            label="Pred",
            linewidth=1.5,
        )
        # start/end markers
        ax.scatter(left_xyz[0, 0], left_xyz[0, 1], left_xyz[0, 2], color="red", marker="o", s=60)
        ax.scatter(
            left_xyz[-1, 0],
            left_xyz[-1, 1],
            left_xyz[-1, 2],
            color="red",
            marker="X",
            s=60,
        )
        # annotate
        ax.text(left_xyz[0, 0], left_xyz[0, 1], left_xyz[0, 2], "L start", color="red")
        ax.text(left_xyz[-1, 0], left_xyz[-1, 1], left_xyz[-1, 2], "L end", color="red")

    if right_xyz.shape[0] > 0:
        ax.plot(
            right_xyz[:, 0],
            right_xyz[:, 1],
            right_xyz[:, 2],
            color="blue",
            label="GT",
            linewidth=1.5,
        )
        ax.scatter(
            right_xyz[0, 0],
            right_xyz[0, 1],
            right_xyz[0, 2],
            color="blue",
            marker="o",
            s=60,
        )
        ax.scatter(
            right_xyz[-1, 0],
            right_xyz[-1, 1],
            right_xyz[-1, 2],
            color="blue",
            marker="X",
            s=60,
        )
        ax.text(right_xyz[0, 0], right_xyz[0, 1], right_xyz[0, 2], "R start", color="blue")
        ax.text(right_xyz[-1, 0], right_xyz[-1, 1], right_xyz[-1, 2], "R end", color="blue")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    ax.grid(visible=True, linestyle="--", alpha=0.4)

    # Set equal aspect ratio for 3D plot
    def _set_axes_equal(ax3d):
        x_limits = ax3d.get_xlim3d()
        y_limits = ax3d.get_ylim3d()
        z_limits = ax3d.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)
        ax3d.set_xlim3d(x_middle - max_range / 2, x_middle + max_range / 2)
        ax3d.set_ylim3d(y_middle - max_range / 2, y_middle + max_range / 2)
        ax3d.set_zlim3d(z_middle - max_range / 2, z_middle + max_range / 2)

    _set_axes_equal(ax)

    subtitle = f"Arm State Trajectories ({subtitle_suffix})"

    fig.suptitle(subtitle)

    assert filename is not None, "filename must be provided for vis_arm_states"
    os.makedirs(vis_dir, exist_ok=True)
    out_path = os.path.join(vis_dir, filename)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved arm states visualization to {out_path}")


def calculate_total_frames(data_loader: _data_loader.DataLoader) -> int:
    inner = getattr(data_loader, "_data_loader", None)
    torch_loader = inner.torch_loader
    ds = getattr(torch_loader, "dataset", None)
    return len(ds)


def visualize_pred_vs_gt(
    pred_seq,
    gt_seq,
    filename: str | None = None,
    use_seconds: bool = False,
    fps: float = DEFAULT_FPS,
):
    """Visualize prediction vs GT for 14 dimensions over time.

    Args:
        pred_seq: array-like shape (T, action_dim) predicted sequence for one example.
        gt_seq: array-like shape (T, action_dim) ground-truth sequence for one example.
        filename: optional filename to save the figure. If None, uses
            `vis_pred_vs_gt_{timestamp}.png`.
        use_seconds: if True, x-axis uses time in seconds; otherwise uses frame numbers.
        fps: frames per second, used to convert frames to seconds.
    """
    pred = np.asarray(pred_seq)
    gt = np.asarray(gt_seq)
    if gt.ndim != 2:
        raise ValueError("gt_seq must have shape (T, action_dim)")
    # pred may be 2D (T, D) for a single continuous line, or 3D (K, H, D) to draw
    # each chunk as an independent line (chunks come from different inferences
    # and shouldn't be visually connected across chunk boundaries).
    if pred.ndim == 3:
        k_chunks, chunk_len, pred_dim = pred.shape
        if pred_dim != gt.shape[1] or k_chunks * chunk_len != gt.shape[0]:
            raise ValueError("3D pred_seq must satisfy K*H == gt T and matching action_dim")
    elif pred.ndim == 2:
        if pred.shape[1] != gt.shape[1]:
            raise ValueError("pred_seq and gt_seq must have the same action_dim")
        if pred.shape[0] != gt.shape[0]:
            raise ValueError("pred_seq and gt_seq must have the same temporal length T")
    else:
        raise ValueError("pred_seq must be 2D (T, D) or 3D (K, H, D)")

    sequence_length = gt.shape[0]
    n = gt.shape[1]
    rows = 2
    cols = n // rows

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    # Determine x-axis values based on use_seconds flag
    if use_seconds:
        t = np.arange(sequence_length) / fps
        x_label = f"time (s) [{fps:g} fps]"
    else:
        t = np.arange(sequence_length)
        x_label = f"frame [{fps:g} fps]"

    for i in range(n):
        ax = axes[i]
        ax.plot(t, gt[:, i], label="GT", color="#1f77b4", linewidth=1.2)
        if pred.ndim == 3:
            for k in range(k_chunks):
                xs = t[k * chunk_len : (k + 1) * chunk_len]
                ax.plot(
                    xs,
                    pred[k, :, i],
                    label="Pred" if k == 0 else None,
                    color="#d62728",
                    linestyle="--",
                    linewidth=1.2,
                )
        else:
            ax.plot(t, pred[:, i], label="Pred", color="#d62728", linestyle="--", linewidth=1.2)
        ax.set_title(f"Dim {i}")
        ax.set_xlabel(x_label)
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Prediction vs Ground Truth ({fps:g} fps)")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close(fig)


def main(
    checkpoint_dir: str,
    dataset_root: str,
    config_name: str,
    num_batches: int = 2,
    norm_stats_path: str | None = None,
    batch_size: int | None = None,
    vis_dir: str = "./open_loop_vis",
    results_dir: str | None = None,
    repo_id: str | None = None,
    sample_steps: int | None = None,
    num_workers: int | None = None,
    *,
    run_subtask_inference: bool = True,
    use_seconds: bool = False,
    fps: float | None = None,
    vis_gap: int = 50,
):
    t0 = time.perf_counter()
    init_logging()

    checkpoint_dir = Path(checkpoint_dir)
    dataset_root = Path(dataset_root)
    checkpoint_base_dir = checkpoint_dir.parent.parent.parent
    # config_name = checkpoint_dir.parent.parent.name
    config = _config.get_config(config_name)
    exp_name = checkpoint_dir.parent.name
    step = checkpoint_dir.name

    replace_kwargs = {
        "checkpoint_base_dir": str(checkpoint_base_dir),
        "exp_name": exp_name,
    }
    if batch_size is not None:
        replace_kwargs["batch_size"] = batch_size
    if num_workers is not None:
        replace_kwargs["num_workers"] = num_workers
    config = dataclasses.replace(config, **replace_kwargs)  # 用提供的字段值覆盖指定字段
    print(f"Using config: {config.name}, exp: {config.exp_name}")

    t1 = time.perf_counter()
    print(f"Timing: config loading took {t1 - t0:.3f}s")

    t0 = time.perf_counter()

    if repo_id is not None:
        assert isinstance(repo_id, str), "repo_id should be a comma-separated string of repo ids"
        # We'll defer creating per-repo data_cfg/data_loader until processing each repo.
        all_repo_ids = [r.strip() for r in repo_id.split(",") if r.strip()]
        # Determine a repo list to process
        first_repo = all_repo_ids[0] if all_repo_ids else repo_id
        assert len(first_repo) > 0, "No valid repo_id found for restore data loader"
    else:
        first_repo = None

    try:
        base_data_cfg = config.data.create(config.assets_dirs, config.model)
        replace_data_kwargs = {"root_dir": str(dataset_root)}
        if first_repo is not None:
            replace_data_kwargs["repo_id"] = [first_repo]

        # Remove TemporalWeightProcessor so every frame has weight=1 during eval
        orig_preprocessors = base_data_cfg.frame_attributes_preprocessors or []
        filtered = [p for p in orig_preprocessors if not isinstance(p, TemporalWeightProcessor)]
        removed = len(orig_preprocessors) - len(filtered)
        if removed > 0:
            logging.info(
                f"Eval: removed {removed} TemporalWeightProcessor(s), keeping {len(filtered)} other processor(s)"
            )
        replace_data_kwargs["frame_attributes_preprocessors"] = filtered or None
        base_data_cfg = dataclasses.replace(base_data_cfg, **replace_data_kwargs)
        norm_stats_path = checkpoint_dir / "assets" / base_data_cfg.asset_id
        if norm_stats_path is not None:
            norm_stats_file = norm_stats_path
            if norm_stats_file.exists():
                loaded = _normalize.load(norm_stats_file)
                base_data_cfg = dataclasses.replace(base_data_cfg, norm_stats=loaded)
                print(f"Loaded norm_stats from {norm_stats_file}")
            else:
                logging.warning(f"Provided norm_stats_path does not exist: {norm_stats_file}")
                raise FileNotFoundError(f"Provided norm_stats_path does not exist: {norm_stats_file}")

        orig_group = base_data_cfg.model_transforms
        if hasattr(orig_group, "inputs") and len(orig_group.inputs) > 0:
            eval_inputs = (
                transforms.InjectEvalSubtaskFlags(),
                *orig_group.inputs,
            )
            base_data_cfg = dataclasses.replace(
                base_data_cfg,
                model_transforms=transforms.Group(inputs=eval_inputs, outputs=orig_group.outputs),
            )

        # create a shallow copy of config with a DataConfigFactory-like object that will return our data_cfg
        class _SimpleFactory:
            def __init__(self, data_cfg):
                self._data_cfg = data_cfg
                self.episode_fail = None
                self.dataset_length = None

            def create(self, assets_dirs, model_config):
                return self._data_cfg

        config = dataclasses.replace(config, data=_SimpleFactory(base_data_cfg))
    except Exception:
        logging.warning("Could not patch data factory; proceeding and hope dataset paths are embedded in config.")

    # Create mesh and sharding same as train.py
    # Prefer using configured devices; otherwise use all local JAX devices for multi-card acceleration
    fsdp_devices = getattr(config, "fsdp_devices", None)
    if not fsdp_devices:
        try:
            fsdp_devices = list(range(jax.device_count()))
            print(f"Auto-configuring fsdp_devices to all local devices: {fsdp_devices}")
        except Exception:
            fsdp_devices = None

    # Create mesh and sharding same as train.py
    mesh = sharding.make_mesh(fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # Params-only inference load: works on slim release checkpoints that only ship
    # `params/` and `assets/` (no `train_state/`), and also on full training checkpoints.
    t0 = time.perf_counter()
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    params = _model.restore_params(
        checkpoint_dir / "params", dtype=jnp.bfloat16, sharding=replicated_sharding
    )
    model = config.model.load(params)
    model.eval()
    sample_fn_jit = nnx_utils.module_jit(model.sample_actions, static_argnames=("run_subtask_inference",))
    rng = jax.random.key(config.seed)

    t1 = time.perf_counter()
    print(f"Timing: model load took {t1 - t0:.3f}s")

    logging.info("Model restored and set to eval mode.")
    logging.info("Open-loop eval sample_steps=%s", sample_steps or 10)
    logging.info("Eval prefix flags injected to disable GT subtask/action tokens.")
    logging.info("Action generation run_subtask_inference=%s", run_subtask_inference)

    # 输出格式转换
    output_fns = []
    output_fns.extend(base_data_cfg.model_transforms.outputs)
    output_fns.append(transforms.Unnormalize(base_data_cfg.norm_stats, use_quantiles=base_data_cfg.use_quantile_norm))
    output_fns.extend(base_data_cfg.data_transforms.outputs)
    output_fns.extend(base_data_cfg.repack_transforms.outputs)

    output_transform = transforms.compose(output_fns)

    t1 = time.perf_counter()
    print(f"Timing: prepare took {t1 - t0:.3f}s")  # 38s
    # Evaluate each repo_id independently, reusing the loaded model
    for repo in all_repo_ids:
        print(f"\nEvaluating repo: {repo}")
        # Build data_cfg for this repo
        try:
            data_cfg = dataclasses.replace(base_data_cfg, root_dir=str(dataset_root), repo_id=[repo])

            # patch a simple factory for create_data_loader
            class _SimpleFactoryLocal:
                def __init__(self, data_cfg):
                    self._data_cfg = data_cfg
                    self.episode_fail = None
                    self.dataset_length = None

                def create(self, assets_dirs, model_config):
                    return self._data_cfg

            cfg_for_loader = dataclasses.replace(config, data=_SimpleFactoryLocal(data_cfg))
        except Exception as e:
            logging.warning(f"Could not create data config for repo {repo}: {e}")
            continue
        meta = LeRobotDatasetMetadata(repo, Path(dataset_root) / repo)
        robot_type = meta.robot_type

        # Read fps from dataset metadata if not specified
        current_fps = fps
        if current_fps is None:
            if hasattr(meta, "fps") and meta.fps is not None:
                current_fps = float(meta.fps)
                print(f"Using FPS from dataset metadata: {current_fps}")
            else:
                current_fps = DEFAULT_FPS
                print(f"Could not read FPS from dataset metadata, using default: {current_fps}")

        data_loader = _data_loader.create_data_loader(
            cfg_for_loader, sharding=data_sharding, shuffle=False, skip_norm_stats=False
        )

        data_iter = iter(data_loader)

        all_preds = []
        all_gts = []
        batch_mses = []

        # Open-loop eval has no previous action chunk, so RTC delay must be 0.
        # Using delay > 0 with zeros action_prefix would overwrite absolute dims
        # (e.g. gripper) to training mean via the RTC prefix injection mechanism.
        delay = 0
        action_prefix = jnp.zeros((batch_size, config.model.action_horizon, config.model.action_dim))

        for i in range(num_batches):
            t0 = time.perf_counter()
            print(f"Current batch: {i}")

            batch = next(data_iter)
            observation, actions, _ = batch
            actions_np = jax.device_get(actions)
            t1 = time.perf_counter()
            print(f"- Timing: get batch data: {t1 - t0:.3f}s")

            t0 = time.perf_counter()
            # build RNG per batch
            rng, subkey = jax.random.split(rng)

            sampled = sample_fn_jit(
                subkey,
                observation,
                action_prefix=action_prefix,
                delay=delay,
                num_steps=sample_steps or 10,
                run_subtask_inference=run_subtask_inference,
            )
            sampled_dict = None
            if isinstance(sampled, dict):
                assert "actions" in sampled, "actions must be in sampled"
                sampled_dict = sampled
                sampled = sampled["actions"]
            sampled = sampled.block_until_ready()
            sampled_np = jax.device_get(sampled)
            subtask_tokens_np = None
            if sampled_dict is not None and "subtask_tokens" in sampled_dict:
                subtask_tokens_np = jax.device_get(sampled_dict["subtask_tokens"])

            t1 = time.perf_counter()
            print(f"- Timing: sample_actions: {t1 - t0:.3f}s")

            t0 = time.perf_counter()

            obs_dict = observation.to_dict() if hasattr(observation, "to_dict") else {}
            obs_host = jax.tree_map(lambda x: np.asarray(x), obs_dict)
            state_host = obs_host.get("state", None)
            # Apply output transform to both prediction and GT to obtain unnormalized absolute actions
            pred_in = {
                "state": state_host,
                "actions": sampled_np,
                "robot_type": robot_type,
            }
            if subtask_tokens_np is not None:
                pred_in["subtask_tokens"] = subtask_tokens_np
            gt_in = {
                "state": state_host,
                "actions": actions_np,
                "robot_type": robot_type,
            }
            pred_trans = output_transform(pred_in)
            final_pred = pred_trans["actions"]
            gt_trans = output_transform(gt_in)
            final_gt = gt_trans["actions"]

            # Compute per-batch MSE
            mse_batch = float(np.mean((final_pred - final_gt) ** 2))

            batch_mses.append(mse_batch)
            all_preds.append(final_pred)
            all_gts.append(final_gt)

            t1 = time.perf_counter()
            print(f"- Timing: Infer + post process: {t1 - t0:.3f}s")
            print(f"- MSE: {round(mse_batch, 4)}")

        jax.clear_caches()
        # Summarize results
        overall_mse = float(np.mean(batch_mses))
        print(f"Overall MSE across {len(batch_mses)} batches for repo {repo}: {round(overall_mse, 4)}")
        preds_arr = np.concatenate(all_preds, axis=0)
        gts_arr = np.concatenate(all_gts, axis=0)
        if results_dir is None:
            save_path_root = checkpoint_base_dir / config_name / exp_name / step / "test_results" / repo
        else:
            save_path_root = Path(results_dir) / repo
        os.makedirs(save_path_root, exist_ok=True)
        np.save(save_path_root / "test_all_preds.npy", preds_arr)
        np.save(save_path_root / "test_all_gts.npy", gts_arr)
        print(f"Saved all preds/gts to [{save_path_root}/test_all_preds.npy] and [{save_path_root}/test_all_gts.npy]")

        # Visualization
        # Trim each inference chunk to vis_gap frames so adjacent chunks are
        # contiguous on the x-axis (no overlap, no gap). Requires
        # vis_gap <= action_horizon (asserted at CLI).
        action_dim = preds_arr.shape[-1]
        preds_arr_gap_chunks = preds_arr[::vis_gap][:, :vis_gap, :]  # (K, vis_gap, D)
        preds_arr_gap = preds_arr_gap_chunks.reshape(-1, action_dim)
        gts_arr_gap = gts_arr[::vis_gap][:, :vis_gap, :].reshape(-1, action_dim)
        filename = f"{vis_dir}/pred_and_gt_{repo}.png"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        visualize_pred_vs_gt(
            preds_arr_gap_chunks, gts_arr_gap, filename=filename, use_seconds=use_seconds, fps=current_fps
        )

    # arm pos visualization
    if ENABLE_BIMANUAL:
        action_dim = 14
        left_arm_pos_gap = np.zeros((preds_arr_gap.shape[0], action_dim // 2 - 1))
        right_arm_pos_gap = np.zeros((preds_arr_gap.shape[0], action_dim // 2 - 1))
        left_arm_pos_gt_gap = np.zeros((gts_arr_gap.shape[0], action_dim // 2 - 1))
        right_arm_pos_gt_gap = np.zeros((gts_arr_gap.shape[0], action_dim // 2 - 1))
        for i in range(preds_arr_gap.shape[0]):
            left_arm_pos_gap[i] = bimanual.forward_kinematics(preds_arr_gap[i, : action_dim // 2 - 1])
            right_arm_pos_gap[i] = bimanual.forward_kinematics(preds_arr_gap[i, action_dim // 2 : action_dim - 1])
            left_arm_pos_gt_gap[i] = bimanual.forward_kinematics(gts_arr_gap[i, : action_dim // 2 - 1])
            right_arm_pos_gt_gap[i] = bimanual.forward_kinematics(gts_arr_gap[i, action_dim // 2 : action_dim - 1])
        arm_pred = np.concatenate([left_arm_pos_gap, right_arm_pos_gap], axis=1)
        arm_gt = np.concatenate(
            [left_arm_pos_gt_gap, right_arm_pos_gt_gap],
            axis=1,
        )

        visualize_pred_vs_gt(arm_pred, arm_gt, filename=f"{vis_dir}/pred_and_gt_arm_poses.png")
        vis_arm_states(
            left_arm_pos_gap,
            left_arm_pos_gt_gap,
            vis_dir=vis_dir,
            filename="predicted_arm_states_left.png",
            subtitle_suffix="pred=red, gt=blue",
        )
        vis_arm_states(
            right_arm_pos_gap,
            right_arm_pos_gt_gap,
            vis_dir=vis_dir,
            filename="predicted_arm_states_right.png",
            subtitle_suffix="pred=red, gt=blue",
        )
        vis_arm_states(
            left_arm_pos_gt_gap,
            right_arm_pos_gt_gap,
            vis_dir=vis_dir,
            filename="predicted_arm_states_gt.png",
            subtitle_suffix="pred=left, gt=right",
        )


if __name__ == "__main__":
    import argparse
    from datetime import datetime
    from datetime import timedelta
    from datetime import timezone

    beijing_tz = timezone(timedelta(hours=8))
    now = datetime.now(beijing_tz)
    time_str = now.strftime("%m%d_%H%M")  # '12_30_20_39'

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)  #
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Override config.num_workers for eval. Set to 0 to avoid the spawn-worker "
            "cold-start (~2000s on first batch). If unset, use the value from the config."
        ),
    )
    parser.add_argument("--run_subtask_inference", type=int, choices=(0, 1), default=1)
    # Vis config
    parser.add_argument("--vis_dir", type=str, default="./open_loop_vis/")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="",
        help="Directory for raw prediction/ground-truth numpy outputs",
    )
    # X-axis config for visualization
    parser.add_argument(
        "--use_seconds",
        action="store_true",
        help="Use seconds for x-axis in visualization (default: use frame numbers)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS for converting frames to seconds. If not specified, reads from dataset metadata. Falls back to 30.",
    )
    parser.add_argument(
        "--vis_gap",
        type=int,
        default=50,
        help="Sampling stride over the concatenated pred/gt sequence for visualization (default: 50).",
    )

    args = parser.parse_args()

    vis_dir = args.vis_dir + time_str + "_" + args.repo_id
    os.makedirs(vis_dir, exist_ok=True)

    assert args.vis_gap <= 50, "vis_gap should be at most 50 to ensure enough temporal resolution in visualizations"
    main(
        args.ckpt_dir,
        args.dataset_root,
        args.config_name,
        args.num_batches,
        batch_size=args.batch_size,
        repo_id=args.repo_id,
        vis_dir=vis_dir,
        results_dir=args.results_dir or None,
        sample_steps=args.sample_steps,
        num_workers=args.num_workers,
        run_subtask_inference=bool(args.run_subtask_inference),
        use_seconds=args.use_seconds,
        fps=args.fps,
        vis_gap=args.vis_gap,
    )
