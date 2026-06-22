#!/usr/bin/env python3
"""Value Model Benchmark — Four-Quadrant Evaluation on Self-Play Data.

Runs value model inference on self-play datasets, classifies episodes into
four quadrants (TP/TN/FP/FN), computes metrics with priority levels, and
generates visualizations.

Usage:
    python scripts/benchmark/run_benchmark.py \
        --ckpt_dir checkpoints/pi06_seatbelt_value_negatives/.../10000 \
        --config_name src/openpi/configs/cfg_pi06_seatbelt_value_negatives.py \
        --dataset_root /mnt/oss_data/.../seatbelt \
        --repo_ids batch.1 batch.2 \
        --output_dir test_results/benchmark/seatbelt_20260324/value_negatives_10k
"""

from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import json
import logging
import os
from pathlib import Path
import sys

import flax.nnx as nnx
import jax
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from openpi.shared import nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader_rl as _data_loader
import openpi.training.sharding as sharding

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from scripts.benchmark.classification_metrics import compute_classification_report
from scripts.benchmark.data_parser import EpisodeInfo
from scripts.benchmark.data_parser import Quadrant
from scripts.benchmark.data_parser import construct_ideal_target
from scripts.benchmark.data_parser import load_episode_metadata
from scripts.benchmark.data_parser import split_by_quadrant
from scripts.benchmark.metrics import MetricPriority
from scripts.benchmark.metrics import compute_episode_metrics
from scripts.benchmark.metrics import compute_quadrant_summary
from scripts.benchmark.viz.curve_plots import save_value_curve
from scripts.train import init_train_state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Episode gap detection (from test_rl.py)
# ---------------------------------------------------------------------------


class FindGap:
    def __init__(self, threshold=2):
        self.last_frame_index = None
        self.threshold = threshold

    def __call__(self, frame_index_np):
        if self.last_frame_index is not None and abs(frame_index_np[0] - self.last_frame_index) > self.threshold:
            self.last_frame_index = frame_index_np[-1]
            return 0
        self.last_frame_index = frame_index_np[-1]
        for i in range(1, len(frame_index_np)):
            if abs(frame_index_np[i] - frame_index_np[i - 1]) > self.threshold:
                return i
        return None


# ---------------------------------------------------------------------------
# Inference — model loaded ONCE, then reused across repos
# ---------------------------------------------------------------------------


def _init_model(config_name: str, ckpt_dir: str, dataset_root: str, first_repo_id: str, batch_size: int):
    """Initialize model, mesh, sharding once. Returns (config, score_fn, mesh, data_sharding, rng)."""
    checkpoint_dir = Path(ckpt_dir)

    checkpoint_base_dir = checkpoint_dir.parent.parent.parent
    config = _config.get_config(config_name)
    exp_name = checkpoint_dir.parent.name
    task_name = checkpoint_dir.parent.parent.name  # actual dir name, may differ from config.name
    step = checkpoint_dir.name

    config = dataclasses.replace(
        config,
        name=task_name,
        checkpoint_base_dir=str(checkpoint_base_dir),
        exp_name=exp_name,
        batch_size=batch_size,
        num_workers=0,  # avoid DataLoader worker thread leaks during init
    )
    print(f"[BENCH] Config: {config.name}, exp: {config.exp_name}", flush=True)

    # Build a data_cfg for model init (need any valid repo to get shapes)
    data_cfg = config.data.create(config.assets_dirs, config.model)
    data_cfg = dataclasses.replace(
        data_cfg,
        root_dir=str(dataset_root),
        episode_fail=[0],
        repo_id=[first_repo_id],
    )

    norm_stats_path = checkpoint_dir / "assets" / data_cfg.asset_id
    if norm_stats_path.exists():
        loaded = _normalize.load(norm_stats_path)
        data_cfg = dataclasses.replace(data_cfg, norm_stats=loaded)
        print(f"[BENCH] Loaded norm_stats from {norm_stats_path}", flush=True)
    else:
        raise FileNotFoundError(f"norm_stats not found: {norm_stats_path}")

    class _SimpleFactory:
        def __init__(self, cfg):
            self._data_cfg = cfg
            self.episode_fail = cfg.episode_fail
            self.dataset_length = None

        def create(self, assets_dirs, model_config):
            return self._data_cfg

    config = dataclasses.replace(config, data=_SimpleFactory(data_cfg))

    print("[BENCH] Making mesh...", flush=True)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    print("[BENCH] Initializing checkpoint manager...", flush=True)
    checkpoint_manager, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )

    print("[BENCH] Creating data loader...", flush=True)
    data_loader = _data_loader.create_rl_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        num_batches=None,
        skip_norm_stats=False,
        drop_last=False,
    )

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    print("[BENCH] init_train_state...", flush=True)
    train_state_shape, _ = init_train_state(config, init_rng, mesh, resume=True)
    print("[BENCH] Restoring checkpoint...", flush=True)
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, data_loader, int(step))

    print("[BENCH] Merging model...", flush=True)
    model = nnx.merge(train_state.model_def, train_state.params)
    model.eval()
    score_observation_jit = nnx_utils.module_jit(model.score_observation)
    print("[BENCH] Model ready.", flush=True)

    # Free train_state to save memory
    del train_state, train_state_shape, data_loader
    jax.clear_caches()

    return config, score_observation_jit, mesh, data_sharding, rng


def _make_data_loader(config, dataset_root: str, repo_id: str, data_sharding, override_prompt: str | None = None):
    """Create a data_loader for a specific repo, reusing norm_stats from config.

    Frame-attribute preprocessors (PruneHeadTail, GripperCount, etc.) are
    intentionally stripped so that every model is evaluated on the same raw
    frame set.  Only the model weights should differ across runs.
    """
    base_data_cfg = config.data.create(config.assets_dirs, config.model)
    data_cfg = dataclasses.replace(
        base_data_cfg,
        repo_id=[repo_id],
        root_dir=str(dataset_root),
        frame_attributes_preprocessors=[],  # uniform raw frames for fair comparison
    )
    if override_prompt is not None:
        from openpi import transforms as _transforms_mod

        # Force-override prompt: replaces InjectDefaultPrompt with a transform
        # that unconditionally sets the prompt, overriding any value from
        # RepackTransform('prompt': 'task') or PromptFromEpisodeTask.
        @dataclasses.dataclass(frozen=True)
        class _ForcePrompt(_transforms_mod.DataTransformFn):
            prompt: str

            def __call__(self, data):
                data["prompt"] = np.asarray(self.prompt)
                return data

        old_model_tf = data_cfg.model_transforms
        new_inputs = []
        for tf in old_model_tf.inputs:
            if isinstance(tf, _transforms_mod.InjectDefaultPrompt):
                new_inputs.append(_ForcePrompt(override_prompt))
            else:
                new_inputs.append(tf)
        data_cfg = dataclasses.replace(
            data_cfg,
            model_transforms=_transforms_mod.Group(inputs=new_inputs, outputs=old_model_tf.outputs),
        )
        print(f"[BENCH] Override prompt (forced): '{override_prompt}'", flush=True)

    class _SimpleFactory:
        def __init__(self, cfg):
            self._data_cfg = cfg
            self.episode_fail = cfg.episode_fail
            self.dataset_length = None

        def create(self, assets_dirs, model_config):
            return self._data_cfg

    cfg = dataclasses.replace(config, data=_SimpleFactory(data_cfg))
    return _data_loader.create_rl_data_loader(
        cfg,
        sharding=data_sharding,
        shuffle=False,
        num_batches=None,
        skip_norm_stats=False,
        drop_last=False,
    )


def run_inference_on_repo(
    score_observation_jit,
    data_loader,
    rng,
    repo_id: str,
    repo_path: Path,
    collect_images: bool = False,
    video_output_dir: Path | None = None,
) -> dict:
    """Run inference on a single repo using an already-loaded model.

    Args:
        score_observation_jit: JIT-compiled score function.
        data_loader: Data loader for the repo.
        rng: JAX random key.
        repo_id: Repository identifier.
        repo_path: Path to the repository.
        collect_images: If True, also collect head camera images for video generation.
        video_output_dir: If set, generate per-episode videos to this directory.

    Returns:
        dict mapping episode_index -> {"pred": np.array, "gt": np.array}
    """
    chunk_dir = repo_path / "data" / "chunk-000"
    total_episodes = sum(1 for f in chunk_dir.iterdir() if f.is_file())
    print(f"[BENCH] Total episodes in {repo_id}: {total_episodes}", flush=True)

    generate_videos = collect_images and video_output_dir is not None

    find_gap = FindGap()
    data_iter = iter(data_loader)
    current_episode = 0
    episode_results: dict = {}

    buf_preds = []
    buf_gts = []
    buf_frames = []
    buf_images = []  # head camera images for video

    max_batches = 10000  # safety limit
    exhausted_naturally = False

    def _finalize_episode(ep_idx, preds, gts, images_head):
        """Save episode results and optionally generate video."""
        all_pred = np.concatenate(preds)
        all_gt = np.concatenate(gts)
        episode_results[ep_idx] = {"pred": all_pred, "gt": all_gt}
        print(f"[BENCH]   Episode {ep_idx}: {len(all_pred)} frames", flush=True)

        if generate_videos and images_head:
            from scripts.benchmark.viz.video import create_value_curve_video

            all_imgs = np.concatenate(images_head, axis=0)
            vid_path = str(video_output_dir / f"ep_{ep_idx:04d}_value_curve.mp4")
            create_value_curve_video(all_imgs, all_pred, all_gt, vid_path, fps=30)
            del all_imgs  # free memory immediately

    for batch_idx in range(max_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            exhausted_naturally = True
            break

        observation, _, _ = batch
        rng, subkey = jax.random.split(rng)
        pred_value = score_observation_jit(subkey, observation)
        pred_value = pred_value.block_until_ready()

        pred_np = jax.device_get(pred_value).ravel()
        gt_np = jax.device_get(observation.returns).ravel()
        frame_np = jax.device_get(observation.frame_index).ravel()

        # Collect head images for video (only when needed)
        if collect_images:
            img_head = jax.device_get(observation.images["base_0_rgb"])
            buf_images.append(img_head)

        gap = find_gap(frame_np)
        if gap is not None:
            if buf_preds:
                imgs_for_ep = [*buf_images[:-1], buf_images[-1][:gap]] if collect_images else []
                _finalize_episode(
                    current_episode,
                    [*buf_preds, pred_np[:gap]],
                    [*buf_gts, gt_np[:gap]],
                    imgs_for_ep,
                )

            current_episode += 1
            if current_episode >= total_episodes:
                # Reached the last episode boundary; clear the buffer so the
                # post-loop "finalize last episode" branch doesn't emit a
                # phantom (n+1)th episode containing only post-tail padding.
                buf_preds = []
                exhausted_naturally = True
                break

            buf_preds = [pred_np[gap:]]
            buf_gts = [gt_np[gap:]]
            buf_frames = [frame_np[gap:]]
            buf_images = [img_head[gap:]] if collect_images else []
        else:
            buf_preds.append(pred_np)
            buf_gts.append(gt_np)
            buf_frames.append(frame_np)

        if batch_idx % 100 == 0:
            print(f"[BENCH]   Batch {batch_idx}, episode {current_episode}/{total_episodes}", flush=True)

    # Finalize last episode
    if buf_preds:
        _finalize_episode(current_episode, buf_preds, buf_gts, buf_images)

    if not exhausted_naturally:
        raise RuntimeError(
            f"max_batches={max_batches} reached for {repo_id} before the data loader was "
            f"exhausted ({len(episode_results)}/{total_episodes} episodes finalized). "
            f"Tail episodes are likely missing — increase max_batches and rerun."
        )

    print(f"[BENCH] Inference complete for {repo_id}: {len(episode_results)} episodes", flush=True)
    return episode_results


# ---------------------------------------------------------------------------
# Benchmark pipeline
# ---------------------------------------------------------------------------


def _resolve_role_alignment(override_prompt: str | None) -> dict[str, bool]:
    """Decide whether each role is aligned with the (possibly overridden) prompt.

    Returns a {role: aligned} mapping. Aligned means "the role's natural behavior
    moves toward the currently-evaluated prompt's goal", which controls whether
    the ideal target trajectory is -1→0 (aligned) or 0→-1 (misaligned).

    Without an override prompt, every role evaluates against its own natural
    task, so all roles are aligned. With an override prompt this is a substring
    heuristic on "take" — see the call site for the rationale.
    """
    if override_prompt is None:
        return {"builder": True, "destroyer": True}
    is_takeoff = "take" in override_prompt.lower()
    # Builder's natural goal is "hang"; destroyer's is "take off".
    return {
        "builder": not is_takeoff,
        "destroyer": is_takeoff,
    }


def run_benchmark(
    config_name: str,
    ckpt_dir: str,
    dataset_root: str,
    repo_ids: list[str],
    output_dir: str,
    batch_size: int = 128,
    max_vis_per_quadrant: int = 5,
    override_prompt: str | None = None,
    generate_videos: bool = False,
    max_video_episodes: int = 3,
    strict: bool = True,
):
    """Full benchmark pipeline: inference → classify → metrics → visualize."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve per-role alignment with the (override) prompt up front so the
    # downstream metric/plot code never has to inspect prompt strings. The
    # benchmark measures `pred` against a prompt-conditioned linear surrogate;
    # `aligned=True` means the role's natural behavior moves toward the
    # currently-evaluated prompt's goal.
    #
    # When override_prompt is None each role uses its own natural prompt, so
    # both roles are aligned. When override_prompt is set we still need a
    # heuristic on prompt text — substring match on "take" — because there is
    # no first-class CLI flag for "is this a take-off prompt". This heuristic
    # is seatbelt-specific; non-seatbelt scenarios should pass natural prompts
    # or extend this resolver.
    aligned_for_role = _resolve_role_alignment(override_prompt)
    print(
        f"[BENCH] Role alignment for prompt={override_prompt!r}: {aligned_for_role}",
        flush=True,
    )

    # 1. Load metadata + classify
    all_episodes: list[EpisodeInfo] = []
    for repo_id in repo_ids:
        repo_path = Path(dataset_root) / repo_id
        episodes = load_episode_metadata(repo_path)
        all_episodes.extend(episodes)
        print(f"[BENCH] {repo_id}: {len(episodes)} episodes loaded", flush=True)

    quadrant_split = split_by_quadrant(all_episodes)
    for q, eps in quadrant_split.items():
        print(f"[BENCH]   {q.value}: {len(eps)} episodes", flush=True)

    # 2. Init model ONCE, then run inference per repo_id
    print("[BENCH] Initializing model (one-time)...", flush=True)
    config, score_fn, mesh, data_sharding, rng = _init_model(
        config_name,
        ckpt_dir,
        dataset_root,
        repo_ids[0],
        batch_size,
    )

    video_dir = output_dir / "visualization" / "videos" if generate_videos else None
    if video_dir is not None:
        video_dir.mkdir(parents=True, exist_ok=True)

    all_inference: dict[str, dict] = {}
    failed_repos: list[str] = []
    for repo_id in repo_ids:
        print(f"[BENCH] Running inference on {repo_id}...", flush=True)
        repo_path = Path(dataset_root) / repo_id
        try:
            dl = _make_data_loader(config, dataset_root, repo_id, data_sharding, override_prompt=override_prompt)
            results = run_inference_on_repo(
                score_fn,
                dl,
                rng,
                repo_id,
                repo_path,
                collect_images=generate_videos,
                video_output_dir=video_dir,
            )
            del dl  # free data loader memory

            metadata = load_episode_metadata(repo_path)
            metadata_indices = {ep.episode_index for ep in metadata}
            missing = metadata_indices - results.keys()
            if missing:
                msg = (
                    f"{repo_id}: {len(missing)} episodes from metadata were not produced by "
                    f"inference (sample: {sorted(missing)[:5]})"
                )
                if strict:
                    raise RuntimeError(msg)
                print(f"[BENCH] WARNING: {msg}", flush=True)

            for ep in metadata:
                local_idx = ep.episode_index
                if local_idx in results:
                    all_inference[f"{repo_id}:{local_idx}"] = {
                        "pred": results[local_idx]["pred"],
                        "gt": results[local_idx]["gt"],
                        "episode_info": ep,
                    }
        except Exception as e:
            if strict:
                raise
            failed_repos.append(repo_id)
            print(f"[BENCH] ERROR processing {repo_id}: {e}", flush=True)
            continue

    if failed_repos:
        print(f"[BENCH] WARNING: {len(failed_repos)} repo(s) failed: {failed_repos}", flush=True)
    print(f"[BENCH] Total inferred episodes: {len(all_inference)}", flush=True)

    # 2b. Rename videos with quadrant prefix (TP_/TN_/FP_/FN_)
    if video_dir is not None and video_dir.exists():
        for key, data in all_inference.items():
            ep_info = data["episode_info"]
            local_idx = ep_info.episode_index
            old_name = video_dir / f"ep_{local_idx:04d}_value_curve.mp4"
            new_name = video_dir / f"{ep_info.quadrant.value}_ep_{local_idx:04d}_value_curve.mp4"
            if old_name.exists():
                old_name.rename(new_name)

    # 3. Compute per-episode metrics against the ideal target trajectory
    quadrant_metrics: dict[str, list[dict]] = {q.value: [] for q in Quadrant}
    episode_details: list[dict] = []

    for key, data in all_inference.items():
        ep_info = data["episode_info"]
        pred = data["pred"]

        # Build the prompt-conditioned ideal target trajectory. We deliberately
        # do not compare against `data["gt"]` (the dataset's recorded returns):
        # the metrics here measure how closely `pred` tracks the surrogate
        # target implied by the evaluation prompt.
        ideal_target = construct_ideal_target(len(pred), aligned=aligned_for_role[ep_info.role])

        metrics = compute_episode_metrics(pred, ideal_target)
        metrics["episode_key"] = key
        metrics["quadrant"] = ep_info.quadrant.value
        metrics["role"] = ep_info.role
        metrics["success"] = ep_info.success
        metrics["value_score"] = ep_info.value_score
        metrics["n_frames"] = len(pred)
        metrics["tail_pred"] = float(pred[-1])
        metrics["tail_target"] = float(ideal_target[-1])

        quadrant_metrics[ep_info.quadrant.value].append(metrics)
        episode_details.append(metrics)

    # 4. Compute quadrant summaries
    quadrant_summaries = {}
    for q_name, eps_metrics in quadrant_metrics.items():
        summary = compute_quadrant_summary(eps_metrics)
        summary["priority"] = MetricPriority.for_quadrant(q_name)
        quadrant_summaries[q_name] = summary
        print(f"\n=== {q_name.upper()} ({summary['n_episodes']} episodes) ===", flush=True)
        print(f"  tail_mse: {summary['mean_tail_mse']:.6f} ± {summary['std_tail_mse']:.6f}", flush=True)
        print(f"  head_mse: {summary['mean_head_mse']:.6f} ± {summary['std_head_mse']:.6f}", flush=True)
        print(f"  mae:      {summary['mean_mae']:.6f} ± {summary['std_mae']:.6f}", flush=True)
        print(f"  rmse:     {summary['mean_rmse']:.6f} ± {summary['std_rmse']:.6f}", flush=True)
        print(f"  pearson:  {summary['mean_pearson']:.6f} ± {summary['std_pearson']:.6f}", flush=True)
        print(f"  r²:       {summary['mean_r_squared']:.6f} ± {summary['std_r_squared']:.6f}", flush=True)

    # 5. Save metrics
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    with open(metrics_dir / "quadrant_summaries.json", "w") as f:
        json.dump(quadrant_summaries, f, indent=2, default=str)

    with open(metrics_dir / "episode_details.json", "w") as f:
        json.dump(episode_details, f, indent=2, default=str)

    # 5b. Classification metrics (per role)
    classification_reports: dict[str, dict] = {}
    for role in ["builder", "destroyer"]:
        role_details = [d for d in episode_details if d["role"] == role]
        cls_report = compute_classification_report(role_details, role)
        if cls_report is not None:
            classification_reports[role] = cls_report
            cls_path = metrics_dir / f"classification_{role}.json"
            with open(cls_path, "w") as f:
                json.dump(cls_report, f, indent=2, default=str)
            print(
                f"[BENCH] Classification ({role}): AUC={cls_report['auc']:.4f}, "
                f"F1={cls_report['optimal_threshold']['f1']:.4f}, "
                f"separation={cls_report['separation_score']:.4f}, "
                f"success={cls_report['n_success']}, failure={cls_report['n_failure']}",
                flush=True,
            )

    # 6. Visualizations
    vis_dir = output_dir / "visualization"

    # 6a. Per-episode value curves (sample per quadrant)
    for q_name, eps_metrics in quadrant_metrics.items():
        q_vis_dir = vis_dir / "per_episode" / q_name
        for i, m in enumerate(eps_metrics[:max_vis_per_quadrant]):
            key = m["episode_key"]
            data = all_inference[key]
            pred = data["pred"]
            ideal_target = construct_ideal_target(len(pred), aligned=aligned_for_role[data["episode_info"].role])
            title = f"{q_name} | {key} | tail_mse={m['tail_mse']:.4f}"
            save_value_curve(pred, ideal_target, str(q_vis_dir / f"ep_{i}_value_curve.png"), title)

    # 6b. Distribution histograms — tail frame pred values per quadrant
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    quadrant_names = [q.value for q in Quadrant]
    for ax, q_name in zip(axes.flat, quadrant_names):
        eps_metrics = quadrant_metrics[q_name]
        if not eps_metrics:
            ax.set_title(f"{q_name} (no data)")
            continue
        tail_preds = []
        for m in eps_metrics:
            key = m["episode_key"]
            pred = all_inference[key]["pred"]
            tail_preds.append(float(pred[-1]))
        ax.hist(tail_preds, bins=20, alpha=0.7, color="#1f77b4", edgecolor="black")
        ax.set_title(f"{q_name} (n={len(eps_metrics)})")
        ax.set_xlabel("Tail Pred Value")
        ax.set_ylabel("Count")
        ax.axvline(x=0.0, color="green", linestyle="--", alpha=0.5, label="target=0 (TP)")
        ax.axvline(x=-1.0, color="red", linestyle="--", alpha=0.5, label="target=-1 (TN)")
        ax.legend(fontsize=8)
    fig.suptitle("Tail Frame Predicted Value Distribution by Quadrant")
    fig.tight_layout()
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(vis_dir / "distribution_hist.png", dpi=150)
    plt.close(fig)

    # 6c. Scatter plot — pred vs target tail values for TP and TN
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, q_name, target_value in zip(axes, ["true_positive", "true_negative"], [0.0, -1.0]):
        eps_metrics = quadrant_metrics[q_name]
        if not eps_metrics:
            continue
        tail_preds = [float(all_inference[m["episode_key"]]["pred"][-1]) for m in eps_metrics]
        tail_targets = [target_value] * len(tail_preds)
        ax.scatter(tail_targets, tail_preds, alpha=0.6, s=30)
        ax.plot([-1.5, 0.5], [-1.5, 0.5], "k--", alpha=0.3, label="y=x")
        ax.set_xlabel("Ideal Tail Target")
        ax.set_ylabel("Pred Tail Value")
        ax.set_title(f"{q_name} (n={len(eps_metrics)})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Tail Value: Pred vs Ideal Target")
    fig.tight_layout()
    fig.savefig(vis_dir / "scatter_tail_pred_gt.png", dpi=150)
    plt.close(fig)

    # 6d. Full scatter — all episodes pred vs ideal target across full trajectory
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    colors = {
        "true_positive": "#2ecc71",
        "true_negative": "#e74c3c",
        "false_positive": "#f39c12",
        "false_negative": "#9b59b6",
    }
    for q_name in quadrant_names:
        eps_metrics = quadrant_metrics[q_name]
        all_pred_vals = []
        all_target_vals = []
        for m in eps_metrics:
            key = m["episode_key"]
            pred = all_inference[key]["pred"]
            ideal_target = construct_ideal_target(
                len(pred), aligned=aligned_for_role[all_inference[key]["episode_info"].role]
            )
            # Subsample for scatter
            step = max(1, len(pred) // 20)
            all_pred_vals.extend(pred[::step].tolist())
            all_target_vals.extend(ideal_target[::step].tolist())
        if all_pred_vals:
            ax.scatter(all_target_vals, all_pred_vals, alpha=0.3, s=10, c=colors[q_name], label=q_name)
    ax.plot([-1.5, 0.5], [-1.5, 0.5], "k--", alpha=0.3)
    ax.set_xlabel("Ideal Target Value")
    ax.set_ylabel("Pred Value")
    ax.set_title("Pred vs Ideal Target (subsampled)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "scatter_pred_gt_all.png", dpi=150)
    plt.close(fig)

    # 6e. Calibration scatter — pred vs ideal target (all frames, subsampled)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    all_pred_flat = []
    all_target_flat = []
    for key, data in all_inference.items():
        pred = data["pred"]
        ideal_target = construct_ideal_target(len(pred), aligned=aligned_for_role[data["episode_info"].role])
        step = max(1, len(pred) // 10)
        all_pred_flat.extend(pred[::step].tolist())
        all_target_flat.extend(ideal_target[::step].tolist())
    ax.scatter(all_target_flat, all_pred_flat, alpha=0.15, s=5, c="#3498db")
    ax.plot([-1.2, 0.2], [-1.2, 0.2], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Ideal Target Value")
    ax.set_ylabel("Pred Value")
    ax.set_title("Calibration: Pred vs Ideal Target (all frames, subsampled)")
    ax.set_xlim(-1.2, 0.2)
    ax.set_ylim(-1.2, 0.2)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "calibration_scatter.png", dpi=150)
    plt.close(fig)

    # 7. Generate markdown report
    _generate_report(output_dir, quadrant_summaries, len(all_inference), classification_reports)

    print(f"\n[BENCH] Benchmark complete. Results saved to {output_dir}", flush=True)
    return quadrant_summaries


def _generate_report(
    output_dir: Path,
    summaries: dict,
    total_episodes: int,
    classification_reports: dict[str, dict] | None = None,
):
    """Generate a markdown summary report."""
    lines = [
        "# Value Model Benchmark Report",
        "",
        f"**Total episodes**: {total_episodes}",
        f"**Generated**: {datetime.now(timezone(timedelta(hours=8))).isoformat()}",
        "",
        "## What this benchmark measures",
        "",
        "All metrics in this report compare model predictions against an **ideal",
        "prompt-conditioned target trajectory** (a linear -1→0 ramp when the role is",
        "aligned with the evaluation prompt, 0→-1 when misaligned). This is a surrogate,",
        "not the dataset's recorded returns. On cleaned data with high success rates,",
        "the surrogate is a reasonable proxy for the true value signal; on noisy data",
        "or for tasks whose state space is not monotonically ordered, treat these",
        "numbers with caution.",
        "",
        "## Quadrant Summary",
        "",
        "| Quadrant | N | Tail MSE | Head MSE | MAE | RMSE | Pearson | R² |",
        "|----------|---|----------|----------|-----|------|---------|----|",
    ]

    for q_name in ["true_positive", "true_negative", "false_positive", "false_negative"]:
        s = summaries.get(q_name, {})
        n = s.get("n_episodes", 0)
        if n == 0:
            lines.append(f"| {q_name} | 0 | - | - | - | - | - | - |")
        else:
            lines.append(
                f"| {q_name} | {n} "
                f"| {s['mean_tail_mse']:.4f}±{s['std_tail_mse']:.4f} "
                f"| {s['mean_head_mse']:.4f}±{s['std_head_mse']:.4f} "
                f"| {s['mean_mae']:.4f}±{s['std_mae']:.4f} "
                f"| {s['mean_rmse']:.4f}±{s['std_rmse']:.4f} "
                f"| {s['mean_pearson']:.4f}±{s['std_pearson']:.4f} "
                f"| {s['mean_r_squared']:.4f}±{s['std_r_squared']:.4f} |"
            )

    # Median table
    lines.extend(
        [
            "",
            "## Median Metrics (robust to outliers)",
            "",
            "| Quadrant | N | Median Tail MSE | Median Pearson | Median MAE |",
            "|----------|---|-----------------|----------------|------------|",
        ]
    )
    for q_name in ["true_positive", "true_negative", "false_positive", "false_negative"]:
        s = summaries.get(q_name, {})
        n = s.get("n_episodes", 0)
        if n == 0:
            lines.append(f"| {q_name} | 0 | - | - | - |")
        else:
            lines.append(
                f"| {q_name} | {n} "
                f"| {s.get('median_tail_mse', float('nan')):.4f} "
                f"| {s.get('median_pearson', float('nan')):.4f} "
                f"| {s.get('median_mae', float('nan')):.4f} |"
            )

    # Length-bucketed Pearson
    lines.extend(
        [
            "",
            "## Pearson by Episode Length",
            "",
            "| Quadrant | Short (<500f) | Medium (500-1000f) | Long (>1000f) |",
            "|----------|---------------|-------------------|---------------|",
        ]
    )
    for q_name in ["true_positive", "true_negative"]:
        s = summaries.get(q_name, {})
        n = s.get("n_episodes", 0)
        if n == 0:
            lines.append(f"| {q_name} | - | - | - |")
        else:
            parts = [f"| {q_name}"]
            for bname in ["short", "medium", "long"]:
                bn = s.get(f"pearson_{bname}_n", 0)
                bm = s.get(f"pearson_{bname}_median", float("nan"))
                parts.append(f" {bm:.3f} (n={bn})" if bn > 0 else " - ")
            lines.append(" |".join(parts) + " |")

    lines.extend(
        [
            "",
            "## Metric Priority",
            "",
            "- **High** (tail MSE for TP/TN): ideal-target endpoint known with certainty",
            "- **Medium** (MAE, RMSE, head MSE): ideal-target trend approximately known",
            "- **Low** (Pearson, R²): trend correlation reference",
        ]
    )

    # Classification Performance section (only when failure samples exist)
    if classification_reports:
        lines.extend(["", "## Classification Performance", ""])
        role_labels = {"builder": "Builder (fold)", "destroyer": "Destroyer (unfold)"}
        for role, report in classification_reports.items():
            opt = report["optimal_threshold"]
            lines.extend(
                [
                    f"### {role_labels.get(role, role)}",
                    "",
                    "| Metric | Value |",
                    "|--------|-------|",
                    f"| AUC-ROC | {report['auc']:.4f} |",
                    f"| Optimal Threshold | {opt['threshold']:.4f} |",
                    f"| F1 | {opt['f1']:.4f} |",
                    f"| Precision | {opt['precision']:.4f} |",
                    f"| Recall | {opt['recall']:.4f} |",
                    f"| Separation Score | {report['separation_score']:.4f} |",
                    f"| Success / Failure | {report['n_success']} / {report['n_failure']} |",
                    "",
                ]
            )

    lines.extend(
        [
            "",
            "## Visualizations",
            "",
            "- `visualization/distribution_hist.png` — Tail value distributions",
            "- `visualization/scatter_tail_pred_gt.png` — Tail pred vs ideal target",
            "- `visualization/scatter_pred_gt_all.png` — Full trajectory scatter (pred vs ideal target)",
            "- `visualization/calibration_scatter.png` — Pred vs ideal-target calibration (all frames)",
            "- `visualization/per_episode/` — Per-episode value curves",
        ]
    )

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"[BENCH] Report saved to {report_path}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Value Model Benchmark")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--repo_ids", type=str, nargs="+", required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_vis_per_quadrant", type=int, default=5)
    parser.add_argument(
        "--override_prompt",
        type=str,
        default=None,
        help="Force all episodes to use this prompt (ignoring episode-level task)",
    )
    parser.add_argument(
        "--generate_videos",
        action="store_true",
        default=False,
        help="Generate per-episode value curve videos (camera | curve split-screen)",
    )
    parser.add_argument(
        "--max_video_episodes", type=int, default=3, help="Max episodes to generate videos for (per repo)"
    )
    parser.add_argument(
        "--no_strict",
        dest="strict",
        action="store_false",
        default=True,
        help="Continue on per-repo errors instead of failing fast (default: strict).",
    )

    args = parser.parse_args()

    run_benchmark(
        config_name=args.config_name,
        ckpt_dir=args.ckpt_dir,
        dataset_root=args.dataset_root,
        repo_ids=args.repo_ids,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_vis_per_quadrant=args.max_vis_per_quadrant,
        override_prompt=args.override_prompt,
        generate_videos=args.generate_videos,
        max_video_episodes=args.max_video_episodes,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
