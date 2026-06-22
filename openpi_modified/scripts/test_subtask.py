"""Minimal JAX-based evaluation script inspired by `scripts/train.py`.

Behaviors:
- Runs model parallel inference offline on user-provided dataset.

Usage (example):
uv run scripts/test.py \
    --ckpt_dir checkpoints/cfg_pi0.5_28_dim.all_public_datasets/cfg_pi0.5_28_dim.all_public_datasets_exp/10000 \
    --dataset_root /mnt/oss_data/ \
    --config_name src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.all_public_datasets.py \
    --num_batches 10 \
    --batch_size 64 \
    --repo_id lerobot/aloha_mobile_cabinet
"""

from collections import Counter
import dataclasses
import logging
import os
from pathlib import Path
import time

import flax.nnx as nnx
import jax
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw

from openpi.models import tokenizer as _tokenizer
from openpi.shared import nnx_utils
import openpi.shared.normalize as _normalize
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.sharding as sharding
import openpi.transforms as transforms


def init_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def calculate_total_frames(data_loader: _data_loader.DataLoader) -> int:
    inner = getattr(data_loader, "_data_loader", None)
    torch_loader = inner.torch_loader
    ds = getattr(torch_loader, "dataset", None)
    return len(ds)


def visualize_pred_vs_gt(pred_seq, gt_seq, filename: str | None = None):
    """Visualize prediction vs GT for 14 dimensions over time.

    Args:
        pred_seq: array-like shape (T, action_dim) predicted sequence for one example.
        gt_seq: array-like shape (T, action_dim) ground-truth sequence for one example.
        filename: optional filename to save the figure. If None, uses
            `vis_pred_vs_gt_{timestamp}.png`.
    """
    pred = np.asarray(pred_seq)
    gt = np.asarray(gt_seq)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != gt.shape[1]:
        raise ValueError("pred_seq and gt_seq must have shape (T, action_dim)")
    if pred.shape[0] != gt.shape[0]:
        raise ValueError("pred_seq and gt_seq must have the same temporal length T")

    num_steps = pred.shape[0]
    n = pred.shape[1]
    rows = 2
    cols = n // rows

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.flatten()

    x = list(range(num_steps))
    for i in range(n):
        ax = axes[i]
        ax.plot(x, gt[:, i], label="GT", color="#1f77b4", linewidth=1.2)
        ax.plot(x, pred[:, i], label="Pred", color="#d62728", linestyle="--", linewidth=1.2)
        ax.set_title(f"Dim {i}")
        ax.set_xlabel("frames")
        ax.grid(visible=True, linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend()

    # Hide any unused axes
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(filename, dpi=300)
    print(f"Saved visualization to {filename}")
    plt.close(fig)


@dataclasses.dataclass
class SubtaskMetrics:
    """Container for subtask prediction metrics."""

    exact_match_accuracy: float
    avg_precision: float
    avg_recall: float
    avg_f1: float
    total_samples: int
    correct_samples: int
    per_sample_matches: list[bool]

    def print_summary(self):
        """Print a formatted summary of metrics."""
        print("\n" + "=" * 50)
        print("=== Subtask Prediction Metrics ===")
        print(f"Total samples: {self.total_samples}")
        print(f"\nExact Match Accuracy: {self.exact_match_accuracy:.4f} ({self.correct_samples}/{self.total_samples})")
        print("\nToken-level Metrics (averaged):")
        print(f"  - Precision: {self.avg_precision:.4f}")
        print(f"  - Recall: {self.avg_recall:.4f}")
        print(f"  - F1 Score: {self.avg_f1:.4f}")
        print("=" * 50 + "\n")


@dataclasses.dataclass
class FASTActionMSEMetrics:
    """Container for FAST action MSE metrics (decode to continuous actions)."""

    mse_pred_vs_gt_orig: float  # pred vs original GT
    mse_pred_vs_gt_recon: float  # pred vs encode-decode GT (quantization space)
    mse_gt_quantization: float  # GT_orig vs GT_recon (FAST discrete token precision loss)
    mae_pred_vs_gt_orig: float
    mae_pred_vs_gt_recon: float
    mae_gt_quantization: float
    total_samples: int
    scale_gt: float  # GT std, for scale-relative interpretation

    def _rel_mae(self, mae: float) -> float:
        """Relative MAE = MAE / scale. <0.1 good, 0.1-0.3 ok, >0.5 poor."""
        return mae / max(self.scale_gt, 1e-8)

    def print_summary(self, title: str = "FAST Action MSE Metrics (decoded)"):
        rel_mae_pred = self._rel_mae(self.mae_pred_vs_gt_orig)
        rel_mae_quant = self._rel_mae(self.mae_gt_quantization)
        print("\n" + "=" * 60)
        print(f"=== {title} ===")
        print(f"Total samples: {self.total_samples}")
        print(f"GT scale (std): {self.scale_gt:.6f}")
        print("\nPred vs Original GT:")
        print(f"  MSE: {self.mse_pred_vs_gt_orig:.6f}  MAE: {self.mae_pred_vs_gt_orig:.6f}")
        _interp = "good" if rel_mae_pred < 0.1 else "ok" if rel_mae_pred < 0.3 else "poor"
        print(f"  MAE/scale (relative): {rel_mae_pred:.4f}  [{_interp}]")
        print("\nPred vs Encode-Decode GT (same quantization space):")
        print(f"  MSE: {self.mse_pred_vs_gt_recon:.6f}  MAE: {self.mae_pred_vs_gt_recon:.6f}")
        print("\nGT Quantization Loss (orig vs encode-decode):")
        print(f"  MSE: {self.mse_gt_quantization:.6f}  MAE: {self.mae_gt_quantization:.6f}")
        print(f"  MAE/scale (relative): {rel_mae_quant:.4f}")
        print("=" * 60 + "\n")


def print_fast_decode_summary(label: str, status_counts: Counter[str]) -> None:
    total_samples = sum(status_counts.values())
    success_count = status_counts.get("ok", 0)
    failure_count = total_samples - success_count
    print("\n" + "=" * 60)
    print(f"=== {label} ===")
    print(f"Total samples: {total_samples}")
    print(f"Successful decodes: {success_count}")
    print(f"Non-success decodes: {failure_count}")
    for status, count in sorted(status_counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {status}: {count}")
    print("=" * 60 + "\n")


def compute_exact_match(pred_text: str, gt_text: str) -> bool:
    """Check if predicted text exactly matches ground truth (case-insensitive)."""
    if pred_text is None or gt_text is None:
        return pred_text == gt_text
    return pred_text.strip().lower() == gt_text.strip().lower()


def compute_token_metrics(pred_text: str, gt_text: str) -> dict:
    """Compute word-level precision, recall, and F1 score."""
    if pred_text is None or gt_text is None:
        if pred_text is None and gt_text is None:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    pred_tokens = set(pred_text.lower().split())
    gt_tokens = set(gt_text.lower().split())

    # Handle empty predictions and ground truth
    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}

    if len(pred_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if len(gt_tokens) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    intersection = pred_tokens & gt_tokens
    precision = len(intersection) / len(pred_tokens)
    recall = len(intersection) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_subtask_metrics(
    pred_tokens_list: list,
    gt_tokens_list: list,
    gt_masks_list: list,
    tokenizer: _tokenizer.FASTTokenizerWithSubtask,
) -> SubtaskMetrics:
    """Compute precision/recall metrics for subtask predictions.

    Args:
        pred_tokens_list: List of predicted token arrays
        gt_tokens_list: List of ground truth token arrays
        gt_masks_list: List of ground truth token masks
        tokenizer: Tokenizer for converting tokens to text

    Returns:
        SubtaskMetrics containing aggregated metrics
    """
    total_samples = len(pred_tokens_list)
    exact_matches = 0
    precision_list = []
    recall_list = []
    f1_list = []
    per_sample_matches = []

    for i in range(total_samples):
        pred_tokens = pred_tokens_list[i]
        gt_tokens = gt_tokens_list[i]
        gt_mask = gt_masks_list[i] if i < len(gt_masks_list) else None

        # Extract text from tokens
        pred_text = tokenizer.extract_subtask(pred_tokens)
        gt_text = (
            tokenizer.extract_subtask(gt_tokens, gt_mask)
            if gt_mask is not None
            else tokenizer.extract_subtask(gt_tokens)
        )

        # Compute metrics
        is_exact_match = compute_exact_match(pred_text, gt_text)
        exact_matches += 1 if is_exact_match else 0
        per_sample_matches.append(is_exact_match)

        token_metrics = compute_token_metrics(pred_text, gt_text)
        precision_list.append(token_metrics["precision"])
        recall_list.append(token_metrics["recall"])
        f1_list.append(token_metrics["f1"])

    exact_match_accuracy = exact_matches / total_samples if total_samples > 0 else 0.0
    avg_precision = np.mean(precision_list) if precision_list else 0.0
    avg_recall = np.mean(recall_list) if recall_list else 0.0
    avg_f1 = np.mean(f1_list) if f1_list else 0.0

    return SubtaskMetrics(
        exact_match_accuracy=exact_match_accuracy,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        avg_f1=avg_f1,
        total_samples=total_samples,
        correct_samples=exact_matches,
        per_sample_matches=per_sample_matches,
    )


def _align_and_mse_mae(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Align shapes and compute MSE, MAE."""
    pred = np.asarray(pred, dtype=np.float32)
    gt = np.asarray(gt, dtype=np.float32)
    if pred.shape != gt.shape:
        min_h = min(pred.shape[0], gt.shape[0])
        min_d = min(pred.shape[1], gt.shape[1])
        pred = pred[:min_h, :min_d]
        gt = gt[:min_h, :min_d]
    diff = pred - gt
    return float(np.mean(diff**2)), float(np.mean(np.abs(diff)))


def compute_fast_action_mse(
    pred_actions_list: list[np.ndarray],
    gt_actions_list: list[np.ndarray],
    gt_recon_actions_list: list[np.ndarray | None] | None = None,
) -> FASTActionMSEMetrics:
    """Compute MSE/MAE between decoded pred actions and GT continuous actions.

    Both ``pred_actions_list`` and ``gt_actions_list`` contain per-sample arrays
    of shape (action_horizon, action_dim).
    If ``gt_recon_actions_list`` is provided (GT after encode-decode), also reports
    pred vs recon and GT quantization loss.
    """
    total = len(pred_actions_list)
    mse_pred_orig: list[float] = []
    mae_pred_orig: list[float] = []
    mse_pred_recon: list[float] = []
    mae_pred_recon: list[float] = []
    mse_quant: list[float] = []
    mae_quant: list[float] = []

    for i in range(total):
        pred = pred_actions_list[i]
        gt = gt_actions_list[i]
        mse_po, mae_po = _align_and_mse_mae(pred, gt)
        mse_pred_orig.append(mse_po)
        mae_pred_orig.append(mae_po)

        if (
            gt_recon_actions_list is not None
            and i < len(gt_recon_actions_list)
            and gt_recon_actions_list[i] is not None
        ):
            gt_recon = gt_recon_actions_list[i]
            mse_pr, mae_pr = _align_and_mse_mae(pred, gt_recon)
            mse_q, mae_q = _align_and_mse_mae(gt, gt_recon)
            mse_pred_recon.append(mse_pr)
            mae_pred_recon.append(mae_pr)
            mse_quant.append(mse_q)
            mae_quant.append(mae_q)

    def _mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    # GT scale (std) for scale-relative interpretation
    all_gt = np.concatenate([np.asarray(g).reshape(-1) for g in gt_actions_list], axis=0)
    scale_gt = float(np.std(all_gt)) if len(all_gt) > 0 else 1.0

    return FASTActionMSEMetrics(
        mse_pred_vs_gt_orig=_mean(mse_pred_orig),
        mse_pred_vs_gt_recon=_mean(mse_pred_recon),
        mse_gt_quantization=_mean(mse_quant),
        mae_pred_vs_gt_orig=_mean(mae_pred_orig),
        mae_pred_vs_gt_recon=_mean(mae_pred_recon),
        mae_gt_quantization=_mean(mae_quant),
        total_samples=total,
        scale_gt=scale_gt,
    )


DEFAULT_TARGET_ACTION_DIM = range(14)


def main(
    checkpoint_dir: str,
    dataset_root: str,
    config_name: str,
    num_batches: int = 2,
    norm_stats_path: str | None = None,
    batch_size: int | None = None,
    vis_dir: str | None = None,
    repo_id: str | None = None,
    sample_steps: int | None = None,
    num_workers: int = 1,
    *,
    save_mismatches_only: bool = False,
    eval_fast_actions: bool = False,
):
    t0 = time.perf_counter()
    init_logging()
    meta = LeRobotDatasetMetadata(repo_id, Path(dataset_root) / repo_id)
    robot_type = meta.robot_type

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

    print(f"Using config: {config.name}, exp: {config.exp_name}, repo_id: {repo_id}, robot_type: {robot_type}")

    t1 = time.perf_counter()
    print(f"Timing: config loading took {t1 - t0:.3f}s")

    t0 = time.perf_counter()
    data_cfg = config.data.create(config.assets_dirs, config.model)
    # # not tokenize subtask in infer stage
    # data_cfg.subtask_info = None
    replace_data_kwargs = {"root_dir": str(dataset_root)}
    if repo_id is not None:
        if isinstance(repo_id, str):
            repo_id = [repo_id]
        assert isinstance(repo_id, list)
        replace_data_kwargs["repo_id"] = repo_id

    data_cfg = dataclasses.replace(data_cfg, **replace_data_kwargs)

    norm_stats_path = checkpoint_dir / "assets" / data_cfg.asset_id
    if norm_stats_path is not None:
        norm_stats_file = norm_stats_path
        if norm_stats_file.exists():
            # _normalize.load expects the directory containing norm_stats.json
            loaded = _normalize.load(norm_stats_file)
            data_cfg = dataclasses.replace(data_cfg, norm_stats=loaded)
            print(f"Loaded norm_stats from {norm_stats_file}")
        else:
            logging.warning(f"Provided norm_stats_path does not exist: {norm_stats_file}")
            raise FileNotFoundError(f"Provided norm_stats_path does not exist: {norm_stats_file}")

    # For subtask eval: prepend InjectEvalSubtaskFlags to avoid encoding GT subtask
    orig_group = data_cfg.model_transforms
    if hasattr(orig_group, "inputs") and len(orig_group.inputs) > 0:
        eval_inputs = (
            transforms.InjectEvalSubtaskFlags(),
            *orig_group.inputs,
        )
        data_cfg = dataclasses.replace(
            data_cfg,
            model_transforms=transforms.Group(inputs=eval_inputs, outputs=orig_group.outputs),
        )
    print(f"data_cfg.model_transforms: {data_cfg.model_transforms}")
    # create a shallow copy of config with a DataConfigFactory-like object that will return our data_cfg
    target_action_dim = getattr(config.data, "target_action_dim", DEFAULT_TARGET_ACTION_DIM)

    class _SimpleFactory:
        def __init__(self, data_cfg, target_action_dim=DEFAULT_TARGET_ACTION_DIM):
            self._data_cfg = data_cfg
            self.target_action_dim = target_action_dim
            self.episode_fail = None
            self.dataset_length = None

        def create(self, assets_dirs, model_config):
            return self._data_cfg

    config = dataclasses.replace(config, data=_SimpleFactory(data_cfg, target_action_dim))

    # Create mesh and sharding same as train.py
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    # Initialize checkpoint manager (resume mode)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    # Create data loader. Use norm stats so inputs/GT align with training preprocessing.
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=False,
        # num_batches=num_batches, # read all data
        skip_norm_stats=False,
    )
    # total_frames_num = calculate_total_frames(data_loader)

    t1 = time.perf_counter()
    print(f"Timing: data loader took {t1 - t0:.3f}s")  # 30s

    t0 = time.perf_counter()
    # init_train_state is defined in scripts/train.py; import here to reuse initialization logic
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from train import init_train_state

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    # Request shapes for train state while indicating resume=True so init_train_state returns shape placeholders
    train_state_shape, state_sharding = init_train_state(config, init_rng, mesh, resume=True)

    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, data_loader, int(step))

    # Merge model def + params into an executable model
    model = nnx.merge(train_state.model_def, train_state.params)
    model.eval()
    sample_fn_jit = nnx_utils.module_jit(model.sample_subtask, static_argnums=(4, 5))

    sample_fast_fn_jit = None
    sample_actions_fast = getattr(model, "_sample_actions_fast", None)
    if eval_fast_actions and sample_actions_fast is not None:
        # max_decoding_steps and temperature must be static (used in jnp.pad pad_width)
        sample_fast_fn_jit = nnx_utils.module_jit(sample_actions_fast, static_argnums=(3, 4))
        logging.info("FAST action evaluation enabled  _sample_actions_fast JIT compiled.")
    elif eval_fast_actions:
        logging.warning("eval_fast_actions=True but model has no _sample_actions_fast; skipping.")
        eval_fast_actions = False

    logging.info("Model restored and set to eval mode.")

    # 输出格式转换
    output_fns = []
    output_fns.extend(data_cfg.model_transforms.outputs)
    output_fns.append(transforms.Unnormalize(data_cfg.norm_stats, use_quantiles=data_cfg.use_quantile_norm))
    output_fns.extend(data_cfg.data_transforms.outputs)
    output_fns.extend(data_cfg.repack_transforms.outputs)

    data_iter = iter(data_loader)

    all_subtask_preds = []
    all_subtask_gts = []
    all_subtask_gt_masks = []
    all_fast_pred_actions: list[np.ndarray] = []
    all_fast_gt_actions: list[np.ndarray] = []
    all_fast_gt_recon_actions: list[np.ndarray | None] = []  # GT after encode-decode (quantization)
    valid_fast_pred_actions: list[np.ndarray] = []
    valid_fast_gt_actions: list[np.ndarray] = []
    valid_fast_gt_recon_actions: list[np.ndarray | None] = []
    pred_fast_decode_status_counts: Counter[str] = Counter()
    gt_fast_decode_status_counts: Counter[str] = Counter()
    if vis_dir is not None:
        vis_batch_head = []
        vis_batch_left = []
        vis_batch_right = []
        vis_batch_episodes = []
        vis_batch_frame_ind = []

    subtask_tokenizer = _tokenizer.FASTTokenizerWithSubtask(
        max_subtask_len=32,
        max_len=160,
        encode_subtask=False,
        encode_actions=False,
    )

    t1 = time.perf_counter()
    print(f"Timing: prepare took {t1 - t0:.3f}s")  # 38s

    for i in range(num_batches):
        t0 = time.perf_counter()
        print(f"Current batch: {i}")

        batch = next(data_iter)
        observation, gt_actions_batch, gt_actions_mask = batch
        t1 = time.perf_counter()
        print(f"- Timing: get batch data: {t1 - t0:.3f}s")

        if vis_dir is not None:
            images = observation.images
            img_head = images["base_0_rgb"]
            img_left_wrist = images["left_wrist_0_rgb"]
            img_right_wrist = images["right_wrist_0_rgb"]
            vis_batch_head.append(img_head)
            vis_batch_left.append(img_left_wrist)
            vis_batch_right.append(img_right_wrist)
            vis_batch_episodes.append(observation.episode_index)
            vis_batch_frame_ind.append(observation.frame_index)

        t0 = time.perf_counter()
        # build RNG per batch
        rng, subkey = jax.random.split(rng)

        sampled = sample_fn_jit(subkey, observation)
        sampled = sampled.block_until_ready()
        sampled_np = jax.device_get(sampled)

        t1 = time.perf_counter()
        print(f"- Timing: sample_subtask: {t1 - t0:.3f}s")

        t0 = time.perf_counter()
        obs_dict = observation.to_dict() if hasattr(observation, "to_dict") else {}
        obs_host = jax.tree_map(lambda x: np.asarray(x), obs_dict)
        subtask_tokens = obs_host.get("subtask_tokens")
        subtask_tokens_mask = obs_host.get("subtask_tokens_mask")
        sampled_np = sampled_np.astype(np.int32)

        all_subtask_preds.append(sampled_np)
        all_subtask_gts.append(subtask_tokens)
        all_subtask_gt_masks.append(subtask_tokens_mask)

        t1 = time.perf_counter()
        print(f"- Timing: Infer + post process: {t1 - t0:.3f}s")

        # --- FAST action evaluation (decode to continuous, compute MSE) ---
        if eval_fast_actions and sample_fast_fn_jit is not None:
            t0 = time.perf_counter()
            rng, fast_subkey = jax.random.split(rng)
            max_decode = getattr(model, "max_decoding_steps", 64)
            fast_result = sample_fast_fn_jit(fast_subkey, observation, max_decode, 0.0, sampled)
            fast_full_tokens = fast_result["fast_action_tokens"]
            fast_full_tokens.block_until_ready()
            fast_full_np = np.asarray(jax.device_get(fast_full_tokens)).astype(np.int32)

            action_horizon = config.model.action_horizon
            gt_act_np = np.asarray(jax.device_get(gt_actions_batch))
            target_action_dim = getattr(config.data, "target_action_dim", range(14))
            effective_action_dim = len(target_action_dim)
            gt_act_slice = gt_act_np[..., target_action_dim]

            for b in range(gt_act_slice.shape[0]):
                gt_single = np.asarray(gt_act_slice[b], dtype=np.float32)
                pred_decode = subtask_tokenizer.extract_actions_with_info(
                    fast_full_np[b], action_horizon, effective_action_dim
                )
                pred_fast_decode_status_counts[pred_decode.status] += 1
                pred_actions = np.asarray(pred_decode.actions, dtype=np.float32)
                all_fast_pred_actions.append(pred_actions)
                all_fast_gt_actions.append(gt_single)

                gt_tokens = subtask_tokenizer.encode_action_to_tokens(gt_single, add_eos=True)
                gt_decode = subtask_tokenizer.extract_actions_with_info(
                    np.asarray(gt_tokens), action_horizon, effective_action_dim
                )
                gt_fast_decode_status_counts[gt_decode.status] += 1
                gt_recon = np.asarray(gt_decode.actions, dtype=np.float32)
                all_fast_gt_recon_actions.append(gt_recon if gt_decode.status == "ok" else None)

                if pred_decode.status == "ok":
                    valid_fast_pred_actions.append(pred_actions)
                    valid_fast_gt_actions.append(gt_single)
                    valid_fast_gt_recon_actions.append(gt_recon if gt_decode.status == "ok" else None)

            t1 = time.perf_counter()
            print(f"- Timing: FAST action inference + decode + GT encode-decode: {t1 - t0:.3f}s")

    jax.clear_caches()

    # Concatenate accumulated results
    def concat_or_empty(lst, axis=0):
        if not lst:
            return np.array([])
        return np.concatenate(lst, axis=axis)

    # Concatenate and compute metrics
    all_subtask_preds_np = concat_or_empty(all_subtask_preds)
    all_subtask_gts_np = concat_or_empty(all_subtask_gts)
    all_subtask_gt_masks_np = concat_or_empty(all_subtask_gt_masks)

    # Compute metrics
    if len(all_subtask_preds_np) > 0 and len(all_subtask_gts_np) > 0:
        # Convert to list for metrics computation
        pred_list = [all_subtask_preds_np[i] for i in range(len(all_subtask_preds_np))]
        gt_list = [all_subtask_gts_np[i] for i in range(len(all_subtask_gts_np))]
        gt_mask_list = [
            all_subtask_gt_masks_np[i] if i < len(all_subtask_gt_masks_np) else None
            for i in range(len(all_subtask_gts_np))
        ]

        metrics = compute_subtask_metrics(pred_list, gt_list, gt_mask_list, subtask_tokenizer)
        metrics.print_summary()

    # FAST action decode diagnostics and MSE metrics.
    if eval_fast_actions and pred_fast_decode_status_counts:
        print_fast_decode_summary(
            "FAST Action Decode Diagnostics (predicted tokens)",
            pred_fast_decode_status_counts,
        )
    if eval_fast_actions and gt_fast_decode_status_counts:
        print_fast_decode_summary(
            "FAST Action Decode Diagnostics (GT encode-decode)",
            gt_fast_decode_status_counts,
        )

    if eval_fast_actions and all_fast_pred_actions and all_fast_gt_actions:
        fast_metrics = compute_fast_action_mse(
            all_fast_pred_actions,
            all_fast_gt_actions,
            all_fast_gt_recon_actions if all_fast_gt_recon_actions else None,
        )
        fast_metrics.print_summary(title="FAST Action MSE Metrics (all samples, decode failures included)")
    if eval_fast_actions and valid_fast_pred_actions and valid_fast_gt_actions:
        valid_fast_metrics = compute_fast_action_mse(
            valid_fast_pred_actions,
            valid_fast_gt_actions,
            valid_fast_gt_recon_actions if valid_fast_gt_recon_actions else None,
        )
        valid_fast_metrics.print_summary(title="FAST Action MSE Metrics (successful predicted decodes only)")
    elif eval_fast_actions:
        print("No successfully decoded FAST action predictions; skipped valid-only FAST metrics.")

    # Visualization of mismatches (do not save during inference; create visuals now)
    if vis_dir:
        vis_batch_head_np = concat_or_empty(vis_batch_head)
        vis_batch_left_np = concat_or_empty(vis_batch_left)
        vis_batch_right_np = concat_or_empty(vis_batch_right)
        vis_batch_episodes = concat_or_empty(vis_batch_episodes)
        vis_batch_frame_ind = concat_or_empty(vis_batch_frame_ind)

        # Ensure shapes match images length
        n_images = vis_batch_head_np.shape[0]

        saved_count = 0
        mismatched_count = 0

        # Save mismatched frames
        for idx in range(n_images):
            try:
                head_img = vis_batch_head_np[idx]
                left_img = vis_batch_left_np[idx]
                right_img = vis_batch_right_np[idx]

                episode_index = vis_batch_episodes[idx, 0]
                frame_index = vis_batch_frame_ind[idx, 0]

                subtask_tokens = all_subtask_gts_np[idx] if idx < len(all_subtask_gts_np) else None
                subtask_tokens_mask = all_subtask_gt_masks_np[idx] if idx < len(all_subtask_gt_masks_np) else None
                if subtask_tokens is not None:
                    subtask_prompt = subtask_tokenizer.extract_subtask(
                        subtask_tokens,
                        subtask_tokens_mask,
                    )
                else:
                    subtask_prompt = None
                pred_subtask_prompt = subtask_tokenizer.extract_subtask(all_subtask_preds_np[idx])

                # Check if GT and Pred match
                is_match = (
                    compute_exact_match(pred_subtask_prompt, subtask_prompt)
                    if subtask_prompt is not None
                    else (pred_subtask_prompt is None)
                )

                # Skip saving if only saving mismatches and this is a match
                if save_mismatches_only and is_match:
                    continue

                mismatched_count += 1 if not is_match else 0
                frame_sec = frame_index / 30.0
                fname = f"episode{episode_index}_frame{frame_index}_time{frame_sec:.1f}s.jpg"
                save_dir = os.path.join(
                    vis_dir,
                    "pred_gt_subtask",
                    repo_id[0] if isinstance(repo_id, list) else repo_id,
                )

                # Add match status to the image text
                _save_mismatch_frame(
                    head_img,
                    left_img,
                    right_img,
                    subtask_prompt,
                    pred_subtask_prompt,
                    save_dir,
                    fname,
                    is_match=is_match,
                )
                saved_count += 1
            except Exception as e:
                print(f"Could not save mismatch frame idx {idx}: {e}")

        print(f"\nVisualization saved to: {vis_dir}/pred_gt_subtask/")
        print(f"  - Total samples: {n_images}")
        print(f"  - Saved samples: {saved_count}")
        if save_mismatches_only:
            print(f"  - Mismatched samples: {mismatched_count} (saved only due to --save_mismatches_only)")
        print()


def _to_pil_image(single_img):
    """Convert a single image array in [-1,1] to a PIL RGB image (uint8).

    Accepts JAX device arrays or numpy arrays. Expects HWC or HW.
    """
    import jax

    a = single_img
    a = jax.device_get(a) if hasattr(a, "device_buffer") or hasattr(a, "device") else np.asarray(a)
    # Convert from [-1,1] to [0,255]
    a = (a + 1.0) * 127.5
    a = np.clip(a, 0, 255).astype(np.uint8)

    # Single-channel -> convert to RGB
    if a.ndim == 2:
        return Image.fromarray(a).convert("RGB")
    if a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
        return Image.fromarray(a).convert("RGB")
    return Image.fromarray(a).convert("RGB")


def _save_mismatch_frame(
    head_img,
    left_img,
    right_img,
    gt_subtask,
    pred_subtask,
    out_dir,
    filename,
    *,
    is_match: bool = False,
):
    """Compose head/left/right images horizontally and draw GT/Pred text, then save.

    Images are expected in [-1,1] numpy/JAX arrays as used elsewhere in this script.

    Args:
        head_img: Head camera image
        left_img: Left wrist camera image
        right_img: Right wrist camera image
        gt_subtask: Ground truth subtask text
        pred_subtask: Predicted subtask text
        out_dir: Output directory path
        filename: Output filename
        is_match: Whether GT and Pred match (for display purposes)
    """
    # Convert arrays to PIL and normalize
    pil_head = _to_pil_image(head_img)
    pil_left = _to_pil_image(left_img)
    pil_right = _to_pil_image(right_img)

    # Normalize heights
    heights = [pil_head.height, pil_left.height, pil_right.height]
    max_h = max(heights)

    def _resize_keep(p):
        if p.height != max_h:
            w = int(p.width * (max_h / p.height))
            return p.resize((w, max_h))
        return p

    r_head, r_left, r_right = (
        _resize_keep(pil_head),
        _resize_keep(pil_left),
        _resize_keep(pil_right),
    )

    total_w = r_head.width + r_left.width + r_right.width
    canvas = Image.new("RGB", (total_w, max_h + 60), (255, 255, 255))
    x = 0
    for p in (r_left, r_head, r_right):
        canvas.paste(p, (x, 0))
        x += p.width

    # Draw overlay text area below the images
    draw = ImageDraw.Draw(canvas)

    # Status indicator
    status_text = "MATCH" if is_match else "MISMATCH"

    text = f"{status_text}\n" f"GT subtask: {gt_subtask}\n" f"Pred subtask: {pred_subtask}"
    # Draw text with status color
    draw.text((4, max_h + 4), text, fill=(0, 0, 0))

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    canvas.save(out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, required=True)  #
    parser.add_argument("--config_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=1)
    # Vis config
    parser.add_argument("--vis_dir", type=str, default="./tmp_open_loop_vis")
    parser.add_argument(
        "--save_mismatches_only",
        action="store_true",
        default=True,
        help="Only save visualization for mismatched samples (GT != Pred)",
    )
    parser.add_argument(
        "--eval_fast_actions",
        action="store_true",
        default=False,
        help="Evaluate FAST discrete action tokens (token accuracy, etc.)",
    )

    args = parser.parse_args()
    os.makedirs(args.vis_dir, exist_ok=True)

    main(
        args.ckpt_dir,
        args.dataset_root,
        args.config_name,
        args.num_batches,
        batch_size=args.batch_size,
        repo_id=args.repo_id,
        vis_dir=args.vis_dir,
        sample_steps=args.sample_steps,
        num_workers=args.num_workers,
        save_mismatches_only=args.save_mismatches_only,
        eval_fast_actions=args.eval_fast_actions,
    )
