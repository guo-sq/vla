"""Curve plotting utilities for value model visualization.

Provides pred vs GT line plots, advantage plots, binned MSE bar charts,
and comprehensive metric visualization suites.
"""

from __future__ import annotations

import os

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_value_curve(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    out_path: str,
    title: str = "",
    fps: int = 30,
) -> None:
    """Save a pred vs GT value curve plot.

    Args:
        pred_values: Predicted values (T,).
        gt_values: Ground truth values (T,).
        out_path: Output file path.
        title: Plot title.
        fps: Frame rate for x-axis time conversion.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    t = np.arange(len(pred_values)) / fps
    ax.plot(t, gt_values, label="GT", color="#1f77b4", linewidth=1.0)
    ax.plot(t, pred_values, label="Pred", color="#d62728", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Value")
    if title:
        ax.set_title(title)
    ax.grid(visible=True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_value_curve_with_advantage(
    pred_values: np.ndarray,
    gt_values: np.ndarray,
    out_path: str,
    title: str = "",
    fps: int = 30,
    advantage_horizon: int = 50,
    episode_length: int = 3310,
) -> None:
    """Save a pred vs GT curve with an advantage subplot below.

    The advantage is computed as:
        adv[i] = -horizon/episode_length + pred[i+horizon] - pred[i]

    If the sequence is shorter than the horizon, only the value curve is shown.

    Args:
        pred_values: Predicted values (T,).
        gt_values: Ground truth values (T,).
        out_path: Output file path.
        title: Plot title.
        fps: Frame rate for x-axis time conversion.
        advantage_horizon: Number of steps for advantage computation.
        episode_length: Total episode length for baseline normalization.
    """
    pred = np.asarray(pred_values).ravel()
    gt = np.asarray(gt_values).ravel()
    t = np.arange(len(pred)) / fps
    has_advantage = len(pred) > advantage_horizon

    n_rows = 2 if has_advantage else 1
    height_ratios = [3, 2] if has_advantage else [1]
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8, 3 + (2 if has_advantage else 0)),
        gridspec_kw={"height_ratios": height_ratios},
        squeeze=False,
    )

    ax_value = axes[0, 0]
    ax_value.plot(t, gt, label="GT", color="#1f77b4", linewidth=1.0)
    ax_value.plot(t, pred, label="Pred", color="#d62728", linestyle="--", linewidth=1.0)
    ax_value.set_ylabel("Value")
    if title:
        ax_value.set_title(title)
    ax_value.grid(visible=True, linestyle="--", alpha=0.4)
    ax_value.legend()

    if has_advantage:
        adv = -advantage_horizon / episode_length + pred[advantage_horizon:] - pred[:-advantage_horizon]
        ax_adv = axes[1, 0]
        ax_adv.plot(t[:-advantage_horizon], adv, color="#e32817", linewidth=1.0)
        ax_adv.set_xlabel("Time (s)")
        ax_adv.set_ylabel("Advantage")
        ax_adv.grid(visible=True, linestyle="--", alpha=0.4)
    else:
        ax_value.set_xlabel("Time (s)")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
