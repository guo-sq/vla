"""Classification metrics for value model benchmark.

Evaluates whether a value model can discriminate successful vs failed
episodes using tail-frame predicted values. Provides AUC, optimal
F1 threshold, and separation score.
"""

from __future__ import annotations

import numpy as np


def compute_tail_auc(
    tail_preds: list[float],
    labels: list[bool],
    role: str,
) -> float:
    """Compute ROC-AUC for discriminating success/failure via tail_pred.

    Args:
        tail_preds: per-episode tail-frame predicted value.
        labels: per-episode success flag.
        role: "builder" or "destroyer".
              Builder: high tail_pred = success, score = tail_pred.
              Destroyer: low tail_pred = success, score = -tail_pred.

    Returns:
        AUC score, or nan if labels are degenerate (all same or len < 2).
    """
    if len(tail_preds) < 2:
        return float("nan")

    labels_arr = np.asarray(labels, dtype=bool)

    # Degenerate: all same label
    if labels_arr.all() or (~labels_arr).all():
        return float("nan")

    preds_arr = np.asarray(tail_preds, dtype=np.float64)
    scores = preds_arr if role == "builder" else -preds_arr

    # Use sklearn if available, otherwise manual implementation
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels_arr.astype(int), scores))
    except ImportError:
        return _manual_roc_auc(scores, labels_arr)


def _manual_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Fallback AUC via Mann-Whitney U statistic."""
    pos = scores[labels]
    neg = scores[~labels]
    n_pos = len(pos)
    n_neg = len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    u = 0.0
    for p in pos:
        u += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return float(u / (n_pos * n_neg))


def compute_optimal_threshold(
    tail_preds: list[float],
    labels: list[bool],
    role: str,
) -> dict:
    """Find the threshold that maximizes F1 score.

    Args:
        tail_preds: per-episode tail-frame predicted value.
        labels: per-episode success flag.
        role: "builder" (pred >= threshold -> success) or
              "destroyer" (pred <= threshold -> success).

    Returns:
        {"threshold": float, "f1": float, "precision": float, "recall": float}
    """
    preds_arr = np.asarray(tail_preds, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=bool)

    # Candidate thresholds: unique sorted pred values
    unique_vals = np.unique(preds_arr)
    # Add midpoints between consecutive values for finer search
    if len(unique_vals) > 1:
        midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2.0
        candidates = np.concatenate([unique_vals, midpoints])
    else:
        candidates = unique_vals

    best = {"threshold": float(candidates[0]), "f1": 0.0, "precision": 0.0, "recall": 0.0}

    for thresh in candidates:
        predicted_success = preds_arr >= thresh if role == "builder" else preds_arr <= thresh

        tp = int(np.sum(predicted_success & labels_arr))
        fp = int(np.sum(predicted_success & ~labels_arr))
        fn = int(np.sum(~predicted_success & labels_arr))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        if f1 > best["f1"]:
            best = {
                "threshold": float(thresh),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }

    return best


def compute_separation_score(
    tail_preds: list[float],
    labels: list[bool],
) -> float:
    """Compute mean(tail_pred | success) - mean(tail_pred | failure).

    Positive for builder (success = high pred), negative for destroyer
    (success = low pred).

    Returns:
        Separation score, or nan if either class is empty.
    """
    preds_arr = np.asarray(tail_preds, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=bool)

    success_preds = preds_arr[labels_arr]
    failure_preds = preds_arr[~labels_arr]

    if len(success_preds) == 0 or len(failure_preds) == 0:
        return float("nan")

    return float(np.mean(success_preds) - np.mean(failure_preds))


def _extract_tail_pred(episode: dict) -> float | None:
    """Extract tail_pred from an episode detail dict.

    Tries in order:
    1. Direct 'tail_pred' key
    2. Last element of 'pred' array
    """
    if "tail_pred" in episode:
        return float(episode["tail_pred"])
    if "pred" in episode:
        pred = np.asarray(episode["pred"])
        if len(pred) > 0:
            return float(pred[-1])
    return None


def compute_classification_report(
    episode_details: list[dict],
    role: str,
) -> dict | None:
    """Compute full classification report from episode details.

    Args:
        episode_details: list of dicts, each with 'success' (bool) and
            either 'tail_pred' (float) or 'pred' (array).
        role: "builder" or "destroyer".

    Returns:
        Dict with auc, optimal_threshold, separation_score, n_success, n_failure.
        Returns None if all labels are the same (no discrimination possible).
    """
    if not episode_details:
        return None

    tail_preds: list[float] = []
    labels: list[bool] = []

    for ep in episode_details:
        tp = _extract_tail_pred(ep)
        if tp is None:
            continue
        tail_preds.append(tp)
        labels.append(bool(ep["success"]))

    if len(tail_preds) < 2:
        return None

    n_success = sum(labels)
    n_failure = len(labels) - n_success

    # Return None if no failures or no successes (cannot discriminate)
    if n_success == 0 or n_failure == 0:
        return None

    auc = compute_tail_auc(tail_preds, labels, role)
    optimal = compute_optimal_threshold(tail_preds, labels, role)
    separation = compute_separation_score(tail_preds, labels)

    return {
        "auc": auc,
        "optimal_threshold": optimal,
        "separation_score": separation,
        "n_success": n_success,
        "n_failure": n_failure,
    }
