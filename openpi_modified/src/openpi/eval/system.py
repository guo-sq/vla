import abc
import dataclasses
from collections.abc import Mapping, Sequence
from itertools import product
from typing import Any

import numpy as np


def _ensure_3d(actions: np.ndarray) -> np.ndarray:
    arr = np.asarray(actions)
    if arr.ndim == 2:
        return arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected action tensor with ndim=2/3, got shape={arr.shape}")
    return arr


class BaseMetric(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def update_batch(self, pred_actions: np.ndarray, gt_actions: np.ndarray) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def num_samples(self) -> int:
        raise NotImplementedError

    def clone(self) -> "BaseMetric":
        return dataclasses.replace(self) if dataclasses.is_dataclass(self) else type(self)()


@dataclasses.dataclass
class TrajectoryADEMetric(BaseMetric):
    _distance_sum: float = 0.0
    _num_trajectories: int = 0

    @property
    def name(self) -> str:
        return "ade"

    def reset(self) -> None:
        self._distance_sum = 0.0
        self._num_trajectories = 0

    def update_batch(self, pred_actions: np.ndarray, gt_actions: np.ndarray) -> None:
        pred = _ensure_3d(pred_actions)
        gt = _ensure_3d(gt_actions)
        if pred.shape != gt.shape:
            raise ValueError(f"Pred/GT shape mismatch: {pred.shape} vs {gt.shape}")

        timestep_l2 = np.linalg.norm(pred - gt, axis=-1)
        per_traj_ade = np.mean(timestep_l2, axis=1)
        self._distance_sum += float(np.sum(per_traj_ade))
        self._num_trajectories += int(pred.shape[0])

    def compute(self) -> float:
        if self._num_trajectories == 0:
            return float("nan")
        return self._distance_sum / self._num_trajectories

    def num_samples(self) -> int:
        return self._num_trajectories


@dataclasses.dataclass
class TrajectoryFDEMetric(BaseMetric):
    _distance_sum: float = 0.0
    _num_trajectories: int = 0

    @property
    def name(self) -> str:
        return "fde"

    def reset(self) -> None:
        self._distance_sum = 0.0
        self._num_trajectories = 0

    def update_batch(self, pred_actions: np.ndarray, gt_actions: np.ndarray) -> None:
        pred = _ensure_3d(pred_actions)
        gt = _ensure_3d(gt_actions)
        if pred.shape != gt.shape:
            raise ValueError(f"Pred/GT shape mismatch: {pred.shape} vs {gt.shape}")

        final_l2 = np.linalg.norm(pred[:, -1, :] - gt[:, -1, :], axis=-1)
        self._distance_sum += float(np.sum(final_l2))
        self._num_trajectories += int(pred.shape[0])

    def compute(self) -> float:
        if self._num_trajectories == 0:
            return float("nan")
        return self._distance_sum / self._num_trajectories

    def num_samples(self) -> int:
        return self._num_trajectories


@dataclasses.dataclass
class TrajectoryRMSEMetric(BaseMetric):
    _squared_sum: float = 0.0
    _num_elements: int = 0
    _num_trajectories: int = 0

    @property
    def name(self) -> str:
        return "rmse"

    def reset(self) -> None:
        self._squared_sum = 0.0
        self._num_elements = 0
        self._num_trajectories = 0

    def update_batch(self, pred_actions: np.ndarray, gt_actions: np.ndarray) -> None:
        pred = _ensure_3d(pred_actions)
        gt = _ensure_3d(gt_actions)
        if pred.shape != gt.shape:
            raise ValueError(f"Pred/GT shape mismatch: {pred.shape} vs {gt.shape}")

        diff = pred - gt
        self._squared_sum += float(np.sum(diff * diff))
        self._num_elements += int(diff.size)
        self._num_trajectories += int(pred.shape[0])

    def compute(self) -> float:
        if self._num_elements == 0:
            return float("nan")
        return float(np.sqrt(self._squared_sum / self._num_elements))

    def num_samples(self) -> int:
        return self._num_trajectories


class BaseBucketizer(abc.ABC):
    @abc.abstractmethod
    def bucket_keys(self, metadata: Mapping[str, Any]) -> list[str]:
        raise NotImplementedError


@dataclasses.dataclass(frozen=True)
class FieldBucketizer(BaseBucketizer):
    field_name: str
    unknown_value: str = "__unknown__"

    def bucket_keys(self, metadata: Mapping[str, Any]) -> list[str]:
        value = metadata.get(self.field_name, self.unknown_value)
        if value is None:
            value = self.unknown_value
        return [f"{self.field_name}={value}"]


@dataclasses.dataclass(frozen=True)
class MetadataFieldsBucketizer(BaseBucketizer):
    field_names: tuple[str, ...]
    unknown_value: str = "__unknown__"

    def bucket_keys(self, metadata: Mapping[str, Any]) -> list[str]:
        key_parts: list[str] = []
        for field in self.field_names:
            value = metadata.get(field, self.unknown_value)
            if value is None:
                value = self.unknown_value
            key_parts.append(f"{field}={value}")
        return ["|".join(key_parts)]


@dataclasses.dataclass(frozen=True)
class CompositeBucketizer(BaseBucketizer):
    bucketizers: tuple[BaseBucketizer, ...]

    def bucket_keys(self, metadata: Mapping[str, Any]) -> list[str]:
        if not self.bucketizers:
            return []
        key_groups = [b.bucket_keys(metadata) for b in self.bucketizers]
        return ["|".join(parts) for parts in product(*key_groups)]


class BaseEvaluator(abc.ABC):
    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def add_batch(
        self,
        pred_actions: np.ndarray,
        gt_actions: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self) -> dict[str, Any]:
        raise NotImplementedError


class BucketedEvaluator(BaseEvaluator):
    def __init__(
        self,
        metrics: Sequence[BaseMetric],
        bucketizer: BaseBucketizer | None = None,
    ):
        if not metrics:
            raise ValueError("metrics must not be empty")
        self._metric_templates = [metric.clone() for metric in metrics]
        self._bucketizer = bucketizer
        self.reset()

    def reset(self) -> None:
        self._global_metrics = [metric.clone() for metric in self._metric_templates]
        for metric in self._global_metrics:
            metric.reset()
        self._bucket_metrics: dict[str, list[BaseMetric]] = {}

    def add_batch(
        self,
        pred_actions: np.ndarray,
        gt_actions: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        pred = _ensure_3d(pred_actions)
        gt = _ensure_3d(gt_actions)
        if pred.shape != gt.shape:
            raise ValueError(f"Pred/GT shape mismatch: {pred.shape} vs {gt.shape}")

        for metric in self._global_metrics:
            metric.update_batch(pred, gt)

        if self._bucketizer is None:
            return

        batch_size = pred.shape[0]
        for idx in range(batch_size):
            item_meta = self._metadata_at_index(metadata or {}, idx, batch_size)
            bucket_keys = self._bucketizer.bucket_keys(item_meta)
            if not bucket_keys:
                continue
            sample_pred = pred[idx : idx + 1]
            sample_gt = gt[idx : idx + 1]
            for key in bucket_keys:
                bucket_metrics = self._bucket_metrics.get(key)
                if bucket_metrics is None:
                    bucket_metrics = [metric.clone() for metric in self._metric_templates]
                    for metric in bucket_metrics:
                        metric.reset()
                    self._bucket_metrics[key] = bucket_metrics
                for metric in bucket_metrics:
                    metric.update_batch(sample_pred, sample_gt)

    def compute(self) -> dict[str, Any]:
        global_payload = {
            metric.name: float(metric.compute()) for metric in self._global_metrics
        }
        global_payload["count"] = int(self._global_metrics[0].num_samples())

        bucket_payload: dict[str, dict[str, float | int]] = {}
        for key, metrics in sorted(self._bucket_metrics.items()):
            payload = {metric.name: float(metric.compute()) for metric in metrics}
            payload["count"] = int(metrics[0].num_samples())
            bucket_payload[key] = payload

        return {
            "global": global_payload,
            "buckets": bucket_payload,
        }

    @staticmethod
    def _metadata_at_index(
        metadata: Mapping[str, Any], idx: int, batch_size: int
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                out[key] = None
                continue
            if isinstance(value, (str, bytes)):
                out[key] = value
                continue
            if isinstance(value, np.ndarray):
                if value.ndim > 0 and value.shape[0] == batch_size:
                    out[key] = value[idx].item() if value.ndim == 1 else value[idx]
                else:
                    out[key] = value
                continue
            if isinstance(value, Sequence):
                if len(value) == batch_size:
                    out[key] = value[idx]
                else:
                    out[key] = value
                continue
            out[key] = value
        return out
