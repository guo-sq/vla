"""Frozen lightweight visual embeddings from LeRobot videos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from diffusion_proxy.utils import episode_video_path
from diffusion_proxy.utils import load_json


GRID_EMBEDDING_VERSION = "cv2_grid_rgb_8x6_stats_v1"
GRID_EMBEDDING_DIM = 152
RESNET18_EMBEDDING_VERSION = "torchvision_resnet18_imagenet_avgpool_v1"
RESNET18_EMBEDDING_DIM = 512
DEFAULT_ENCODER = "resnet18"


def embedding_dim(encoder: str) -> int:
    if encoder == "grid":
        return GRID_EMBEDDING_DIM
    if encoder == "resnet18":
        return RESNET18_EMBEDDING_DIM
    raise ValueError(f"Unknown encoder: {encoder}")


def embedding_version(encoder: str) -> str:
    if encoder == "grid":
        return GRID_EMBEDDING_VERSION
    if encoder == "resnet18":
        return RESNET18_EMBEDDING_VERSION
    raise ValueError(f"Unknown encoder: {encoder}")


def embedding_cache_path(cache_dir: Path, repo_id: str, episode_index: int, camera: str, encoder: str = DEFAULT_ENCODER) -> Path:
    safe_camera = camera.replace("/", "_").replace(".", "_")
    return cache_dir / encoder / repo_id / safe_camera / f"episode_{episode_index:06d}.npz"


def frame_embedding(frame_bgr: np.ndarray, *, grid_width: int = 8, grid_height: int = 6) -> np.ndarray:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected BGR frame with shape HxWx3, got {frame_bgr.shape}")
    small = cv2.resize(frame_bgr, (grid_width, grid_height), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    rgb = small[:, :, ::-1]
    flat_rgb = rgb.reshape(-1)
    mean = rgb.reshape(-1, 3).mean(axis=0)
    std = rgb.reshape(-1, 3).std(axis=0)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray_small = cv2.resize(gray, (grid_width, grid_height), interpolation=cv2.INTER_AREA)
    sobel_x = cv2.Sobel(gray_small, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_small, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    edge_stats = np.array([float(edge.mean()), float(edge.std())], dtype=np.float32)
    out = np.concatenate([flat_rgb, mean, std, edge_stats], axis=0).astype(np.float32)
    if out.shape[0] != GRID_EMBEDDING_DIM:
        raise ValueError(f"Unexpected embedding dim {out.shape[0]} != {GRID_EMBEDDING_DIM}")
    return out


def _load_resnet18(device: str | None = None):
    import torch
    from torch import nn
    from torchvision import models

    model = models.resnet18(weights=None)
    cache_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet18-f37072fd.pth"
    if cache_path.exists():
        state = torch.load(cache_path, map_location="cpu")
        model.load_state_dict(state)
    else:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    encoder = nn.Sequential(*list(model.children())[:-1])
    device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    encoder.to(device_obj)
    encoder.eval()
    return encoder, device_obj


def _resnet_preprocess(frame_bgr: np.ndarray) -> "torch.Tensor":
    import torch

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


def _extract_video_embeddings_grid(cap: cv2.VideoCapture) -> list[np.ndarray]:
    embeddings: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        embeddings.append(frame_embedding(frame))
    return embeddings


def _extract_video_embeddings_resnet18(cap: cv2.VideoCapture, *, batch_size: int = 128, device: str | None = None) -> list[np.ndarray]:
    import torch

    model, device_obj = _load_resnet18(device)
    embeddings: list[np.ndarray] = []
    batch = []
    with torch.inference_mode():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            batch.append(_resnet_preprocess(frame))
            if len(batch) >= batch_size:
                x = torch.stack(batch, dim=0).to(device_obj, non_blocking=True)
                y = model(x).flatten(1).detach().cpu().numpy().astype(np.float32)
                embeddings.extend([row for row in y])
                batch = []
        if batch:
            x = torch.stack(batch, dim=0).to(device_obj, non_blocking=True)
            y = model(x).flatten(1).detach().cpu().numpy().astype(np.float32)
            embeddings.extend([row for row in y])
    return embeddings


def extract_video_embeddings(
    video_path: Path,
    *,
    expected_length: int | None = None,
    encoder: str = DEFAULT_ENCODER,
    batch_size: int = 128,
    device: str | None = None,
) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if encoder == "grid":
        embeddings = _extract_video_embeddings_grid(cap)
    elif encoder == "resnet18":
        embeddings = _extract_video_embeddings_resnet18(cap, batch_size=batch_size, device=device)
    else:
        raise ValueError(f"Unknown encoder: {encoder}")
    cap.release()
    if not embeddings:
        raise ValueError(f"No frames decoded from {video_path}")
    values = np.stack(embeddings, axis=0).astype(np.float32)
    raw_length = int(len(values))
    adjusted = False
    if expected_length is not None and raw_length != int(expected_length):
        adjusted = True
        if raw_length > expected_length:
            values = values[:expected_length]
        else:
            pad = np.repeat(values[-1:], int(expected_length) - raw_length, axis=0)
            values = np.concatenate([values, pad], axis=0)
    meta = {
        "video_path": str(video_path),
        "encoder": encoder,
        "embedding_version": embedding_version(encoder),
        "embedding_dim": embedding_dim(encoder),
        "fps": fps,
        "width": width,
        "height": height,
        "video_frame_count": frame_count,
        "decoded_frame_count": raw_length,
        "expected_length": expected_length,
        "adjusted_to_expected_length": adjusted,
    }
    return values, meta


def save_episode_embeddings(
    *,
    root_dir: Path,
    repo_id: str,
    episode_meta: dict,
    cache_dir: Path,
    camera: str,
    encoder: str = DEFAULT_ENCODER,
    batch_size: int = 128,
    device: str | None = None,
) -> Path:
    repo_root = root_dir / repo_id
    info = load_json(repo_root / "meta" / "info.json")
    episode_index = int(episode_meta["episode_index"])
    expected_length = int(episode_meta["length"])
    video_path = episode_video_path(repo_root, info, episode_index, camera)
    embeddings, meta = extract_video_embeddings(video_path, expected_length=expected_length, encoder=encoder, batch_size=batch_size, device=device)
    path = embedding_cache_path(cache_dir, repo_id, episode_index, camera, encoder)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        embeddings=embeddings.astype(np.float32),
        repo_id=np.array(repo_id),
        episode_index=np.array(episode_index, dtype=np.int32),
        camera=np.array(camera),
        metadata=np.array(meta, dtype=object),
    )
    return path


def load_episode_embeddings(
    cache_dir: Path,
    repo_id: str,
    episode_index: int,
    camera: str,
    *,
    expected_length: int,
    encoder: str = DEFAULT_ENCODER,
) -> np.ndarray:
    path = embedding_cache_path(cache_dir, repo_id, episode_index, camera, encoder)
    if not path.exists():
        raise FileNotFoundError(f"Missing visual embedding cache: {path}")
    with np.load(path, allow_pickle=True) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if len(embeddings) != int(expected_length):
        raise ValueError(f"{path}: embedding length {len(embeddings)} != expected {expected_length}")
    expected_dim = embedding_dim(encoder)
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        raise ValueError(f"{path}: expected shape [T,{expected_dim}], got {embeddings.shape}")
    return embeddings
