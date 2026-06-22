"""Image conversion and composition utilities for value model visualization.

Shared between run_benchmark.py and test_rl.py to avoid duplication.
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image
from PIL import ImageDraw


def to_pil_image(single_img) -> Image.Image:
    """Convert a single image array in [-1, 1] to a PIL RGB image (uint8).

    Accepts JAX device arrays or numpy arrays. Input shape: (H, W, C), (H, W),
    or (H, W, 1).
    """
    a = single_img
    if hasattr(a, "device") or hasattr(a, "device_buffer"):
        import jax

        a = jax.device_get(a)
    a = np.asarray(a)

    a = (a + 1.0) * 127.5
    a = np.clip(a, 0, 255).astype(np.uint8)

    if a.ndim == 2:
        return Image.fromarray(a).convert("RGB")
    if a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    return Image.fromarray(a).convert("RGB")


def to_uint8_array(img_array) -> np.ndarray:
    """Convert an image array in [-1, 1] to uint8 numpy array [0, 255].

    Accepts JAX device arrays or numpy arrays.
    """
    a = img_array
    if hasattr(a, "device") or hasattr(a, "device_buffer"):
        import jax

        a = jax.device_get(a)
    a = np.asarray(a, dtype=np.float64)

    a = (a + 1.0) * 127.5
    return np.clip(a, 0, 255).astype(np.uint8)


def save_composite_frame(
    img_left,
    img_head,
    img_right,
    pred_val: float,
    gt_val: float,
    out_path: str,
    title_height: int = 28,
) -> None:
    """Save left|head|right composite image with pred/gt title bar.

    Args:
        img_left: Left camera image array in [-1, 1].
        img_head: Head camera image array in [-1, 1].
        img_right: Right camera image array in [-1, 1].
        pred_val: Predicted value scalar.
        gt_val: Ground truth value scalar.
        out_path: Output file path.
        title_height: Height of the title bar in pixels.
    """
    pil_left = to_pil_image(img_left)
    pil_head = to_pil_image(img_head)
    pil_right = to_pil_image(img_right)

    max_h = max(pil_left.height, pil_head.height, pil_right.height)
    pil_left = _resize_to_height(pil_left, max_h)
    pil_head = _resize_to_height(pil_head, max_h)
    pil_right = _resize_to_height(pil_right, max_h)

    total_w = pil_left.width + pil_head.width + pil_right.width
    composite = Image.new("RGB", (total_w, max_h + title_height), (255, 255, 255))

    x = 0
    composite.paste(pil_left, (x, title_height))
    x += pil_left.width
    composite.paste(pil_head, (x, title_height))
    x += pil_head.width
    composite.paste(pil_right, (x, title_height))

    draw = ImageDraw.Draw(composite)
    title = f"pred {float(pred_val):.4f} vs gt {float(gt_val):.4f}"
    draw.text((8, 6), title, fill=(0, 0, 0))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    composite.save(out_path)


def make_image_row(
    images: list,
    target_height: int | None = None,
) -> Image.Image:
    """Stitch images horizontally into a single row.

    Args:
        images: List of image arrays in [-1, 1] or PIL Images.
        target_height: If set, resize all images to this height. Otherwise uses
            the maximum height among inputs.

    Returns:
        A single PIL Image with all inputs side by side.

    Raises:
        ValueError: If images list is empty.
    """
    if not images:
        raise ValueError("images list must not be empty")

    pil_images = []
    for img in images:
        if isinstance(img, Image.Image):
            pil_images.append(img)
        else:
            pil_images.append(to_pil_image(img))

    if target_height is None:
        target_height = max(p.height for p in pil_images)

    resized = [_resize_to_height(p, target_height) for p in pil_images]

    total_w = sum(p.width for p in resized)
    row = Image.new("RGB", (total_w, target_height), (255, 255, 255))
    x = 0
    for p in resized:
        row.paste(p, (x, 0))
        x += p.width

    return row


def _resize_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Resize image to target height, preserving aspect ratio."""
    if img.height == target_h:
        return img
    w = int(img.width * (target_h / img.height))
    return img.resize((w, target_h))
