"""Status bar rendering for value model video visualization.

Draws a color-coded bar showing frame index, predicted/GT values, and MSE.
"""

from __future__ import annotations

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

# Color convention: Pred=Red, GT=Blue
_COLOR_PRED = "#d62728"
_COLOR_GT = "#1f77b4"
_COLOR_MSE = "#666666"
_COLOR_BG = "#1a1a2e"
_COLOR_TEXT = "#e0e0e0"


def _get_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try loading a monospace font, fallback to default."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ]
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_status_bar(
    width: int,
    frame_idx: int,
    pred_val: float,
    gt_val: float,
    mse_val: float | None = None,
    bar_height: int = 36,
) -> Image.Image:
    """Draw a color-coded status bar for a video frame.

    Layout: [Frame: N] [Pred: -0.XXX (red)] [GT: -0.XXX (blue)] [MSE: 0.XXXX (gray)]

    Args:
        width: Bar width in pixels.
        frame_idx: Current frame index.
        pred_val: Predicted value.
        gt_val: Ground truth value.
        mse_val: Optional MSE value.
        bar_height: Height of the bar.

    Returns:
        PIL Image of the status bar.
    """
    bar = Image.new("RGB", (width, bar_height), _COLOR_BG)
    draw = ImageDraw.Draw(bar)
    font = _get_font(13)

    x = 8
    y = (bar_height - 16) // 2

    # Frame index
    draw.text((x, y), f"Frame: {frame_idx}", fill=_COLOR_TEXT, font=font)
    x += 120

    # Pred value (red)
    draw.text((x, y), f"Pred: {pred_val:+.4f}", fill=_COLOR_PRED, font=font)
    x += 160

    # GT value (blue)
    draw.text((x, y), f"GT: {gt_val:+.4f}", fill=_COLOR_GT, font=font)
    x += 150

    # MSE (gray, optional)
    if mse_val is not None:
        draw.text((x, y), f"MSE: {mse_val:.4f}", fill=_COLOR_MSE, font=font)

    return bar
