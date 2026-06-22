"""
Self-contained multi-category HTML visualising the entire WAIC fold_box corpus
listed in a cfg_pi05_base_finetune_box_recap_pt_*.py file.

Layout
------
  • Page header (centered, independent)
  • Top hero with overview KPIs across ALL categories.
  • Tab bar of category cards grouped into 6 buckets:
      全流程类 / 半流程类 / 局部强化类 / 错误恢复类 / 新增动作类 / 推理类
  • Each category panel: per-category KPIs, filmstrip, weight chart,
    weight distribution.

Caches under .viz_cache/<MMDD>/ next to this script so re-runs are fast:
.per_cat_cache.json, .thumbs_cache.json. Final HTML auto-published to
project root config_viz/viz_<MMDD>.html.

Usage:
    # default cfg → config_viz/viz_0526.html
    python tools/fold_box/make_cfg_viz.py
    # any other cfg → config_viz/viz_<MMDD>.html
    python tools/fold_box/make_cfg_viz.py \
        --cfg src/openpi/configs/cfg_pi05_base_finetune_box_recap_pt_0525.py
"""

from __future__ import annotations

import argparse
import base64
import colorsys
import glob
import json
import math
import os
import re
import shutil
import time
from io import BytesIO

import cv2
import pandas as pd
from PIL import Image

ROOT = "/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/fold_box_WAIC"
DEFAULT_CFG = "/mnt/workspace/zengqi/openpi_modified/src/openpi/configs/cfg_pi05_base_finetune_box_recap_pt_0526.py"
# Cache root — per-cfg subdirs live under here (one isolated cache per cfg).
CACHE_ROOT = os.path.join(os.path.dirname(__file__), ".viz_cache")
# After writing viz.html, also publish a date-tagged copy here so all corpus
# overviews live under one canonical project-level directory.
PUBLISH_DIR = "/mnt/workspace/zengqi/openpi_modified/config_viz"
# cfg basename → MMDD tag (e.g. cfg_pi05_base_finetune_box_recap_pt_0525.py → "0525")
CFG_DATE_RE = re.compile(r"_pt_(\d{4})\.py$")

# Mutable globals (rebound by main() from CLI args)
CFG_PATH = DEFAULT_CFG
OUT_DIR = CACHE_ROOT  # placeholder; main() resolves per-cfg subdir

DEFAULT_WEIGHT = 1
CAMERA_KEY = "observation.images.head"

N_THUMBS = 13
THUMB_DISPLAY_H_PX = 180
THUMB_RENDER_H_PX = 240

VB_WIDTH = 1800
VB_HEIGHT = 280
YAXIS_W = 60
STRIP_W = VB_WIDTH - YAXIS_W

PALETTE = {
    0: "#94a3b8",
    1: "#60a5fa",
    2: "#fb923c",
    4: "#ef4444",
}
WEIGHT_NOTES = {0: "masked", 1: "default", 2: "", 4: "max"}

# Match BOTH the standard "<dir>/fold_box_from_scratch.<cat>...." pattern AND
# the inference replay pattern "infer/<sub>/fold_box_from_scratch_infer.<task>...".
CFG_ENTRY_RE = re.compile(
    r'\(\s*"([a-zA-Z0-9_/.-]+?/fold_box_from_scratch(?:_infer)?\.[^"]+)"\s*,\s*(\[.*?\])\s*\)',
    re.S,
)

# (group_name, english_subtitle, accent_text, accent_bg)
CATEGORY_GROUPS = [
    ("全流程类",   "Whole-process workflows",  "#15803d", "rgba(34,197,94,0.14)"),
    ("半流程类",   "Half-process workflows",   "#1d4ed8", "rgba(59,130,246,0.14)"),
    ("局部强化类", "Targeted reinforcement",   "#b45309", "rgba(245,158,11,0.14)"),
    ("错误恢复类", "Error-recovery",           "#b91c1c", "rgba(239,68,68,0.14)"),
    ("新增动作类", "Newly added actions",      "#6d28d9", "rgba(139,92,246,0.14)"),
    ("推理类",     "Inference replays",        "#0e7490", "rgba(20,184,166,0.14)"),
    ("其他",       "Other",                    "#475569", "rgba(100,116,139,0.14)"),
]


def _classify_normal_cat(cat: str) -> str:
    if cat in ("total_steps", "total_steps2"):
        return "全流程类"
    if cat.startswith("second_half"):
        return "半流程类"
    if cat in ("rein_38", "rein_38_2", "step24"):
        return "局部强化类"
    if cat.startswith("recover"):
        return "错误恢复类"
    if "last_step" in cat:
        return "新增动作类"
    return "其他"


# ---------- weight helpers ----------

def weight_at(t: float, segs) -> int:
    for s, e, w, _fps in segs:
        if s <= t < e:
            return int(w)
    return DEFAULT_WEIGHT


def build_step_segments(duration: float, segs):
    boundaries = {0.0, duration}
    for s, e, _w, _fps in segs:
        if 0 <= s <= duration:
            boundaries.add(float(s))
        if 0 <= e <= duration:
            boundaries.add(float(e))
    bs = sorted(boundaries)
    out = []
    for i in range(len(bs) - 1):
        a, b = bs[i], bs[i + 1]
        if b <= a:
            continue
        w = weight_at((a + b) / 2, segs)
        if out and out[-1][2] == w:
            out[-1] = (out[-1][0], b, w)
        else:
            out.append((a, b, w))
    return out


def episode_buckets(n_frames: int, segs):
    fps = 30.0
    duration = n_frames / fps
    bs_set = {0.0, duration}
    for s, e, _w, _fps in segs:
        if 0 <= s <= duration:
            bs_set.add(float(s))
        if 0 <= e <= duration:
            bs_set.add(float(e))
    bs = sorted(bs_set)
    buckets: dict[int, int] = {}
    total_w = 0
    consumed = 0
    for i in range(len(bs) - 1):
        a, b = bs[i], bs[i + 1]
        if b <= a:
            continue
        fa = int(round(a * fps))
        fb = int(round(b * fps))
        cnt = max(0, fb - fa)
        if cnt == 0:
            continue
        w = weight_at((a + b) / 2, segs)
        buckets[w] = buckets.get(w, 0) + cnt
        total_w += cnt * w
        consumed += cnt
    if consumed != n_frames and bs:
        last_w = weight_at((bs[-2] + bs[-1]) / 2, segs) if len(bs) >= 2 else DEFAULT_WEIGHT
        delta = n_frames - consumed
        buckets[last_w] = buckets.get(last_w, 0) + delta
        total_w += delta * last_w
    return buckets, total_w


# ---------- cfg parsing + per-category aggregation ----------

def parse_cfg_entries():
    text = open(CFG_PATH).read()
    out = []
    for m in CFG_ENTRY_RE.finditer(text):
        rel = m.group(1)
        segs_str = m.group(2)
        line_start = text.rfind("\n", 0, m.start()) + 1
        if text[line_start:m.start()].lstrip().startswith("#"):
            continue
        try:
            segs = eval(segs_str)
        except Exception:
            continue

        # Inference replays: each sub-bucket (good1-25, good_all, ...) becomes
        # its own category, all routed into the "推理类" group.
        if rel.startswith("infer/"):
            parts = rel.split("/")
            cat = parts[1] if len(parts) >= 2 else "infer"
            group = "推理类"
            out.append((cat, group, rel, segs))
            continue

        # Category = the directory prefix, UNLESS the file-name category is a
        # more-specific extension of the directory (e.g. new_last_step/ holds
        # both new_last_step and new_last_step2 files; we want them split).
        dir_cat = rel.split("/", 1)[0]
        name = rel.split("/")[-1]
        try:
            name_cat = name.split("fold_box_from_scratch.", 1)[1].split(".")[0]
        except IndexError:
            continue
        if name_cat.startswith(dir_cat):
            cat = name_cat
        else:
            cat = dir_cat
        out.append((cat, _classify_normal_cat(cat), rel, segs))
    return out


def aggregate_per_category(force: bool = False) -> dict:
    cache_path = os.path.join(OUT_DIR, ".per_cat_cache.json")
    cfg_mtime = os.path.getmtime(CFG_PATH)
    if not force and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
            if cache.get("cfg_mtime") == cfg_mtime and cache.get("cfg_path") == CFG_PATH:
                print(f"  per-cat aggregate: cache hit ({len(cache['per_cat'])} categories)")
                return cache["per_cat"]
        except Exception:
            pass

    entries = parse_cfg_entries()
    by_cat: dict[str, list] = {}
    cat_to_group: dict[str, str] = {}
    for cat, group, rel, segs in entries:
        by_cat.setdefault(cat, []).append((rel, segs))
        cat_to_group[cat] = group

    per_cat: dict[str, dict] = {}
    t0 = time.time()
    for cat, items in by_cat.items():
        n_ds = n_eps = total_raw = total_weighted = 0
        buckets: dict[int, int] = {}
        durations: list[float] = []
        sample_path = None
        sample_segs = None
        for rel, segs in items:
            files = sorted(glob.glob(os.path.join(ROOT, rel, "data/chunk-000/episode_*.parquet")))
            if not files:
                continue
            if sample_path is None:
                vp = os.path.join(ROOT, rel, "videos/chunk-000", CAMERA_KEY, "episode_000000.mp4")
                if os.path.exists(vp):
                    sample_path = rel
                    sample_segs = segs
            n_ds += 1
            for f in files:
                n = pd.read_parquet(f, columns=["frame_index"]).shape[0]
                n_eps += 1
                total_raw += n
                durations.append(n / 30.0)
                b, tw = episode_buckets(n, segs)
                for k, v in b.items():
                    buckets[k] = buckets.get(k, 0) + v
                total_weighted += tw
        if n_ds == 0:
            continue
        per_cat[cat] = dict(
            cat=cat, group=cat_to_group.get(cat, "其他"),
            n_ds=n_ds, n_eps=n_eps,
            total_raw=total_raw, total_weighted=total_weighted,
            amplification=total_weighted / max(total_raw, 1),
            avg_duration_s=sum(durations) / len(durations) if durations else 0,
            buckets={str(k): v for k, v in sorted(buckets.items())},
            sample_path=sample_path,
            sample_segs=sample_segs,
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump({"cfg_path": CFG_PATH, "cfg_mtime": cfg_mtime, "per_cat": per_cat}, f, indent=2)
    print(f"  per-cat aggregate: {len(per_cat)} categories in {time.time() - t0:.1f}s")
    return per_cat


# ---------- thumbnail extraction + cache ----------

def extract_thumbs_for(sample_path: str, episode: int = 0):
    cache_path = os.path.join(OUT_DIR, ".thumbs_cache.json")
    cache: dict = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}
    key = f"{sample_path}|ep{episode:06d}|n{N_THUMBS}|h{THUMB_RENDER_H_PX}"
    if key in cache:
        return cache[key]["duration"], cache[key]["thumbs"]

    path = os.path.join(ROOT, sample_path, "videos/chunk-000", CAMERA_KEY, f"episode_{episode:06d}.mp4")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = n_frames / fps
    thumbs = []
    for i in range(N_THUMBS):
        t = (i + 0.5) * duration / N_THUMBS
        f_idx = max(0, min(n_frames - 1, int(round(t * fps))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame_rgb)
        h_target = THUMB_RENDER_H_PX
        w_target = int(round(pil.width * h_target / pil.height))
        pil = pil.resize((w_target, h_target), Image.LANCZOS)
        buf = BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        thumbs.append([t, f_idx, b64])
    cap.release()
    cache[key] = {"duration": duration, "thumbs": thumbs}
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    return duration, thumbs


# ---------- SVG ----------

def build_svg(duration: float, segs) -> str:
    step_segs = build_step_segments(duration, segs)
    if not step_segs:
        return ""
    y_max = max(4, max(w for _, _, w in step_segs)) + 1
    plot_top, plot_bottom = 18, VB_HEIGHT - 56
    plot_h = plot_bottom - plot_top

    def x_of(t: float) -> float:
        return YAXIS_W + (t / max(duration, 1e-6)) * STRIP_W

    def y_of(w: float) -> float:
        return plot_bottom - (w / y_max) * plot_h

    parts: list[str] = []

    parts.append(
        f'<rect x="{YAXIS_W}" y="{plot_top}" width="{STRIP_W}" height="{plot_h}" '
        f'fill="#fafbfc" stroke="none"/>'
    )

    for w in range(0, y_max + 1):
        y = y_of(w)
        parts.append(f'<line x1="{YAXIS_W}" x2="{VB_WIDTH}" y1="{y:.2f}" y2="{y:.2f}" '
                     f'stroke="#e2e8f0" stroke-width="0.6"/>')
        parts.append(f'<text x="{YAXIS_W - 10}" y="{y + 4:.2f}" font-size="12" text-anchor="end" '
                     f'fill="#64748b" font-family="ui-monospace, Menlo, Consolas, monospace">{w}</text>')

    step_s = 5 if duration <= 30 else 10
    t = 0
    while t <= duration + 1e-6:
        x = x_of(t)
        parts.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{plot_top}" y2="{plot_bottom}" '
                     f'stroke="#eef2f6" stroke-width="0.5"/>')
        parts.append(f'<text x="{x:.2f}" y="{plot_bottom + 16:.2f}" font-size="11" text-anchor="middle" '
                     f'fill="#64748b" font-family="ui-monospace, Menlo, Consolas, monospace">{int(round(t))}</text>')
        t += step_s
    parts.append(
        f'<text x="{VB_WIDTH - 4:.2f}" y="{plot_bottom + 16:.2f}" font-size="11" text-anchor="end" '
        f'fill="#94a3b8" font-family="sans-serif">[s]</text>'
    )

    parts.append(f'<line x1="{YAXIS_W}" x2="{VB_WIDTH}" y1="{plot_bottom}" y2="{plot_bottom}" '
                 f'stroke="#475569" stroke-width="1"/>')
    parts.append(f'<line x1="{YAXIS_W}" x2="{YAXIS_W}" y1="{plot_top}" y2="{plot_bottom}" '
                 f'stroke="#475569" stroke-width="1"/>')

    cy = (plot_top + plot_bottom) / 2
    parts.append(
        f'<text x="{YAXIS_W - 40}" y="{cy:.2f}" font-size="12" text-anchor="middle" '
        f'transform="rotate(-90 {YAXIS_W - 40} {cy:.2f})" '
        f'fill="#475569" font-family="sans-serif" font-weight="600" letter-spacing="0.04em">FRAME WEIGHT</text>'
    )

    for a, b, w in step_segs:
        x0, x1 = x_of(a), x_of(b)
        y = y_of(w)
        color = PALETTE.get(w, "#475569")
        parts.append(f'<rect x="{x0:.2f}" y="{y:.2f}" width="{x1 - x0:.2f}" height="{plot_bottom - y:.2f}" '
                     f'fill="{color}" fill-opacity="0.18" stroke="none"/>')
        parts.append(f'<line x1="{x0:.2f}" x2="{x1:.2f}" y1="{y:.2f}" y2="{y:.2f}" '
                     f'stroke="{color}" stroke-width="2.8" stroke-linecap="round"/>')
        cx = (x0 + x1) / 2
        parts.append(
            f'<text x="{cx:.2f}" y="{y - 8:.2f}" font-size="13" text-anchor="middle" '
            f'fill="{color}" font-weight="700" font-family="ui-monospace, Menlo, Consolas, monospace" '
            f'paint-order="stroke" stroke="#fff" stroke-width="3">w={w}</text>'
        )

    for i in range(1, len(step_segs)):
        tb = step_segs[i][0]
        x = x_of(tb)
        parts.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{plot_top - 4}" y2="{plot_bottom + 4}" '
            f'stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="4,3"/>'
        )
        badge_w = 32
        bx = x - badge_w / 2
        by = plot_top - 14
        parts.append(
            f'<rect x="{bx:.2f}" y="{by:.2f}" width="{badge_w}" height="14" rx="3" ry="3" fill="#0f172a"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{by + 10:.2f}" font-size="10" text-anchor="middle" fill="#fff" '
            f'font-family="ui-monospace, Menlo, Consolas, monospace">@{tb:.0f}s</text>'
        )

    return (
        f'<svg viewBox="0 0 {VB_WIDTH} {VB_HEIGHT}" preserveAspectRatio="xMidYMid meet" '
        f'style="width: 100%; height: auto; display: block;">'
        + "\n".join(parts) + "</svg>"
    )


# ---------- HTML helpers ----------

def _fmt_int(n: int) -> str:
    return f"{int(n):,}"


def _fmt_compact(n: int) -> str:
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def shade(base_hex: str, k: int, n: int) -> str:
    """Return the k-th (0-based) of n evenly-spread shades of base_hex.
    Spread is along HLS lightness; saturation/hue preserved."""
    if n <= 1:
        return base_hex
    h = base_hex.lstrip("#")
    r = int(h[0:2], 16) / 255
    g = int(h[2:4], 16) / 255
    b = int(h[4:6], 16) / 255
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    l_min = max(0.22, ll - 0.18)
    l_max = min(0.78, ll + 0.28)
    new_l = l_min + (l_max - l_min) * (k / (n - 1))
    nr, ng, nb = colorsys.hls_to_rgb(hh, new_l, ss)
    return f"#{int(round(nr * 255)):02x}{int(round(ng * 255)):02x}{int(round(nb * 255)):02x}"


def _focus_pie_slices(focus_cats, other_cats, metric_key, base_color, mute_color="#cbd5e1"):
    """Build slice dicts for a pie that emphasises `focus_cats` against a single
    combined 'other' wedge in gray. Focus slices carry data_cat so the page JS
    can pop them when the matching tab is selected."""
    sorted_focus = sorted(focus_cats, key=lambda c: -c[metric_key])
    n = len(sorted_focus)
    slices = []
    for i, c in enumerate(sorted_focus):
        if c[metric_key] <= 0:
            continue
        slices.append({
            "name": c["cat"], "value": c[metric_key],
            "color": shade(base_color, i, n) if n > 1 else base_color,
            "data_cat": c["cat"],
        })
    other_total = sum(c[metric_key] for c in other_cats)
    if other_total > 0:
        slices.append({"name": "其他大类合计", "value": other_total, "color": mute_color})
    total = sum(s["value"] for s in slices)
    return slices, total


def build_pie_block(title: str, slices: list[dict], total: int) -> str:
    """Render an SVG pie + legend in the order slices are passed in.
    Each slice carries `data-mid` (its mid-angle in radians) plus optional
    `data-cat` / `data-group` so the page JS can "pop" matching slices
    radially when a tab is selected."""
    cx, cy, r = 100, 100, 88
    paths = []
    start_angle = -math.pi / 2  # start at top, sweep clockwise
    for s in slices:
        if s["value"] <= 0 or total <= 0:
            continue
        sweep = (s["value"] / total) * 2 * math.pi
        mid_angle = start_angle + sweep / 2
        if sweep >= 2 * math.pi - 1e-9:
            d = (f"M {cx - r:.2f},{cy:.2f} "
                 f"A {r},{r} 0 1,1 {cx + r:.2f},{cy:.2f} "
                 f"A {r},{r} 0 1,1 {cx - r:.2f},{cy:.2f} Z")
        else:
            end_angle = start_angle + sweep
            x1 = cx + r * math.cos(start_angle)
            y1 = cy + r * math.sin(start_angle)
            x2 = cx + r * math.cos(end_angle)
            y2 = cy + r * math.sin(end_angle)
            large_arc = 1 if sweep > math.pi else 0
            d = (f"M {cx},{cy} L {x1:.2f},{y1:.2f} "
                 f"A {r},{r} 0 {large_arc},1 {x2:.2f},{y2:.2f} Z")
        start_angle += sweep
        pct = s["value"] / total * 100
        data_attrs = [f'data-mid="{mid_angle:.4f}"']
        if "data_cat" in s:
            data_attrs.append(f'data-cat="{s["data_cat"]}"')
        if "data_group" in s:
            data_attrs.append(f'data-group="{s["data_group"]}"')
        paths.append(
            f'<path class="pie-slice" d="{d}" fill="{s["color"]}" stroke="#fff" '
            f'stroke-width="1.5" {" ".join(data_attrs)}>'
            f'<title>{s["name"]}: {s["value"]:,} ({pct:.1f}%)</title></path>'
        )
    pie_svg = (
        f'<svg viewBox="0 0 200 200" style="width:170px; height:170px; display:block; overflow:visible;">'
        + "".join(paths)
        + "</svg>"
    )
    legend_items = []
    for s in slices:
        pct = s["value"] / total * 100 if total else 0
        legend_items.append(
            f'<div class="pie-legend-item">'
            f'<span class="pie-dot" style="background:{s["color"]}"></span>'
            f'<span class="pie-legend-name">{s["name"]}</span>'
            f'<span class="pie-legend-pct">{pct:.1f}%</span>'
            f'<span class="pie-legend-val">{_fmt_compact(s["value"])}</span>'
            f'</div>'
        )
    return (
        '<div class="pie-block">'
        f'<div class="pie-title">{title}</div>'
        '<div class="pie-body">'
        f'<div class="pie-svg">{pie_svg}</div>'
        f'<div class="pie-legend">{"".join(legend_items)}</div>'
        '</div></div>'
    )


def build_kpi_row(kpis) -> str:
    return "".join(
        f'<div class="kpi"><div class="kpi-value">{v}</div>'
        f'<div class="kpi-label">{lbl}</div>'
        f'<div class="kpi-hint">{hint}</div></div>'
        for (lbl, v, hint) in kpis
    )


def build_strip(duration, thumbs, segs) -> str:
    items = []
    for t, f_idx, b64 in thumbs:
        w = weight_at(t, segs)
        color = PALETTE.get(w, "#475569")
        items.append(
            f'<div class="thumb">'
            f'<div class="thumb-img-wrap">'
            f'<img src="data:image/jpeg;base64,{b64}" alt="t={t:.1f}s"/>'
            f'<div class="thumb-time-badge">{t:.1f}s</div>'
            f'</div>'
            f'<div class="thumb-wband" style="background:{color}">w={w}</div>'
            f'</div>'
        )
    return f'<div class="stripwrap">{"".join(items)}</div>'


def build_legend() -> str:
    chips = "".join(
        f'<span class="legend-chip">'
        f'<span class="dot" style="background:{PALETTE[w]}"></span>w = {w}'
        f'{" · " + WEIGHT_NOTES[w] if WEIGHT_NOTES[w] else ""}'
        f'</span>'
        for w in (0, 1, 2, 4)
    )
    return f'<div class="legend"><span class="legend-label">Weight</span>{chips}</div>'


def build_distribution(buckets: dict[int, int], total_raw: int, total_weighted: int) -> str:
    weighted_contrib = {w: cnt * w for w, cnt in buckets.items()}
    sorted_w_desc = sorted(buckets.keys(), reverse=True)

    def _bar(metric_dict, total, exclude_zero=False):
        segs_html = []
        for w in sorted_w_desc:
            v = metric_dict[w]
            if v == 0 or (exclude_zero and w == 0):
                continue
            pct = v / total * 100 if total else 0
            color = PALETTE.get(w, "#475569")
            segs_html.append(
                f'<div class="stack-seg" style="width:{pct:.4f}%; background:{color}" '
                f'title="w={w}: {v:,} ({pct:.2f}%)">'
                f'<div class="stack-seg-inner">'
                f'<span class="stack-seg-label">w={w}</span>'
                f'<span class="stack-seg-pct">{pct:.1f}%</span>'
                f'</div></div>'
            )
        return f'<div class="stack-bar">{"".join(segs_html)}</div>'

    raw_bar = _bar(buckets, total_raw, exclude_zero=False)
    wt_bar = _bar(weighted_contrib, total_weighted, exclude_zero=True)

    rows = []
    for w in sorted_w_desc:
        raw = buckets[w]
        contrib = weighted_contrib[w]
        raw_pct = raw / total_raw * 100 if total_raw else 0
        wt_pct = contrib / total_weighted * 100 if total_weighted else 0
        c = PALETTE.get(w, "#475569")
        note = WEIGHT_NOTES.get(w, "")
        note_html = f' <span class="muted-note">{note}</span>' if note else ""
        rows.append(
            f'<tr>'
            f'<td><span class="wchip" style="background:{c}">w = {w}</span>{note_html}</td>'
            f'<td class="mono right">{raw:,}</td>'
            f'<td class="mono right">{raw_pct:.1f}%</td>'
            f'<td class="mono right strong">{contrib:,}</td>'
            f'<td class="mono right">{wt_pct:.1f}%</td>'
            f'</tr>'
        )
    table = (
        '<table class="aggtable"><thead>'
        '<tr><th>weight</th>'
        '<th class="right">raw frames</th>'
        '<th class="right">% of raw</th>'
        '<th class="right">weighted frames</th>'
        '<th class="right">% of weighted</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + f'<tr class="total-row">'
          f'<td class="strong">total</td>'
          f'<td class="mono right strong">{total_raw:,}</td>'
          f'<td class="mono right">100.0%</td>'
          f'<td class="mono right strong">{total_weighted:,}</td>'
          f'<td class="mono right">100.0%</td>'
          f'</tr></tbody></table>'
    )
    return f"""
    <div class="bar-block">
      <div class="bar-label">RAW frame distribution by weight</div>
      {raw_bar}
    </div>
    <div class="bar-block">
      <div class="bar-label">WEIGHTED training-budget distribution by weight</div>
      {wt_bar}
    </div>
    {table}
    """


def build_per_group_breakdown_panel(sorted_cats: list) -> str:
    """For each major group: a row with two pies (raw + weighted) where the
    focus group's sub-cats use shades of the group color and all other groups
    collapse into one gray 'other' wedge."""
    rows_html = []
    for (gname, gsub, gfg, gbg) in CATEGORY_GROUPS:
        focus_cats = [c for c in sorted_cats if c.get("group") == gname]
        if not focus_cats:
            continue
        other_cats = [c for c in sorted_cats if c.get("group") != gname]

        raw_slices, raw_total = _focus_pie_slices(focus_cats, other_cats, "total_raw", gfg)
        wt_slices, wt_total = _focus_pie_slices(focus_cats, other_cats, "total_weighted", gfg)

        focus_eps = sum(c["n_eps"] for c in focus_cats)
        focus_raw = sum(c["total_raw"] for c in focus_cats)
        focus_wt = sum(c["total_weighted"] for c in focus_cats)
        rows_html.append(
            '<div class="per-group-row">'
            '<div class="per-group-row-header">'
            f'<span class="group-chip" style="background:{gbg}; color:{gfg}">{gname}</span>'
            f'<span class="per-group-row-subtitle">{gsub}</span>'
            '<span class="per-group-row-stats">'
            f'<span>{len(focus_cats)} 子类</span>'
            f'<span>· {focus_eps:,} eps</span>'
            f'<span>· raw {_fmt_compact(focus_raw)}</span>'
            f'<span>· 加权 {_fmt_compact(focus_wt)}</span>'
            '</span>'
            '</div>'
            '<div class="per-group-pies">'
            + build_pie_block(f"{gname} · 原始帧细分", raw_slices, raw_total)
            + build_pie_block(f"{gname} · 加权帧细分", wt_slices, wt_total)
            + '</div>'
            '</div>'
        )
    return (
        '<div class="panel">'
        '<div class="panel-header">'
        '<div class="panel-title">各大类内部占比 · sub-category breakdown</div>'
        '<div class="panel-subtitle">'
        '每行一个大类 · focus 子类用该大类色彩的不同深浅, 其他大类合并为灰色弱化'
        '</div>'
        '</div>'
        f'<div class="per-group-rows">{"".join(rows_html)}</div>'
        '</div>'
    )


def build_category_panel(cat_data: dict, is_active: bool) -> str:
    cat = cat_data["cat"]
    sample_path = cat_data["sample_path"]
    sample_segs = cat_data["sample_segs"]
    buckets = {int(k): v for k, v in cat_data["buckets"].items()}

    if sample_path is None:
        film_html = '<div class="empty">no video sample available for this category</div>'
        chart_html = ""
        sample_meta = "—"
    else:
        duration, thumbs = extract_thumbs_for(sample_path)
        film_html = build_strip(duration, thumbs, sample_segs) + build_legend()
        chart_html = build_svg(duration, sample_segs)
        sample_meta = (
            f'shown: <code>{sample_path}</code> ep000 · {N_THUMBS} 帧均匀采样 (t<sub>i</sub> = (i+0.5)·T/N) · '
            f'duration ≈ {duration:.1f}s'
        )

    kpis = [
        ("Datasets",       _fmt_int(cat_data["n_ds"]),         f"cfg 中 {cat}/"),
        ("Episodes",       _fmt_int(cat_data["n_eps"]),        "累加所有 batch"),
        ("Total raw",      _fmt_int(cat_data["total_raw"]),    f"≈ {_fmt_compact(cat_data['total_raw'])} @ 30 fps"),
        ("Total weighted", _fmt_int(cat_data["total_weighted"]),"Σ raw × weight"),
        ("Amplification",  f"{cat_data['amplification']:.2f}×","weighted / raw"),
        ("Avg ep length",  f"{cat_data['avg_duration_s']:.1f}s",f"≈ {int(round(cat_data['avg_duration_s'] * 30))} 帧"),
    ]
    return f"""
    <div class="cat-panel{' active' if is_active else ''}" id="panel-{cat}">

      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">{cat} · KPI 概览</div>
          <div class="panel-subtitle">本类别所有未注释 cfg 条目的累加统计</div>
        </div>
        <div class="kpis kpis-inpanel">{build_kpi_row(kpis)}</div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">Filmstrip · 1 representative of {cat_data["n_eps"]:,}</div>
          <div class="panel-subtitle">{sample_meta}</div>
        </div>
        {film_html}
      </div>

      {f'''<div class="panel">
        <div class="panel-header">
          <div class="panel-title">Per-frame weight timeline (this batch)</div>
          <div class="panel-subtitle">该 batch 的 TemporalWeight tuple · 黑徽章 = 权重切换 · 横轴像素与 filmstrip 对齐</div>
        </div>
        {chart_html}
      </div>''' if chart_html else ''}

      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">{cat} 类别内 weight 分布 · {cat_data["n_eps"]:,} 个 episode 汇总</div>
          <div class="panel-subtitle">上条 = 原始帧占比, 下条 = 加权训练预算占比</div>
        </div>
        {build_distribution(buckets, cat_data["total_raw"], cat_data["total_weighted"])}
      </div>

    </div>
    """


def build_html(per_cat: dict) -> str:
    sorted_cats = sorted(per_cat.values(), key=lambda c: -c["total_weighted"])
    cat_names = [c["cat"] for c in sorted_cats]
    active = cat_names[0] if cat_names else ""

    g_ds = sum(c["n_ds"] for c in sorted_cats)
    g_eps = sum(c["n_eps"] for c in sorted_cats)
    g_raw = sum(c["total_raw"] for c in sorted_cats)
    g_wt = sum(c["total_weighted"] for c in sorted_cats)
    g_amp = g_wt / max(g_raw, 1)

    hero_kpis = [
        ("Categories",     _fmt_int(len(sorted_cats)),  "cfg 中带 weight 的类别"),
        ("Datasets",       _fmt_int(g_ds),              "累加所有类别"),
        ("Episodes",       _fmt_int(g_eps),             "累加所有类别"),
        ("Total raw",      _fmt_int(g_raw),             f"≈ {_fmt_compact(g_raw)} @ 30 fps"),
        ("Total weighted", _fmt_int(g_wt),              f"≈ {_fmt_compact(g_wt)}"),
        ("Amplification",  f"{g_amp:.2f}×",             "weighted / raw"),
    ]

    # Per-group totals for the two pie charts (raw vs weighted by major group)
    group_raw: dict[str, int] = {}
    group_wt: dict[str, int] = {}
    for c in sorted_cats:
        gname = c.get("group", "其他")
        group_raw[gname] = group_raw.get(gname, 0) + c["total_raw"]
        group_wt[gname] = group_wt.get(gname, 0) + c["total_weighted"]
    group_color = {gn: fg for (gn, _sub, fg, _bg) in CATEGORY_GROUPS}
    raw_slices = [
        {"name": gn, "value": group_raw.get(gn, 0), "color": group_color.get(gn, "#475569"),
         "data_group": gn}
        for (gn, _sub, _fg, _bg) in CATEGORY_GROUPS
        if group_raw.get(gn, 0) > 0
    ]
    wt_slices = [
        {"name": gn, "value": group_wt.get(gn, 0), "color": group_color.get(gn, "#475569"),
         "data_group": gn}
        for (gn, _sub, _fg, _bg) in CATEGORY_GROUPS
        if group_wt.get(gn, 0) > 0
    ]
    pies_html = (
        '<div class="hero-pies">'
        + build_pie_block("原始帧分布 · 按大类", raw_slices, g_raw)
        + build_pie_block("加权帧分布 · 按大类", wt_slices, g_wt)
        + '</div>'
    )

    # Tab cards grouped by CATEGORY_GROUPS — each group also gets two focus
    # pies (raw + weighted) embedded below its cards.
    tab_groups_html = []
    for group_name, group_subtitle, accent_fg, accent_bg in CATEGORY_GROUPS:
        group_cats = [c for c in sorted_cats if c.get("group") == group_name]
        if not group_cats:
            continue
        other_cats = [c for c in sorted_cats if c.get("group") != group_name]
        g2_eps = sum(c["n_eps"] for c in group_cats)
        g2_wt = sum(c["total_weighted"] for c in group_cats)
        cards_html = []
        for c in group_cats:
            is_active = c["cat"] == active
            cards_html.append(
                f'<button class="tab{" active" if is_active else ""}" '
                f'data-cat="{c["cat"]}" data-group="{group_name}">'
                f'  <div class="tab-name">{c["cat"]}</div>'
                f'  <div class="tab-stats">'
                f'    <span class="ts-eps">{c["n_eps"]:,} eps</span>'
                f'    <span class="ts-wt">w·frm {_fmt_compact(c["total_weighted"])}</span>'
                f'  </div>'
                f'</button>'
            )
        raw_slices, raw_total = _focus_pie_slices(group_cats, other_cats, "total_raw", accent_fg)
        wt_slices, wt_total = _focus_pie_slices(group_cats, other_cats, "total_weighted", accent_fg)
        pies_html_g = (
            '<div class="tabgroup-pies">'
            + build_pie_block(f"{group_name} · 原始帧细分", raw_slices, raw_total)
            + build_pie_block(f"{group_name} · 加权帧细分", wt_slices, wt_total)
            + '</div>'
        )
        tab_groups_html.append(
            f'<div class="tabgroup">'
            f'  <div class="tabgroup-header">'
            f'    <div class="tabgroup-title-wrap">'
            f'      <span class="group-chip" style="background:{accent_bg}; color:{accent_fg}">{group_name}</span>'
            f'      <span class="tabgroup-subtitle">{group_subtitle}</span>'
            f'    </div>'
            f'    <div class="tabgroup-stats">'
            f'      <span>{len(group_cats)} 类</span>'
            f'      <span>· {g2_eps:,} eps</span>'
            f'      <span>· 加权 {_fmt_compact(g2_wt)}</span>'
            f'    </div>'
            f'  </div>'
            f'  <div class="tabgroup-cards">{"".join(cards_html)}</div>'
            f'  {pies_html_g}'
            f'</div>'
        )

    panels = "\n".join(
        build_category_panel(c, is_active=(c["cat"] == active))
        for c in sorted_cats
    )

    css = f"""
    :root {{
      --bg: #f1f5f9;
      --card: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #64748b;
      --subtle: #94a3b8;
      --accent: #0ea5e9;
      --accent-soft: #e0f2fe;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      padding: 28px 20px;
      background: var(--bg); color: var(--text);
      line-height: 1.45;
      font-feature-settings: "tnum" 1, "lnum" 1;
    }}
    .container {{ max-width: {VB_WIDTH}px; margin: 0 auto; }}

    /* Page header (centered, independent) */
    .page-header {{
      text-align: center; margin: 12px 0 30px; padding: 6px 0;
    }}
    .page-header .eyebrow {{
      font-size: 11px; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.18em;
      margin-bottom: 10px;
    }}
    .page-header .accent-line {{
      width: 38px; height: 2px; background: var(--accent);
      margin: 0 auto 14px; border-radius: 1px;
    }}
    .page-header h1 {{
      margin: 0 0 10px; font-size: 32px; font-weight: 800;
      letter-spacing: -0.02em; line-height: 1.15; color: var(--text);
    }}
    .page-header .tagline {{
      font-size: 13px; color: var(--muted); line-height: 1.6;
      max-width: 760px; margin: 0 auto;
    }}
    .page-header .tagline .pill {{
      display: inline-block; background: #f1f5f9; color: #334155;
      padding: 2px 9px; margin: 0 2px; border-radius: 6px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px;
      word-break: break-all;
    }}

    /* Hero (KPI block) */
    .hero {{
      background: var(--card); border: 1px solid var(--border); border-radius: 14px;
      padding: 18px 20px; margin-bottom: 16px;
      box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }}
    .kpis {{
      display: grid; grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
    }}
    .kpis-inpanel {{ margin: 14px 20px; }}
    .kpi {{
      background: linear-gradient(180deg, #fbfcfd 0%, #f1f5f9 100%);
      border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
    }}
    .kpi-value {{ font-size: 22px; font-weight: 700; color: #0f172a; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; }}
    .kpi-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-top: 4px; font-weight: 600; }}
    .kpi-hint  {{ font-size: 11px; color: var(--subtle); margin-top: 2px; }}

    /* Pies in hero */
    .hero-pies {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
      margin-top: 16px; padding-top: 16px;
      border-top: 1px solid var(--border);
    }}
    .pie-block {{
      background: linear-gradient(180deg, #fbfcfd 0%, #f1f5f9 100%);
      border: 1px solid var(--border); border-radius: 10px;
      padding: 14px 18px;
    }}
    .pie-title {{
      font-size: 11px; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.06em;
      margin-bottom: 12px;
    }}
    .pie-body {{ display: flex; align-items: center; gap: 18px; }}
    .pie-svg {{ flex: 0 0 170px; }}
    .pie-slice {{ transition: transform 0.22s ease; }}
    .pie-legend {{
      flex: 1; display: flex; flex-direction: column; gap: 4px;
      font-size: 12px;
    }}
    .pie-legend-item {{
      display: grid;
      grid-template-columns: 12px 1fr auto auto;
      gap: 8px; align-items: center;
      padding: 3px 0;
    }}
    .pie-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
    .pie-legend-name {{ color: #334155; font-weight: 600; }}
    .pie-legend-pct {{
      font-variant-numeric: tabular-nums;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      color: var(--muted);
    }}
    .pie-legend-val {{
      font-variant-numeric: tabular-nums;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      color: #0f172a; font-weight: 700;
      min-width: 54px; text-align: right;
    }}

    /* Per-group breakdown panel */
    .per-group-rows {{
      display: flex; flex-direction: column; gap: 14px;
      padding: 14px 20px;
    }}
    .per-group-row {{
      border: 1px solid var(--border); border-radius: 12px;
      padding: 14px 16px; background: #fcfdfe;
    }}
    .per-group-row-header {{
      display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
      padding-bottom: 12px; margin-bottom: 12px;
      border-bottom: 1px solid var(--border);
    }}
    .per-group-row-subtitle {{
      font-size: 11px; color: var(--subtle);
      text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
    }}
    .per-group-row-stats {{
      margin-left: auto;
      font-size: 11px; color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      display: inline-flex; gap: 6px;
    }}
    .per-group-pies {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
    }}

    /* Tab cards (grouped) */
    .tabbar-wrap {{
      background: var(--card); border: 1px solid var(--border); border-radius: 14px;
      padding: 12px; margin-bottom: 16px;
      box-shadow: 0 1px 3px rgba(15,23,42,0.04);
    }}
    .tabbar-label {{
      font-size: 11px; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.06em;
      padding: 0 4px 8px; margin-bottom: 4px; border-bottom: 1px solid var(--border);
    }}
    .tabbar {{
      display: flex; flex-direction: column; gap: 12px; padding-top: 10px;
    }}
    .tabgroup {{
      padding: 4px 0 4px;
    }}
    .tabgroup + .tabgroup {{
      border-top: 1px solid var(--border);
      padding-top: 12px;
    }}
    .tabgroup-header {{
      display: flex; align-items: baseline; justify-content: space-between;
      padding: 0 4px 8px;
      gap: 12px; flex-wrap: wrap;
    }}
    .tabgroup-title-wrap {{
      display: inline-flex; align-items: baseline; gap: 10px;
    }}
    .group-chip {{
      display: inline-block;
      padding: 3px 10px; border-radius: 999px;
      font-size: 12px; font-weight: 700;
      letter-spacing: 0.02em;
    }}
    .tabgroup-subtitle {{
      font-size: 11px; color: var(--subtle);
      text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
    }}
    .tabgroup-stats {{
      font-size: 11px; color: var(--muted);
      font-variant-numeric: tabular-nums;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      display: inline-flex; gap: 6px;
    }}
    .tabgroup-cards {{
      display: flex; flex-wrap: wrap; gap: 8px;
    }}
    .tabgroup-pies {{
      display: none;
      grid-template-columns: 1fr 1fr; gap: 10px;
      margin-top: 12px; padding-top: 12px;
      border-top: 1px dashed var(--border);
    }}
    .tabgroup.expanded .tabgroup-pies {{ display: grid; }}
    .tab {{
      flex: 0 0 auto;
      min-width: 130px;
      background: #f8fafc;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 14px;
      cursor: pointer;
      text-align: left;
      font-family: inherit; color: inherit;
      transition: all 0.12s ease;
    }}
    .tab:hover {{
      border-color: #cbd5e1;
      background: #f1f5f9;
    }}
    .tab.active {{
      border-color: var(--accent);
      background: var(--accent-soft);
      box-shadow: 0 0 0 3px rgba(14,165,233,0.12);
    }}
    .tab .tab-name {{
      font-size: 13.5px; font-weight: 700; color: #0f172a;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      letter-spacing: -0.005em;
    }}
    .tab .tab-stats {{
      display: flex; gap: 8px; margin-top: 4px;
      font-size: 10.5px; color: var(--muted);
      font-variant-numeric: tabular-nums;
    }}
    .tab .ts-wt {{
      font-weight: 700; color: #0f172a;
      font-family: ui-monospace, Menlo, Consolas, monospace;
    }}
    .tab.active .ts-wt {{ color: var(--accent); }}

    /* Panel */
    .cat-panel {{ display: none; }}
    .cat-panel.active {{ display: block; }}
    .panel {{
      background: var(--card); border: 1px solid var(--border); border-radius: 14px;
      padding: 16px 0 16px; margin-bottom: 16px;
      box-shadow: 0 1px 3px rgba(15,23,42,0.04); overflow: hidden;
    }}
    .panel-header {{ padding: 0 20px 14px; border-bottom: 1px solid var(--border); margin-bottom: 14px; }}
    .panel-title {{ font-size: 14px; font-weight: 700; color: #0f172a; margin: 0 0 2px;
      text-transform: uppercase; letter-spacing: 0.04em; }}
    .panel-subtitle {{ font-size: 12px; color: var(--muted); }}
    .panel-subtitle code {{ background: #f1f5f9; padding: 1px 5px; border-radius: 4px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 11px; word-break: break-all; }}

    /* Filmstrip */
    .stripwrap {{ display: flex; padding-left: {YAXIS_W / VB_WIDTH * 100:.4f}%; padding-right: 0; }}
    .thumb {{ flex: 1 1 0; min-width: 0; padding: 0 2px; }}
    .thumb-img-wrap {{ position: relative; }}
    .thumb img {{
      width: 100%; height: {THUMB_DISPLAY_H_PX}px; object-fit: cover; display: block;
      border: 1px solid var(--border); border-bottom: 0; border-radius: 5px 5px 0 0;
    }}
    .thumb-time-badge {{
      position: absolute; top: 5px; left: 5px;
      background: rgba(15,23,42,0.78); color: #fff;
      font-size: 10px; padding: 2px 6px; border-radius: 4px;
      font-variant-numeric: tabular-nums;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    }}
    .thumb-wband {{
      height: 22px; display: flex; align-items: center; justify-content: center;
      color: #fff; font-size: 11px; font-weight: 700;
      border-radius: 0 0 5px 5px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      text-shadow: 0 0 2px rgba(0,0,0,0.25);
    }}

    /* Legend */
    .legend {{
      display: flex; flex-wrap: wrap; gap: 8px; align-items: center;
      padding: 14px 20px 4px;
    }}
    .legend-label {{
      font-size: 11px; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.06em; margin-right: 4px;
    }}
    .legend-chip {{
      font-size: 12px; color: #334155;
      display: inline-flex; align-items: center; gap: 7px;
      padding: 4px 12px; background: #f8fafc; border: 1px solid var(--border);
      border-radius: 999px;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    }}
    .legend-chip .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}

    /* Bars */
    .bar-block {{ padding: 8px 20px 14px; }}
    .bar-label {{
      font-size: 11px; font-weight: 600; color: var(--muted);
      text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
    }}
    .stack-bar {{
      display: flex; width: 100%; height: 42px; border-radius: 8px; overflow: hidden;
      border: 1px solid var(--border);
    }}
    .stack-seg {{
      display: flex; align-items: center; justify-content: center; color: #fff;
      min-width: 0; overflow: hidden;
    }}
    .stack-seg-inner {{ display: flex; flex-direction: column; align-items: center; line-height: 1.1;
      text-shadow: 0 0 2px rgba(0,0,0,0.35); padding: 0 4px; white-space: nowrap; }}
    .stack-seg-label {{ font-size: 11px; font-weight: 700;
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
    .stack-seg-pct {{ font-size: 11px; font-variant-numeric: tabular-nums; opacity: 0.95; }}

    /* Table */
    .aggtable {{
      width: calc(100% - 40px); margin: 4px 20px 0;
      border-collapse: collapse; font-size: 13px;
    }}
    .aggtable th, .aggtable td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }}
    .aggtable th {{
      text-align: left; color: var(--muted); font-weight: 600;
      font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
      background: #f8fafc;
    }}
    .aggtable th.right, .aggtable td.right {{ text-align: right; }}
    .aggtable .mono {{
      font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
      font-variant-numeric: tabular-nums;
    }}
    .aggtable .strong {{ font-weight: 700; color: #0f172a; }}
    .aggtable .muted-note {{ color: var(--subtle); font-size: 11px; margin-left: 4px; }}
    .aggtable .wchip {{
      display: inline-block; min-width: 56px; padding: 2px 10px;
      border-radius: 999px; color: #fff; font-weight: 700; font-size: 11px;
      text-align: center; font-family: ui-monospace, Menlo, Consolas, monospace;
    }}
    .aggtable .total-row td {{ border-top: 2px solid #cbd5e1; background: #f8fafc; padding-top: 10px; }}

    .empty {{ padding: 22px 20px; color: var(--subtle); font-style: italic; text-align: center; }}
    """

    js = """
    const POP_PX = 10;
    function popSlicesForActive() {
      const tab = document.querySelector('.tab.active');
      document.querySelectorAll('.pie-slice').forEach(p => { p.style.transform = ''; });
      if (!tab) return;
      const cat = tab.dataset.cat;
      const group = tab.dataset.group;
      const pop = (selector) => {
        document.querySelectorAll(selector).forEach(p => {
          const mid = parseFloat(p.dataset.mid);
          const dx = Math.cos(mid) * POP_PX;
          const dy = Math.sin(mid) * POP_PX;
          p.style.transform = `translate(${dx}px, ${dy}px)`;
        });
      };
      if (cat)   pop('.pie-slice[data-cat="' + cat + '"]');     // tabgroup focus pies
      if (group) pop('.pie-slice[data-group="' + group + '"]'); // hero pies
    }
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const cat = tab.dataset.cat;
        const tabgroup = tab.closest('.tabgroup');
        const wasActiveSameTab = tab.classList.contains('active') && tabgroup.classList.contains('expanded');
        document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t === tab));
        document.querySelectorAll('.cat-panel').forEach(p => {
          p.classList.toggle('active', p.id === 'panel-' + cat);
        });
        // Expand the clicked tab's group, fold all others. Re-clicking the
        // already-active tab in the already-expanded group folds it back.
        document.querySelectorAll('.tabgroup').forEach(g => {
          if (g === tabgroup) g.classList.toggle('expanded', !wasActiveSameTab);
          else g.classList.remove('expanded');
        });
        popSlicesForActive();
      });
    });
    window.addEventListener('load', popSlicesForActive);
    """

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>叠纸盒数据汇总</title>
<style>{css}</style>
</head><body>
<div class="container">

  <div class="page-header">
    <div class="eyebrow">Fold Box · Corpus Overview</div>
    <div class="accent-line"></div>
    <h1>叠纸盒数据汇总</h1>
    <div class="tagline">
      cfg: <span class="pill">{os.path.basename(CFG_PATH)}</span>
    </div>
  </div>

  <div class="hero">
    <div class="kpis">{build_kpi_row(hero_kpis)}</div>
    {pies_html}
  </div>

  <div class="tabbar-wrap">
    <div class="tabbar-label">Categories · 按类别分组 · 各组内按加权帧数降序 · 点击切换</div>
    <div class="tabbar">{"".join(tab_groups_html)}</div>
  </div>

  {panels}

</div>
<script>{js}</script>
</body></html>
"""


def _cfg_tag(cfg_path: str) -> str:
    """MMDD tag from cfg basename, falling back to basename-without-.py."""
    m = CFG_DATE_RE.search(os.path.basename(cfg_path))
    return m.group(1) if m else os.path.basename(cfg_path).replace(".py", "")


def main() -> None:
    global CFG_PATH, OUT_DIR
    parser = argparse.ArgumentParser(
        description="Build a cfg-driven multi-category corpus visualisation. "
                    "Cache lives under .viz_cache/<cfg_tag>/, final HTML at config_viz/viz_<cfg_tag>.html."
    )
    parser.add_argument("--cfg", default=DEFAULT_CFG,
                        help="path to cfg_pi05_base_finetune_box_recap_pt_*.py")
    parser.add_argument("--out", default=None,
                        help="cache dir override (default: .viz_cache/<cfg_tag>/ next to this script)")
    args = parser.parse_args()
    CFG_PATH = os.path.abspath(args.cfg)
    if not os.path.exists(CFG_PATH):
        raise SystemExit(f"cfg not found: {CFG_PATH}")

    tag = _cfg_tag(CFG_PATH)
    OUT_DIR = os.path.abspath(args.out) if args.out else os.path.join(CACHE_ROOT, tag)

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"cfg:   {CFG_PATH}")
    print(f"cache: {OUT_DIR}")
    print("aggregating cfg per-category stats...")
    per_cat = aggregate_per_category()
    print(f"  categories found: {sorted(per_cat.keys())}")
    print("extracting representative thumbnails per category...")
    html = build_html(per_cat)
    out_path = os.path.join(OUT_DIR, "viz.html")
    with open(out_path, "w") as f:
        f.write(html)
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"saved -> {out_path}  ({size_kb:.1f} KB)")

    # Publish the canonical user-facing copy.
    os.makedirs(PUBLISH_DIR, exist_ok=True)
    publish_path = os.path.join(PUBLISH_DIR, f"viz_{tag}.html")
    shutil.copyfile(out_path, publish_path)
    print(f"published -> {publish_path}")
    print(f"open with:  file://{publish_path}")


if __name__ == "__main__":
    main()
