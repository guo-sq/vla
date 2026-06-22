"""
HTML comparison between two cfg_pi05_base_finetune_box_recap_pt_*.py files.

Layout
------
  * Page header (eyebrow + h1 "A → B")
  * Hero: 6 KPI Δ tiles + two side-by-side pies (加权帧分布 by 大类, A vs B)
  * Change Pattern cards, grouped by category — main view
        each card: representative filmstrip + A/B dual timeline + diff snippet
  * Added / Removed entries panels (if any)
  * Category summary table
  * Full entry-level diff table (collapsed by default in <details>)

Caches under .viz_cache/diff_<tagA>_vs_<tagB>/ for thumbs + per-rel frame counts.
HTML auto-published to config_viz/diff_<tagA>_vs_<tagB>.html.

Usage:
    # default 0522 → 0525
    python tools/fold_box/make_cfg_diff_viz.py
    # any other pair
    python tools/fold_box/make_cfg_diff_viz.py \\
        --cfg-a src/openpi/configs/cfg_pi05_base_finetune_box_recap_pt_0522.py \\
        --cfg-b src/openpi/configs/cfg_pi05_base_finetune_box_recap_pt_0525.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field

import pandas as pd

# Reuse the single-cfg viz helpers (palette, parsing, SVG, pie, kpis, ...).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
import make_cfg_viz as mcv  # noqa: E402


DEFAULT_CFG_A = (
    "/mnt/workspace/zengqi/openpi_modified/src/openpi/configs/"
    "cfg_pi05_base_finetune_box_recap_pt_0522.py"
)
DEFAULT_CFG_B = (
    "/mnt/workspace/zengqi/openpi_modified/src/openpi/configs/"
    "cfg_pi05_base_finetune_box_recap_pt_0525.py"
)


# ---------- cfg parsing (independent of mcv globals) ----------

def parse_cfg_entries_path(path: str) -> dict:
    """Re-parse a cfg using the shared CFG_ENTRY_RE without mutating mcv.CFG_PATH.

    Returns {rel: (cat, group, segs_tuple)}. segs is normalised to a tuple of
    tuples so it can be used as a dict key.
    """
    text = open(path).read()
    out: dict[str, tuple] = {}
    for m in mcv.CFG_ENTRY_RE.finditer(text):
        rel = m.group(1)
        segs_str = m.group(2)
        line_start = text.rfind("\n", 0, m.start()) + 1
        if text[line_start:m.start()].lstrip().startswith("#"):
            continue
        try:
            segs = eval(segs_str)
        except Exception:
            continue

        if rel.startswith("infer/"):
            parts = rel.split("/")
            cat = parts[1] if len(parts) >= 2 else "infer"
            group = "推理类"
        else:
            dir_cat = rel.split("/", 1)[0]
            name = rel.split("/")[-1]
            try:
                name_cat = name.split("fold_box_from_scratch.", 1)[1].split(".")[0]
            except IndexError:
                continue
            cat = name_cat if name_cat.startswith(dir_cat) else dir_cat
            group = mcv._classify_normal_cat(cat)

        out[rel] = (cat, group, tuple(tuple(s) for s in segs))
    return out


# ---------- per-rel frame count caching ----------

def load_frames_cache(out_dir: str) -> dict:
    path = os.path.join(out_dir, ".frames_cache.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_frames_cache(out_dir: str, cache: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, ".frames_cache.json"), "w") as f:
        json.dump(cache, f)


def get_frames_for_rel(rel: str, cache: dict) -> list[int]:
    if rel in cache:
        return cache[rel]
    files = sorted(glob.glob(os.path.join(mcv.ROOT, rel, "data/chunk-000/episode_*.parquet")))
    counts: list[int] = []
    for f in files:
        try:
            n = pd.read_parquet(f, columns=["frame_index"]).shape[0]
            counts.append(n)
        except Exception:
            continue
    cache[rel] = counts
    return counts


def weighted_for(frame_counts: list[int], segs) -> tuple[int, int, dict[int, int]]:
    """Return (total_raw, total_weighted, buckets[w -> frames]) for a single
    cfg-entry across all its episodes."""
    total_raw = 0
    total_weighted = 0
    buckets: dict[int, int] = {}
    for n in frame_counts:
        total_raw += n
        b, tw = mcv.episode_buckets(n, segs)
        for k, v in b.items():
            buckets[k] = buckets.get(k, 0) + v
        total_weighted += tw
    return total_raw, total_weighted, buckets


# ---------- change pattern aggregation ----------

@dataclass
class ChangePattern:
    cat: str
    group: str
    segs_a: tuple
    segs_b: tuple
    rels: list[str] = field(default_factory=list)
    representative: str | None = None  # rel with a usable video
    rep_duration: float = 0.0
    n_eps: int = 0
    total_raw: int = 0
    weighted_a: int = 0
    weighted_b: int = 0


def find_representative(rels: list[str]) -> str | None:
    """First rel that has an episode_000000.mp4 head camera video."""
    for rel in rels:
        vp = os.path.join(mcv.ROOT, rel, "videos/chunk-000", mcv.CAMERA_KEY, "episode_000000.mp4")
        if os.path.exists(vp):
            return rel
    return None


def aggregate_diff(cfg_a: str, cfg_b: str, out_dir: str) -> dict:
    print(f"  parsing {os.path.basename(cfg_a)} ...")
    entries_a = parse_cfg_entries_path(cfg_a)
    print(f"  parsing {os.path.basename(cfg_b)} ...")
    entries_b = parse_cfg_entries_path(cfg_b)

    all_rels = sorted(set(entries_a) | set(entries_b))
    print(f"  union rels: {len(all_rels)} (A {len(entries_a)} / B {len(entries_b)})")

    print("  loading per-rel frame counts (parquet, cached)...")
    frames_cache = load_frames_cache(out_dir)
    miss = 0
    t0 = time.time()
    for rel in all_rels:
        if rel not in frames_cache:
            miss += 1
            get_frames_for_rel(rel, frames_cache)
    if miss:
        save_frames_cache(out_dir, frames_cache)
    print(f"    cache miss: {miss} (filled in {time.time() - t0:.1f}s)")

    # Per-rel weighted stats under each cfg's segs.
    per_rel_a: dict[str, tuple] = {}  # rel -> (raw, weighted, buckets)
    per_rel_b: dict[str, tuple] = {}
    for rel, (_cat, _grp, segs) in entries_a.items():
        per_rel_a[rel] = weighted_for(frames_cache.get(rel, []), segs)
    for rel, (_cat, _grp, segs) in entries_b.items():
        per_rel_b[rel] = weighted_for(frames_cache.get(rel, []), segs)

    changed: list[tuple] = []   # (cat, group, rel, segs_a, segs_b, raw, wt_a, wt_b)
    added: list[tuple] = []     # (cat, group, rel, segs_b, raw, wt_b)
    removed: list[tuple] = []   # (cat, group, rel, segs_a, raw, wt_a)
    unchanged_n = 0

    for rel in all_rels:
        in_a = rel in entries_a
        in_b = rel in entries_b
        if in_a and in_b:
            cat_a, grp_a, segs_a = entries_a[rel]
            cat_b, grp_b, segs_b = entries_b[rel]
            if segs_a == segs_b:
                unchanged_n += 1
                continue
            raw_a, wt_a, _ = per_rel_a[rel]
            _, wt_b, _ = per_rel_b[rel]
            # Use B's metadata for cat/group (newer cfg wins for display).
            changed.append((cat_b, grp_b, rel, segs_a, segs_b, raw_a, wt_a, wt_b))
        elif in_b:
            cat, grp, segs = entries_b[rel]
            raw, wt, _ = per_rel_b[rel]
            added.append((cat, grp, rel, segs, raw, wt))
        else:
            cat, grp, segs = entries_a[rel]
            raw, wt, _ = per_rel_a[rel]
            removed.append((cat, grp, rel, segs, raw, wt))

    print(f"  changed {len(changed)}  added {len(added)}  removed {len(removed)}  unchanged {unchanged_n}")

    # Group changed entries by (cat, segs_a, segs_b) → ChangePattern.
    patterns_map: dict[tuple, ChangePattern] = {}
    for cat, group, rel, segs_a, segs_b, raw, wt_a, wt_b in changed:
        key = (cat, segs_a, segs_b)
        cp = patterns_map.get(key)
        if cp is None:
            cp = ChangePattern(cat=cat, group=group, segs_a=segs_a, segs_b=segs_b)
            patterns_map[key] = cp
        cp.rels.append(rel)
        cp.n_eps += len(frames_cache.get(rel, []))
        cp.total_raw += raw
        cp.weighted_a += wt_a
        cp.weighted_b += wt_b

    for cp in patterns_map.values():
        cp.representative = find_representative(cp.rels)

    patterns = sorted(patterns_map.values(), key=lambda p: -(abs(p.weighted_a - p.weighted_b)))

    # Global KPIs.
    n_eps_a = sum(len(frames_cache.get(rel, [])) for rel in entries_a)
    n_eps_b = sum(len(frames_cache.get(rel, [])) for rel in entries_b)
    raw_a = sum(per_rel_a[r][0] for r in entries_a)
    raw_b = sum(per_rel_b[r][0] for r in entries_b)
    wt_a = sum(per_rel_a[r][1] for r in entries_a)
    wt_b = sum(per_rel_b[r][1] for r in entries_b)
    cats_a = sorted({entries_a[r][0] for r in entries_a})
    cats_b = sorted({entries_b[r][0] for r in entries_b})

    # Per-group raw + weighted totals (for the two hero pies + two bar charts).
    grp_raw_a: dict[str, int] = {}
    grp_raw_b: dict[str, int] = {}
    grp_wt_a: dict[str, int] = {}
    grp_wt_b: dict[str, int] = {}
    for rel, (_c, grp, _s) in entries_a.items():
        grp_raw_a[grp] = grp_raw_a.get(grp, 0) + per_rel_a[rel][0]
        grp_wt_a[grp] = grp_wt_a.get(grp, 0) + per_rel_a[rel][1]
    for rel, (_c, grp, _s) in entries_b.items():
        grp_raw_b[grp] = grp_raw_b.get(grp, 0) + per_rel_b[rel][0]
        grp_wt_b[grp] = grp_wt_b.get(grp, 0) + per_rel_b[rel][1]

    # Per-category roll-up for the summary table.
    cat_rollup: dict[str, dict] = {}
    for rel, (cat, grp, _s) in entries_a.items():
        d = cat_rollup.setdefault(cat, {"group": grp, "n_eps_a": 0, "raw_a": 0, "wt_a": 0,
                                       "n_eps_b": 0, "raw_b": 0, "wt_b": 0})
        d["n_eps_a"] += len(frames_cache.get(rel, []))
        d["raw_a"] += per_rel_a[rel][0]
        d["wt_a"] += per_rel_a[rel][1]
    for rel, (cat, grp, _s) in entries_b.items():
        d = cat_rollup.setdefault(cat, {"group": grp, "n_eps_a": 0, "raw_a": 0, "wt_a": 0,
                                       "n_eps_b": 0, "raw_b": 0, "wt_b": 0})
        d["group"] = grp
        d["n_eps_b"] += len(frames_cache.get(rel, []))
        d["raw_b"] += per_rel_b[rel][0]
        d["wt_b"] += per_rel_b[rel][1]

    return {
        "entries_a": entries_a,
        "entries_b": entries_b,
        "frames_cache": frames_cache,
        "patterns": patterns,
        "changed": changed,
        "added": added,
        "removed": removed,
        "unchanged_n": unchanged_n,
        "kpis": {
            "n_cats_a": len(cats_a), "n_cats_b": len(cats_b),
            "n_ds_a": len(entries_a), "n_ds_b": len(entries_b),
            "n_eps_a": n_eps_a, "n_eps_b": n_eps_b,
            "raw_a": raw_a, "raw_b": raw_b,
            "wt_a": wt_a, "wt_b": wt_b,
        },
        "grp_raw_a": grp_raw_a,
        "grp_raw_b": grp_raw_b,
        "grp_wt_a": grp_wt_a,
        "grp_wt_b": grp_wt_b,
        "cat_rollup": cat_rollup,
    }


# ---------- pretty printers ----------

def _segs_pretty(segs, highlight_diff_with=None) -> str:
    """Render a segs list inline. If `highlight_diff_with` is given, segments that
    *differ* element-wise from the corresponding element of the other segs get
    a <strong> wrap around the weight value."""
    parts = []
    if highlight_diff_with is None or len(segs) != len(highlight_diff_with):
        # No alignment possible — render plainly.
        for s in segs:
            if len(s) >= 3:
                a, b, w = s[0], s[1], s[2]
                parts.append(f"({_fmt_num(a)},{_fmt_num(b)},{_fmt_num(w)})")
        return " ".join(parts)
    for s, o in zip(segs, highlight_diff_with):
        a, b, w = s[0], s[1], s[2]
        oa, ob, ow = o[0], o[1], o[2]
        if (a, b) != (oa, ob):
            # Boundaries differ — wrap the whole tuple.
            parts.append(f"<strong>({_fmt_num(a)},{_fmt_num(b)},{_fmt_num(w)})</strong>")
        elif w != ow:
            parts.append(f"({_fmt_num(a)},{_fmt_num(b)},<strong>{_fmt_num(w)}</strong>)")
        else:
            parts.append(f"({_fmt_num(a)},{_fmt_num(b)},{_fmt_num(w)})")
    return " ".join(parts)


def _fmt_num(x) -> str:
    if isinstance(x, float) and x.is_integer():
        return str(int(x))
    return str(x)


def _fmt_signed(n: int) -> str:
    if n > 0:
        return f"+{n:,}"
    return f"{n:,}"


def _fmt_signed_compact(n: int) -> str:
    sign = "+" if n > 0 else ("−" if n < 0 else "")
    return sign + mcv._fmt_compact(abs(n))


def _pct_change(a: int, b: int) -> str:
    if a == 0:
        return "—"
    delta = (b - a) / a * 100
    return f"{delta:+.1f}%"


def _nice_y_upper(max_val: float) -> float:
    """Round max_val up to a "nice" axis upper bound (1, 2, 5, 10 × 10^k)."""
    import math as _math
    if max_val <= 0:
        return 1.0
    exp = _math.floor(_math.log10(max_val))
    base = 10 ** exp
    n = max_val / base
    if n <= 1:
        nice = 1
    elif n <= 2:
        nice = 2
    elif n <= 5:
        nice = 5
    else:
        nice = 10
    return nice * base


def build_group_bars_svg(title: str, data_a: dict, data_b: dict,
                         tag_a: str, tag_b: str) -> str:
    """Side-by-side grouped bar chart per major group: CFG-A bar + CFG-B bar.

    data_a, data_b: {group_name: int}. Iteration follows mcv.CATEGORY_GROUPS so
    both charts share an identical x-axis ordering.
    """
    groups = [(gn, fg, bg) for (gn, _sub, fg, bg) in mcv.CATEGORY_GROUPS
              if data_a.get(gn, 0) > 0 or data_b.get(gn, 0) > 0]
    if not groups:
        return ""

    vb_w, vb_h = 880, 360
    margin_l, margin_r, margin_t, margin_b = 70, 16, 30, 96
    plot_x = margin_l
    plot_y = margin_t
    plot_w = vb_w - margin_l - margin_r
    plot_h = vb_h - margin_t - margin_b
    plot_bottom = plot_y + plot_h

    max_val = max(max(data_a.get(gn, 0), data_b.get(gn, 0)) for (gn, _, _) in groups)
    y_upper = _nice_y_upper(max_val)

    n_groups = len(groups)
    col_w = plot_w / n_groups
    bar_w = min(40, col_w * 0.36)
    gap = 4

    parts: list[str] = []

    # plot background
    parts.append(
        f'<rect x="{plot_x}" y="{plot_y}" width="{plot_w}" height="{plot_h}" '
        f'fill="#fafbfc" stroke="none"/>'
    )

    # horizontal gridlines + y-axis labels (5 divisions)
    for i in range(6):
        v = y_upper * i / 5
        y = plot_bottom - (i / 5) * plot_h
        parts.append(
            f'<line x1="{plot_x}" x2="{plot_x + plot_w}" '
            f'y1="{y:.2f}" y2="{y:.2f}" stroke="#e2e8f0" stroke-width="0.6"/>'
        )
        parts.append(
            f'<text x="{plot_x - 10}" y="{y + 4:.2f}" font-size="11" '
            f'text-anchor="end" fill="#64748b" '
            f'font-family="ui-monospace, Menlo, Consolas, monospace">'
            f'{mcv._fmt_compact(int(v))}</text>'
        )

    # axes
    parts.append(
        f'<line x1="{plot_x}" x2="{plot_x + plot_w}" '
        f'y1="{plot_bottom}" y2="{plot_bottom}" stroke="#475569" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{plot_x}" x2="{plot_x}" '
        f'y1="{plot_y}" y2="{plot_bottom}" stroke="#475569" stroke-width="1"/>'
    )

    # bars
    for i, (gn, fg, _bg) in enumerate(groups):
        col_cx = plot_x + col_w * (i + 0.5)
        va = data_a.get(gn, 0)
        vb = data_b.get(gn, 0)
        ha = (va / y_upper) * plot_h
        hb = (vb / y_upper) * plot_h
        x_a = col_cx - bar_w - gap / 2
        x_b = col_cx + gap / 2

        # A bar — lighter, with outline
        parts.append(
            f'<rect x="{x_a:.2f}" y="{plot_bottom - ha:.2f}" '
            f'width="{bar_w:.2f}" height="{ha:.2f}" '
            f'fill="{fg}" fill-opacity="0.32" stroke="{fg}" stroke-width="1"/>'
        )
        if va > 0:
            parts.append(
                f'<text x="{x_a + bar_w/2:.2f}" y="{plot_bottom - ha - 5:.2f}" '
                f'font-size="10" text-anchor="middle" fill="#475569" '
                f'font-family="ui-monospace, Menlo, Consolas, monospace">'
                f'{mcv._fmt_compact(va)}</text>'
            )

        # B bar — solid
        parts.append(
            f'<rect x="{x_b:.2f}" y="{plot_bottom - hb:.2f}" '
            f'width="{bar_w:.2f}" height="{hb:.2f}" '
            f'fill="{fg}" stroke="none"/>'
        )
        if vb > 0:
            parts.append(
                f'<text x="{x_b + bar_w/2:.2f}" y="{plot_bottom - hb - 5:.2f}" '
                f'font-size="10" text-anchor="middle" fill="#0f172a" '
                f'font-weight="700" '
                f'font-family="ui-monospace, Menlo, Consolas, monospace">'
                f'{mcv._fmt_compact(vb)}</text>'
            )

        # group name label
        parts.append(
            f'<text x="{col_cx:.2f}" y="{plot_bottom + 18:.2f}" '
            f'font-size="11" text-anchor="middle" fill="#334155" '
            f'font-weight="700">{gn}</text>'
        )

        # delta label
        delta = vb - va
        if delta != 0 and va > 0:
            pct = (delta / va) * 100
            sign = "+" if delta > 0 else "−"
            delta_str = sign + mcv._fmt_compact(abs(delta))
            color = "#15803d" if delta > 0 else "#b91c1c"
            parts.append(
                f'<text x="{col_cx:.2f}" y="{plot_bottom + 34:.2f}" '
                f'font-size="10" text-anchor="middle" fill="{color}" '
                f'font-family="ui-monospace, Menlo, Consolas, monospace" '
                f'font-weight="700">Δ {delta_str} ({pct:+.0f}%)</text>'
            )
        elif delta != 0:
            sign = "+" if delta > 0 else "−"
            delta_str = sign + mcv._fmt_compact(abs(delta))
            color = "#15803d" if delta > 0 else "#b91c1c"
            parts.append(
                f'<text x="{col_cx:.2f}" y="{plot_bottom + 34:.2f}" '
                f'font-size="10" text-anchor="middle" fill="{color}" '
                f'font-family="ui-monospace, Menlo, Consolas, monospace" '
                f'font-weight="700">Δ {delta_str}</text>'
            )

    # legend (A vs B)
    legend_y = vb_h - 18
    parts.append(
        f'<rect x="{plot_x}" y="{legend_y - 10}" width="14" height="10" '
        f'fill="#64748b" fill-opacity="0.32" stroke="#64748b" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{plot_x + 20}" y="{legend_y - 1}" font-size="11.5" '
        f'fill="#475569" font-weight="600" '
        f'font-family="ui-monospace, Menlo, Consolas, monospace">CFG-{tag_a}</text>'
    )
    parts.append(
        f'<rect x="{plot_x + 130}" y="{legend_y - 10}" width="14" height="10" '
        f'fill="#64748b"/>'
    )
    parts.append(
        f'<text x="{plot_x + 150}" y="{legend_y - 1}" font-size="11.5" '
        f'fill="#0f172a" font-weight="700" '
        f'font-family="ui-monospace, Menlo, Consolas, monospace">CFG-{tag_b}</text>'
    )

    return (
        '<div class="bar-chart-block">'
        f'<div class="bar-chart-title">{title}</div>'
        f'<svg viewBox="0 0 {vb_w} {vb_h}" '
        f'style="width: 100%; height: auto; display: block;">'
        + "\n".join(parts) +
        '</svg></div>'
    )


def _delta_class(a: int, b: int) -> str:
    if b > a:
        return "delta-up"
    if b < a:
        return "delta-down"
    return "delta-zero"


# ---------- HTML sections ----------

def build_hero(kpis: dict, grp_raw_a: dict, grp_raw_b: dict,
               grp_wt_a: dict, grp_wt_b: dict, n_changed: int,
               n_added: int, n_removed: int, tag_a: str, tag_b: str) -> str:
    tiles = []

    def _tile(label, va, vb, hint=""):
        cls = _delta_class(va if isinstance(va, (int, float)) else 0,
                           vb if isinstance(vb, (int, float)) else 0)
        delta_html = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != vb:
            delta_html = (
                f'<div class="diff-tile-delta {cls}">'
                f'{_fmt_signed_compact(vb - va)} ({_pct_change(va, vb)})</div>'
            )
        return (
            '<div class="diff-tile">'
            f'<div class="diff-tile-label">{label}</div>'
            f'<div class="diff-tile-pair">'
            f'<span class="dt-a">{va}</span>'
            f'<span class="dt-arrow">→</span>'
            f'<span class="dt-b">{vb}</span>'
            f'</div>'
            f'{delta_html}'
            f'<div class="diff-tile-hint">{hint}</div>'
            '</div>'
        )

    tiles.append(_tile(
        "Categories", kpis["n_cats_a"], kpis["n_cats_b"], "cfg 中带 weight 的类别数"
    ))
    tiles.append(_tile(
        "Datasets",
        mcv._fmt_int(kpis["n_ds_a"]), mcv._fmt_int(kpis["n_ds_b"]),
        "cfg 中未注释的数据集条目数"
    ))
    tiles.append(_tile(
        "Episodes",
        mcv._fmt_int(kpis["n_eps_a"]), mcv._fmt_int(kpis["n_eps_b"]),
        "所有 batch 的 episode 总和"
    ))
    # Standalone counter for changed datasets — no A/B pair.
    tiles.append(
        '<div class="diff-tile">'
        '<div class="diff-tile-label">Datasets changed</div>'
        f'<div class="diff-tile-pair single"><span class="dt-b">{n_changed:,}</span></div>'
        '<div class="diff-tile-delta">'
        f'<span class="badge badge-add">+{n_added}</span> '
        f'<span class="badge badge-rm">−{n_removed}</span></div>'
        '<div class="diff-tile-hint">weight tuple 改动 · 新增/移除数据集</div>'
        '</div>'
    )
    tiles.append(_tile(
        "Total raw frames",
        mcv._fmt_int(kpis["raw_a"]), mcv._fmt_int(kpis["raw_b"]),
        f"≈ {mcv._fmt_compact(kpis['raw_a'])} → {mcv._fmt_compact(kpis['raw_b'])} @ 30 fps"
    ))
    tiles.append(_tile(
        "Total weighted",
        mcv._fmt_int(kpis["wt_a"]), mcv._fmt_int(kpis["wt_b"]),
        f"Σ raw × weight · ≈ {mcv._fmt_compact(kpis['wt_a'])} → {mcv._fmt_compact(kpis['wt_b'])}"
    ))
    amp_a = kpis["wt_a"] / max(kpis["raw_a"], 1)
    amp_b = kpis["wt_b"] / max(kpis["raw_b"], 1)
    cls = _delta_class(amp_a, amp_b)
    delta_amp = ""
    if amp_a != amp_b:
        delta_amp = (
            f'<div class="diff-tile-delta {cls}">'
            f'{amp_b - amp_a:+.2f}× ({(amp_b - amp_a)/amp_a*100:+.1f}%)</div>'
        )
    tiles.append(
        '<div class="diff-tile">'
        '<div class="diff-tile-label">Amplification</div>'
        '<div class="diff-tile-pair">'
        f'<span class="dt-a">{amp_a:.2f}×</span>'
        '<span class="dt-arrow">→</span>'
        f'<span class="dt-b">{amp_b:.2f}×</span>'
        '</div>'
        f'{delta_amp}'
        '<div class="diff-tile-hint">weighted / raw</div>'
        '</div>'
    )

    # Hero pies: weighted by group, A vs B. Keep a stable colour per group.
    group_color = {gn: fg for (gn, _sub, fg, _bg) in mcv.CATEGORY_GROUPS}
    total_a = sum(grp_wt_a.values())
    total_b = sum(grp_wt_b.values())
    slices_a, slices_b = [], []
    for (gn, _sub, _fg, _bg) in mcv.CATEGORY_GROUPS:
        va = grp_wt_a.get(gn, 0)
        vb = grp_wt_b.get(gn, 0)
        c = group_color.get(gn, "#475569")
        if va > 0:
            slices_a.append({"name": gn, "value": va, "color": c})
        if vb > 0:
            slices_b.append({"name": gn, "value": vb, "color": c})
    pies_html = (
        '<div class="hero-pies">'
        + mcv.build_pie_block(f"加权帧分布 · CFG-{tag_a} · Σ {mcv._fmt_compact(total_a)}",
                              slices_a, total_a)
        + mcv.build_pie_block(f"加权帧分布 · CFG-{tag_b} · Σ {mcv._fmt_compact(total_b)}",
                              slices_b, total_b)
        + '</div>'
    )

    bars_html = (
        '<div class="hero-bars">'
        + build_group_bars_svg("原始帧数 · 按大类对比 (不加权)",
                               grp_raw_a, grp_raw_b, tag_a, tag_b)
        + build_group_bars_svg("加权帧数 · 按大类对比",
                               grp_wt_a, grp_wt_b, tag_a, tag_b)
        + '</div>'
    )

    return (
        '<div class="hero">'
        f'<div class="diff-tiles">{"".join(tiles)}</div>'
        f'{pies_html}'
        f'{bars_html}'
        '</div>'
    )


def build_change_pattern_card(cp: ChangePattern, tag_a: str, tag_b: str) -> str:
    # Δ summary
    delta_wt = cp.weighted_b - cp.weighted_a
    amp_a = cp.weighted_a / max(cp.total_raw, 1)
    amp_b = cp.weighted_b / max(cp.total_raw, 1)
    cls = _delta_class(cp.weighted_a, cp.weighted_b)
    delta_pct = _pct_change(cp.weighted_a, cp.weighted_b)

    # Filmstrip + dual timeline (only if we have a representative with video).
    body_html = ""
    if cp.representative is not None:
        try:
            duration, thumbs = mcv.extract_thumbs_for(cp.representative)
        except Exception:
            duration, thumbs = 0.0, []
        if thumbs:
            # Filmstrip coloured by CFG-<tag_b>'s segs (post-change palette).
            strip = mcv.build_strip(duration, thumbs, list(cp.segs_b))
            tl_a = mcv.build_svg(duration, list(cp.segs_a))
            tl_b = mcv.build_svg(duration, list(cp.segs_b))
            body_html = (
                f'<div class="cp-sample-meta">'
                f'代表数据集: <code>{cp.representative}</code> · '
                f'ep000 · duration ≈ {duration:.1f}s · '
                f'filmstrip 以 <strong>CFG-{tag_b}</strong> 的 segs 上色(新版)</div>'
                f'{strip}'
                f'<div class="cp-tracks">'
                f'<div class="cp-track">'
                f'<div class="cp-track-label cp-track-label-a">CFG-{tag_a}</div>'
                f'{tl_a}'
                f'</div>'
                f'<div class="cp-track">'
                f'<div class="cp-track-label cp-track-label-b">CFG-{tag_b}</div>'
                f'{tl_b}'
                f'</div>'
                f'</div>'
            )
        else:
            body_html = (
                f'<div class="empty">无法抽取 representative 视频帧:'
                f' {cp.representative}</div>'
            )
    else:
        body_html = '<div class="empty">本 pattern 下没有可用视频样本</div>'

    # Diff line (text)
    diff_text = (
        '<div class="cp-diff-line">'
        f'<span class="cp-diff-side cp-diff-a">{tag_a}:</span>'
        f'<span class="cp-diff-segs">{_segs_pretty(cp.segs_a, cp.segs_b)}</span>'
        '</div>'
        '<div class="cp-diff-line">'
        f'<span class="cp-diff-side cp-diff-b">{tag_b}:</span>'
        f'<span class="cp-diff-segs">{_segs_pretty(cp.segs_b, cp.segs_a)}</span>'
        '</div>'
    )

    rels_block = ""
    if len(cp.rels) > 1:
        rels_block = (
            f'<details class="cp-rels"><summary>'
            f'{len(cp.rels)} 个数据集 · 点击展开列表</summary>'
            f'<ul class="cp-rels-list">'
            + "".join(f'<li><code>{os.path.basename(r)}</code></li>' for r in cp.rels)
            + '</ul></details>'
        )

    return (
        '<div class="cp-card">'
        '<div class="cp-card-header">'
        f'<div class="cp-card-title">'
        f'<span class="cp-cat-chip">{cp.cat}</span>'
        f'<span class="cp-card-subtitle">{cp.group}</span>'
        '</div>'
        '<div class="cp-card-kpis">'
        f'<span class="cp-kpi"><span class="cp-kpi-lab">eps</span>'
        f'<span class="cp-kpi-val">{cp.n_eps:,}</span></span>'
        f'<span class="cp-kpi"><span class="cp-kpi-lab">数据集</span>'
        f'<span class="cp-kpi-val">{len(cp.rels):,}</span></span>'
        f'<span class="cp-kpi"><span class="cp-kpi-lab">Δ weighted</span>'
        f'<span class="cp-kpi-val {cls}">{_fmt_signed_compact(delta_wt)} ({delta_pct})</span>'
        '</span>'
        f'<span class="cp-kpi"><span class="cp-kpi-lab">amp</span>'
        f'<span class="cp-kpi-val">{amp_a:.2f}× → {amp_b:.2f}×</span></span>'
        '</div>'
        '</div>'
        '<div class="cp-card-body">'
        f'{diff_text}'
        f'{body_html}'
        f'{rels_block}'
        '</div>'
        '</div>'
    )


def build_added_removed_panels(added: list, removed: list, tag_a: str, tag_b: str) -> str:
    out_parts = []
    for kind, items, css_cls, label in (
        ("added", added, "added", f"新增数据集 · 仅 CFG-{tag_b} 有 · {len(added)} 个"),
        ("removed", removed, "removed", f"移除数据集 · 仅 CFG-{tag_a} 有 · {len(removed)} 个"),
    ):
        if not items:
            continue
        rows = []
        for cat, group, rel, segs, raw, wt in items:
            rows.append(
                f'<tr class="{css_cls}-row">'
                f'<td><span class="cp-cat-chip cp-cat-chip-sm">{cat}</span></td>'
                f'<td class="mono"><code>{os.path.basename(rel)}</code></td>'
                f'<td class="mono">{_segs_pretty(segs)}</td>'
                f'<td class="mono right">{raw:,}</td>'
                f'<td class="mono right strong">{wt:,}</td>'
                '</tr>'
            )
        out_parts.append(
            '<div class="panel">'
            '<div class="panel-header">'
            f'<div class="panel-title">{label}</div>'
            '<div class="panel-subtitle">'
            + (f"仅出现在 CFG-{tag_b} 中的数据集 — 新增 entry / 新的 weight tuple"
               if kind == "added" else
               f"仅出现在 CFG-{tag_a} 中的数据集 — 在 CFG-{tag_b} 中被删除或注释掉")
            + '</div>'
            '</div>'
            '<table class="aggtable">'
            '<thead><tr>'
            '<th>category</th><th>dataset basename</th><th>segs</th>'
            '<th class="right">raw frames</th><th class="right">weighted frames</th>'
            '</tr></thead><tbody>'
            + "".join(rows)
            + '</tbody></table></div>'
        )
    return "".join(out_parts)


def build_category_rollup(cat_rollup: dict, tag_a: str, tag_b: str) -> str:
    rows = []
    cats = sorted(
        cat_rollup.items(),
        key=lambda kv: -abs(kv[1]["wt_b"] - kv[1]["wt_a"]),
    )
    for cat, d in cats:
        delta_wt = d["wt_b"] - d["wt_a"]
        cls = _delta_class(d["wt_a"], d["wt_b"])
        amp_a = d["wt_a"] / max(d["raw_a"], 1)
        amp_b = d["wt_b"] / max(d["raw_b"], 1)
        rows.append(
            '<tr>'
            f'<td><span class="cp-cat-chip cp-cat-chip-sm">{cat}</span></td>'
            f'<td class="mono right">{d["n_eps_a"]:,}</td>'
            f'<td class="mono right">{d["n_eps_b"]:,}</td>'
            f'<td class="mono right">{d["wt_a"]:,}</td>'
            f'<td class="mono right">{d["wt_b"]:,}</td>'
            f'<td class="mono right {cls}"><strong>{_fmt_signed_compact(delta_wt)}</strong> '
            f'<span class="muted-note">({_pct_change(d["wt_a"], d["wt_b"])})</span></td>'
            f'<td class="mono right">{amp_a:.2f}× → {amp_b:.2f}×</td>'
            '</tr>'
        )
    return (
        '<div class="panel">'
        '<div class="panel-header">'
        '<div class="panel-title">Category Summary · 按类别汇总</div>'
        f'<div class="panel-subtitle">按 |Δ weighted| 降序 · 显示每个 category 在 CFG-{tag_a} 与 CFG-{tag_b} 下的累计加权帧</div>'
        '</div>'
        '<table class="aggtable"><thead><tr>'
        '<th>category</th>'
        f'<th class="right">{tag_a} eps</th><th class="right">{tag_b} eps</th>'
        f'<th class="right">{tag_a} weighted</th><th class="right">{tag_b} weighted</th>'
        '<th class="right">Δ weighted</th>'
        f'<th class="right">amp {tag_a} → {tag_b}</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table></div>'
    )


def build_full_entry_table(changed: list, tag_a: str, tag_b: str) -> str:
    if not changed:
        return ""
    # Sort by category, then by |Δ weighted| descending.
    sorted_changed = sorted(
        changed,
        key=lambda r: (r[0], -abs(r[7] - r[6])),
    )
    rows = []
    for cat, group, rel, segs_a, segs_b, raw, wt_a, wt_b in sorted_changed:
        delta_wt = wt_b - wt_a
        cls = _delta_class(wt_a, wt_b)
        rows.append(
            '<tr>'
            f'<td><span class="cp-cat-chip cp-cat-chip-sm">{cat}</span></td>'
            f'<td class="mono"><code>{os.path.basename(rel)}</code></td>'
            f'<td class="mono">{_segs_pretty(segs_a, segs_b)}</td>'
            f'<td class="mono">{_segs_pretty(segs_b, segs_a)}</td>'
            f'<td class="mono right {cls}"><strong>{_fmt_signed_compact(delta_wt)}</strong></td>'
            '</tr>'
        )
    return (
        '<div class="panel">'
        f'<details class="full-diff-details">'
        f'<summary>'
        f'<span class="full-diff-summary-title">逐数据集 Diff 总表 · 共 {len(changed)} 个数据集有改动</span>'
        f'<span class="full-diff-summary-hint">点击展开 · 按 category × |Δ weighted| 排序</span>'
        f'</summary>'
        '<table class="aggtable">'
        '<thead><tr>'
        '<th>category</th><th>dataset basename</th>'
        f'<th>CFG-{tag_a} segs</th><th>CFG-{tag_b} segs</th>'
        '<th class="right">Δ weighted</th>'
        '</tr></thead><tbody>'
        + "".join(rows)
        + '</tbody></table></details></div>'
    )


# ---------- top-level HTML assembly ----------

EXTRA_CSS = """
/* Diff-specific CSS, layered on top of single-cfg viz CSS. */

.diff-tiles {
  display: grid; grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 10px;
}
@media (max-width: 1500px) { .diff-tiles { grid-template-columns: repeat(4, minmax(0, 1fr)); } }

.diff-tile {
  background: linear-gradient(180deg, #fbfcfd 0%, #f1f5f9 100%);
  border: 1px solid var(--border); border-radius: 10px; padding: 12px 14px;
}
.diff-tile-label {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px;
}
.diff-tile-pair {
  display: flex; align-items: baseline; gap: 6px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  font-size: 16px; font-weight: 700; color: #0f172a; line-height: 1.15;
  flex-wrap: wrap;
}
.diff-tile-pair.single { font-size: 22px; }
.dt-a { color: #64748b; }
.dt-arrow { color: var(--subtle); font-size: 13px; padding: 0 2px; }
.dt-b { color: #0f172a; }
.diff-tile-delta {
  font-family: ui-monospace, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
  font-size: 12px; margin-top: 6px; font-weight: 700;
}
.diff-tile-hint {
  font-size: 11px; color: var(--subtle); margin-top: 4px;
}

.delta-up   { color: #15803d; }
.delta-down { color: #b91c1c; }
.delta-zero { color: var(--subtle); }

.badge {
  display: inline-block;
  padding: 1px 8px; border-radius: 999px;
  font-size: 11px; font-weight: 700;
  font-family: ui-monospace, Menlo, Consolas, monospace;
}
.badge-add { background: rgba(34,197,94,0.14); color: #15803d; }
.badge-rm  { background: rgba(239,68,68,0.14); color: #b91c1c; }

/* Grouped bar charts below the hero pies */
.hero-bars {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
  margin-top: 14px; padding-top: 14px;
  border-top: 1px solid var(--border);
}
.bar-chart-block {
  background: linear-gradient(180deg, #fbfcfd 0%, #f1f5f9 100%);
  border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 18px;
}
.bar-chart-title {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin-bottom: 8px;
}

/* Group-level <details> wrapping a set of cat-sections */
.cp-group {
  border: 1px solid var(--border); border-radius: 12px;
  background: #fcfdfe;
  margin: 10px 0;
  overflow: hidden;
}
.cp-group + .cp-group { margin-top: 10px; }
.cp-group-summary {
  list-style: none;
  cursor: pointer;
  display: flex; align-items: baseline; gap: 14px; flex-wrap: wrap;
  padding: 14px 18px;
  background: #f8fafc;
  transition: background 0.15s ease;
}
.cp-group-summary::-webkit-details-marker { display: none; }
.cp-group-summary:hover { background: #f1f5f9; }
.cp-group[open] > .cp-group-summary { border-bottom: 1px solid var(--border); }
.cp-group-subtitle {
  font-size: 11px; color: var(--subtle);
  text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
}
.cp-group-stats {
  margin-left: auto;
  font-size: 12px; color: var(--muted);
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  display: inline-flex; gap: 6px;
  flex-wrap: wrap;
}
.cp-group-stats strong { color: inherit; }
.cp-group-toggle {
  display: inline-block; width: 18px; text-align: center;
  font-size: 11px; color: var(--muted);
  transition: transform 0.18s ease;
}
.cp-group[open] > .cp-group-summary .cp-group-toggle {
  transform: rotate(90deg);
}
.cp-group-body {
  padding: 14px 18px 4px;
}
.group-chip {
  display: inline-block;
  padding: 3px 10px; border-radius: 999px;
  font-size: 13px; font-weight: 700;
  letter-spacing: 0.02em;
}

/* Change-pattern cards */
.cat-section {
  margin: 6px 0 18px;
}
.cat-section-header {
  display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap;
  padding: 6px 4px 10px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 14px;
}
.cat-section-title {
  font-size: 16px; font-weight: 800; color: #0f172a; letter-spacing: -0.01em;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.cat-section-meta {
  margin-left: auto;
  font-size: 11px; color: var(--muted);
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, Menlo, Consolas, monospace;
}

.cp-card {
  background: var(--card); border: 1px solid var(--border); border-radius: 12px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
  margin-bottom: 14px; overflow: hidden;
}
.cp-card-header {
  display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
  padding: 14px 18px; background: #f8fafc; border-bottom: 1px solid var(--border);
}
.cp-card-title { display: inline-flex; align-items: baseline; gap: 10px; }
.cp-cat-chip {
  display: inline-block; padding: 3px 10px; border-radius: 999px;
  background: var(--accent-soft); color: var(--accent);
  font-size: 12px; font-weight: 700;
  font-family: ui-monospace, Menlo, Consolas, monospace;
}
.cp-cat-chip-sm {
  padding: 2px 8px; font-size: 11px;
}
.cp-card-subtitle {
  font-size: 11px; color: var(--subtle);
  text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
}
.cp-card-kpis { display: inline-flex; gap: 16px; align-items: baseline; margin-left: auto; flex-wrap: wrap; }
.cp-kpi { display: inline-flex; gap: 6px; align-items: baseline; }
.cp-kpi-lab {
  font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em;
  font-weight: 600;
}
.cp-kpi-val {
  font-size: 13px; font-weight: 700;
  font-family: ui-monospace, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; color: #0f172a;
}

.cp-card-body { padding: 0; }
.cp-sample-meta {
  font-size: 12px; color: var(--muted); padding: 14px 20px 8px;
}
.cp-sample-meta code {
  background: #f1f5f9; padding: 1px 5px; border-radius: 4px;
  font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11px;
}

.cp-diff-line {
  display: flex; gap: 10px; padding: 4px 20px;
  font-family: ui-monospace, Menlo, Consolas, monospace;
  font-size: 12.5px;
  font-variant-numeric: tabular-nums;
  align-items: baseline;
}
.cp-diff-line:first-of-type { padding-top: 10px; }
.cp-diff-side {
  flex: 0 0 28px; font-weight: 700;
}
.cp-diff-a { color: #64748b; }
.cp-diff-b { color: #0f172a; }
.cp-diff-segs { color: #334155; word-break: break-all; }
.cp-diff-segs strong { color: #b91c1c; background: rgba(254,243,199,0.6); padding: 0 2px; border-radius: 2px; }
.cp-diff-line + .cp-diff-line { border-top: 1px dashed #f1f5f9; }

.cp-tracks {
  display: flex; flex-direction: column; gap: 10px; padding: 8px 0 14px;
}
.cp-track {
  display: flex; flex-direction: column;
}
.cp-track-label {
  font-size: 11px; font-weight: 700; padding: 4px 20px;
  letter-spacing: 0.08em; text-transform: uppercase;
}
.cp-track-label-a { color: #64748b; }
.cp-track-label-b { color: var(--accent); }

.cp-rels {
  padding: 8px 20px 14px; font-size: 12px; color: var(--muted);
}
.cp-rels summary {
  cursor: pointer; color: var(--accent); font-weight: 600;
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;
}
.cp-rels-list {
  margin: 8px 0 0; padding-left: 18px; max-height: 220px; overflow-y: auto;
}
.cp-rels-list code {
  font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 11.5px;
  background: #f8fafc; padding: 1px 4px; border-radius: 3px;
}

/* added / removed rows */
.added-row td { background: rgba(34,197,94,0.06); }
.removed-row td { background: rgba(239,68,68,0.06); }

/* full-diff details */
.full-diff-details summary {
  cursor: pointer; padding: 14px 20px;
  display: flex; align-items: baseline; gap: 14px;
  flex-wrap: wrap;
}
.full-diff-summary-title {
  font-size: 14px; font-weight: 700; color: #0f172a;
  text-transform: uppercase; letter-spacing: 0.04em;
}
.full-diff-summary-hint {
  font-size: 12px; color: var(--muted);
}

/* Header arrow */
.page-header h1 .arrow {
  color: var(--accent); margin: 0 8px; font-weight: 600;
}
.page-header .tagline .arrow {
  color: var(--accent); margin: 0 6px; font-weight: 600;
}
"""


def build_diff_html(cfg_a_path: str, cfg_b_path: str, data: dict, tag_a: str, tag_b: str) -> str:
    # First level: 大类组 (from mcv.CATEGORY_GROUPS). Second level: category.
    # Each group is a <details> block; categories inside flow as section blocks.
    patterns_by_cat: dict[str, list] = {}
    for cp in data["patterns"]:
        patterns_by_cat.setdefault(cp.cat, []).append(cp)

    # Decide which group to open by default — the one with the largest |Δ weighted|.
    group_total_abs_dwt: dict[str, int] = {}
    group_to_cats: dict[str, list[str]] = {}
    for cat, cps in patterns_by_cat.items():
        gname = cps[0].group
        group_to_cats.setdefault(gname, []).append(cat)
        group_total_abs_dwt[gname] = group_total_abs_dwt.get(gname, 0) + sum(
            abs(p.weighted_b - p.weighted_a) for p in cps
        )
    default_open_group = (
        max(group_total_abs_dwt.items(), key=lambda kv: kv[1])[0]
        if group_total_abs_dwt else None
    )

    group_blocks_html = []
    # Iterate in the canonical CATEGORY_GROUPS order so the page reads consistently.
    for (gname, gsub, gfg, gbg) in mcv.CATEGORY_GROUPS:
        cats_in_group = group_to_cats.get(gname, [])
        if not cats_in_group:
            continue

        # Sort cats within group by |Δ weighted| descending.
        cats_sorted = sorted(
            cats_in_group,
            key=lambda c: -sum(abs(p.weighted_b - p.weighted_a) for p in patterns_by_cat[c]),
        )

        group_total_dwt = sum(
            (p.weighted_b - p.weighted_a)
            for c in cats_in_group for p in patterns_by_cat[c]
        )
        group_eps = sum(p.n_eps for c in cats_in_group for p in patterns_by_cat[c])
        group_n_patterns = sum(len(patterns_by_cat[c]) for c in cats_in_group)
        gcls = _delta_class(0, group_total_dwt) if group_total_dwt != 0 else "delta-zero"

        cat_sections_in_group = []
        for cat in cats_sorted:
            cps = patterns_by_cat[cat]
            total_dwt = sum(p.weighted_b - p.weighted_a for p in cps)
            total_eps = sum(p.n_eps for p in cps)
            cls = _delta_class(0, total_dwt) if total_dwt != 0 else "delta-zero"
            sec_html = (
                '<div class="cat-section">'
                '<div class="cat-section-header">'
                f'<span class="cp-cat-chip">{cat}</span>'
                '<span class="cat-section-meta">'
                f'<span>{len(cps)} pattern{"s" if len(cps) != 1 else ""}</span>'
                f'<span> · {total_eps:,} eps</span>'
                f'<span> · Δ weighted: <span class="{cls}"><strong>{_fmt_signed_compact(total_dwt)}</strong></span></span>'
                '</span>'
                '</div>'
                + "".join(build_change_pattern_card(cp, tag_a, tag_b) for cp in cps)
                + '</div>'
            )
            cat_sections_in_group.append(sec_html)

        is_open = (gname == default_open_group)
        open_attr = " open" if is_open else ""
        group_blocks_html.append(
            f'<details class="cp-group"{open_attr}>'
            '<summary class="cp-group-summary">'
            f'<span class="group-chip" style="background:{gbg}; color:{gfg}">{gname}</span>'
            f'<span class="cp-group-subtitle">{gsub}</span>'
            '<span class="cp-group-stats">'
            f'<span>{len(cats_in_group)} 子类</span>'
            f'<span> · {group_n_patterns} pattern{"s" if group_n_patterns != 1 else ""}</span>'
            f'<span> · {group_eps:,} eps</span>'
            f'<span> · Δ weighted: <span class="{gcls}"><strong>{_fmt_signed_compact(group_total_dwt)}</strong></span></span>'
            '</span>'
            '<span class="cp-group-toggle">▶</span>'
            '</summary>'
            '<div class="cp-group-body">'
            + "".join(cat_sections_in_group)
            + '</div></details>'
        )

    hero_html = build_hero(
        data["kpis"],
        data["grp_raw_a"], data["grp_raw_b"],
        data["grp_wt_a"], data["grp_wt_b"],
        n_changed=len(data["changed"]),
        n_added=len(data["added"]),
        n_removed=len(data["removed"]),
        tag_a=tag_a, tag_b=tag_b,
    )

    added_removed_html = build_added_removed_panels(data["added"], data["removed"], tag_a, tag_b)
    cat_rollup_html = build_category_rollup(data["cat_rollup"], tag_a, tag_b)
    full_table_html = build_full_entry_table(data["changed"], tag_a, tag_b)

    main_css = _BASE_CSS  # see below
    css = main_css + EXTRA_CSS

    cfg_a_name = os.path.basename(cfg_a_path)
    cfg_b_name = os.path.basename(cfg_b_path)

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>cfg diff · {tag_a} → {tag_b}</title>
<style>{css}</style>
</head><body>
<div class="container">

  <div class="page-header">
    <div class="eyebrow">Fold Box · Config Diff · Weight Budget Rebalance</div>
    <div class="accent-line"></div>
    <h1>数据分布对比</h1>
    <div class="tagline">
      <span class="pill">{cfg_a_name}</span>
      <span class="arrow">→</span>
      <span class="pill">{cfg_b_name}</span>
    </div>
  </div>

  {hero_html}

  <div class="panel" style="padding: 14px 20px;">
    <div class="panel-header" style="border-bottom: none; padding: 0 0 6px;">
      <div class="panel-title">Change Patterns · 改动模式</div>
      <div class="panel-subtitle">按大类组折叠 · 点击组标题展开 · 组内按 (category × {tag_a} 段 × {tag_b} 段) 聚合 · CFG-{tag_a} 在上、CFG-{tag_b} 在下</div>
    </div>
    {"".join(group_blocks_html)}
  </div>

  {added_removed_html}

  {cat_rollup_html}

  {full_table_html}

</div>
</body></html>
"""


# ---------- copied base CSS (kept in sync with make_cfg_viz.py palette) ----------

_BASE_CSS = """
:root {
  --bg: #f1f5f9;
  --card: #ffffff;
  --border: #e2e8f0;
  --text: #0f172a;
  --muted: #64748b;
  --subtle: #94a3b8;
  --accent: #0ea5e9;
  --accent-soft: #e0f2fe;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  padding: 28px 20px;
  background: var(--bg); color: var(--text);
  line-height: 1.45;
  font-feature-settings: "tnum" 1, "lnum" 1;
}
.container { max-width: 1800px; margin: 0 auto; }

/* Page header */
.page-header { text-align: center; margin: 12px 0 30px; padding: 6px 0; }
.page-header .eyebrow {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.18em;
  margin-bottom: 10px;
}
.page-header .accent-line {
  width: 38px; height: 2px; background: var(--accent);
  margin: 0 auto 14px; border-radius: 1px;
}
.page-header h1 {
  margin: 0 0 10px; font-size: 32px; font-weight: 800;
  letter-spacing: -0.02em; line-height: 1.15; color: var(--text);
}
.page-header .tagline {
  font-size: 13px; color: var(--muted); line-height: 1.6;
  max-width: 1100px; margin: 0 auto;
}
.page-header .tagline .pill {
  display: inline-block; background: #f1f5f9; color: #334155;
  padding: 2px 9px; margin: 0 2px; border-radius: 6px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 12px;
  word-break: break-all;
}

/* Hero */
.hero {
  background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  padding: 18px 20px; margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04);
}
.hero-pies {
  display: grid; grid-template-columns: 1fr 1fr; gap: 14px;
  margin-top: 16px; padding-top: 16px;
  border-top: 1px solid var(--border);
}
.pie-block {
  background: linear-gradient(180deg, #fbfcfd 0%, #f1f5f9 100%);
  border: 1px solid var(--border); border-radius: 10px;
  padding: 14px 18px;
}
.pie-title {
  font-size: 11px; font-weight: 600; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.06em;
  margin-bottom: 12px;
}
.pie-body { display: flex; align-items: center; gap: 18px; }
.pie-svg { flex: 0 0 170px; }
.pie-slice { transition: transform 0.22s ease; }
.pie-legend { flex: 1; display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
.pie-legend-item {
  display: grid; grid-template-columns: 12px 1fr auto auto;
  gap: 8px; align-items: center; padding: 3px 0;
}
.pie-dot { width: 10px; height: 10px; border-radius: 50%; }
.pie-legend-name { color: #334155; font-weight: 600; }
.pie-legend-pct {
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  color: var(--muted);
}
.pie-legend-val {
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  color: #0f172a; font-weight: 700;
  min-width: 54px; text-align: right;
}

/* Panel */
.panel {
  background: var(--card); border: 1px solid var(--border); border-radius: 14px;
  padding: 16px 0 16px; margin-bottom: 16px;
  box-shadow: 0 1px 3px rgba(15,23,42,0.04); overflow: hidden;
}
.panel-header { padding: 0 20px 14px; border-bottom: 1px solid var(--border); margin-bottom: 14px; }
.panel-title { font-size: 14px; font-weight: 700; color: #0f172a; margin: 0 0 2px;
  text-transform: uppercase; letter-spacing: 0.04em; }
.panel-subtitle { font-size: 12px; color: var(--muted); }
.panel-subtitle code { background: #f1f5f9; padding: 1px 5px; border-radius: 4px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 11px; word-break: break-all; }

/* Filmstrip */
.stripwrap { display: flex; padding-left: 3.3333%; padding-right: 0; }
.thumb { flex: 1 1 0; min-width: 0; padding: 0 2px; }
.thumb-img-wrap { position: relative; }
.thumb img {
  width: 100%; height: 180px; object-fit: cover; display: block;
  border: 1px solid var(--border); border-bottom: 0; border-radius: 5px 5px 0 0;
}
.thumb-time-badge {
  position: absolute; top: 5px; left: 5px;
  background: rgba(15,23,42,0.78); color: #fff;
  font-size: 10px; padding: 2px 6px; border-radius: 4px;
  font-variant-numeric: tabular-nums;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.thumb-wband {
  height: 22px; display: flex; align-items: center; justify-content: center;
  color: #fff; font-size: 11px; font-weight: 700;
  border-radius: 0 0 5px 5px;
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  text-shadow: 0 0 2px rgba(0,0,0,0.25);
}

/* Table */
.aggtable {
  width: calc(100% - 40px); margin: 4px 20px 0;
  border-collapse: collapse; font-size: 13px;
}
.aggtable th, .aggtable td { padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }
.aggtable th {
  text-align: left; color: var(--muted); font-weight: 600;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
  background: #f8fafc;
}
.aggtable th.right, .aggtable td.right { text-align: right; }
.aggtable .mono {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}
.aggtable .strong { font-weight: 700; color: #0f172a; }
.aggtable .muted-note { color: var(--subtle); font-size: 11px; margin-left: 4px; }

.empty { padding: 22px 20px; color: var(--subtle); font-style: italic; text-align: center; }
"""


# ---------- main ----------

def _cfg_tag(cfg_path: str) -> str:
    """Reuse the same regex as make_cfg_viz for MMDD."""
    return mcv._cfg_tag(cfg_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two cfg_pi05_base_finetune_box_recap_pt_*.py files "
                    "and produce config_viz/diff_<tagA>_vs_<tagB>.html.",
    )
    parser.add_argument("--cfg-a", default=DEFAULT_CFG_A,
                        help="path to the OLD cfg (left side)")
    parser.add_argument("--cfg-b", default=DEFAULT_CFG_B,
                        help="path to the NEW cfg (right side)")
    parser.add_argument("--out", default=None,
                        help="cache dir override (default: .viz_cache/diff_<tagA>_vs_<tagB>/)")
    args = parser.parse_args()

    cfg_a = os.path.abspath(args.cfg_a)
    cfg_b = os.path.abspath(args.cfg_b)
    for p in (cfg_a, cfg_b):
        if not os.path.exists(p):
            raise SystemExit(f"cfg not found: {p}")

    tag_a = _cfg_tag(cfg_a)
    tag_b = _cfg_tag(cfg_b)
    diff_tag = f"diff_{tag_a}_vs_{tag_b}"

    out_dir = os.path.abspath(args.out) if args.out else os.path.join(mcv.CACHE_ROOT, diff_tag)
    os.makedirs(out_dir, exist_ok=True)
    # mcv.extract_thumbs_for reads from mcv.OUT_DIR — point it at our diff cache.
    mcv.OUT_DIR = out_dir

    print(f"cfg-a: {cfg_a}  (tag {tag_a})")
    print(f"cfg-b: {cfg_b}  (tag {tag_b})")
    print(f"cache: {out_dir}")
    print("aggregating diff...")
    t0 = time.time()
    data = aggregate_diff(cfg_a, cfg_b, out_dir)
    print(f"  diff aggregated in {time.time() - t0:.1f}s")
    print(f"  change patterns: {len(data['patterns'])}")

    print("rendering HTML...")
    html = build_diff_html(cfg_a, cfg_b, data, tag_a, tag_b)
    out_path = os.path.join(out_dir, "diff.html")
    with open(out_path, "w") as f:
        f.write(html)
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"saved -> {out_path}  ({size_kb:.1f} KB)")

    os.makedirs(mcv.PUBLISH_DIR, exist_ok=True)
    publish_path = os.path.join(mcv.PUBLISH_DIR, f"{diff_tag}.html")
    shutil.copyfile(out_path, publish_path)
    print(f"published -> {publish_path}")
    print(f"open with:  file://{publish_path}")


if __name__ == "__main__":
    main()
