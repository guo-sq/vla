#!/usr/bin/env python3
"""Compare per-frame model intended action (from unified_record.log) against
the parquet-recorded executed action and the robot state at that frame.

Built for investigating cases where the robot executed something the model did
not ask for — e.g. the 2026-06-02 episode where the arm violently jumped at
start and almost flipped the table.

Alignment math (per record_unified.py run_record_loop):
    At every step N the loop does:
        action = buffer.get_next_action()   # pops buffer[0]
        remain = buffer.get_subsequent_chunk()  # buffer[0:available] AFTER pop
        log("Got action ... step={N} ... action_chunk: {remain}")
        robot.send_action(action)
    Therefore log[step=N].chunk[joint][0] equals the value that WOULD have been
    popped at step N+1 had no new inference fused in between. So:
        intended_action[frame M][joint] = log[step M-1].chunk[joint][0]
    For frame 0 there is no preceding log line — value left NaN.

Outputs (under --out-dir, defaults to --episode-dir):
    inference_vs_executed.csv          full episode, per-joint cols
    inference_vs_executed_first_5s.png 14 arm joints over first --plot-frames frames
plus stdout summary (max diff per inference event + top-3 episode-wide).
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


GOT_ACTION_RE = re.compile(
    r"^Got action from buffer at step (\d+), infer_delay: (\d+), action_chunk: (\{.*\})$"
)
RUN_INF_RE = re.compile(r"^inference_worker: Running inference at step_id=(\d+)")


def parse_log(log_path: Path):
    chunks: dict[int, tuple[int, dict[str, list[float]]]] = {}
    inference_events: list[int] = []
    skipped = 0
    with log_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = GOT_ACTION_RE.match(line)
            if m:
                step = int(m.group(1))
                infer_delay = int(m.group(2))
                try:
                    chunk = ast.literal_eval(m.group(3))
                except (SyntaxError, ValueError) as e:
                    skipped += 1
                    print(f"WARN: chunk parse failed at step {step}: {e}", file=sys.stderr)
                    continue
                chunks[step] = (infer_delay, chunk)
                continue
            m = RUN_INF_RE.match(line)
            if m:
                inference_events.append(int(m.group(1)))
    if skipped:
        print(f"WARN: skipped {skipped} malformed chunk lines", file=sys.stderr)
    return chunks, inference_events


def load_parquet(episode_dir: Path):
    info = json.loads((episode_dir / "meta" / "info.json").read_text())
    joint_order = info["features"]["action"]["names"]
    state_order = info["features"]["observation.state"]["names"]
    if joint_order != state_order:
        raise SystemExit("action.names and observation.state.names disagree — abort")
    pq_files = sorted((episode_dir / "data").rglob("*.parquet"))
    if len(pq_files) != 1:
        raise SystemExit(f"expected exactly 1 parquet, found {pq_files}")
    df = pq.read_table(pq_files[0]).to_pandas()
    return df, joint_order, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--episode-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: same as --episode-dir")
    ap.add_argument("--plot-name", type=str, default="inference_vs_executed.png",
                    help="full-episode PNG filename (default: inference_vs_executed.png)")
    ap.add_argument("--zoom-plot-name", type=str, default="inference_vs_executed_first_2s.png",
                    help="zoom PNG filename (default: inference_vs_executed_first_2s.png)")
    ap.add_argument("--zoom-seconds", type=float, default=2.0,
                    help="zoom window length in seconds (default: 2.0)")
    ap.add_argument("--transition-steps", type=int, default=15,
                    help="record_unified.py transition_steps; shaded as danger window (default 15)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.episode_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing log: {args.log}")
    chunks, inference_events = parse_log(args.log)
    print(f"  parsed {len(chunks)} chunk entries, {len(inference_events)} inference fires")
    if not chunks:
        raise SystemExit("no chunks parsed — log path or format wrong")

    print(f"Loading parquet: {args.episode_dir}")
    df, joint_order, info = load_parquet(args.episode_dir)
    total_frames = len(df)
    print(f"  {total_frames} frames, parquet joints = {len(joint_order)}")
    if total_frames != info["total_frames"]:
        raise SystemExit("parquet row count != info.json total_frames")

    sample_chunk = next(iter(chunks.values()))[1]
    model_joints = list(sample_chunk.keys())
    missing = [j for j in model_joints if j not in joint_order]
    if missing:
        raise SystemExit(f"chunk has joints absent from parquet schema: {missing}")
    joint_to_pq_idx = {j: i for i, j in enumerate(joint_order)}
    horizon = len(sample_chunk[model_joints[0]])
    print(f"  model joints ({len(model_joints)}): {model_joints}")
    print(f"  chunk horizon (after-pop): {horizon}")

    # Build intended-action arrays of shape (T, n_model_joints)
    intended = np.full((total_frames, len(model_joints)), np.nan, dtype=np.float64)
    source_step = np.full(total_frames, -1, dtype=np.int64)
    for M in range(total_frames):
        prev = M - 1
        entry = chunks.get(prev)
        if entry is None:
            continue
        _, chunk = entry
        for j_idx, joint in enumerate(model_joints):
            vals = chunk.get(joint)
            if vals:
                intended[M, j_idx] = vals[0]
        source_step[M] = prev

    exec_arr = np.stack([np.asarray(a, dtype=np.float64) for a in df["action"].to_numpy()])
    state_arr = np.stack([np.asarray(a, dtype=np.float64) for a in df["observation.state"].to_numpy()])

    # CSV
    rows = {
        "frame_index": df["frame_index"].to_numpy(),
        "timestamp": df["timestamp"].to_numpy(),
        "is_human_intervention": df["is_human_intervention"].to_numpy(),
        "source_log_step": source_step,
    }
    for j_idx, joint in enumerate(model_joints):
        pq_idx = joint_to_pq_idx[joint]
        rows[f"executed_{joint}"] = exec_arr[:, pq_idx]
        rows[f"model_{joint}"] = intended[:, j_idx]
        rows[f"state_{joint}"] = state_arr[:, pq_idx]
        rows[f"diff_{joint}"] = exec_arr[:, pq_idx] - intended[:, j_idx]
    csv_df = pd.DataFrame(rows)
    out_csv = out_dir / "inference_vs_executed.csv"
    csv_df.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"Wrote {out_csv} ({len(csv_df)} rows x {len(csv_df.columns)} cols)")

    # Per-inference-event summary
    print()
    print("=== Per inference event: worst |executed - model| within next 20 frames ===")
    for ev_step in inference_events:
        lo = ev_step + 1
        hi = min(ev_step + 20, total_frames - 1)
        if lo > hi:
            continue
        worst = (0.0, None, None)
        for j_idx, joint in enumerate(model_joints):
            pq_idx = joint_to_pq_idx[joint]
            diff = exec_arr[lo:hi+1, pq_idx] - intended[lo:hi+1, j_idx]
            if np.all(np.isnan(diff)):
                continue
            absd = np.nanargmax(np.abs(diff))
            v = diff[absd]
            if abs(v) > worst[0]:
                worst = (abs(v), joint, (lo + absd, v))
        if worst[1] is not None:
            joint = worst[1]
            frame, signed = worst[2]
            print(f"  inference@step={ev_step:4d}  worst: {joint:30s} diff={signed:+.4f} @frame {frame}")

    # Top-3 episode-wide
    print()
    print("=== Top-3 |executed - model| across episode ===")
    all_diffs = []
    for j_idx, joint in enumerate(model_joints):
        pq_idx = joint_to_pq_idx[joint]
        diff = exec_arr[:, pq_idx] - intended[:, j_idx]
        mask = ~np.isnan(diff)
        for fi in np.where(mask)[0]:
            all_diffs.append((abs(diff[fi]), int(fi), joint,
                              exec_arr[fi, pq_idx], intended[fi, j_idx], state_arr[fi, pq_idx]))
    all_diffs.sort(reverse=True)
    for absd, fi, joint, ex, mo, st in all_diffs[:3]:
        print(f"  frame={fi:4d}  {joint:30s}  executed={ex:+.4f}  model={mo:+.4f}  state={st:+.4f}  |diff|={absd:.4f}")

    # Plot: 2x7 grid — left arm row 0, right arm row 1.
    if not HAVE_MPL:
        print()
        print("matplotlib not available — skipping plot. CSV is the source of truth.")
        return

    fps = info["fps"]
    t = np.arange(total_frames) / fps
    transition_t = args.transition_steps / fps  # default 0.5s

    def joint_key(side: str, idx: int) -> str:
        return f"{side}_arm_joint_{idx}.pos"

    def make_plot(t_start: float, t_end: float, out_png: Path, scope_label: str):
        rows_spec = [("left", 0), ("right", 1)]
        ncols = 7
        fig, axes = plt.subplots(2, ncols, figsize=(28, 7), squeeze=False, sharex=True)

        i_start = max(int(np.floor(t_start * fps)), 0)
        i_end = min(int(np.ceil(t_end * fps)), total_frames)
        t_view = t[i_start:i_end]

        handles_for_legend = None
        for side, row in rows_spec:
            for col in range(ncols):
                joint = joint_key(side, col + 1)
                ax = axes[row][col]
                if joint not in joint_to_pq_idx or joint not in model_joints:
                    ax.set_title(f"{joint} (missing)", fontsize=9)
                    ax.axis("off")
                    continue
                pq_idx = joint_to_pq_idx[joint]
                j_idx = model_joints.index(joint)
                # Shade only the visible portion of the transition window
                if t_start < transition_t:
                    ax.axvspan(max(0, t_start), min(transition_t, t_end),
                               color="red", alpha=0.12)
                h_state, = ax.plot(t_view, state_arr[i_start:i_end, pq_idx],
                                   color="#444", lw=1.6, label="state")
                h_exec, = ax.plot(t_view, exec_arr[i_start:i_end, pq_idx],
                                  color="tab:blue", lw=1.2, label="executed (parquet)")
                h_model, = ax.plot(t_view, intended[i_start:i_end, j_idx],
                                   color="tab:orange", lw=1.0, linestyle="--",
                                   label="model intended")
                ax.set_xlim(t_start, t_end)
                ax.set_title(joint, fontsize=10)
                ax.grid(alpha=0.3)
                if row == 1:
                    ax.set_xlabel("time (s)", fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{side} arm  (rad)", fontsize=10)
                if handles_for_legend is None and ax.patches:
                    handles_for_legend = [h_state, h_exec, h_model, ax.patches[0]]
                elif handles_for_legend is None:
                    handles_for_legend = [h_state, h_exec, h_model]

        if handles_for_legend is not None:
            labels = ["state (joint position)", "executed (parquet action)",
                      "model intended (from log)"]
            if len(handles_for_legend) == 4:
                labels.append(f"transition window [0, {transition_t:.2f}s]")
            fig.legend(handles_for_legend, labels,
                       loc="upper center", ncol=4,
                       bbox_to_anchor=(0.5, 0.985), fontsize=10)

        fig.suptitle(
            f"Inference vs Executed — {scope_label} ({total_frames} frames @ {fps}fps)\n"
            f"Top row: left arm joints 1–7   |   Bottom row: right arm joints 1–7",
            fontsize=11, y=1.04,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(out_png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {out_png}")

    print()
    full_t = total_frames / fps
    make_plot(0.0, full_t,
              out_dir / args.plot_name,
              f"full episode ({full_t:.1f}s)")
    make_plot(0.0, args.zoom_seconds,
              out_dir / args.zoom_plot_name,
              f"first {args.zoom_seconds:g}s zoom")


if __name__ == "__main__":
    main()
