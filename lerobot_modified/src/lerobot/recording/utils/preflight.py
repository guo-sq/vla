"""Preflight checks for recording sessions.

Two operator-facing checks run before a recording job starts:

  1. ``cameras [<config_path>]`` — capture one frame per camera declared
     under ``hardware_meta.cameras`` and show them in a labeled grid so
     the operator can visually confirm each named slot maps to the
     correct physical camera. With no path, falls back to the legacy
     "list all detected OpenCV cameras" behavior.
  2. ``session <session_config_path>`` — load + validate the operator's
     session config and show the parsed task description, role prompts
     (template-substituted), and operator metadata for confirmation.
  3. ``collection_info <path> [<task_spec_path>]`` — legacy entry point
     for the pre-session-config workflow. Same UI as ``session``.

The first positional argument for ``cameras``/``session`` may be either
a session config or a legacy collection_info file — both expose the
same ``hardware_meta.cameras`` / ``task_meta.*`` shape, so the loader
auto-detects.

UI is a Tk popup with Confirm / Cancel buttons. When ``$DISPLAY`` is unset
(headless SSH session, no X forwarding), each check degrades to a terminal
fallback so the same scripts work locally and over SSH:

  - cameras (with path):  frames are written to /tmp/lerobot_preflight_cameras/
                          and their paths are printed for the operator to
                          inspect (e.g. via scp + image viewer)
  - cameras (no path):    text list of detected cameras
  - session:              pretty-printed text + [y/N] prompt

Exit code 0 = confirmed, 1 = rejected or validation failed.
"""

from __future__ import annotations

import io
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from lerobot.recording.task.collection_info import CollectionInfo, CollectionInfoError
from lerobot.recording.task.session_config import SessionConfig, SessionConfigError


# ---------------------------------------------------------------------------
# Confirmation UI
# ---------------------------------------------------------------------------

def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


# Fonts in preference order. First one whose family is installed wins.
# The leading entries are CJK-capable so Chinese task descriptions / object
# names render instead of showing tofu boxes. Last entry is a guaranteed
# fallback (TkDefaultFont always exists).
_CJK_FONT_CANDIDATES = (
    "Noto Sans CJK SC",
    "Noto Sans SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "WenQuanYi Micro Hei",
    "Microsoft YaHei",
    "PingFang SC",
    "Hiragino Sans GB",
    "DejaVu Sans",
    "TkDefaultFont",
)


def _pick_ui_font_family() -> str:
    try:
        from tkinter import font as tkfont
    except Exception:
        return "TkDefaultFont"
    try:
        available = set(tkfont.families())
    except Exception:
        return "TkDefaultFont"
    for fam in _CJK_FONT_CANDIDATES:
        if fam in available:
            return fam
    return "TkDefaultFont"


# Inline markers for "highlight this run of text" in the rendered body.
# The ASCII control chars STX/ETX (0x02/0x03) are vanishingly unlikely to
# appear in real config content, so we don't need a fancier escape scheme.
_HL_OPEN = "\x02"
_HL_CLOSE = "\x03"


def _h(text: object) -> str:
    """Wrap ``text`` so it gets highlighted in the confirmation popup."""
    return f"{_HL_OPEN}{text}{_HL_CLOSE}"


def _setup_root_dpi(root) -> None:
    """Scale Tk widgets to the actual display DPI.

    Tk's nominal DPI is 72; on a HiDPI display ``winfo_fpixels('1i')`` returns
    something like 144 or 192, but Tk itself stays at 1.0 unless we override
    it, which is why text looks tiny/blurry on those screens. Only scale up,
    never down, so a regular 96 DPI monitor isn't affected.
    """
    try:
        ratio = root.winfo_fpixels("1i") / 72.0
    except Exception:
        return
    if ratio > 1.1:
        try:
            root.tk.call("tk", "scaling", ratio)
        except Exception:
            pass


def _terminal_confirm(title: str, body: str) -> bool:
    # ANSI bold + yellow for highlighted runs when stdout is a TTY; otherwise
    # strip the markers so log files / pipes stay clean.
    if sys.stdout.isatty():
        rendered = body.replace(_HL_OPEN, "\x1b[1;33m").replace(_HL_CLOSE, "\x1b[0m")
    else:
        rendered = body.replace(_HL_OPEN, "").replace(_HL_CLOSE, "")
    print(f"\n{'='*70}\n{title}\n{'='*70}")
    print(rendered)
    print("=" * 70)
    if not sys.stdin.isatty():
        # Non-interactive shells (CI, piped) auto-confirm so the recording
        # pipeline doesn't deadlock; the operator-facing scripts always run
        # in a TTY.
        print("(non-TTY — auto-confirm)")
        return True
    try:
        ans = input("Confirm? [y/N]: ").strip().lower()
    except EOFError:
        return False
    return ans == "y"


def _insert_with_highlights(txt_widget, body: str, *, base_font) -> None:
    """Insert ``body`` into a Tk Text widget, applying a 'highlight' tag to
    anything wrapped in _HL_OPEN/_HL_CLOSE. The tag is configured for bold +
    a soft yellow background so the operator's eyes land on robot id, operator
    name, task description, prompt, etc."""
    family = base_font[0] if isinstance(base_font, tuple) else "TkDefaultFont"
    size = base_font[1] if isinstance(base_font, tuple) and len(base_font) > 1 else 12
    txt_widget.tag_configure(
        "highlight",
        background="#FFF59D",  # light amber
        foreground="#000000",
        font=(family, size, "bold"),
    )
    pos = 0
    while True:
        start = body.find(_HL_OPEN, pos)
        if start < 0:
            txt_widget.insert("end", body[pos:])
            break
        end = body.find(_HL_CLOSE, start + 1)
        if end < 0:
            # Unmatched open marker — render the rest as plain text so we
            # never silently drop content.
            txt_widget.insert("end", body[pos:].replace(_HL_OPEN, ""))
            break
        if start > pos:
            txt_widget.insert("end", body[pos:start])
        txt_widget.insert("end", body[start + 1:end], "highlight")
        pos = end + 1


def _tk_confirm(title: str, body: str) -> bool:
    try:
        import tkinter as tk
        from tkinter import scrolledtext
    except Exception:
        return _terminal_confirm(title, body)

    result = {"ok": False}
    root = tk.Tk()
    root.title(title)
    _setup_root_dpi(root)

    family = _pick_ui_font_family()
    body_font = (family, 12)
    button_font = (family, 12)

    # Resizable so a long task description with many objects can be expanded.
    root.minsize(720, 520)
    root.geometry("960x680")

    txt = scrolledtext.ScrolledText(root, width=96, height=30, wrap=tk.WORD, font=body_font)
    _insert_with_highlights(txt, body, base_font=body_font)
    txt.configure(state="disabled")
    txt.pack(padx=12, pady=12, fill="both", expand=True)

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=(0, 12))

    def _ok():
        result["ok"] = True
        root.destroy()

    def _cancel():
        result["ok"] = False
        root.destroy()

    tk.Button(
        btn_frame, text="Confirm", width=14, command=_ok, font=button_font
    ).pack(side="left", padx=8)
    tk.Button(
        btn_frame, text="Cancel", width=14, command=_cancel, font=button_font
    ).pack(side="left", padx=8)

    root.bind("<Return>", lambda _e: _ok())
    root.bind("<Escape>", lambda _e: _cancel())
    root.mainloop()
    return result["ok"]


def confirm(title: str, body: str) -> bool:
    """Show a confirmation dialog. Tk if DISPLAY available, else terminal."""
    if _has_display():
        return _tk_confirm(title, body)
    return _terminal_confirm(title, body)


# ---------------------------------------------------------------------------
# Camera capture
# ---------------------------------------------------------------------------

# Number of warm-up frames to discard before keeping the preview frame.
# Many USB webcams need a few frames before exposure stabilizes.
_CAMERA_WARMUP_FRAMES = 5
_CAMERA_OPEN_TIMEOUT_S = 5.0


def _capture_camera_frame(spec: dict) -> tuple[Any | None, str | None]:
    """Open a camera once, return ``(frame, error)``. Frame is BGR (OpenCV).

    On failure returns ``(None, message)``. The caller is expected to render
    a placeholder for the failed slot so the operator notices.
    """
    try:
        import cv2
    except ImportError as e:
        return None, f"cv2 unavailable: {e}"

    idx = spec.get("index_or_path")
    width = spec.get("width")
    height = spec.get("height")
    if idx is None:
        return None, "index_or_path missing"

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        return None, f"cv2.VideoCapture({idx!r}) could not open"
    try:
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        # Warm up — discard a few frames.
        frame = None
        for _ in range(_CAMERA_WARMUP_FRAMES):
            ok, frame = cap.read()
            if not ok:
                frame = None
        if frame is None:
            return None, f"camera {idx!r} opened but read() never returned a frame"
        return frame, None
    finally:
        cap.release()


def _capture_all_cameras(cameras: dict) -> list[tuple[str, Any | None, str | None]]:
    """Capture one frame per declared camera in dict-iteration order."""
    out = []
    for name, spec in cameras.items():
        # Underscore-prefixed entries are inline annotations and not cameras.
        if isinstance(name, str) and name.startswith("_"):
            continue
        frame, err = _capture_camera_frame(spec)
        out.append((name, frame, err))
    return out


# ---------------------------------------------------------------------------
# Camera grid display
# ---------------------------------------------------------------------------

_THUMB_W, _THUMB_H = 320, 240


def _to_pil_thumbnail(frame_bgr) -> Any | None:
    """Convert BGR frame to a Pillow Image scaled to the thumbnail size."""
    try:
        import cv2
        from PIL import Image
    except ImportError:
        return None
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img.thumbnail((_THUMB_W, _THUMB_H))
    return img


def _tk_camera_grid(captures: list[tuple[str, Any | None, str | None]]) -> bool:
    """Show captured frames in a labeled grid; ask Confirm / Cancel.

    Each tile shows the camera *name* (head / left_wrist / ...) above the
    captured frame so the operator can visually verify "this image at the
    'head' slot is what I expect for the head camera."
    """
    try:
        import tkinter as tk
        from PIL import ImageTk
    except Exception as e:
        return _terminal_camera_grid_fallback(captures, fallback_reason=str(e))

    result = {"ok": False}
    root = tk.Tk()
    root.title("Confirm cameras")
    _setup_root_dpi(root)

    family = _pick_ui_font_family()
    label_font = (family, 12, "bold")
    info_font = (family, 11)
    button_font = (family, 12)

    # Lay out in a roughly-square grid: ceil(sqrt(N)) columns.
    import math
    n = max(len(captures), 1)
    cols = max(1, int(math.ceil(math.sqrt(n))))

    # Hold references to PhotoImages so they aren't garbage-collected.
    photos: list[Any] = []
    grid = tk.Frame(root, padx=10, pady=10)
    grid.pack()

    for i, (name, frame, err) in enumerate(captures):
        cell = tk.Frame(grid, padx=6, pady=6, borderwidth=1, relief="solid")
        cell.grid(row=i // cols, column=i % cols, padx=4, pady=4)
        tk.Label(cell, text=name, font=label_font).pack()
        if err is not None or frame is None:
            tk.Label(
                cell, text=f"FAILED\n{err or 'no frame'}",
                width=40, height=10, fg="red", justify="center", font=info_font,
            ).pack()
        else:
            img = _to_pil_thumbnail(frame)
            photo = ImageTk.PhotoImage(img)
            photos.append(photo)
            tk.Label(cell, image=photo).pack()
            spec_w = frame.shape[1]
            spec_h = frame.shape[0]
            tk.Label(cell, text=f"{spec_w}x{spec_h}", font=info_font).pack()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=(0, 10))

    def _ok():
        result["ok"] = True
        root.destroy()

    def _cancel():
        result["ok"] = False
        root.destroy()

    tk.Button(
        btn_frame, text="Confirm", width=14, command=_ok, font=button_font
    ).pack(side="left", padx=8)
    tk.Button(
        btn_frame, text="Cancel", width=14, command=_cancel, font=button_font
    ).pack(side="left", padx=8)
    root.bind("<Return>", lambda _e: _ok())
    root.bind("<Escape>", lambda _e: _cancel())
    root.mainloop()
    return result["ok"]


def _terminal_camera_grid_fallback(
    captures: list[tuple[str, Any | None, str | None]],
    fallback_reason: str = "headless environment",
) -> bool:
    """Headless fallback: write each frame to a known location and prompt."""
    out_dir = Path("/tmp/lerobot_preflight_cameras")
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"Camera preview saved to {out_dir} ({fallback_reason}):",
        "Open them on the workstation (e.g. via scp) and verify each "
        "named slot matches the physical camera.",
        "",
    ]
    try:
        import cv2
    except ImportError:
        cv2 = None

    for name, frame, err in captures:
        if err is not None or frame is None:
            lines.append(f"  [{name}] FAILED: {err or 'no frame'}")
            continue
        path = out_dir / f"{name}.png"
        if cv2 is not None:
            cv2.imwrite(str(path), frame)
            lines.append(f"  [{name}] {path}  ({frame.shape[1]}x{frame.shape[0]})")
        else:
            lines.append(f"  [{name}] cv2 unavailable, cannot write preview")

    return _terminal_confirm("Confirm cameras", "\n".join(lines))


# ---------------------------------------------------------------------------
# Camera check entry points
# ---------------------------------------------------------------------------

def cameras_check_legacy() -> int:
    """Old behavior: list all OpenCV cameras detected. Useful when picking
    indices for a fresh anyverse_collection_info.json."""
    from lerobot.find_cameras import find_and_print_cameras

    buf = io.StringIO()
    with redirect_stdout(buf):
        cams = find_and_print_cameras("opencv")
    body = buf.getvalue() or "(no cameras detected)"
    if not cams:
        body += (
            "\nNo OpenCV cameras detected. The recording will fail if the script "
            "expects a camera. Confirm only if you intend to run without cameras."
        )
    return 0 if confirm("Detected OpenCV cameras", body) else 1


def _load_collection_info_any(path: str | Path) -> CollectionInfo:
    """Load a CollectionInfo from either a session config or a legacy
    collection_info JSON. Both share the same hardware_meta / collection_meta
    / task_meta / robot{type,id} / task_description shape, so we can detect
    the format by looking for ``schema_version``.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if "schema_version" in data and "session_id" in data:
        sess = SessionConfig.from_dict(data, source_path=str(p))
        return sess.to_collection_info()
    return CollectionInfo.from_dict(data)


def cameras_check_with_preview(config_path: str) -> int:
    """For each camera declared in the JSON, capture a frame and show it in
    a labeled grid so the operator can verify name ↔ image mapping.

    Accepts either a session config or a legacy collection_info JSON.
    """
    p = Path(config_path)
    if not p.is_file():
        print(f"config file not found: {config_path}", file=sys.stderr)
        return 1
    try:
        info = _load_collection_info_any(p)
    except Exception as e:
        print(f"Failed to load {config_path}: {e}", file=sys.stderr)
        return 1

    cameras = info.hardware_meta.cameras
    if not cameras:
        print(
            f"hardware_meta.cameras is empty in {config_path}; nothing to preview.",
            file=sys.stderr,
        )
        return 1

    captures = _capture_all_cameras(cameras)
    if _has_display():
        ok = _tk_camera_grid(captures)
    else:
        ok = _terminal_camera_grid_fallback(captures)
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# Collection info check
# ---------------------------------------------------------------------------

def _build_template_context(info: CollectionInfo) -> dict:
    """Mirror ``record._build_template_context`` so the prompt the operator
    sees in the popup is identical to what the recorder will load."""
    return {
        **info.task_meta.objects,
        "task_name": info.task_meta.task_name,
        "operator": info.collection_meta.operator_name,
        "city": info.collection_meta.city,
        "site": info.collection_meta.site_location,
    }


def _format_end_effector_set(ee_set) -> str:
    """Render a per-arm EndEffectorSet for the confirmation popup.

    Collapses to a single 'type / model' line when both arms match (common
    case), otherwise shows ``left=type/model, right=type/model``.
    """
    left, right = ee_set.left, ee_set.right
    if left.type == right.type and left.model == right.model:
        return f"{left.type} / {left.model}"
    return (
        f"left={left.type}/{left.model}, "
        f"right={right.type}/{right.model}"
    )


def _format_collection_info(info: CollectionInfo, task_spec=None) -> str:
    cm = info.collection_meta
    tm = info.task_meta
    # Bilingual labels: 中文 (English) so the data collector sees Chinese first
    # while engineers can still cross-reference field names. Highlighted values
    # (robot id, operator, all task_meta, task_description, prompts) are wrapped
    # with _h() so the confirmation popup draws attention to them.
    objects_text = (
        json.dumps(tm.objects, indent=2, ensure_ascii=False) if tm.objects else "  (none)"
    )
    lines = [
        f"任务名称 (Task name)         : {_h(tm.task_name)}",
        f"任务阶段 (Stage)             : {_h(tm.task_stage.mode)}"
        + (f" — {_h(tm.task_stage.stages)}" if tm.task_stage.stages else ""),
        f"采集模式 (Mode)              : {cm.mode}",
        f"操作员 (Operator)            : {_h(cm.operator_name)}"
        + (f" + {_h(cm.adversary_operator)}" if cm.is_adversary else ""),
        f"站点 (Site)                  : {cm.site_location} ({cm.city})",
        f"机器人 (Robot)               : {info.robot_type} (id={_h(info.robot_id)})",
        f"末端执行器 (End-effector)    : "
        + _format_end_effector_set(info.hardware_meta.end_effector),
        "",
        "物体 (Objects):",
        _h(objects_text),
    ]
    if info.task_description:
        lines += ["", "任务描述 (Task description):", _h(info.task_description)]

    if task_spec is not None:
        lines += [
            "",
            f"任务规格 ID (Task spec id) : {task_spec.task_id}",
            "提示词 (Prompts, template-substituted):",
        ]
        for role_name, role in task_spec.roles.items():
            lines.append(f"  [{role_name}] {_h(role.prompt)}")
    return "\n".join(lines)


def session_check(session_path: str) -> int:
    """Load + validate the session config and show a confirmation popup.

    The popup includes the same task_meta + objects + task_description
    summary as the legacy collection_info popup, *plus* the resolved role
    prompts (template-substituted), so the operator confirms exactly what
    the recorder will pass to the policy.
    """
    p = Path(session_path)
    if not p.is_file():
        print(f"session config not found: {session_path}", file=sys.stderr)
        return 1
    try:
        sess = SessionConfig.from_json(p)
        sess.validate()
    except SessionConfigError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed to load {session_path}: {e}", file=sys.stderr)
        return 1

    info = sess.to_collection_info()
    task_spec = sess.task_spec
    if task_spec is not None:
        task_spec = task_spec.apply_template(_build_template_context(info))

    body = _format_session(sess, info, task_spec)
    return 0 if confirm("Confirm session config", body) else 1


def _format_session(
    sess: SessionConfig,
    info: CollectionInfo,
    task_spec=None,
) -> str:
    rec = sess.recording
    body = _format_collection_info(info, task_spec=task_spec)
    extras = [
        "",
        f"会话 ID (Session id)         : {sess.session_id}",
        f"回合数 (Episodes)            : {rec.get('num_episodes')}"
        f"  ({rec.get('episode_time_s')}s each, reset {rec.get('reset_time_s')}s)",
        f"数据根目录 (Data root)       : {sess.data_root}",
    ]
    return body + "\n" + "\n".join(extras)


def collection_info_check(path: str, task_spec_path: str | None = None) -> int:
    """Load + validate the collection_info JSON. When ``task_spec_path`` is
    given, also load the task_spec, apply the same template substitution
    the recorder would, and show the resolved prompts in the popup so the
    operator confirms what will actually be recorded."""
    p = Path(path)
    if not p.is_file():
        print(f"collection_info file not found: {path}", file=sys.stderr)
        return 1
    try:
        info = CollectionInfo.from_json(p)
        info.validate()
    except CollectionInfoError as e:
        print(str(e), file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed to load {path}: {e}", file=sys.stderr)
        return 1

    task_spec = None
    if task_spec_path:
        try:
            from lerobot.recording.task.task_spec import TaskSpec
            task_spec = TaskSpec.from_json(task_spec_path)
            task_spec = task_spec.apply_template(_build_template_context(info))
        except Exception as e:
            # Don't fail preflight on a bad task_spec — show the warning and
            # continue with collection_info-only confirmation. The recorder
            # will surface the real error if the spec actually breaks.
            print(
                f"warning: failed to load task_spec {task_spec_path}: {e}",
                file=sys.stderr,
            )
            task_spec = None

    body = _format_collection_info(info, task_spec=task_spec)
    return 0 if confirm("Confirm collection info", body) else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_usage() -> None:
    print(
        "Usage:\n"
        "  python -m lerobot.recording.utils.preflight cameras [<config_path>]\n"
        "  python -m lerobot.recording.utils.preflight session <session_config_path>\n"
        "  python -m lerobot.recording.utils.preflight collection_info <path> [<task_spec_path>]",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        _print_usage()
        return 2

    cmd = args[0]
    if cmd == "cameras":
        if len(args) >= 2:
            return cameras_check_with_preview(args[1])
        return cameras_check_legacy()
    if cmd == "session":
        if len(args) < 2:
            _print_usage()
            return 2
        return session_check(args[1])
    if cmd == "collection_info":
        if len(args) < 2:
            _print_usage()
            return 2
        task_spec_path = args[2] if len(args) >= 3 else None
        return collection_info_check(args[1], task_spec_path=task_spec_path)

    _print_usage()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
