"""Build SQLite index from ROOT_DIR/<repo_id>/meta/tasks.jsonl."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time

from .parse_tasks import parse_tasks_jsonl_content


def _repo_family(repo_id: str) -> str:
    parts = repo_id.strip("/").split("/")
    if len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else "unknown"


def _parse_meta_info(repo_root: Path) -> tuple[str | None, int | None, float | None]:
    """Read meta/info.json: robot_type, total_episodes, duration_hours from total_frames/fps."""
    p = repo_root / "meta" / "info.json"
    if not p.is_file():
        return None, None, None
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return None, None, None
    rt_raw = data.get("robot_type")
    robot_type: str | None = None
    if rt_raw is not None and not isinstance(rt_raw, bool):
        if isinstance(rt_raw, int | float):
            robot_type = str(rt_raw)
        else:
            s = str(rt_raw).strip()
            robot_type = s or None

    total_episodes: int | None = None
    te = data.get("total_episodes")
    if te is not None:
        try:
            total_episodes = int(te)
        except (TypeError, ValueError):
            total_episodes = None

    duration_hours: float | None = None
    tf = data.get("total_frames")
    fps = data.get("fps")
    if tf is not None and fps is not None:
        try:
            tf_f = float(tf)
            fps_f = float(fps)
            if fps_f > 0:
                duration_hours = (tf_f / fps_f) / 3600.0
        except (TypeError, ValueError):
            pass

    return robot_type, total_episodes, duration_hours


@dataclass
class IndexStats:
    repos_scanned: int
    repos_ok: int
    repos_missing_tasks: int
    repos_errors: int
    task_rows: int


def ensure_repos_columns(conn: sqlite3.Connection) -> None:
    """Add columns to repos if missing (older index DBs)."""
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='repos'")
    if cur.fetchone() is None:
        return
    cur = conn.execute("PRAGMA table_info(repos)")
    cols = {row[1] for row in cur.fetchall()}
    if "robot_type" not in cols:
        conn.execute("ALTER TABLE repos ADD COLUMN robot_type TEXT")
    if "total_episodes" not in cols:
        conn.execute("ALTER TABLE repos ADD COLUMN total_episodes INTEGER")
    if "duration_hours" not in cols:
        conn.execute("ALTER TABLE repos ADD COLUMN duration_hours REAL")


# Backwards-compatible name
def ensure_repos_robot_type_column(conn: sqlite3.Connection) -> None:
    ensure_repos_columns(conn)


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE IF NOT EXISTS repos (
            repo_id TEXT PRIMARY KEY,
            family TEXT,
            robot_type TEXT,
            total_episodes INTEGER,
            duration_hours REAL,
            has_tasks_jsonl INTEGER NOT NULL,
            error TEXT,
            tasks_jsonl_mtime REAL,
            indexed_at REAL NOT NULL,
            task_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id TEXT NOT NULL,
            line_no INTEGER NOT NULL,
            task_index INTEGER,
            raw_json TEXT NOT NULL,
            normalized_text TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_repo ON tasks(repo_id);
        """
    )
    ensure_repos_columns(conn)
    conn.commit()


def _process_one_repo(args: tuple[str, Path]) -> dict:
    repo_id, root = args
    meta_path = root / "meta" / "tasks.jsonl"
    robot_type, total_episodes, duration_hours = _parse_meta_info(root)
    out: dict = {
        "repo_id": repo_id,
        "family": _repo_family(repo_id),
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "duration_hours": duration_hours,
        "has_tasks_jsonl": False,
        "error": None,
        "mtime": None,
        "tasks": [],
    }
    if not meta_path.is_file():
        out["error"] = "missing_tasks_jsonl"
        return out
    out["has_tasks_jsonl"] = True
    try:
        out["mtime"] = meta_path.stat().st_mtime
        text = meta_path.read_text(encoding="utf-8", errors="replace")
        rows = parse_tasks_jsonl_content(text)
        for r in rows:
            out["tasks"].append(
                {
                    "line_no": r["line_no"],
                    "task_index": r["task_index"],
                    "raw_json": json.dumps(r["raw"], ensure_ascii=False),
                    "normalized_text": r["normalized_text"],
                }
            )
    except OSError as e:
        out["error"] = f"os_error:{e}"
    except Exception as e:
        out["error"] = f"read_error:{e}"
    return out


def build_index(
    root_dir: Path,
    repo_ids: list[str],
    db_path: Path,
    *,
    workers: int = 8,
    incremental: bool = True,
    progress: Callable[[str], None] | None = None,
) -> IndexStats:
    root_dir = root_dir.resolve()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON")
    init_schema(conn)

    existing: dict[str, float] = {}
    if incremental:
        cur = conn.execute("SELECT repo_id, tasks_jsonl_mtime FROM repos WHERE has_tasks_jsonl = 1")
        existing = {r[0]: r[1] or 0.0 for r in cur.fetchall()}

    to_scan: list[str] = []
    for rid in repo_ids:
        meta = root_dir / rid / "meta" / "tasks.jsonl"
        if not incremental:
            to_scan.append(rid)
            continue
        if not meta.is_file():
            to_scan.append(rid)
            continue
        try:
            mtime = meta.stat().st_mtime
        except OSError:
            to_scan.append(rid)
            continue
        if rid not in existing or abs(existing.get(rid, 0) - mtime) > 1e-6:
            to_scan.append(rid)

    if progress:
        progress(f"scanning {len(to_scan)} repos (incremental={incremental})")

    if workers <= 1:
        results = [_process_one_repo((rid, root_dir / rid)) for rid in to_scan]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_process_one_repo, (rid, root_dir / rid)) for rid in to_scan]
            results = [f.result() for f in futs]

    now = time.time()
    repos_ok = 0
    repos_missing = 0
    repos_err = 0
    task_rows = 0

    for r in results:
        rid = r["repo_id"]
        conn.execute("DELETE FROM tasks WHERE repo_id = ?", (rid,))
        conn.execute("DELETE FROM repos WHERE repo_id = ?", (rid,))

        err = r.get("error")
        has_meta = r["has_tasks_jsonl"]
        if err == "missing_tasks_jsonl" or not has_meta:
            repos_missing += 1
            conn.execute(
                """
                INSERT INTO repos (repo_id, family, robot_type, total_episodes, duration_hours, has_tasks_jsonl, error, tasks_jsonl_mtime, indexed_at, task_count)
                VALUES (?, ?, ?, ?, ?, 0, ?, NULL, ?, 0)
                """,
                (
                    rid,
                    r["family"],
                    r.get("robot_type"),
                    r.get("total_episodes"),
                    r.get("duration_hours"),
                    err or "missing_tasks_jsonl",
                    now,
                ),
            )
            continue
        if err:
            repos_err += 1
            conn.execute(
                """
                INSERT INTO repos (repo_id, family, robot_type, total_episodes, duration_hours, has_tasks_jsonl, error, tasks_jsonl_mtime, indexed_at, task_count)
                VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, 0)
                """,
                (
                    rid,
                    r["family"],
                    r.get("robot_type"),
                    r.get("total_episodes"),
                    r.get("duration_hours"),
                    err,
                    r.get("mtime"),
                    now,
                ),
            )
            continue

        repos_ok += 1
        tasks = r["tasks"]
        task_count = len(tasks)
        task_rows += task_count
        conn.execute(
            """
            INSERT INTO repos (repo_id, family, robot_type, total_episodes, duration_hours, has_tasks_jsonl, error, tasks_jsonl_mtime, indexed_at, task_count)
            VALUES (?, ?, ?, ?, ?, 1, NULL, ?, ?, ?)
            """,
            (
                rid,
                r["family"],
                r.get("robot_type"),
                r.get("total_episodes"),
                r.get("duration_hours"),
                r.get("mtime"),
                now,
                task_count,
            ),
        )
        for t in tasks:
            conn.execute(
                """
                INSERT INTO tasks (repo_id, line_no, task_index, raw_json, normalized_text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    rid,
                    t["line_no"],
                    t["task_index"],
                    t["raw_json"],
                    t["normalized_text"],
                ),
            )

    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("indexed_at", str(now)),
    )
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("root_dir", str(root_dir)),
    )
    conn.commit()
    conn.close()

    return IndexStats(
        repos_scanned=len(to_scan),
        repos_ok=repos_ok,
        repos_missing_tasks=repos_missing,
        repos_errors=repos_err,
        task_rows=task_rows,
    )


def full_rebuild(
    root_dir: Path,
    repo_ids: list[str],
    db_path: Path,
    *,
    workers: int = 8,
    progress: Callable[[str], None] | None = None,
) -> IndexStats:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    return build_index(
        root_dir,
        repo_ids,
        db_path,
        workers=workers,
        incremental=False,
        progress=progress,
    )
