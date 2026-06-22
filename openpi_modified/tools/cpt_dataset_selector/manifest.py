"""Generate repo_id manifest from one or more training config files (REPO_ID)."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC
from datetime import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_public_config_path() -> Path:
    return _repo_root() / "src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.all_public_datasets.py"


def parse_config_paths_csv(value: str | None) -> list[Path] | None:
    """Split comma-separated config paths into ``list[Path]`` (whitespace trimmed, empty parts skipped).

    Returns ``None`` if ``value`` is None or only whitespace, so callers can treat it as "use default".
    """
    if value is None or not str(value).strip():
        return None
    parts = [p.strip() for p in str(value).split(",")]
    paths = [Path(p) for p in parts if p]
    return paths if paths else None


def load_repo_ids_from_config(config_path: Path) -> tuple[str, ...]:
    """Load REPO_ID from a single training config .py (tuple or list)."""
    path = Path(config_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)

    key = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    mod_name = f"cpt_manifest_cfg_{key}"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    repo_id = getattr(mod, "REPO_ID", None)
    if repo_id is None:
        raise ValueError(f"REPO_ID not found in config {path}")
    if isinstance(repo_id, tuple):
        return tuple(str(x) for x in repo_id)
    if isinstance(repo_id, list):
        return tuple(str(x) for x in repo_id)
    raise TypeError(f"REPO_ID has unexpected type {type(repo_id)}")


def load_repo_ids_merged(config_paths: Iterable[Path]) -> tuple[str, ...]:
    """Merge REPO_ID from several configs; order preserved, duplicates skipped (first wins)."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in config_paths:
        path = Path(raw).resolve()
        for rid in load_repo_ids_from_config(path):
            if rid not in seen:
                seen.add(rid)
                out.append(rid)
    return tuple(out)


def load_all_public_datasets_repo_ids(config_path: Path | None = None) -> tuple[str, ...]:
    """Load REPO_ID from a single config (default: all_public_datasets)."""
    path = config_path or _default_public_config_path()
    return load_repo_ids_from_config(path)


def git_rev(root: Path | None = None) -> str | None:
    root = root or _repo_root()
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        pass
    return None


def write_manifest(
    out_path: Path,
    *,
    config_paths: list[Path] | None = None,
    root_dir: str | None = None,
) -> dict:
    """Write manifest.json with repo_ids and metadata.

    ``config_paths``: training config .py files whose ``REPO_ID`` are merged in order
    (deduped). ``None`` or ``[]`` means only the default ``all_public_datasets`` config.
    """
    if config_paths:
        paths = [Path(p).resolve() for p in config_paths]
        repo_ids = load_repo_ids_merged(paths)
    else:
        paths = [_default_public_config_path()]
        repo_ids = load_repo_ids_from_config(paths[0])
    config_paths_str = [str(p) for p in paths]

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "git_rev": git_rev(),
        "config_paths": config_paths_str,
        "root_dir_env": root_dir or os.environ.get("OPENPI_ROOT_DIR"),
        "count": len(repo_ids),
        "repo_ids": list(repo_ids),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
