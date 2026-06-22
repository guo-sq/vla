"""Data parser for value model benchmark — four-quadrant classification and GT construction.

Parses self-play episode_metadata.jsonl and classifies episodes into four quadrants:
- True Positive (TP): builder + success
- True Negative (TN): destroyer + success
- False Positive (FP): builder + failure
- False Negative (FN): destroyer + failure
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Quadrant(str, Enum):
    TRUE_POSITIVE = "true_positive"
    TRUE_NEGATIVE = "true_negative"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"


@dataclass(frozen=True)
class EpisodeInfo:
    episode_index: int
    role: str
    success: bool
    quadrant: Quadrant
    value_score: float
    end_reason: str = ""
    label_source: str = "metadata"


VALID_ROLES = {"builder", "destroyer"}
ROLE_ALIASES = {"folder": "builder", "disturber": "destroyer"}


_BUILDER_KEYWORDS = {"fold", "collar", "lay"}
_DESTROYER_KEYWORDS = {"disarrange", "messy"}


def _normalize_role(role: str) -> str:
    """Map role aliases to canonical names."""
    return ROLE_ALIASES.get(role, role)


def infer_role_from_task(task_text: str) -> str:
    """Infer role from task description text.

    - Contains "fold", "collar", or "lay" -> "builder"
    - Contains "disarrange" or "messy" -> "destroyer"
    - Otherwise -> raises ValueError
    """
    lower = task_text.lower()
    for kw in _BUILDER_KEYWORDS:
        if kw in lower:
            return "builder"
    for kw in _DESTROYER_KEYWORDS:
        if kw in lower:
            return "destroyer"
    raise ValueError(f"Cannot infer role from task text: {task_text!r}")


def classify_episode(role: str, success: bool) -> Quadrant:
    """Classify an episode into one of four quadrants based on role and success."""
    role = _normalize_role(role)
    if role not in VALID_ROLES:
        raise ValueError(f"Unknown role: {role!r}. Expected one of {VALID_ROLES}")

    if role == "builder" and success:
        return Quadrant.TRUE_POSITIVE
    if role == "destroyer" and success:
        return Quadrant.TRUE_NEGATIVE
    if role == "builder" and not success:
        return Quadrant.FALSE_POSITIVE
    return Quadrant.FALSE_NEGATIVE


def load_episode_metadata(repo_path: str | Path) -> list[EpisodeInfo]:
    """Load episode metadata from a repo directory.

    Expects ``repo_path/meta/episode_metadata.jsonl``.

    Episodes missing the explicit ``success`` field are skipped with a warning.
    Inferring success from path substrings or end-reason text is the data
    quality pipeline's responsibility, not the benchmark's.

    Malformed JSON lines raise immediately with file/line context — silently
    skipping them would let the benchmark drop episodes without anyone noticing.
    """
    repo_path = Path(repo_path)
    meta_file = repo_path / "meta" / "episode_metadata.jsonl"
    if not meta_file.exists():
        meta_file = repo_path / "meta" / "episodes.jsonl"
    if not meta_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found in {repo_path / 'meta'} (tried episode_metadata.jsonl and episodes.jsonl)"
        )

    episodes: list[EpisodeInfo] = []
    skipped_missing_success = 0
    with open(meta_file) as f:
        for lineno, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Malformed JSON in {meta_file} line {lineno}: {e.msg}") from e
            label_source = "metadata"

            # --- role resolution ---
            raw_role = record.get("role")
            if raw_role is not None:
                role = _normalize_role(raw_role)
            else:
                tasks = record.get("tasks", [])
                if not tasks:
                    raise ValueError(f"Record has no 'role' and no 'tasks': {record}")
                role = infer_role_from_task(tasks[0])
                label_source = "inferred"

            # --- success resolution ---
            success = record.get("success")
            if success is None:
                skipped_missing_success += 1
                logger.warning(
                    "Episode %s in %s missing 'success' field (end_reason=%r); "
                    "skipping — fix in data quality pipeline",
                    record.get("episode_index", "?"),
                    meta_file.parent.parent.name,
                    record.get("end_reason"),
                )
                continue

            quadrant = classify_episode(role, success)
            episodes.append(
                EpisodeInfo(
                    episode_index=record["episode_index"],
                    role=role,
                    success=success,
                    quadrant=quadrant,
                    value_score=record.get("value_score", 0.0),
                    end_reason=record.get("end_reason", ""),
                    label_source=label_source,
                )
            )

    if skipped_missing_success:
        logger.warning(
            "%s: skipped %d episodes missing 'success' field",
            meta_file.parent.parent.name,
            skipped_missing_success,
        )
    return episodes


def split_by_quadrant(episodes: list[EpisodeInfo]) -> dict[Quadrant, list[EpisodeInfo]]:
    """Split episodes into four quadrant groups."""
    result: dict[Quadrant, list[EpisodeInfo]] = {q: [] for q in Quadrant}
    for ep in episodes:
        result[ep.quadrant].append(ep)
    return result


def construct_ideal_target(num_steps: int, *, aligned: bool) -> np.ndarray:
    """Build the ideal value-target trajectory under the prompt alignment assumption.

    The benchmark measures `pred` against a prompt-conditioned linear surrogate, not
    against the dataset's recorded returns. This function returns that surrogate:

      - aligned=True  → -1 → 0 (linear), behavior moves toward the prompted goal
      - aligned=False → 0 → -1 (linear), behavior moves away from the prompted goal

    The caller decides what "aligned" means for a given (role, override_prompt). The
    benchmark deliberately does NOT inspect prompt strings here — substring heuristics
    silently break for any task whose prompt vocabulary differs from the seatbelt
    take-off / hang convention.
    """
    start, end = (-1.0, 0.0) if aligned else (0.0, -1.0)
    return np.linspace(start, end, num_steps)
