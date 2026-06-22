"""Episode lifecycle management: multi-episode loop, metadata, stats, role switching.

Handles the outer loop of recording sessions: preparing episodes,
saving data, tracking statistics, and managing role alternation
in self-play mode.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


class RoleManager:
    """Manages role alternation for self-play.

    Cycles through roles defined in the TaskSpec (builder → destroyer → builder ...).
    """

    def __init__(self, task_spec: Any):
        self._spec = task_spec
        self._role_names = task_spec.role_names
        self._role_idx = 0

    @property
    def current_role(self) -> str:
        return self._role_names[self._role_idx]

    @property
    def current_prompt(self) -> str:
        return self._spec.roles[self.current_role].prompt

    def prompt_for_episode(self, episode_idx: int) -> str:
        return self.current_prompt

    def advance_role(self) -> str:
        """Advance to the next role. Returns the new role name."""
        if len(self._role_names) > 1:
            self._role_idx = (self._role_idx + 1) % len(self._role_names)
        return self.current_role


class SelfPlayStatsAggregator:
    """Tracks aggregate statistics across self-play episodes."""

    def __init__(self):
        self.episodes_total: int = 0
        self.takeover_count_total: int = 0
        self.takeover_duration_total_s: float = 0.0
        self.success_count: int = 0
        self.duration_total_s: float = 0.0
        self._role_stats: dict[str, dict] = {}

    def update_episode(
        self,
        episode_role: str | None,
        task_success: bool | None,
        episode_duration_s: float,
        intervention_duration_s: float,
        intervention_count: int,
        time_to_home_s: float | None = None,
        end_reason: str | None = None,
        collision_count: int = 0,
        home_duration_s: float = 0.0,
    ) -> None:
        self.episodes_total += 1
        self.takeover_count_total += intervention_count
        self.takeover_duration_total_s += intervention_duration_s
        self.duration_total_s += episode_duration_s

        if task_success is True:
            self.success_count += 1

        if episode_role:
            if episode_role not in self._role_stats:
                self._role_stats[episode_role] = {
                    "count": 0,
                    "success": 0,
                    "fail": 0,
                    "duration_total": 0.0,
                    "time_to_home_values": [],
                    "success_active_durations": [],  # episode_duration - home_duration for successes
                    "end_reasons": {},
                    "collision_count": 0,
                    "intervention_episodes": 0,
                }
            rs = self._role_stats[episode_role]
            rs["count"] += 1
            rs["duration_total"] += episode_duration_s
            rs["collision_count"] += collision_count
            if task_success is True:
                rs["success"] += 1
                active_time = max(0.0, episode_duration_s - home_duration_s)
                rs["success_active_durations"].append(active_time)
            elif task_success is False:
                rs["fail"] += 1
            if time_to_home_s is not None:
                rs["time_to_home_values"].append(time_to_home_s)
            if end_reason:
                rs["end_reasons"][end_reason] = rs["end_reasons"].get(end_reason, 0) + 1
            if intervention_count > 0:
                rs["intervention_episodes"] += 1

    def to_dict(self) -> dict:
        takeover_rate = (
            self.takeover_count_total / self.episodes_total
            if self.episodes_total > 0
            else 0.0
        )
        takeover_time_ratio = (
            self.takeover_duration_total_s / self.duration_total_s
            if self.duration_total_s > 0
            else 0.0
        )
        avg_duration = (
            self.duration_total_s / self.episodes_total
            if self.episodes_total > 0
            else 0.0
        )

        # Build per-role summary with computed fields
        role_summary = {}
        for role_name, rs in self._role_stats.items():
            count = rs["count"]
            success_rate = rs["success"] / count if count > 0 else 0.0
            avg_time_to_home = (
                sum(rs["time_to_home_values"]) / len(rs["time_to_home_values"])
                if rs["time_to_home_values"]
                else None
            )
            avg_success_active_duration = (
                sum(rs["success_active_durations"]) / len(rs["success_active_durations"])
                if rs["success_active_durations"]
                else None
            )
            role_summary[role_name] = {
                "count": count,
                "success": rs["success"],
                "fail": rs["fail"],
                "success_rate": round(success_rate, 4),
                "avg_duration_s": round(rs["duration_total"] / count, 3) if count > 0 else 0.0,
                "avg_success_active_duration_s": round(avg_success_active_duration, 3) if avg_success_active_duration is not None else None,
                "avg_time_to_home_s": round(avg_time_to_home, 3) if avg_time_to_home is not None else None,
                "collision_count": rs["collision_count"],
                "intervention_episodes": rs["intervention_episodes"],
                "end_reasons": rs["end_reasons"],
            }

        return {
            "episodes_total": self.episodes_total,
            "success_count": self.success_count,
            "success_rate": round(self.success_count / self.episodes_total, 4) if self.episodes_total > 0 else 0.0,
            "avg_episode_duration_s": round(avg_duration, 3),
            "takeover_count_total": self.takeover_count_total,
            "takeover_rate_per_episode": round(takeover_rate, 4),
            "takeover_duration_total_s": round(self.takeover_duration_total_s, 3),
            "takeover_time_ratio": round(takeover_time_ratio, 4),
            "role_stats": role_summary,
        }

    @staticmethod
    def resolve_task_success(
        task_success: bool | None,
        episode_role: str | None,
        final_value: float | None,
        final_is_home: bool,
        timeout_fail: bool,
        value_threshold_task_complete: float,
        value_threshold_back_to_start: float,
    ) -> tuple[bool | None, str]:
        """Determine task success from available signals.

        Returns (success, reason_string).
        """
        if task_success is not None:
            return task_success, "explicit"

        if timeout_fail:
            return False, "timeout"

        if final_value is not None and episode_role:
            if episode_role == "builder" and final_value >= value_threshold_task_complete:
                return True, "value_gte_threshold"
            if episode_role == "destroyer" and final_value <= value_threshold_back_to_start:
                return True, "value_lte_threshold"
            # Value didn't meet threshold
            if final_is_home:
                return False, "value_at_home_not_met"

        return None, "undetermined"

    @staticmethod
    def build_episode_metadata(
        episode_role: str | None,
        end_reason: str | None,
        final_value: float | None,
        final_is_home: bool,
        intervention_count: int,
        intervention_duration_s: float,
        episode_duration_s: float,
        time_to_home_s: float | None = None,
        home_duration_s: float = 0.0,
        collision_count: int = 0,
        collision_max_recovery: str = "none",
        collision_events: list | None = None,
    ) -> dict:
        return {
            "role": episode_role,
            "end_reason": end_reason,
            "value_score": final_value,
            "is_home": final_is_home,
            "intervention_count": intervention_count,
            "intervention_duration_s": round(intervention_duration_s, 3),
            "episode_duration_s": round(episode_duration_s, 3),
            "time_to_home_s": round(time_to_home_s, 3) if time_to_home_s is not None else None,
            "home_duration_s": round(home_duration_s, 3),
            "collision_count": collision_count,
            "collision_max_recovery": collision_max_recovery,
            "collision_events": collision_events or [],
        }


class EpisodeMetadataWriter:
    """Writes per-episode metadata to episode_metadata.jsonl.

    This supplements the standard episodes.jsonl (which only has
    episode_index, tasks, length) with self-play specific data.
    """

    def __init__(self, meta_dir: str):
        self._path = os.path.join(meta_dir, "episode_metadata.jsonl")

    def write(self, episode_index: int, metadata: dict) -> None:
        record = {"episode_index": episode_index, **metadata}
        line = json.dumps(record, ensure_ascii=False, default=str)
        with open(self._path, "a") as f:
            f.write(line + "\n")
