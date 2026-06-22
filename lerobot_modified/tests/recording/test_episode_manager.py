"""Tests for recording.task.episode_manager — episode lifecycle, metadata, stats."""

import json
import os
import tempfile
import time

import pytest

from lerobot.recording.task.episode_manager import (
    EpisodeMetadataWriter,
    RoleManager,
    SelfPlayStatsAggregator,
)
from lerobot.recording.task.task_spec import (
    ResetConfig,
    RoleSpec,
    SuccessCondition,
    TaskSpec,
)


def _make_self_play_spec():
    return TaskSpec(
        task_id="test",
        roles={
            "builder": RoleSpec(
                prompt="build it", max_time_s=60,
                success_when=SuccessCondition(at_home=True, value_gte=0.8),
            ),
            "destroyer": RoleSpec(
                prompt="destroy it", max_time_s=45,
                success_when=SuccessCondition(at_home=True, value_lte=-0.9),
            ),
        },
        reset=ResetConfig(reference_pose=[0.0], threshold=0.1, speed_threshold=0.01),
    )


# ---------------------------------------------------------------------------
# RoleManager tests
# ---------------------------------------------------------------------------

class TestRoleManager:
    def test_initial_role(self):
        spec = _make_self_play_spec()
        rm = RoleManager(spec)
        assert rm.current_role == "builder"
        assert rm.current_prompt == "build it"

    def test_advance_role_alternates(self):
        spec = _make_self_play_spec()
        rm = RoleManager(spec)
        assert rm.current_role == "builder"

        rm.advance_role()
        assert rm.current_role == "destroyer"
        assert rm.current_prompt == "destroy it"

        rm.advance_role()
        assert rm.current_role == "builder"
        assert rm.current_prompt == "build it"

    def test_prompt_for_episode(self):
        spec = _make_self_play_spec()
        rm = RoleManager(spec)
        p = rm.prompt_for_episode(0)
        assert p == "build it"

    def test_single_role_stays_same(self):
        spec = TaskSpec(
            task_id="single",
            roles={"operator": RoleSpec(prompt="do it", max_time_s=30)},
            reset=ResetConfig(reference_pose=[0.0], threshold=0.1, speed_threshold=0.01),
        )
        rm = RoleManager(spec)
        assert rm.current_role == "operator"
        rm.advance_role()
        assert rm.current_role == "operator"

    def test_three_roles_cycle(self):
        spec = TaskSpec(
            task_id="tri",
            roles={
                "a": RoleSpec(prompt="do a", max_time_s=10),
                "b": RoleSpec(prompt="do b", max_time_s=10),
                "c": RoleSpec(prompt="do c", max_time_s=10),
            },
            reset=ResetConfig(reference_pose=[0.0], threshold=0.1, speed_threshold=0.01),
        )
        rm = RoleManager(spec)
        assert rm.current_role == "a"
        rm.advance_role()
        assert rm.current_role == "b"
        rm.advance_role()
        assert rm.current_role == "c"
        rm.advance_role()
        assert rm.current_role == "a"


# ---------------------------------------------------------------------------
# SelfPlayStatsAggregator tests
# ---------------------------------------------------------------------------

class TestSelfPlayStatsAggregator:
    def test_empty_stats(self):
        stats = SelfPlayStatsAggregator()
        info = stats.to_dict()
        assert info["episodes_total"] == 0
        assert info["takeover_count_total"] == 0

    def test_update_single_episode(self):
        stats = SelfPlayStatsAggregator()
        stats.update_episode(
            episode_role="builder",
            task_success=True,
            episode_duration_s=30.0,
            intervention_duration_s=0.0,
            intervention_count=0,
            time_to_home_s=10.0,
        )
        info = stats.to_dict()
        assert info["episodes_total"] == 1
        assert info["takeover_count_total"] == 0

    def test_update_multiple_episodes(self):
        stats = SelfPlayStatsAggregator()
        for i in range(5):
            stats.update_episode(
                episode_role="builder" if i % 2 == 0 else "destroyer",
                task_success=i % 2 == 0,
                episode_duration_s=20.0 + i,
                intervention_duration_s=1.0 if i == 3 else 0.0,
                intervention_count=1 if i == 3 else 0,
                time_to_home_s=5.0,
            )
        info = stats.to_dict()
        assert info["episodes_total"] == 5
        assert info["takeover_count_total"] == 1

    def test_resolve_success_builder_value(self):
        success, reason = SelfPlayStatsAggregator.resolve_task_success(
            task_success=None,
            episode_role="builder",
            final_value=0.9,
            final_is_home=True,
            timeout_fail=False,
            value_threshold_task_complete=0.8,
            value_threshold_back_to_start=-0.9,
        )
        assert success is True
        assert "value" in reason.lower()

    def test_resolve_success_destroyer_value(self):
        success, reason = SelfPlayStatsAggregator.resolve_task_success(
            task_success=None,
            episode_role="destroyer",
            final_value=-0.95,
            final_is_home=True,
            timeout_fail=False,
            value_threshold_task_complete=0.8,
            value_threshold_back_to_start=-0.9,
        )
        assert success is True

    def test_resolve_timeout_fail(self):
        success, reason = SelfPlayStatsAggregator.resolve_task_success(
            task_success=None,
            episode_role="builder",
            final_value=0.3,
            final_is_home=False,
            timeout_fail=True,
            value_threshold_task_complete=0.8,
            value_threshold_back_to_start=-0.9,
        )
        assert success is False
        assert "timeout" in reason.lower()

    def test_resolve_preserves_explicit_success(self):
        success, reason = SelfPlayStatsAggregator.resolve_task_success(
            task_success=True,
            episode_role="builder",
            final_value=0.3,
            final_is_home=True,
            timeout_fail=False,
            value_threshold_task_complete=0.8,
            value_threshold_back_to_start=-0.9,
        )
        assert success is True
        assert "explicit" in reason.lower()

    def test_build_episode_metadata(self):
        meta = SelfPlayStatsAggregator.build_episode_metadata(
            episode_role="builder",
            end_reason="value_success",
            final_value=0.9,
            final_is_home=True,
            intervention_count=1,
            intervention_duration_s=5.0,
            episode_duration_s=30.0,
            time_to_home_s=12.0,
            home_duration_s=8.0,
        )
        assert meta["role"] == "builder"
        assert meta["end_reason"] == "value_success"
        assert meta["value_score"] == 0.9
        assert meta["intervention_count"] == 1


# ---------------------------------------------------------------------------
# EpisodeMetadataWriter tests
# ---------------------------------------------------------------------------

class TestEpisodeMetadataWriter:
    def test_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = EpisodeMetadataWriter(tmpdir)
            writer.write(0, {"role": "builder", "success": True, "value_score": 0.9})
            writer.write(1, {"role": "destroyer", "success": False, "value_score": -0.5})

            path = os.path.join(tmpdir, "episode_metadata.jsonl")
            assert os.path.isfile(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2

            d0 = json.loads(lines[0])
            assert d0["episode_index"] == 0
            assert d0["role"] == "builder"

            d1 = json.loads(lines[1])
            assert d1["episode_index"] == 1

    def test_append_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer1 = EpisodeMetadataWriter(tmpdir)
            writer1.write(0, {"role": "builder"})

            writer2 = EpisodeMetadataWriter(tmpdir)
            writer2.write(1, {"role": "destroyer"})

            path = os.path.join(tmpdir, "episode_metadata.jsonl")
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 2
