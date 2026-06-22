"""Tests for benchmark data parser — four-quadrant classification and GT value construction."""

import json

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

from scripts.benchmark.data_parser import EpisodeInfo
from scripts.benchmark.data_parser import Quadrant
from scripts.benchmark.data_parser import classify_episode
from scripts.benchmark.data_parser import construct_ideal_target
from scripts.benchmark.data_parser import infer_role_from_task
from scripts.benchmark.data_parser import load_episode_metadata
from scripts.benchmark.data_parser import split_by_quadrant

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metadata_lines():
    """Realistic metadata lines covering all four quadrants."""
    return [
        {"episode_index": 0, "role": "builder", "end_reason": "value_success", "value_score": -0.017, "success": True},
        {
            "episode_index": 1,
            "role": "destroyer",
            "end_reason": "value_success",
            "value_score": -0.992,
            "success": True,
        },
        {"episode_index": 2, "role": "builder", "end_reason": "timeout", "value_score": -0.5, "success": False},
        {"episode_index": 3, "role": "destroyer", "end_reason": "timeout", "value_score": -0.3, "success": False},
    ]


@pytest.fixture
def metadata_file(sample_metadata_lines, tmp_path):
    """Write sample metadata to a temp jsonl file."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    fpath = meta_dir / "episode_metadata.jsonl"
    with open(fpath, "w") as f:
        for line in sample_metadata_lines:
            f.write(json.dumps(line) + "\n")
    return tmp_path


# ---------------------------------------------------------------------------
# Test: classify_episode
# ---------------------------------------------------------------------------


class TestClassifyEpisode:
    def test_builder_success_is_true_positive(self):
        assert classify_episode(role="builder", success=True) == Quadrant.TRUE_POSITIVE

    def test_destroyer_success_is_true_negative(self):
        assert classify_episode(role="destroyer", success=True) == Quadrant.TRUE_NEGATIVE

    def test_builder_failure_is_false_positive(self):
        assert classify_episode(role="builder", success=False) == Quadrant.FALSE_POSITIVE

    def test_destroyer_failure_is_false_negative(self):
        assert classify_episode(role="destroyer", success=False) == Quadrant.FALSE_NEGATIVE

    def test_unknown_role_raises(self):
        with pytest.raises(ValueError, match="role"):
            classify_episode(role="unknown", success=True)


# ---------------------------------------------------------------------------
# Test: load_episode_metadata
# ---------------------------------------------------------------------------


class TestLoadEpisodeMetadata:
    def test_loads_all_episodes(self, metadata_file):
        episodes = load_episode_metadata(metadata_file)
        assert len(episodes) == 4

    def test_episode_info_fields(self, metadata_file):
        episodes = load_episode_metadata(metadata_file)
        ep0 = episodes[0]
        assert isinstance(ep0, EpisodeInfo)
        assert ep0.episode_index == 0
        assert ep0.role == "builder"
        assert ep0.success is True
        assert ep0.quadrant == Quadrant.TRUE_POSITIVE
        assert ep0.value_score == pytest.approx(-0.017)

    def test_empty_file_returns_empty(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        fpath = meta_dir / "episode_metadata.jsonl"
        fpath.write_text("")
        episodes = load_episode_metadata(tmp_path)
        assert episodes == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_episode_metadata(tmp_path / "nonexistent")

    def test_missing_success_logs_warning_and_skips_episode(self, tmp_path, caplog):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "role": "builder", "success": True})
            + "\n"
            + json.dumps({"episode_index": 1, "role": "builder", "end_reason": "exit_early"})
            + "\n"
        )
        with caplog.at_level("WARNING"):
            episodes = load_episode_metadata(tmp_path)
        assert len(episodes) == 1
        assert episodes[0].episode_index == 0
        assert any("missing 'success'" in r.message for r in caplog.records)

    def test_malformed_jsonl_raises_with_context(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "role": "builder", "success": True}) + "\n" + "{not valid json\n"
        )
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_episode_metadata(tmp_path)

    def test_path_containing_error_substring_does_not_force_failure(self, tmp_path):
        """Regression: prior heuristic flipped success=False whenever 'error' was in
        the repo path. The new contract is that success must come from the record."""
        repo_dir = tmp_path / "anyverse_data_with_error_recovery_batch_03"
        meta_dir = repo_dir / "meta"
        meta_dir.mkdir(parents=True)
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "role": "builder", "success": True}) + "\n"
        )
        episodes = load_episode_metadata(repo_dir)
        assert len(episodes) == 1
        assert episodes[0].success is True
        assert episodes[0].quadrant == Quadrant.TRUE_POSITIVE


# ---------------------------------------------------------------------------
# Test: episodes.jsonl fallback
# ---------------------------------------------------------------------------


class TestEpisodesJsonlFallback:
    def test_reads_episodes_jsonl_when_no_episode_metadata(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "role": "builder", "success": True}) + "\n"
        )
        episodes = load_episode_metadata(tmp_path)
        assert len(episodes) == 1
        assert episodes[0].role == "builder"

    def test_prefers_episode_metadata_over_episodes(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "episode_metadata.jsonl").write_text(
            json.dumps({"episode_index": 0, "role": "builder", "success": True}) + "\n"
        )
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "role": "destroyer", "success": True}) + "\n"
        )
        episodes = load_episode_metadata(tmp_path)
        assert episodes[0].role == "builder"

    def test_neither_file_raises(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        with pytest.raises(FileNotFoundError):
            load_episode_metadata(tmp_path)


# ---------------------------------------------------------------------------
# Test: role aliases (folder/disturber)
# ---------------------------------------------------------------------------


class TestRoleAliases:
    def test_folder_classified_as_builder(self):
        assert classify_episode(role="folder", success=True) == Quadrant.TRUE_POSITIVE
        assert classify_episode(role="folder", success=False) == Quadrant.FALSE_POSITIVE

    def test_disturber_classified_as_destroyer(self):
        assert classify_episode(role="disturber", success=True) == Quadrant.TRUE_NEGATIVE
        assert classify_episode(role="disturber", success=False) == Quadrant.FALSE_NEGATIVE

    def test_load_with_folder_disturber_roles(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        (meta_dir / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "role": "folder", "success": True})
            + "\n"
            + json.dumps({"episode_index": 1, "role": "disturber", "success": True})
            + "\n"
        )
        episodes = load_episode_metadata(tmp_path)
        assert len(episodes) == 2
        assert episodes[0].role == "builder"
        assert episodes[0].quadrant == Quadrant.TRUE_POSITIVE
        assert episodes[1].role == "destroyer"
        assert episodes[1].quadrant == Quadrant.TRUE_NEGATIVE


# ---------------------------------------------------------------------------
# Test: split_by_quadrant
# ---------------------------------------------------------------------------


class TestSplitByQuadrant:
    def test_splits_correctly(self, metadata_file):
        episodes = load_episode_metadata(metadata_file)
        split = split_by_quadrant(episodes)
        assert len(split[Quadrant.TRUE_POSITIVE]) == 1
        assert len(split[Quadrant.TRUE_NEGATIVE]) == 1
        assert len(split[Quadrant.FALSE_POSITIVE]) == 1
        assert len(split[Quadrant.FALSE_NEGATIVE]) == 1

    def test_empty_input(self):
        split = split_by_quadrant([])
        for q in Quadrant:
            assert split[q] == []


# ---------------------------------------------------------------------------
# Test: construct_ideal_target
# ---------------------------------------------------------------------------


class TestConstructIdealTarget:
    def test_aligned_goes_minus_one_to_zero(self):
        T = 100
        target = construct_ideal_target(T, aligned=True)
        assert target.shape == (T,)
        assert target[0] == pytest.approx(-1.0)
        assert target[-1] == pytest.approx(0.0)
        assert np.all(np.diff(target) >= 0)

    def test_misaligned_goes_zero_to_minus_one(self):
        T = 100
        target = construct_ideal_target(T, aligned=False)
        assert target.shape == (T,)
        assert target[0] == pytest.approx(0.0)
        assert target[-1] == pytest.approx(-1.0)
        assert np.all(np.diff(target) <= 0)

    def test_single_step(self):
        target_aligned = construct_ideal_target(1, aligned=True)
        assert target_aligned.shape == (1,)
        assert target_aligned[0] == pytest.approx(-1.0)
        target_misaligned = construct_ideal_target(1, aligned=False)
        assert target_misaligned[0] == pytest.approx(0.0)

    def test_linearity(self):
        T = 50
        np.testing.assert_allclose(construct_ideal_target(T, aligned=True), np.linspace(-1.0, 0.0, T))
        np.testing.assert_allclose(construct_ideal_target(T, aligned=False), np.linspace(0.0, -1.0, T))

    def test_aligned_keyword_only(self):
        """`aligned` is keyword-only — guards against positional confusion at call sites."""
        with pytest.raises(TypeError):
            construct_ideal_target(10, True)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Test: infer_role_from_task
# ---------------------------------------------------------------------------


class TestInferRole:
    def test_infer_role_fold_task(self):
        assert infer_role_from_task("fold the shirt neatly") == "builder"

    def test_infer_role_collar_task(self):
        assert infer_role_from_task("adjust the collar") == "builder"

    def test_infer_role_lay_task(self):
        assert infer_role_from_task("lay flat on the table") == "builder"

    def test_infer_role_disarrange_task(self):
        assert infer_role_from_task("disarrange the clothes") == "destroyer"

    def test_infer_role_messy_task(self):
        assert infer_role_from_task("make it messy") == "destroyer"

    def test_infer_role_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot infer role"):
            infer_role_from_task("pick up the box")

    def test_infer_role_case_insensitive(self):
        assert infer_role_from_task("FOLD the Collar") == "builder"
        assert infer_role_from_task("DISARRANGE everything") == "destroyer"


# ---------------------------------------------------------------------------
# Test: load_episode_metadata with missing role (inferred from task)
# ---------------------------------------------------------------------------


class TestLoadMetadataMissingRole:
    def test_load_metadata_missing_role_infers_from_task(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        record = {
            "episode_index": 0,
            "tasks": ["fold the shirt neatly", "lay it flat"],
            "success": True,
        }
        (meta_dir / "episodes.jsonl").write_text(json.dumps(record) + "\n")
        episodes = load_episode_metadata(tmp_path)
        assert len(episodes) == 1
        assert episodes[0].role == "builder"
        assert episodes[0].label_source == "inferred"

    def test_load_metadata_missing_role_destroyer_task(self, tmp_path):
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        record = {
            "episode_index": 0,
            "tasks": ["disarrange the pile"],
            "success": False,
        }
        (meta_dir / "episodes.jsonl").write_text(json.dumps(record) + "\n")
        episodes = load_episode_metadata(tmp_path)
        assert episodes[0].role == "destroyer"
        assert episodes[0].label_source == "inferred"


# ---------------------------------------------------------------------------
# Test: label_source field
# ---------------------------------------------------------------------------


class TestLabelSource:
    def test_label_source_metadata_default(self, tmp_path):
        """When role and success are explicit, label_source is 'metadata'."""
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        record = {
            "episode_index": 0,
            "role": "builder",
            "success": True,
        }
        (meta_dir / "episodes.jsonl").write_text(json.dumps(record) + "\n")
        episodes = load_episode_metadata(tmp_path)
        assert episodes[0].label_source == "metadata"

    def test_label_source_inferred_from_task(self, tmp_path):
        """When role is inferred from task, label_source is 'inferred'."""
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()
        record = {
            "episode_index": 0,
            "tasks": ["fold the shirt"],
            "success": True,
        }
        (meta_dir / "episodes.jsonl").write_text(json.dumps(record) + "\n")
        episodes = load_episode_metadata(tmp_path)
        assert episodes[0].label_source == "inferred"

    def test_existing_episodes_have_label_source_metadata(self, metadata_file):
        """All episodes from the standard fixture should have label_source='metadata'."""
        episodes = load_episode_metadata(metadata_file)
        for ep in episodes:
            assert ep.label_source == "metadata"
