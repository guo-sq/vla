"""ValueReturnsPreprocessor unit tests.

Tests for a FrameAttributeProcessor that classifies episodes as positive/negative
and assigns episode_boundary and prompt_map. Does NOT compute returns (that is
done by compute_episode_returns in rl_dataset.py).
"""

from __future__ import annotations

from pathlib import Path
import textwrap
from unittest.mock import Mock

import numpy as np
import pytest

from openpi.training.frame_attributes_preprocessors.base import EXTRA_EPISODE_PROMPT_MAP
from openpi.training.frame_attributes_preprocessors.base import DatasetContext
from openpi.training.frame_attributes_preprocessors.base import EpisodeBoundary
from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.value_returns_preprocessor import EpisodeClass
from openpi.training.frame_attributes_preprocessors.value_returns_preprocessor import ValueReturnsPreprocessor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_dataset(num_frames: int) -> Mock:
    """Create a minimal mock HuggingFace Dataset with *num_frames* rows."""
    ds = Mock()
    ds.__len__ = Mock(return_value=num_frames)
    return ds


def _make_meta(episodes: dict[int, dict]) -> Mock:
    """Create a mock LeRobotDatasetMeta with the given episode metadata."""
    meta = Mock()
    meta.episodes = episodes
    return meta


def _make_ctx(
    episode_lengths: list[int],
    episodes_meta: dict[int, dict],
    repo_id: str = "test_repo",
    root: str | None = None,
) -> DatasetContext:
    """Build a DatasetContext from a list of per-episode frame counts."""
    total = sum(episode_lengths)
    from_indices: list[int] = []
    to_indices: list[int] = []
    offset = 0
    for length in episode_lengths:
        from_indices.append(offset)
        to_indices.append(offset + length)
        offset += length

    return DatasetContext(
        repo_id=repo_id,
        hf_dataset=_make_hf_dataset(total),
        episode_data_index={"from": from_indices, "to": to_indices},
        meta=_make_meta(episodes_meta),
        delta_indices=None,
        root=root,
    )


def _write_yaml(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBothConfirmed:
    """state_confirmation='both': episode_boundary=BOTH_CONFIRMED."""

    def test_both_confirmed_positive(self, tmp_path: Path):
        """Positive episode with both ends confirmed: episode_boundary=BOTH_CONFIRMED, not negative."""
        num_frames = 100
        yaml_path = tmp_path / "classification.yaml"
        _write_yaml(
            yaml_path,
            """\
            positive_datasets:
              - "test_repo"
            negative_datasets: []
            default_type: "positive"
            """,
        )

        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["hang_clothes"], "role": "builder", "success": True}},
            repo_id="test_repo",
            root=str(tmp_path),
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="dataset",
            state_confirmation="both",
            dataset_type_config_path=str(yaml_path),
        )
        proc(ctx, attrs)

        # Classification outputs
        assert attrs.is_negative_episode is not None
        assert not np.any(attrs.is_negative_episode)
        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.BOTH_CONFIRMED)

    def test_both_confirmed_negative(self, tmp_path: Path):
        """Negative episode with both ends confirmed."""
        num_frames = 100
        yaml_path = tmp_path / "classification.yaml"
        _write_yaml(
            yaml_path,
            """\
            positive_datasets: []
            negative_datasets:
              - "test_repo"
            default_type: "negative"
            """,
        )

        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": True}},
            repo_id="test_repo",
            root=str(tmp_path),
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="dataset",
            state_confirmation="both",
            dataset_type_config_path=str(yaml_path),
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert np.all(attrs.is_negative_episode)
        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.BOTH_CONFIRMED)


class TestPerRoleEpisodeBoundary:
    """Per-role episode boundary: auto + episode mode assigns BOTH to builder, END to destroyer."""

    def test_mixed_roles_get_different_boundaries(self):
        """Builder→BOTH_CONFIRMED, destroyer→END_CONFIRMED, builder-failure→UNCONFIRMED_NEGATIVE_END."""
        ep_lens = [40, 50, 30]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["take_off"], "role": "destroyer", "success": True},
                2: {"tasks": ["hang"], "role": "builder", "success": False},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="auto",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        # builder success → BOTH_CONFIRMED
        np.testing.assert_array_equal(attrs.episode_boundary[:40], EpisodeBoundary.BOTH_CONFIRMED)
        # destroyer success → END_CONFIRMED
        np.testing.assert_array_equal(attrs.episode_boundary[40:90], EpisodeBoundary.END_CONFIRMED)
        # builder failure → UNCONFIRMED_NEGATIVE_END (FP)
        np.testing.assert_array_equal(attrs.episode_boundary[90:120], EpisodeBoundary.UNCONFIRMED_NEGATIVE_END)

    def test_explicit_both_overrides_per_role(self):
        """state_confirmation='both' should give BOTH_CONFIRMED to all, ignoring role."""
        ep_lens = [40, 50]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["take_off"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="both",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary[:40], EpisodeBoundary.BOTH_CONFIRMED)
        np.testing.assert_array_equal(attrs.episode_boundary[40:90], EpisodeBoundary.BOTH_CONFIRMED)

    def test_explicit_end_only_overrides_per_role(self):
        """state_confirmation='end_only' should give END_CONFIRMED to all, ignoring role."""
        ep_lens = [40, 50]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["take_off"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary[:40], EpisodeBoundary.END_CONFIRMED)
        np.testing.assert_array_equal(attrs.episode_boundary[40:90], EpisodeBoundary.END_CONFIRMED)

    def test_no_role_metadata_defaults_to_builder(self):
        """Episodes without 'role' metadata should default to builder→BOTH_CONFIRMED."""
        ctx = _make_ctx(
            episode_lengths=[50],
            episodes_meta={0: {"tasks": ["hang"], "success": True}},  # no 'role'
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="auto",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.BOTH_CONFIRMED)


class TestEndOnlyConfirmed:
    """state_confirmation='end_only': episode_boundary=END_CONFIRMED."""

    def test_end_only_positive_tp(self):
        """Builder success (true positive): END_CONFIRMED, not negative."""
        num_frames = 100

        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["hang_clothes"], "role": "builder", "success": True}},
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert not np.any(attrs.is_negative_episode)
        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.END_CONFIRMED)

    def test_end_only_negative_tn(self):
        """Destroyer success (true negative): END_CONFIRMED, is_negative."""
        num_frames = 100

        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": True}},
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert np.all(attrs.is_negative_episode)
        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.END_CONFIRMED)


class TestFailureCases:
    """Failure episodes: episode_boundary split into UNCONFIRMED_NEGATIVE_END / UNCONFIRMED_POSITIVE_END."""

    def test_failure_fp(self):
        """Builder failure (FP): UNCONFIRMED_NEGATIVE_END boundary (GT=-1 heuristic)."""
        num_frames = 100
        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["hang_clothes"], "role": "builder", "success": False}},
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.UNCONFIRMED_NEGATIVE_END)

    def test_failure_fn(self):
        """Destroyer failure (FN): UNCONFIRMED_POSITIVE_END boundary (GT=0 heuristic)."""
        num_frames = 80
        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": False}},
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.UNCONFIRMED_POSITIVE_END)


class TestEpisodeClassification:
    """Episode-level classification: builder=positive, destroyer=negative."""

    def test_episode_classification(self):
        """Two episodes: builder (positive) and destroyer (negative) correctly classified."""
        ep0_len = 50  # builder
        ep1_len = 60  # destroyer

        ctx = _make_ctx(
            episode_lengths=[ep0_len, ep1_len],
            episodes_meta={
                0: {"tasks": ["hang_clothes"], "role": "builder", "success": True},
                1: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert len(attrs.is_negative_episode) == ep0_len + ep1_len
        assert not np.any(attrs.is_negative_episode[:ep0_len]), "builder episode should not be negative"
        assert np.all(attrs.is_negative_episode[ep0_len:]), "destroyer episode should be negative"


class TestEpisodeBoundaryAssignment:
    """Verify episode_boundary field is correctly assigned per episode."""

    def test_episode_boundary_assignment(self):
        """3 episodes: builder success, destroyer success, builder failure."""
        ep_lens = [40, 50, 30]

        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang_clothes"], "role": "builder", "success": True},
                1: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": True},
                2: {"tasks": ["hang_clothes"], "role": "builder", "success": False},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        total = sum(ep_lens)
        assert len(attrs.episode_boundary) == total

        # Episode 0 (builder success, end_only) -> END_CONFIRMED (2)
        np.testing.assert_array_equal(attrs.episode_boundary[:40], EpisodeBoundary.END_CONFIRMED)
        # Episode 1 (destroyer success, end_only) -> END_CONFIRMED (2)
        np.testing.assert_array_equal(attrs.episode_boundary[40:90], EpisodeBoundary.END_CONFIRMED)
        # Episode 2 (builder failure = FP) -> UNCONFIRMED_NEGATIVE_END
        np.testing.assert_array_equal(attrs.episode_boundary[90:120], EpisodeBoundary.UNCONFIRMED_NEGATIVE_END)


class TestEpisodePromptMap:
    """Verify ctx.extras[EXTRA_EPISODE_PROMPT_MAP] is populated."""

    def test_episode_prompt_map(self):
        """Builder maps to positive_prompt, destroyer maps to negative_prompt."""
        ctx = _make_ctx(
            episode_lengths=[30, 40],
            episodes_meta={
                0: {"tasks": ["hang_clothes"], "role": "builder", "success": True},
                1: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
        )
        proc(ctx, attrs)

        prompt_map = ctx.extras.get(EXTRA_EPISODE_PROMPT_MAP)
        assert prompt_map is not None, "EXTRA_EPISODE_PROMPT_MAP should be set in ctx.extras"
        assert isinstance(prompt_map, dict)
        assert 0 in prompt_map
        assert prompt_map[0] == proc.positive_prompt
        assert 1 in prompt_map
        assert prompt_map[1] == proc.negative_prompt
        assert prompt_map[0] != prompt_map[1], "positive and negative prompts should differ"


class TestAutoStateConfirmation:
    """state_confirmation='auto' should resolve based on classification_mode and role."""

    def test_auto_with_episode_mode_builder_resolves_to_both(self):
        """auto + episode mode + builder -> auto_per_role -> BOTH_CONFIRMED."""
        num_frames = 50
        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["hang_clothes"], "role": "builder", "success": True}},
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="auto",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.BOTH_CONFIRMED)

    def test_auto_with_episode_mode_destroyer_resolves_to_end(self):
        """auto + episode mode + destroyer -> auto_per_role -> END_CONFIRMED."""
        num_frames = 50
        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": True}},
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="auto",
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.END_CONFIRMED)

    def test_auto_with_dataset_mode_resolves_to_both(self, tmp_path: Path):
        """auto + classification_mode='dataset' -> both -> BOTH_CONFIRMED."""
        num_frames = 50
        yaml_path = tmp_path / "classification.yaml"
        _write_yaml(
            yaml_path,
            """\
            positive_datasets:
              - "test_repo"
            negative_datasets: []
            default_type: "positive"
            """,
        )

        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["hang_clothes"], "role": "builder", "success": True}},
            repo_id="test_repo",
            root=str(tmp_path),
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="dataset",
            state_confirmation="auto",
            dataset_type_config_path=str(yaml_path),
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.BOTH_CONFIRMED)


class TestStateConfirmationByRole:
    """state_confirmation_by_role: configurable per-role boundary mapping."""

    def test_builder_both_destroyer_end(self):
        """builder→BOTH_CONFIRMED, destroyer→END_CONFIRMED via config dict."""
        ep_lens = [40, 50, 30]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["take_off"], "role": "destroyer", "success": True},
                2: {"tasks": ["hang"], "role": "builder", "success": False},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation_by_role={"builder": "both", "destroyer": "end_only"},
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary[:40], EpisodeBoundary.BOTH_CONFIRMED)
        np.testing.assert_array_equal(attrs.episode_boundary[40:90], EpisodeBoundary.END_CONFIRMED)
        np.testing.assert_array_equal(attrs.episode_boundary[90:120], EpisodeBoundary.UNCONFIRMED_NEGATIVE_END)

    def test_all_both_via_by_role(self):
        """Both roles set to 'both' → all success episodes get BOTH_CONFIRMED."""
        ep_lens = [40, 50]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["take_off"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation_by_role={"builder": "both", "destroyer": "both"},
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary[:40], EpisodeBoundary.BOTH_CONFIRMED)
        np.testing.assert_array_equal(attrs.episode_boundary[40:90], EpisodeBoundary.BOTH_CONFIRMED)

    def test_unknown_role_falls_back_to_global(self):
        """Role not in dict falls back to state_confirmation global value."""
        ctx = _make_ctx(
            episode_lengths=[50],
            episodes_meta={0: {"tasks": ["task"], "role": "observer", "success": True}},
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
            state_confirmation_by_role={"builder": "both"},  # no "observer"
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.END_CONFIRMED)

    def test_by_role_overrides_auto(self):
        """state_confirmation_by_role takes precedence over auto resolution."""
        ep_lens = [40, 50]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["take_off"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="auto",
            state_confirmation_by_role={"builder": "end_only", "destroyer": "both"},
        )
        proc(ctx, attrs)

        assert attrs.episode_boundary is not None
        # Reversed from default: builder=END, destroyer=BOTH
        np.testing.assert_array_equal(attrs.episode_boundary[:40], EpisodeBoundary.END_CONFIRMED)
        np.testing.assert_array_equal(attrs.episode_boundary[40:90], EpisodeBoundary.BOTH_CONFIRMED)


class TestNegativeRoles:
    """negative_roles: configurable role→classification mapping."""

    def test_custom_negative_role(self):
        """Custom negative_roles=['attacker'] classifies 'attacker' as negative."""
        ep_lens = [40, 50]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["build"], "role": "helper", "success": True},
                1: {"tasks": ["attack"], "role": "attacker", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="both",
            negative_roles=["attacker"],
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert not np.any(attrs.is_negative_episode[:40]), "helper should be positive"
        assert np.all(attrs.is_negative_episode[40:90]), "attacker should be negative"

    def test_default_negative_roles_backward_compatible(self):
        """Default negative_roles=['destroyer'] maintains backward compatibility."""
        ep_lens = [40, 50]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["take_off"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="both",
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert not np.any(attrs.is_negative_episode[:40]), "builder should be positive"
        assert np.all(attrs.is_negative_episode[40:90]), "destroyer should be negative"

    def test_multiple_negative_roles(self):
        """Multiple roles can be marked as negative."""
        ep_lens = [30, 40, 50]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["build"], "role": "builder", "success": True},
                1: {"tasks": ["destroy"], "role": "destroyer", "success": True},
                2: {"tasks": ["sabotage"], "role": "saboteur", "success": True},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="both",
            negative_roles=["destroyer", "saboteur"],
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert not np.any(attrs.is_negative_episode[:30]), "builder should be positive"
        assert np.all(attrs.is_negative_episode[30:70]), "destroyer should be negative"
        assert np.all(attrs.is_negative_episode[70:120]), "saboteur should be negative"


class TestAutoClassificationMode:
    """classification_mode='auto': detect episode vs dataset mode by checking role metadata."""

    def test_auto_detects_episode_mode_when_role_present(self, tmp_path: Path):
        """When episodes have 'role' metadata, auto should use episode classification + end_only."""
        ep0_len = 50  # builder
        ep1_len = 60  # destroyer

        yaml_path = tmp_path / "classification.yaml"
        _write_yaml(
            yaml_path,
            """\
            positive_datasets:
              - "test_repo"
            negative_datasets: []
            default_type: "positive"
            """,
        )

        ctx = _make_ctx(
            episode_lengths=[ep0_len, ep1_len],
            episodes_meta={
                0: {"tasks": ["hang_clothes"], "role": "builder", "success": True},
                1: {"tasks": ["take_off_clothes"], "role": "destroyer", "success": True},
            },
            repo_id="test_repo",
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="auto",
            dataset_type_config_path=str(yaml_path),
            state_confirmation="auto",
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert not np.any(attrs.is_negative_episode[:ep0_len])
        assert np.all(attrs.is_negative_episode[ep0_len:])

        # auto + episode -> auto_per_role: builder=BOTH_CONFIRMED, destroyer=END_CONFIRMED
        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary[:ep0_len], EpisodeBoundary.BOTH_CONFIRMED)
        np.testing.assert_array_equal(attrs.episode_boundary[ep0_len:], EpisodeBoundary.END_CONFIRMED)

    def test_auto_detects_dataset_mode_when_no_role(self, tmp_path: Path):
        """When episodes lack 'role' metadata, auto should use dataset classification + both."""
        num_frames = 50

        yaml_path = tmp_path / "classification.yaml"
        _write_yaml(
            yaml_path,
            """\
            positive_datasets:
              - "seatbelt.single.hang.*"
            negative_datasets:
              - "seatbelt.single.take_off*"
            default_type: "positive"
            """,
        )

        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["hang_clothes"]}},  # no 'role' key
            repo_id="seatbelt.single.hang.test.20260205.batch.1",
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="auto",
            dataset_type_config_path=str(yaml_path),
            state_confirmation="auto",
        )
        proc(ctx, attrs)

        # episode_boundary should be BOTH_CONFIRMED (3)
        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.BOTH_CONFIRMED)

    def test_auto_negative_dataset_no_role(self, tmp_path: Path):
        """Auto mode with negative dataset pattern and no role -> dataset mode + both + negative."""
        num_frames = 40

        yaml_path = tmp_path / "classification.yaml"
        _write_yaml(
            yaml_path,
            """\
            positive_datasets:
              - "seatbelt.single.hang.*"
            negative_datasets:
              - "seatbelt.single.take_off*"
            default_type: "positive"
            """,
        )

        ctx = _make_ctx(
            episode_lengths=[num_frames],
            episodes_meta={0: {"tasks": ["take_off_clothes"]}},  # no 'role'
            repo_id="seatbelt.single.take_off_move.test.20260228.batch.1",
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="auto",
            dataset_type_config_path=str(yaml_path),
            state_confirmation="auto",
        )
        proc(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert np.all(attrs.is_negative_episode)
        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.BOTH_CONFIRMED)


class TestExcludeFailures:
    """exclude_failures=True writes valid_mask=False for failure episodes.

    This is the primary defence against the v3 collapse bug: UNCONFIRMED failure
    episodes contribute GT=-1 to every frame, dragging the value model to
    predict -1 everywhere. Dropping them at the valid_mask layer keeps
    rl_dataset.calc_episode from ever surfacing them to the training loop.
    """

    def test_exclude_failures_sets_valid_mask_false_on_failure_frames(self):
        """Success episode kept, failure episodes marked invalid."""
        ctx = _make_ctx(
            episode_lengths=[40, 30, 40],
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["hang"], "role": "builder", "success": False},
                2: {"tasks": ["hang"], "role": "destroyer", "success": False},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation_by_role={"builder": "both", "destroyer": "end_only"},
            exclude_failures=True,
        )
        proc(ctx, attrs)

        assert attrs.valid_mask is not None
        assert attrs.valid_mask.dtype == bool
        # success episode kept
        assert attrs.valid_mask[0:40].all()
        # builder failure dropped
        assert not attrs.valid_mask[40:70].any()
        # destroyer failure dropped
        assert not attrs.valid_mask[70:110].any()

    def test_exclude_failures_false_leaves_valid_mask_untouched(self):
        """Default exclude_failures=False must NOT touch valid_mask."""
        ctx = _make_ctx(
            episode_lengths=[40, 30],
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["hang"], "role": "builder", "success": False},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation_by_role={"builder": "both", "destroyer": "end_only"},
            exclude_failures=False,
        )
        proc(ctx, attrs)

        # valid_mask either untouched (None) or all-True; never partially masked by this proc
        assert attrs.valid_mask is None or attrs.valid_mask.all()

    def test_exclude_failures_only_masks_failures_not_negative_roles(self):
        """Destroyer success episode is negative but still valid; only success=False drops frames."""
        ctx = _make_ctx(
            episode_lengths=[40, 40, 30],
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["hang"], "role": "destroyer", "success": True},
                2: {"tasks": ["hang"], "role": "destroyer", "success": False},
            },
        )
        attrs = FrameAttributes()

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation_by_role={"builder": "both", "destroyer": "end_only"},
            exclude_failures=True,
        )
        proc(ctx, attrs)

        assert attrs.valid_mask is not None
        # builder success kept
        assert attrs.valid_mask[0:40].all()
        # destroyer success kept (negative ≠ failure)
        assert attrs.valid_mask[40:80].all()
        # destroyer failure dropped
        assert not attrs.valid_mask[80:110].any()
        # negative classification is orthogonal: destroyer success still gets is_negative=True
        assert attrs.is_negative_episode is not None
        assert attrs.is_negative_episode[40:80].all()

    def test_exclude_failures_preserves_existing_valid_mask(self):
        """When a prior preprocessor has already written valid_mask, exclude_failures must
        only flip failure frames to False and preserve all other entries unchanged.

        Simulates a pipeline like [GripperCountPreprocessor, ValueReturnsPreprocessor]:
        the upstream processor masks out some non-failure frames (e.g. wrong gripper
        count), then ValueReturnsPreprocessor must not reset those bits when dropping
        failure episodes.
        """
        ep_lens = [40, 30, 40]  # ep0 success, ep1 failure, ep2 success
        total = sum(ep_lens)
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["hang"], "role": "builder", "success": False},
                2: {"tasks": ["hang"], "role": "destroyer", "success": True},
            },
        )
        attrs = FrameAttributes()
        # Upstream processor: mask out frames 5..10 in ep0 and 85..95 in ep2 (non-failure)
        attrs.valid_mask = np.ones(total, dtype=bool)
        attrs.valid_mask[5:10] = False
        attrs.valid_mask[85:95] = False

        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation_by_role={"builder": "both", "destroyer": "end_only"},
            exclude_failures=True,
        )
        proc(ctx, attrs)

        assert attrs.valid_mask is not None
        # Upstream-masked frames in success episodes are still False
        assert not attrs.valid_mask[5:10].any()
        assert not attrs.valid_mask[85:95].any()
        # Other success frames in ep0 remain True
        assert attrs.valid_mask[0:5].all()
        assert attrs.valid_mask[10:40].all()
        # Failure episode (ep1) entirely dropped regardless of upstream state
        assert not attrs.valid_mask[40:70].any()
        # Other success frames in ep2 remain True
        assert attrs.valid_mask[70:85].all()
        assert attrs.valid_mask[95:110].all()


# ---------------------------------------------------------------------------
# Part 2: FAILURE_FP / FAILURE_FN split (Red phase tests)
# ---------------------------------------------------------------------------


class TestFailureSubclasses:
    """Part 2 Red phase: EpisodeClass.FAILURE is being split into FAILURE_FP
    (builder-failure, visual terminal state typically NOT at target → GT=-1)
    and FAILURE_FN (destroyer-failure, visual terminal state typically AT
    target because 'pull failed' → GT=0).

    Heuristic approximation only — see PR description C1/C5 limitation.
    """

    def _require_new_enums(self):
        if not hasattr(EpisodeClass, "FAILURE_FP"):
            pytest.fail("EpisodeClass.FAILURE_FP not yet defined (Red phase)")
        if not hasattr(EpisodeClass, "FAILURE_FN"):
            pytest.fail("EpisodeClass.FAILURE_FN not yet defined (Red phase)")
        if not hasattr(EpisodeBoundary, "UNCONFIRMED_NEGATIVE_END"):
            pytest.fail("EpisodeBoundary.UNCONFIRMED_NEGATIVE_END not yet defined (Red phase)")
        if not hasattr(EpisodeBoundary, "UNCONFIRMED_POSITIVE_END"):
            pytest.fail("EpisodeBoundary.UNCONFIRMED_POSITIVE_END not yet defined (Red phase)")

    def test_builder_failure_boundary_is_unconfirmed_negative_end(self):
        """Builder failure → UNCONFIRMED_NEGATIVE_END (FP class, GT=-1)."""
        self._require_new_enums()
        ctx = _make_ctx(
            episode_lengths=[60],
            episodes_meta={0: {"tasks": ["hang"], "role": "builder", "success": False}},
        )
        attrs = FrameAttributes()
        ValueReturnsPreprocessor(classification_mode="episode", state_confirmation="end_only")(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.UNCONFIRMED_NEGATIVE_END)

    def test_destroyer_failure_boundary_is_unconfirmed_positive_end(self):
        """Destroyer failure → UNCONFIRMED_POSITIVE_END (FN class, GT=0)."""
        self._require_new_enums()
        ctx = _make_ctx(
            episode_lengths=[60],
            episodes_meta={0: {"tasks": ["take_off"], "role": "destroyer", "success": False}},
        )
        attrs = FrameAttributes()
        ValueReturnsPreprocessor(classification_mode="episode", state_confirmation="end_only")(ctx, attrs)

        assert attrs.episode_boundary is not None
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.UNCONFIRMED_POSITIVE_END)

    def test_builder_failure_is_negative_false(self):
        """FP (builder failure) must keep is_negative_episode=False so the GT=-1
        branch is taken in compute_episode_returns (C4 assert aligns with this)."""
        self._require_new_enums()
        ctx = _make_ctx(
            episode_lengths=[40],
            episodes_meta={0: {"tasks": ["hang"], "role": "builder", "success": False}},
        )
        attrs = FrameAttributes()
        ValueReturnsPreprocessor(classification_mode="episode", state_confirmation="end_only")(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert not np.any(attrs.is_negative_episode)

    def test_destroyer_failure_is_negative_true(self):
        """FN (destroyer failure) must set is_negative_episode=True so the GT=0
        branch is taken in compute_episode_returns (C4 assert aligns with this)."""
        self._require_new_enums()
        ctx = _make_ctx(
            episode_lengths=[40],
            episodes_meta={0: {"tasks": ["take_off"], "role": "destroyer", "success": False}},
        )
        attrs = FrameAttributes()
        ValueReturnsPreprocessor(classification_mode="episode", state_confirmation="end_only")(ctx, attrs)

        assert attrs.is_negative_episode is not None
        assert np.all(attrs.is_negative_episode)

    def test_classify_episodes_by_role_returns_FP_for_builder_failure(self):  # noqa: N802
        """_classify_episodes_by_role must emit FAILURE_FP for builder failure."""
        self._require_new_enums()
        ctx = _make_ctx(
            episode_lengths=[20],
            episodes_meta={0: {"tasks": ["hang"], "role": "builder", "success": False}},
        )
        proc = ValueReturnsPreprocessor(classification_mode="episode")
        classes = proc._classify_episodes_by_role(ctx, num_episodes=1)  # noqa: SLF001
        assert classes == [EpisodeClass.FAILURE_FP]

    def test_classify_episodes_by_role_returns_FN_for_destroyer_failure(self):  # noqa: N802
        """_classify_episodes_by_role must emit FAILURE_FN for destroyer failure."""
        self._require_new_enums()
        ctx = _make_ctx(
            episode_lengths=[20],
            episodes_meta={0: {"tasks": ["take_off"], "role": "destroyer", "success": False}},
        )
        proc = ValueReturnsPreprocessor(classification_mode="episode")
        classes = proc._classify_episodes_by_role(ctx, num_episodes=1)  # noqa: SLF001
        assert classes == [EpisodeClass.FAILURE_FN]

    def test_mixed_FP_FN_in_same_batch(self):  # noqa: N802
        """Mixed batch: builder success / destroyer success / builder failure / destroyer failure.

        The two failure rows must receive the NEW split boundaries, NOT the
        legacy single UNCONFIRMED value.
        """
        self._require_new_enums()
        ep_lens = [30, 40, 20, 25]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["take_off"], "role": "destroyer", "success": True},
                2: {"tasks": ["hang"], "role": "builder", "success": False},
                3: {"tasks": ["take_off"], "role": "destroyer", "success": False},
            },
        )
        attrs = FrameAttributes()
        ValueReturnsPreprocessor(classification_mode="episode", state_confirmation="end_only")(ctx, attrs)

        assert attrs.episode_boundary is not None
        # ep0 builder success → END_CONFIRMED (unchanged)
        np.testing.assert_array_equal(attrs.episode_boundary[:30], EpisodeBoundary.END_CONFIRMED)
        # ep1 destroyer success → END_CONFIRMED (unchanged)
        np.testing.assert_array_equal(attrs.episode_boundary[30:70], EpisodeBoundary.END_CONFIRMED)
        # ep2 builder failure → UNCONFIRMED_NEGATIVE_END
        np.testing.assert_array_equal(attrs.episode_boundary[70:90], EpisodeBoundary.UNCONFIRMED_NEGATIVE_END)
        # ep3 destroyer failure → UNCONFIRMED_POSITIVE_END
        np.testing.assert_array_equal(attrs.episode_boundary[90:115], EpisodeBoundary.UNCONFIRMED_POSITIVE_END)

        # is_negative_episode: ep0 False, ep1 True, ep2 False (FP), ep3 True (FN)
        assert attrs.is_negative_episode is not None
        assert not attrs.is_negative_episode[:30].any()
        assert attrs.is_negative_episode[30:70].all()
        assert not attrs.is_negative_episode[70:90].any()
        assert attrs.is_negative_episode[90:115].all()

    def test_dataset_mode_never_produces_failure(self, tmp_path: Path):
        """Dataset mode ignores the 'success' field — classification is
        purely repo_id-based, so failure subclasses must not surface."""
        self._require_new_enums()
        yaml_path = tmp_path / "classification.yaml"
        _write_yaml(
            yaml_path,
            """\
            positive_datasets:
              - "test_repo"
            negative_datasets: []
            default_type: "positive"
            """,
        )
        ctx = _make_ctx(
            episode_lengths=[30],
            episodes_meta={0: {"tasks": ["hang"], "role": "builder", "success": False}},
            repo_id="test_repo",
            root=str(tmp_path),
        )
        attrs = FrameAttributes()
        ValueReturnsPreprocessor(
            classification_mode="dataset",
            state_confirmation="both",
            dataset_type_config_path=str(yaml_path),
        )(ctx, attrs)

        assert attrs.episode_boundary is not None
        # No failure subclass anywhere — every frame is BOTH_CONFIRMED.
        np.testing.assert_array_equal(attrs.episode_boundary, EpisodeBoundary.BOTH_CONFIRMED)

    def test_exclude_failures_drops_both_FP_and_FN(self):  # noqa: N802
        """exclude_failures=True must mask both failure subclasses."""
        self._require_new_enums()
        ep_lens = [30, 20, 25]
        ctx = _make_ctx(
            episode_lengths=ep_lens,
            episodes_meta={
                0: {"tasks": ["hang"], "role": "builder", "success": True},
                1: {"tasks": ["hang"], "role": "builder", "success": False},  # FP
                2: {"tasks": ["take_off"], "role": "destroyer", "success": False},  # FN
            },
        )
        attrs = FrameAttributes()
        ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
            exclude_failures=True,
        )(ctx, attrs)

        assert attrs.valid_mask is not None
        assert attrs.valid_mask[0:30].all()  # success kept
        assert not attrs.valid_mask[30:50].any()  # FP dropped
        assert not attrs.valid_mask[50:75].any()  # FN dropped

    def test_custom_negative_roles_drive_FP_FN_split(self):  # noqa: N802
        """The FP/FN split must honour custom negative_roles, not hard-code 'destroyer'."""
        self._require_new_enums()
        ctx = _make_ctx(
            episode_lengths=[20, 20],
            episodes_meta={
                0: {"tasks": ["a"], "role": "helper", "success": False},  # custom role → FP
                1: {"tasks": ["b"], "role": "saboteur", "success": False},  # custom negative → FN
            },
        )
        proc = ValueReturnsPreprocessor(
            classification_mode="episode",
            state_confirmation="end_only",
            negative_roles=["saboteur"],
        )
        classes = proc._classify_episodes_by_role(ctx, num_episodes=2)  # noqa: SLF001
        assert classes == [EpisodeClass.FAILURE_FP, EpisodeClass.FAILURE_FN]
