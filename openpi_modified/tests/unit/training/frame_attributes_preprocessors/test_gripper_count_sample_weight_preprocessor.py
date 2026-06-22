"""GripperCountSampleWeightPreprocessor unit tests.

Gripper convention: 0 = fully open, high value = fully closed.
open event: value drops below open_threshold.
close event: value rises above close_threshold.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import (
    GripperCountSampleWeightPreprocessor,
)
from openpi.training.frame_attributes_preprocessors.sample_weight_preprocessor import GripperCountSampleWeightRule

# Thresholds: open when < 0.5, closed when > 3.0
OPEN_THRE = 0.5
CLOSE_THRE = 3.0

# Canonical values
CLOSED = 3.5  # above close_threshold → closed
OPEN = 0.0  # below open_threshold → open


def _make_ctx(repo_id: str, states: np.ndarray, episode_boundaries: list[tuple[int, int]] | None = None):
    """Build a mock DatasetContext."""
    n = len(states)
    ctx = Mock()
    ctx.repo_id = repo_id
    ctx.robot_type = "arxx5_bimanual"
    ctx.extras = {}

    hf = Mock()
    hf.__len__ = Mock(return_value=n)
    hf.__getitem__ = Mock(side_effect=lambda key: states.tolist() if key == "observation.state" else None)
    hf.features = {}
    ctx.hf_dataset = hf

    if episode_boundaries is None:
        episode_boundaries = [(0, n)]
    ctx.episode_data_index = {
        "from": [s for s, _ in episode_boundaries],
        "to": [e for _, e in episode_boundaries],
    }
    return ctx


def _make_14d_states(n: int, left_gripper: np.ndarray | None = None, right_gripper: np.ndarray | None = None):
    """Build (n, 14) state array. Default gripper value is CLOSED (3.5)."""
    states = np.full((n, 14), 1.75, dtype=np.float32)
    states[:, 6] = CLOSED
    states[:, 13] = CLOSED
    if left_gripper is not None:
        states[:, 6] = left_gripper
    if right_gripper is not None:
        states[:, 13] = right_gripper
    return states


# ---------------------------------------------------------------------------
# GripperCountSampleWeightRule validation tests
# ---------------------------------------------------------------------------


class TestGripperCountSampleWeightRule:
    def test_invalid_gripper(self):
        with pytest.raises(ValueError, match="gripper"):
            GripperCountSampleWeightRule(batch_contains="foo", gripper="middle", event="open", count=1, region="before")

    def test_invalid_event(self):
        with pytest.raises(ValueError, match="event"):
            GripperCountSampleWeightRule(
                batch_contains="foo", gripper="left", event="squeeze", count=1, region="before"
            )

    def test_invalid_count_zero(self):
        with pytest.raises(ValueError, match="count must be non-zero"):
            GripperCountSampleWeightRule(batch_contains="foo", gripper="left", event="open", count=0, region="before")

    def test_negative_count_allowed(self):
        rule = GripperCountSampleWeightRule(
            batch_contains="foo", gripper="left", event="open", count=-1, region="before"
        )
        assert rule.count == -1

    def test_invalid_region(self):
        with pytest.raises(ValueError, match="region"):
            GripperCountSampleWeightRule(batch_contains="foo", gripper="left", event="open", count=1, region="during")

    def test_invalid_weight_zero(self):
        with pytest.raises(ValueError, match="weight"):
            GripperCountSampleWeightRule(
                batch_contains="foo", gripper="left", event="open", count=1, region="before", weight=0
            )

    def test_invalid_weight_exceeds_max(self):
        with pytest.raises(ValueError, match="weight"):
            GripperCountSampleWeightRule(
                batch_contains="foo", gripper="left", event="open", count=1, region="before", weight=1001
            )

    def test_invalid_duration_s(self):
        with pytest.raises(ValueError, match="duration_s"):
            GripperCountSampleWeightRule(
                batch_contains="foo", gripper="left", event="open", count=1, region="before", duration_s=-1.0
            )


# ---------------------------------------------------------------------------
# GripperCountSampleWeightPreprocessor tests
# ---------------------------------------------------------------------------


class TestGripperCountSampleWeightPreprocessor:
    def test_threshold_validation(self):
        with pytest.raises(ValueError, match=r"close_threshold.*must be"):
            GripperCountSampleWeightPreprocessor(open_threshold=3.0, close_threshold=1.0)

    def test_no_matching_rules_noop(self):
        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="recover_2_n", gripper="right", event="open", count=3, region="after", weight=3
                )
            ],
        )
        states = _make_14d_states(10)
        ctx = _make_ctx("some_other_batch", states)
        attrs = FrameAttributes(sample_weight=np.ones(10, dtype=np.int32))
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, np.ones(10, dtype=np.int32))

    def test_upweight_after_right_open_3rd(self):
        """Upweight frames after the right gripper opens the 3rd time."""
        n = 15
        right_g = np.full(n, CLOSED, dtype=np.float32)
        # 1st open at frame 2, close at frame 4
        right_g[2:4] = OPEN
        # 2nd open at frame 6, close at frame 8
        right_g[6:8] = OPEN
        # 3rd open at frame 10, stays open
        right_g[10:13] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="recover_2_n",
                    gripper="right",
                    event="open",
                    count=3,
                    region="after",
                    weight=5,
                ),
            ],
        )
        ctx = _make_ctx("seatbelt.single.recover_2_n.op.20260301.batch.1", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # 3rd open at frame 10; frames 11..14 get weight 5, rest get 1
        expected = np.ones(n, dtype=np.int32)
        expected[11:] = 5
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_upweight_before_left_close_2nd(self):
        """Upweight frames before the left gripper closes the 2nd time."""
        n = 15
        left_g = np.full(n, CLOSED, dtype=np.float32)
        # 1st open at frame 1, close at frame 3
        left_g[1:3] = OPEN
        # 2nd open at frame 5, close at frame 8
        left_g[5:8] = OPEN
        states = _make_14d_states(n, left_gripper=left_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="recover_2_left",
                    gripper="left",
                    event="close",
                    count=2,
                    region="before",
                    weight=3,
                ),
            ],
        )
        ctx = _make_ctx("seatbelt.single.recover_2_left.op.20260301.batch.1", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # 2nd close at frame 8; frames 0..7 get weight 3, rest get 1
        expected = np.ones(n, dtype=np.int32)
        expected[:8] = 3
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_not_enough_events_no_change(self):
        """If fewer events than count, weight stays 1."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[3:5] = OPEN  # only 1 open event
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="recover", gripper="right", event="open", count=3, region="after", weight=5
                ),
            ],
        )
        ctx = _make_ctx("recover_2_n_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, np.ones(n, dtype=np.int32))

    def test_multiplies_existing_sample_weight(self):
        """New weight is multiplied with any existing sample_weight."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[3:5] = OPEN  # open at frame 3
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="recover", gripper="right", event="open", count=1, region="after", weight=4
                ),
            ],
        )
        ctx = _make_ctx("recover_batch", states)
        existing = np.full(n, 2, dtype=np.int32)
        attrs = FrameAttributes(sample_weight=existing.copy())
        proc(ctx, attrs)

        # frames 4..9 get multiplied by 4; frames 0..3 stay at 2
        expected = np.full(n, 2, dtype=np.int32)
        expected[4:] = 8  # 2 * 4
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_multiple_rules_multiply(self):
        """Multiple matching rules multiply together."""
        n = 20
        left_g = np.full(n, CLOSED, dtype=np.float32)
        left_g[2:4] = OPEN  # open@2, close@4
        left_g[8:10] = OPEN  # open@8, close@10

        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[5:7] = OPEN  # open@5, close@7
        right_g[12:14] = OPEN  # open@12, close@14

        states = _make_14d_states(n, left_gripper=left_g, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                # frames before right 1st open (frame 5) get weight 2 → frames 0..4
                GripperCountSampleWeightRule(
                    batch_contains="combo", gripper="right", event="open", count=1, region="before", weight=2
                ),
                # frames after left 2nd close (frame 10) get weight 3 → frames 11..19
                GripperCountSampleWeightRule(
                    batch_contains="combo", gripper="left", event="close", count=2, region="after", weight=3
                ),
            ],
        )
        ctx = _make_ctx("combo_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        expected = np.ones(n, dtype=np.int32)
        expected[:5] = 2  # rule 1: before right open 1st
        expected[11:] = 3  # rule 2: after left close 2nd
        # No overlap between [0..4] and [11..19], so no multiplication
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_overlapping_rules_multiply(self):
        """Overlapping regions from multiple rules multiply."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[2:4] = OPEN  # open@2, close@4
        right_g[6:8] = OPEN  # open@6, close@8
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                # after 1st open (frame 2): frames 3..9 get weight 2
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="after", weight=2
                ),
                # after 1st close (frame 4): frames 5..9 get weight 3
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="close", count=1, region="after", weight=3
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        expected = np.ones(n, dtype=np.int32)
        expected[3:5] = 2  # only rule 1
        expected[5:] = 6  # both rules: 2 * 3
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_multi_episode(self):
        """Rules apply independently per episode."""
        n = 20
        right_g = np.full(n, CLOSED, dtype=np.float32)
        # Ep0: open at frame 2
        right_g[2:4] = OPEN
        # Ep1: open at frame 12
        right_g[12:14] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="before", weight=5
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states, episode_boundaries=[(0, 10), (10, 20)])
        attrs = FrameAttributes()
        proc(ctx, attrs)

        expected = np.ones(n, dtype=np.int32)
        expected[:2] = 5  # ep0: before open at 2
        expected[10:12] = 5  # ep1: before open at 12
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_initializes_sample_weight_if_none(self):
        """If attrs.sample_weight is None, initialize from scratch."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[5:7] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="after", weight=3
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()  # sample_weight is None
        proc(ctx, attrs)

        expected = np.ones(n, dtype=np.int32)
        expected[6:] = 3
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_event_at_second_frame(self):
        """Event near start: 'before' region has 1 frame, 'after' covers rest."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[1:3] = OPEN  # open at frame 1
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="after", weight=4
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # open at frame 1, after → frames 2..9 get weight 4
        expected = np.ones(n, dtype=np.int32)
        expected[2:] = 4
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_event_at_last_frame(self):
        """Event at last frame: 'after' region is empty, 'before' covers rest."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[9] = OPEN  # open at last frame
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="before", weight=3
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # open at frame 9, before → frames 0..8 get weight 3
        expected = np.ones(n, dtype=np.int32)
        expected[:9] = 3
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_negative_count_last_event(self):
        """count=-1 selects the last event."""
        n = 15
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[2:4] = OPEN  # 1st open at 2
        right_g[6:8] = OPEN  # 2nd open at 6
        right_g[10:12] = OPEN  # 3rd (last) open at 10
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=-1, region="before", weight=5
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # last open at frame 10, before → frames 0..9 get weight 5
        expected = np.ones(n, dtype=np.int32)
        expected[:10] = 5
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_negative_count_not_enough_events(self):
        """count=-3 with only 1 event keeps weight unchanged."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[3:5] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=-3, region="before", weight=5
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.sample_weight, np.ones(n, dtype=np.int32))

    def test_duration_s_before(self):
        """duration_s limits weighting to a time window before the event."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[20:22] = OPEN  # open at frame 20
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test",
                    gripper="right",
                    event="open",
                    count=1,
                    region="before",
                    duration_s=0.5,
                    weight=4,
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # 0.5s at 10fps = 5 frames before pivot(20) → frames 15..19 get weight 4
        expected = np.ones(n, dtype=np.int32)
        expected[15:20] = 4
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_duration_s_after(self):
        """duration_s limits weighting to a time window after the event."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[5:7] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test",
                    gripper="right",
                    event="open",
                    count=1,
                    region="after",
                    duration_s=1.0,
                    weight=3,
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # 1.0s at 10fps = 10 frames after pivot(5) → frames 6..15 get weight 3
        expected = np.ones(n, dtype=np.int32)
        expected[6:16] = 3
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_duration_s_with_negative_count(self):
        """Combine negative count and duration_s: weight 2s before last open."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[5:7] = OPEN
        right_g[15:17] = OPEN
        right_g[25:27] = OPEN  # last open at 25
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test",
                    gripper="right",
                    event="open",
                    count=-1,
                    region="before",
                    duration_s=0.5,
                    weight=2,
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # last open at 25, 0.5s at 10fps = 5 frames → frames 20..24 get weight 2
        expected = np.ones(n, dtype=np.int32)
        expected[20:25] = 2
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_16dim_state(self):
        """16-DOF state (7-DOF arm): gripper at indices 7 and 15."""
        n = 15
        states = np.full((n, 16), 1.75, dtype=np.float32)
        states[:, 7] = CLOSED  # left gripper
        states[:, 15] = CLOSED  # right gripper
        states[5:7, 15] = OPEN  # right open at frame 5

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="after", weight=3
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        expected = np.ones(n, dtype=np.int32)
        expected[6:] = 3
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_unsupported_state_dim_raises(self):
        """Non-14/16 dim state raises ValueError."""
        states = np.zeros((10, 10), dtype=np.float32)
        proc = GripperCountSampleWeightPreprocessor(
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="before", weight=2
                )
            ],
        )
        ctx = _make_ctx("test_batch", states)
        with pytest.raises(ValueError, match=r"supports state_dim.*got 10"):
            proc(ctx, FrameAttributes())

    def test_head_margin_preserves_start(self):
        """head_margin_s keeps weight=1 at episode start."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[15:17] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            head_margin_s=1.0,  # 10 frames at start unmodified
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="before", weight=5
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # Without margin: frames 0..14 get weight 5. With head_margin=1.0: frames 0..9 stay 1
        expected = np.ones(n, dtype=np.int32)
        expected[10:15] = 5
        np.testing.assert_array_equal(attrs.sample_weight, expected)

    def test_tail_margin_preserves_end(self):
        """tail_margin_s keeps weight=1 at episode end."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[10:12] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountSampleWeightPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            tail_margin_s=0.5,  # 5 frames at end unmodified
            rules=[
                GripperCountSampleWeightRule(
                    batch_contains="test", gripper="right", event="open", count=1, region="after", weight=4
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # Without margin: frames 11..29 get weight 4. With tail_margin=0.5: frames 25..29 stay 1
        expected = np.ones(n, dtype=np.int32)
        expected[11:25] = 4
        np.testing.assert_array_equal(attrs.sample_weight, expected)
