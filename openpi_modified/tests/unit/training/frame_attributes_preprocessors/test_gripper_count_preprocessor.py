"""GripperCountValidMaskPreprocessor unit tests.

Gripper convention: 0 = fully open, high value = fully closed.
open event: value drops below open_threshold.
close event: value rises above close_threshold.
"""

from unittest.mock import Mock

import numpy as np
import pytest

from openpi.training.frame_attributes_preprocessors.base import FrameAttributes
from openpi.training.frame_attributes_preprocessors.utils import detect_gripper_events
from openpi.training.frame_attributes_preprocessors.valid_mask_preprocessor import GripperCountRule
from openpi.training.frame_attributes_preprocessors.valid_mask_preprocessor import GripperCountValidMaskPreprocessor

# Thresholds: open when < 0.5, closed when > 3.0
OPEN_THRE = 0.5
CLOSE_THRE = 3.0

# Canonical values
CLOSED = 3.5  # above close_threshold → closed
OPEN = 0.0  # below open_threshold → open


def _make_ctx(repo_id: str, states: np.ndarray, episode_boundaries: list[tuple[int, int]] | None = None):
    """Build a mock DatasetContext with the given states and episode boundaries."""
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
    states = np.full((n, 14), 1.75, dtype=np.float32)  # joints at neutral
    states[:, 6] = CLOSED  # left gripper default: closed
    states[:, 13] = CLOSED  # right gripper default: closed
    if left_gripper is not None:
        states[:, 6] = left_gripper
    if right_gripper is not None:
        states[:, 13] = right_gripper
    return states


# ---------------------------------------------------------------------------
# detect_gripper_events tests
# ---------------------------------------------------------------------------


class TestDetectGripperEvents:
    def test_no_events_all_closed(self):
        values = np.array([3.5, 3.2, 3.5, 3.3, 3.5])
        opens, closes = detect_gripper_events(values, OPEN_THRE, CLOSE_THRE)
        assert opens == []
        assert closes == []

    def test_no_events_in_hysteresis_band(self):
        # Value oscillates between thresholds — no events should fire
        values = np.array([3.5, 2.0, 1.5, 2.5, 1.2, 2.8, 1.0])
        opens, closes = detect_gripper_events(values, OPEN_THRE, CLOSE_THRE)
        assert opens == []
        assert closes == []

    def test_single_open_close_cycle(self):
        # closed(3.5) -> open at frame 2, close at frame 4
        values = np.array([CLOSED, CLOSED, OPEN, OPEN, CLOSED, CLOSED])
        opens, closes = detect_gripper_events(values, OPEN_THRE, CLOSE_THRE)
        assert opens == [2]
        assert closes == [4]

    def test_multiple_cycles(self):
        # 3 open/close cycles
        values = np.array([CLOSED, OPEN, CLOSED, OPEN, CLOSED, OPEN, CLOSED])
        opens, closes = detect_gripper_events(values, OPEN_THRE, CLOSE_THRE)
        assert opens == [1, 3, 5]
        assert closes == [2, 4, 6]

    def test_starts_open(self):
        # Starts below open_threshold (already open)
        values = np.array([OPEN, OPEN, CLOSED, OPEN, CLOSED])
        opens, closes = detect_gripper_events(values, OPEN_THRE, CLOSE_THRE)
        # No open event at start (already open), close@2, open@3, close@4
        assert opens == [3]
        assert closes == [2, 4]

    def test_empty_array(self):
        opens, closes = detect_gripper_events(np.array([]), OPEN_THRE, CLOSE_THRE)
        assert opens == []
        assert closes == []

    def test_event_at_first_frame(self):
        # First frame is open (below threshold), so no open *event* at start
        # but close event at frame 1
        values = np.array([OPEN, CLOSED, CLOSED])
        opens, closes = detect_gripper_events(values, OPEN_THRE, CLOSE_THRE)
        assert opens == []
        assert closes == [1]

    def test_event_at_last_frame(self):
        values = np.array([CLOSED, CLOSED, OPEN])
        opens, closes = detect_gripper_events(values, OPEN_THRE, CLOSE_THRE)
        assert opens == [2]
        assert closes == []

    def test_threshold_validation(self):
        with pytest.raises(ValueError, match=r"close_threshold.*must be > open_threshold"):
            detect_gripper_events(np.array([0.0]), 3.0, 1.0)

    def test_equal_thresholds_raises(self):
        with pytest.raises(ValueError, match=r"close_threshold.*must be > open_threshold"):
            detect_gripper_events(np.array([0.0]), 2.0, 2.0)


# ---------------------------------------------------------------------------
# GripperCountRule validation tests
# ---------------------------------------------------------------------------


class TestGripperCountRule:
    def test_invalid_gripper(self):
        with pytest.raises(ValueError, match="gripper"):
            GripperCountRule(batch_contains="foo", gripper="middle", event="open", count=1, invalidate="before")

    def test_invalid_event(self):
        with pytest.raises(ValueError, match="event"):
            GripperCountRule(batch_contains="foo", gripper="left", event="squeeze", count=1, invalidate="before")

    def test_invalid_count_zero(self):
        with pytest.raises(ValueError, match="count must be non-zero"):
            GripperCountRule(batch_contains="foo", gripper="left", event="open", count=0, invalidate="before")

    def test_negative_count_allowed(self):
        rule = GripperCountRule(batch_contains="foo", gripper="left", event="open", count=-1, invalidate="before")
        assert rule.count == -1

    def test_invalid_invalidate(self):
        with pytest.raises(ValueError, match="invalidate"):
            GripperCountRule(batch_contains="foo", gripper="left", event="open", count=1, invalidate="during")

    def test_invalid_duration_s(self):
        with pytest.raises(ValueError, match="duration_s"):
            GripperCountRule(
                batch_contains="foo", gripper="left", event="open", count=1, invalidate="before", duration_s=-1.0
            )

    def test_duration_s_zero_raises(self):
        with pytest.raises(ValueError, match="duration_s"):
            GripperCountRule(
                batch_contains="foo", gripper="left", event="open", count=1, invalidate="before", duration_s=0.0
            )


# ---------------------------------------------------------------------------
# GripperCountValidMaskPreprocessor tests
# ---------------------------------------------------------------------------


class TestGripperCountValidMaskPreprocessor:
    def test_threshold_validation(self):
        with pytest.raises(ValueError, match=r"close_threshold.*must be"):
            GripperCountValidMaskPreprocessor(open_threshold=3.0, close_threshold=1.0)

    def test_no_matching_rules_noop(self):
        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(
                    batch_contains="recover_2_n", gripper="right", event="open", count=3, invalidate="before"
                )
            ],
        )
        states = _make_14d_states(10)
        ctx = _make_ctx("some_other_batch", states)
        attrs = FrameAttributes(valid_mask=np.ones(10, dtype=bool))
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.valid_mask, np.ones(10, dtype=bool))

    def test_invalidate_before_right_open_3rd(self):
        """Invalid all frames before the right gripper opens the 3rd time."""
        n = 15
        right_g = np.full(n, CLOSED, dtype=np.float32)
        # 1st open at frame 2, close at frame 4
        right_g[2:4] = OPEN
        # 2nd open at frame 6, close at frame 8
        right_g[6:8] = OPEN
        # 3rd open at frame 10, stays open
        right_g[10:13] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(
                    batch_contains="recover_2_n",
                    gripper="right",
                    event="open",
                    count=3,
                    invalidate="before",
                ),
            ],
        )
        ctx = _make_ctx("seatbelt.single.recover_2_n.op.20260301.batch.1", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # Frames 0..9 should be invalid, frames 10..14 valid
        expected = np.ones(n, dtype=bool)
        expected[:10] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_invalidate_after_left_close_2nd(self):
        """Invalid all frames after the left gripper closes the 2nd time."""
        n = 15
        left_g = np.full(n, CLOSED, dtype=np.float32)
        # 1st open at frame 1, close at frame 3
        left_g[1:3] = OPEN
        # 2nd open at frame 5, close at frame 8
        left_g[5:8] = OPEN
        states = _make_14d_states(n, left_gripper=left_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(
                    batch_contains="recover_2_left",
                    gripper="left",
                    event="close",
                    count=2,
                    invalidate="after",
                ),
            ],
        )
        ctx = _make_ctx("seatbelt.single.recover_2_left.op.20260301.batch.1", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # 2nd close at frame 8, frames 9..14 invalid
        expected = np.ones(n, dtype=bool)
        expected[9:] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_not_enough_events_keeps_all_valid(self):
        """If fewer events than count, all frames stay valid."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[3:5] = OPEN  # only 1 open event
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(batch_contains="recover", gripper="right", event="open", count=3, invalidate="before"),
            ],
        )
        ctx = _make_ctx("recover_2_n_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.valid_mask, np.ones(n, dtype=bool))

    def test_ands_with_existing_valid_mask(self):
        """New mask is AND-ed with any existing valid_mask."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[3:5] = OPEN  # open at frame 3
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(batch_contains="recover", gripper="right", event="open", count=1, invalidate="before"),
            ],
        )
        ctx = _make_ctx("recover_batch", states)
        existing = np.ones(n, dtype=bool)
        existing[8:] = False  # already invalid at tail
        attrs = FrameAttributes(valid_mask=existing.copy())
        proc(ctx, attrs)

        expected = np.ones(n, dtype=bool)
        expected[:3] = False  # invalidated by rule (before 1st open)
        expected[8:] = False  # already invalid
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_multiple_rules_applied_cumulatively(self):
        """Multiple matching rules are AND-ed together."""
        n = 20
        left_g = np.full(n, CLOSED, dtype=np.float32)
        left_g[2:4] = OPEN  # open@2, close@4
        left_g[8:10] = OPEN  # open@8, close@10

        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[5:7] = OPEN  # open@5, close@7
        right_g[12:14] = OPEN  # open@12, close@14

        states = _make_14d_states(n, left_gripper=left_g, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                # invalidate before right opens 1st time (frame 5) → frames 0..4 invalid
                GripperCountRule(batch_contains="combo", gripper="right", event="open", count=1, invalidate="before"),
                # invalidate after left closes 2nd time (frame 10) → frames 11..19 invalid
                GripperCountRule(batch_contains="combo", gripper="left", event="close", count=2, invalidate="after"),
            ],
        )
        ctx = _make_ctx("combo_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        expected = np.ones(n, dtype=bool)
        expected[:5] = False  # before right open 1st
        expected[11:] = False  # after left close 2nd
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_multi_episode(self):
        """Rules apply independently per episode."""
        # Episode 0: frames 0..9, Episode 1: frames 10..19
        n = 20
        right_g = np.full(n, CLOSED, dtype=np.float32)
        # Ep0: open at frame 2
        right_g[2:4] = OPEN
        # Ep1: open at frame 12
        right_g[12:14] = OPEN
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(batch_contains="test", gripper="right", event="open", count=1, invalidate="before"),
            ],
        )
        ctx = _make_ctx("test_batch", states, episode_boundaries=[(0, 10), (10, 20)])
        attrs = FrameAttributes()
        proc(ctx, attrs)

        expected = np.ones(n, dtype=bool)
        expected[:2] = False  # ep0: before open at 2
        expected[10:12] = False  # ep1: before open at 12
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_negative_count_last_event(self):
        """count=-1 selects the last event."""
        n = 15
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[2:4] = OPEN  # 1st open at 2
        right_g[6:8] = OPEN  # 2nd open at 6
        right_g[10:12] = OPEN  # 3rd (last) open at 10
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(batch_contains="test", gripper="right", event="open", count=-1, invalidate="before"),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # last open at frame 10, invalidate before → frames 0..9 invalid
        expected = np.ones(n, dtype=bool)
        expected[:10] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_negative_count_second_to_last(self):
        """count=-2 selects the second to last event."""
        n = 15
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[2:4] = OPEN  # 1st open at 2
        right_g[6:8] = OPEN  # 2nd open at 6
        right_g[10:12] = OPEN  # 3rd open at 10
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(batch_contains="test", gripper="right", event="open", count=-2, invalidate="after"),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # 2nd to last open at frame 6, invalidate after → frames 7..14 invalid
        expected = np.ones(n, dtype=bool)
        expected[7:] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_negative_count_not_enough_events(self):
        """count=-3 with only 1 event keeps all valid."""
        n = 10
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[3:5] = OPEN  # only 1 open
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(batch_contains="test", gripper="right", event="open", count=-3, invalidate="before"),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)
        np.testing.assert_array_equal(attrs.valid_mask, np.ones(n, dtype=bool))

    def test_duration_s_before(self):
        """duration_s limits invalidation to a time window before the event."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[20:22] = OPEN  # open at frame 20
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,  # 10 fps → 2s = 20 frames, but only 5 frames window here
            rules=[
                GripperCountRule(
                    batch_contains="test",
                    gripper="right",
                    event="open",
                    count=1,
                    invalidate="before",
                    duration_s=0.5,
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # 0.5s at 10fps = 5 frames before pivot(20) → frames 15..19 invalid
        expected = np.ones(n, dtype=bool)
        expected[15:20] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_duration_s_after(self):
        """duration_s limits invalidation to a time window after the event."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[5:7] = OPEN  # open at frame 5
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            rules=[
                GripperCountRule(
                    batch_contains="test",
                    gripper="right",
                    event="open",
                    count=1,
                    invalidate="after",
                    duration_s=1.0,
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # 1.0s at 10fps = 10 frames after pivot(5) → frames 6..15 invalid
        expected = np.ones(n, dtype=bool)
        expected[6:16] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_duration_s_with_negative_count(self):
        """Combine negative count and duration_s."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[5:7] = OPEN  # 1st open at 5
        right_g[15:17] = OPEN  # 2nd open at 15
        right_g[25:27] = OPEN  # 3rd (last) open at 25
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            rules=[
                GripperCountRule(
                    batch_contains="test",
                    gripper="right",
                    event="open",
                    count=-1,
                    invalidate="before",
                    duration_s=0.5,
                ),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # last open at 25, 0.5s at 10fps = 5 frames → frames 20..24 invalid
        expected = np.ones(n, dtype=bool)
        expected[20:25] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_16dim_state(self):
        """16-DOF state (7-DOF arm): gripper at indices 7 and 15."""
        n = 15
        states = np.full((n, 16), 1.75, dtype=np.float32)
        states[:, 7] = CLOSED  # left gripper
        states[:, 15] = CLOSED  # right gripper
        # right gripper open at frame 5
        states[5:7, 15] = OPEN

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            rules=[
                GripperCountRule(batch_contains="test", gripper="right", event="open", count=1, invalidate="before"),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        expected = np.ones(n, dtype=bool)
        expected[:5] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_unsupported_state_dim_raises(self):
        """Non-14/16 dim state raises ValueError."""
        states = np.zeros((10, 10), dtype=np.float32)
        proc = GripperCountValidMaskPreprocessor(
            rules=[
                GripperCountRule(batch_contains="test", gripper="right", event="open", count=1, invalidate="before")
            ],
        )
        ctx = _make_ctx("test_batch", states)
        with pytest.raises(ValueError, match=r"supports state_dim.*got 10"):
            proc(ctx, FrameAttributes())

    def test_head_margin_preserves_start(self):
        """head_margin_s keeps the first N seconds valid."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[15:17] = OPEN  # open at frame 15
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            head_margin_s=1.0,  # 10 frames at start always valid
            rules=[
                GripperCountRule(batch_contains="test", gripper="right", event="open", count=1, invalidate="before"),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # Without margin: frames 0..14 invalid. With head_margin_s=1.0: frames 0..9 restored valid
        expected = np.ones(n, dtype=bool)
        expected[10:15] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)

    def test_tail_margin_preserves_end(self):
        """tail_margin_s keeps the last N seconds valid."""
        n = 30
        right_g = np.full(n, CLOSED, dtype=np.float32)
        right_g[10:12] = OPEN  # open at frame 10
        states = _make_14d_states(n, right_gripper=right_g)

        proc = GripperCountValidMaskPreprocessor(
            open_threshold=OPEN_THRE,
            close_threshold=CLOSE_THRE,
            fps=10,
            tail_margin_s=0.5,  # 5 frames at end always valid
            rules=[
                GripperCountRule(batch_contains="test", gripper="right", event="open", count=1, invalidate="after"),
            ],
        )
        ctx = _make_ctx("test_batch", states)
        attrs = FrameAttributes()
        proc(ctx, attrs)

        # Without margin: frames 11..29 invalid. With tail_margin_s=0.5: frames 25..29 restored valid
        expected = np.ones(n, dtype=bool)
        expected[11:25] = False
        np.testing.assert_array_equal(attrs.valid_mask, expected)
