"""Unit tests for piper_leader.send_feedback key validation (M2).

Locks the MR !10 round 2/3 C2 fix: send_feedback MUST raise KeyError on
missing keys rather than silently defaulting missing joints to 0.0. The
original bug was a physical safety hazard — a 0.0 default could command
the arm to jump to its zero pose when upstream passed an incomplete dict.

These tests run without hardware by stubbing the piper_sdk / pinocchio /
scipy C extensions and bypassing Teleoperator.__init__ (which would
otherwise try to create a calibration directory).
"""

import sys
from unittest.mock import MagicMock

import pytest

# Stub hardware-only deps BEFORE importing PiperLeader. `setdefault` avoids
# clobbering real modules if they happen to be installed.
for _mod in ("pinocchio", "piper_sdk", "scipy", "scipy.interpolate"):
    sys.modules.setdefault(_mod, MagicMock())

from lerobot.errors import DeviceNotConnectedError  # noqa: E402
from lerobot.teleoperators.piper_leader.piper_leader import PiperLeader  # noqa: E402


def _make_leader_without_init(connected: bool = True) -> PiperLeader:
    """Build a PiperLeader instance bypassing Teleoperator.__init__.

    The base class __init__ creates a calibration directory side effect we
    do not need here — send_feedback only touches `self.sdk` and
    `self.is_connected`. `id` is only needed for __str__ in error messages.
    """
    leader = PiperLeader.__new__(PiperLeader)
    leader.id = "test_leader"
    leader.sdk = MagicMock() if connected else None
    return leader


class TestSendFeedbackKeyValidation:
    def _full_feedback(self) -> dict[str, float]:
        return {f"joint_{i}.pos": float(i * 10) for i in range(7)}

    def test_send_feedback_happy_path_forwards_7_positions(self):
        """With all 7 keys present, positions are forwarded to the SDK in order."""
        leader = _make_leader_without_init(connected=True)
        feedback = self._full_feedback()

        leader.send_feedback(feedback)

        leader.sdk.set_joint_positions.assert_called_once()
        (positions,), _ = leader.sdk.set_joint_positions.call_args
        assert positions == [float(i * 10) for i in range(7)]

    def test_send_feedback_raises_on_missing_key(self):
        """Missing a single key must raise KeyError — NOT silently default to 0.0.

        Regression guard: the previous behavior was `feedback.get(k, 0.0)`,
        which could command the arm to jump to 0° on partial data. Commit
        message labelled this a 'physical safety hazard'.
        """
        leader = _make_leader_without_init(connected=True)
        feedback = self._full_feedback()
        del feedback["joint_3.pos"]

        with pytest.raises(KeyError) as exc_info:
            leader.send_feedback(feedback)

        # Error message must name the missing key so operators can debug.
        assert "joint_3.pos" in str(exc_info.value)
        # SDK must NOT have been called — failure happens before dispatch.
        leader.sdk.set_joint_positions.assert_not_called()

    def test_send_feedback_raises_on_multiple_missing_keys(self):
        """Error lists ALL missing keys, not just the first one."""
        leader = _make_leader_without_init(connected=True)
        feedback = {"joint_0.pos": 1.0, "joint_1.pos": 2.0}  # 5 missing

        with pytest.raises(KeyError) as exc_info:
            leader.send_feedback(feedback)

        msg = str(exc_info.value)
        for i in range(2, 7):
            assert f"joint_{i}.pos" in msg, (
                f"expected joint_{i}.pos in error message, got: {msg}"
            )
        leader.sdk.set_joint_positions.assert_not_called()

    def test_send_feedback_raises_on_empty_dict(self):
        """Empty dict (e.g. from key mismatch in leader_sync_loop) raises.

        This is the exact scenario I1 guards against — if leader_sync_loop
        regressed, send_feedback({}) would surface here as a loud error.
        """
        leader = _make_leader_without_init(connected=True)

        with pytest.raises(KeyError):
            leader.send_feedback({})

        leader.sdk.set_joint_positions.assert_not_called()

    def test_send_feedback_raises_when_not_connected(self):
        """Calling send_feedback on a disconnected leader raises DeviceNotConnectedError."""
        leader = _make_leader_without_init(connected=False)

        with pytest.raises(DeviceNotConnectedError):
            leader.send_feedback(self._full_feedback())

    def test_send_feedback_extra_keys_ignored(self):
        """Extra keys beyond the 7 required are silently accepted — only the
        7 required joints are forwarded to the SDK. This keeps send_feedback
        permissive for callers that compose feedback from larger dicts.
        """
        leader = _make_leader_without_init(connected=True)
        feedback = self._full_feedback()
        feedback["extra_key"] = 999.0
        feedback["velocity_0"] = 1.0

        leader.send_feedback(feedback)

        leader.sdk.set_joint_positions.assert_called_once()
        (positions,), _ = leader.sdk.set_joint_positions.call_args
        assert positions == [float(i * 10) for i in range(7)]
