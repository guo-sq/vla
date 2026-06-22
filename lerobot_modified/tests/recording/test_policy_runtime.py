"""Tests for recording.runtime.policy_runtime — ActionBuffer, PolicyRuntime."""

import queue
import threading
import time

import numpy as np
import pytest

from lerobot.recording.runtime.policy_runtime import (
    ActionBuffer,
    PolicyRuntime,
    inference_worker,
    queue_put_replace_oldest,
)
from lerobot.recording.utils.logging import AsyncLogger


# ---------------------------------------------------------------------------
# ActionBuffer tests
# ---------------------------------------------------------------------------

class TestActionBuffer:
    def _make_buffer(self, action_dim=3, action_horizon=4, fusion_type="linear"):
        keys = [f"joint_{i}.pos" for i in range(action_dim)]
        return ActionBuffer(
            robot_state_keys=keys,
            action_dim=action_dim,
            action_horizon=action_horizon,
            fusion_type=fusion_type,
        )

    def test_init_buffer(self):
        buf = self._make_buffer(action_dim=3, action_horizon=4)
        chunk = np.ones((4, 3), dtype=np.float32)
        buf.init_buffer(chunk)
        assert not buf.is_empty()

    def test_get_next_action(self):
        buf = self._make_buffer(action_dim=3, action_horizon=4)
        chunk = np.arange(12, dtype=np.float32).reshape(4, 3)
        buf.init_buffer(chunk)

        action = buf.get_next_action()
        assert action is not None
        assert len(action) == 3
        # First action should be first row
        assert list(action.keys()) == ["joint_0.pos", "joint_1.pos", "joint_2.pos"]

    def test_buffer_empties_after_consuming_all(self):
        buf = self._make_buffer(action_dim=2, action_horizon=3)
        chunk = np.ones((3, 2), dtype=np.float32)
        buf.init_buffer(chunk)

        for _ in range(3):
            action = buf.get_next_action()
            assert action is not None

        assert buf.is_empty()

    def test_get_next_action_when_empty_returns_none(self):
        buf = self._make_buffer()
        assert buf.is_empty()
        assert buf.get_next_action() is None

    def test_clear(self):
        buf = self._make_buffer(action_dim=2, action_horizon=3)
        buf.init_buffer(np.ones((3, 2), dtype=np.float32))
        assert not buf.is_empty()
        buf.clear()
        assert buf.is_empty()

    def test_update_from_inference_appends(self):
        buf = self._make_buffer(action_dim=2, action_horizon=4)
        initial = np.ones((4, 2), dtype=np.float32)
        buf.init_buffer(initial)

        # Consume 2 actions
        buf.get_next_action()
        buf.get_next_action()

        # New chunk arrives
        new_chunk = np.ones((4, 2), dtype=np.float32) * 2.0
        buf.update_from_inference(new_chunk, start_step=2, current_step=2)

        # Should have actions available
        assert not buf.is_empty()

    def test_get_future_actions(self):
        buf = self._make_buffer(action_dim=2, action_horizon=4)
        chunk = np.arange(8, dtype=np.float32).reshape(4, 2)
        buf.init_buffer(chunk)
        buf.get_next_action()  # consume one

        prefix, delay, valid_len = buf.get_future_actions(default_delay=0)
        assert isinstance(prefix, np.ndarray)
        assert isinstance(delay, int)

    def test_thread_safety_concurrent_access(self):
        buf = self._make_buffer(action_dim=2, action_horizon=10)
        buf.init_buffer(np.ones((10, 2), dtype=np.float32))
        errors = []

        def reader():
            try:
                for _ in range(5):
                    buf.get_next_action()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for i in range(5):
                    buf.update_from_inference(
                        np.ones((10, 2), dtype=np.float32) * i,
                        start_step=i * 10,
                        current_step=i * 10,
                    )
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0

    def test_fusion_linear(self):
        buf = self._make_buffer(action_dim=2, action_horizon=4, fusion_type="linear")
        buf.init_buffer(np.ones((4, 2), dtype=np.float32))
        # Consume 1
        buf.get_next_action()
        # Update with fusion enabled
        new_chunk = np.ones((4, 2), dtype=np.float32) * 3.0
        buf.update_from_inference(new_chunk, start_step=1, current_step=1, enable_fusion=True)
        action = buf.get_next_action()
        assert action is not None
        # With fusion, values should be blended (not exactly 1.0 or 3.0)

    def test_fusion_exponential(self):
        buf = self._make_buffer(action_dim=2, action_horizon=4, fusion_type="exponential")
        buf.init_buffer(np.ones((4, 2), dtype=np.float32))
        buf.get_next_action()
        new_chunk = np.ones((4, 2), dtype=np.float32) * 5.0
        buf.update_from_inference(new_chunk, start_step=1, current_step=1, enable_fusion=True)
        action = buf.get_next_action()
        assert action is not None


# ---------------------------------------------------------------------------
# queue_put_replace_oldest tests
# ---------------------------------------------------------------------------

class TestQueuePutReplaceOldest:
    def test_put_to_empty_queue(self):
        q = queue.Queue(maxsize=2)
        replaced = queue_put_replace_oldest(q, "item1")
        assert replaced is False
        assert q.qsize() == 1

    def test_replace_when_full(self):
        q = queue.Queue(maxsize=1)
        q.put("old")
        replaced = queue_put_replace_oldest(q, "new")
        assert replaced is True
        assert q.get() == "new"

    def test_preserves_newest_item(self):
        q = queue.Queue(maxsize=1)
        q.put("first")
        queue_put_replace_oldest(q, "second")
        assert q.get() == "second"


# ---------------------------------------------------------------------------
# inference_worker tests
# ---------------------------------------------------------------------------

class TestInferenceWorker:
    def test_processes_items(self, tmp_path):
        logger = AsyncLogger(str(tmp_path), enabled=True)
        input_q = queue.Queue()
        output_q = queue.Queue()
        stop_event = threading.Event()

        class FakeClient:
            def get_action(self, obs, prefix, delay, valid_len, lang):
                return np.ones((4, 2), dtype=np.float32)

        client = FakeClient()
        thread = threading.Thread(
            target=inference_worker,
            args=(client, input_q, output_q, stop_event, logger),
            daemon=True,
        )
        thread.start()

        obs = {"state": np.zeros(2)}
        prefix = np.zeros((4, 2), dtype=np.float32)
        input_q.put(("prompt", obs, prefix, 0, 0, 0))

        result = output_q.get(timeout=5.0)
        assert result is not None
        chunk, step_id = result
        assert isinstance(chunk, np.ndarray)
        assert step_id == 0

        stop_event.set()
        input_q.put(None)
        thread.join(timeout=2.0)
        logger.close()

    def test_signals_failure_on_exception(self, tmp_path):
        logger = AsyncLogger(str(tmp_path), enabled=True)
        input_q = queue.Queue()
        output_q = queue.Queue()
        stop_event = threading.Event()

        class FailingClient:
            def get_action(self, obs, prefix, delay, valid_len, lang):
                raise RuntimeError("inference failed")

        client = FailingClient()
        thread = threading.Thread(
            target=inference_worker,
            args=(client, input_q, output_q, stop_event, logger),
            daemon=True,
        )
        thread.start()

        input_q.put(("prompt", {}, np.zeros((4, 2)), 0, 0, 42))
        result = output_q.get(timeout=5.0)
        chunk, step_id = result
        assert chunk is None  # signals failure
        assert step_id == 42

        stop_event.set()
        input_q.put(None)
        thread.join(timeout=2.0)
        logger.close()

    def test_stops_on_stop_event(self, tmp_path):
        logger = AsyncLogger(str(tmp_path), enabled=True)
        input_q = queue.Queue()
        output_q = queue.Queue()
        stop_event = threading.Event()

        class FakeClient:
            def get_action(self, obs, prefix, delay, valid_len, lang):
                return np.ones((4, 2), dtype=np.float32)

        thread = threading.Thread(
            target=inference_worker,
            args=(FakeClient(), input_q, output_q, stop_event, logger),
            daemon=True,
        )
        thread.start()
        stop_event.set()
        thread.join(timeout=3.0)
        assert not thread.is_alive()
        logger.close()

    def test_stops_on_none_sentinel(self, tmp_path):
        logger = AsyncLogger(str(tmp_path), enabled=True)
        input_q = queue.Queue()
        output_q = queue.Queue()
        stop_event = threading.Event()

        class FakeClient:
            def get_action(self, obs, prefix, delay, valid_len, lang):
                return np.ones((4, 2), dtype=np.float32)

        thread = threading.Thread(
            target=inference_worker,
            args=(FakeClient(), input_q, output_q, stop_event, logger),
            daemon=True,
        )
        thread.start()
        input_q.put(None)
        thread.join(timeout=3.0)
        assert not thread.is_alive()
        logger.close()
