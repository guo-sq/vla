"""Tests for recording.utils.logging — AsyncLogger and SelfPlayLogger."""

import json
import os
import tempfile
import threading
import time

import pytest

from lerobot.recording.utils.logging import AsyncLogger, SelfPlayLogger


class TestAsyncLogger:
    def test_creates_log_directory_and_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "test_logs")
            logger = AsyncLogger(log_dir, enabled=True)
            logger.log("hello")
            logger.close()
            assert os.path.isdir(log_dir)
            log_file = os.path.join(log_dir, "log.txt")
            assert os.path.isfile(log_file)
            with open(log_file) as f:
                content = f.read()
            assert "hello" in content

    def test_disabled_logger_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "test_logs")
            logger = AsyncLogger(log_dir, enabled=False)
            logger.log("should not appear")
            logger.close()
            assert not os.path.exists(log_dir)

    def test_log_messages_are_timestamped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "test_logs")
            logger = AsyncLogger(log_dir, enabled=True)
            logger.log("timestamped message")
            logger.close()
            with open(os.path.join(log_dir, "log.txt")) as f:
                content = f.read()
            # Timestamp format: YYYY-MM-DD HH:MM:SS
            assert "20" in content  # year prefix
            assert "timestamped message" in content

    def test_multiple_messages_in_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "test_logs")
            logger = AsyncLogger(log_dir, enabled=True)
            for i in range(10):
                logger.log(f"message_{i}")
            logger.close()
            with open(os.path.join(log_dir, "log.txt")) as f:
                lines = f.readlines()
            messages = [line for line in lines if "message_" in line]
            assert len(messages) == 10
            for i, line in enumerate(messages):
                assert f"message_{i}" in line

    def test_thread_safety_concurrent_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "test_logs")
            logger = AsyncLogger(log_dir, enabled=True)
            errors = []

            def writer(thread_id):
                try:
                    for i in range(50):
                        logger.log(f"thread_{thread_id}_msg_{i}")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            logger.close()

            assert len(errors) == 0
            with open(os.path.join(log_dir, "log.txt")) as f:
                content = f.read()
            # All 250 messages should be present
            for t in range(5):
                for i in range(50):
                    assert f"thread_{t}_msg_{i}" in content

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "test_logs")
            logger = AsyncLogger(log_dir, enabled=True)
            logger.log("msg")
            logger.close()
            logger.close()  # should not raise

    def test_log_after_close_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "test_logs")
            logger = AsyncLogger(log_dir, enabled=True)
            logger.close()
            logger.log("after close")  # should not raise


class TestSelfPlayLogger:
    def test_creates_jsonl_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SelfPlayLogger(tmpdir, enabled=True)
            logger.log("test_event", key1="value1")
            logger.close()
            log_file = os.path.join(tmpdir, "self_play_events.jsonl")
            assert os.path.isfile(log_file)
            with open(log_file) as f:
                line = f.readline()
            data = json.loads(line)
            assert data["event"] == "test_event"
            assert data["key1"] == "value1"
            assert "ts" in data

    def test_disabled_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SelfPlayLogger(tmpdir, enabled=False)
            logger.log("test_event")
            logger.close()
            log_file = os.path.join(tmpdir, "self_play_events.jsonl")
            assert not os.path.isfile(log_file)

    def test_multiple_events_ordered(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SelfPlayLogger(tmpdir, enabled=True)
            for i in range(5):
                logger.log(f"event_{i}", index=i)
            logger.close()
            log_file = os.path.join(tmpdir, "self_play_events.jsonl")
            with open(log_file) as f:
                lines = f.readlines()
            assert len(lines) == 5
            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["event"] == f"event_{i}"
                assert data["index"] == i

    def test_thread_safety(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SelfPlayLogger(tmpdir, enabled=True)
            errors = []

            def writer(tid):
                try:
                    for i in range(20):
                        logger.log("evt", tid=tid, idx=i)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            logger.close()

            assert len(errors) == 0
            with open(os.path.join(tmpdir, "self_play_events.jsonl")) as f:
                lines = f.readlines()
            assert len(lines) == 100

    def test_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = SelfPlayLogger(tmpdir, enabled=True)
            logger.log("evt")
            logger.close()
            logger.close()
