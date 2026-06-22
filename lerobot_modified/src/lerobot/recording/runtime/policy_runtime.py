"""Policy inference runtime: client, action buffer, background thread.

Encapsulates all inference-related state and provides the ActionSource
protocol for the control loop.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from typing import Any

import numpy as np

from lerobot.recording.utils.logging import ActionChunkLogger, AsyncLogger, debug_print


# ---------------------------------------------------------------------------
# Auto-optimal infer_interval
# ---------------------------------------------------------------------------

def compute_optimal_infer_interval(
    latencies: list[float],
    effective_fps: int,
    action_horizon: int,
    margin: int = 3,
) -> dict:
    """Compute optimal infer_interval from measured inference latencies.

    Args:
        latencies: List of measured inference wall-clock times (seconds).
        effective_fps: Control loop frequency (after speedup).
        action_horizon: Number of actions per inference chunk.
        margin: Base safety margin (steps).

    Returns:
        Dict with keys: infer_interval, latency_steps_mean, latency_steps_3sigma,
        feasible, warning (optional).
    """
    mu = sum(latencies) / len(latencies)
    sigma = (sum((t - mu) ** 2 for t in latencies) / len(latencies)) ** 0.5
    # Floor sigma at 15% of mean to guard against small warmup sample
    sigma = max(sigma, mu * 0.15)

    L_mean = math.ceil(mu * effective_fps)
    L_3sigma = math.ceil((mu + 3 * sigma) * effective_fps)

    result: dict = {
        "latency_mean_ms": round(mu * 1000, 1),
        "latency_sigma_ms": round(sigma * 1000, 1),
        "latency_steps_mean": L_mean,
        "latency_steps_3sigma": L_3sigma,
    }

    if L_3sigma >= action_horizon:
        result["feasible"] = False
        result["infer_interval"] = max(1, action_horizon - L_mean - margin)
        result["warning"] = (
            f"Inference too slow: L_3σ={L_3sigma} >= H={action_horizon}. "
            f"Reduce inference_speedup or increase action_horizon."
        )
        return result

    I_max = action_horizon - L_3sigma - margin
    I_min = L_mean
    I_optimal = max(I_min, I_max)

    result["feasible"] = True
    result["infer_interval"] = I_optimal
    return result


# ---------------------------------------------------------------------------
# Gaussian smoothing (pure numpy, no scipy dependency)
# ---------------------------------------------------------------------------

def _gaussian_smooth_1d(data: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1-D Gaussian smoothing along axis 0.  Pure numpy implementation.

    Args:
        data: Array of shape (T, D) — time steps × action dims.
        sigma: Standard deviation of Gaussian kernel.

    Returns:
        Smoothed array, same shape as input.
    """
    if sigma <= 0 or len(data) <= 1:
        return data
    radius = int(3 * sigma + 0.5)  # ±3σ covers 99.7%
    if radius == 0:
        return data
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    # Pad with edge values to avoid boundary artifacts
    padded = np.pad(data, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(data, dtype=np.float64)
    for i in range(len(kernel)):
        out += kernel[i] * padded[i : i + len(data)]
    return out.astype(data.dtype)


# ---------------------------------------------------------------------------
# ActionBuffer
# ---------------------------------------------------------------------------

class ActionBuffer:
    """Thread-safe time-aligned action buffer with optional fusion.

    Stores a chunk of future actions from policy inference and dispenses
    them one at a time.  When a new chunk arrives, it can be fused with
    remaining buffered actions using linear or exponential weights.
    """

    def __init__(
        self,
        robot_state_keys: list[str],
        action_dim: int,
        action_horizon: int,
        fusion_type: str = "linear",
        fusion_exp_decay: float = 0.5,
        smooth_sigma: float = 0.0,
        fusion_window: int = 5,
    ):
        self._keys = list(robot_state_keys)
        self._action_dim = action_dim
        self._action_horizon = action_horizon
        self._fusion_type = fusion_type
        self._fusion_exp_decay = fusion_exp_decay
        self._smooth_sigma = smooth_sigma
        self._fusion_window = fusion_window
        self._lock = threading.Lock()
        self._buffer: np.ndarray | None = None
        self._available_cnt: int = 0
        self._is_cleared: bool = False

    @property
    def available_count(self) -> int:
        with self._lock:
            return self._available_cnt

    def init_buffer(self, chunk: np.ndarray) -> None:
        with self._lock:
            self._buffer = np.copy(np.array(chunk, dtype=np.float32))
            self._available_cnt = len(self._buffer)
            self._is_cleared = False

    def is_empty(self) -> bool:
        with self._lock:
            return self._available_cnt <= 0

    def clear(self) -> None:
        with self._lock:
            if self._buffer is not None:
                self._buffer.fill(0.0)
            self._available_cnt = 0
            self._is_cleared = True

    def get_next_action(self) -> dict[str, Any] | None:
        """Consume the next action from the front of the buffer (shift-left)."""
        with self._lock:
            if self._available_cnt <= 0:
                return None
            row = self._buffer[0].copy()
            # Shift buffer left by one
            self._buffer[:-1] = self._buffer[1:]
            self._buffer[-1] = 0.0
            self._available_cnt -= 1
            return {k: float(row[i]) for i, k in enumerate(self._keys)}

    def get_future_actions(self, default_delay: int = 0) -> tuple[np.ndarray, int, int]:
        """Return the full buffer as prefix for the next inference call.

        Returns the entire buffer (including consumed/zero slots) as the
        old code did — the server uses action_mask to know which entries
        are valid.

        Returns:
            (prefix_array, infer_delay, valid_length)
        """
        with self._lock:
            if self._available_cnt <= 0:
                return np.zeros((self._action_horizon, self._action_dim), dtype=np.float32), 0, 0
            actual_delay = min(default_delay, self._available_cnt) if default_delay > 0 else 0
            return np.copy(self._buffer), actual_delay, self._available_cnt

    def update_from_inference(
        self,
        new_chunk: np.ndarray,
        start_step: int,
        current_step: int,
    ) -> None:
        """Update buffer with a new inference result.

        Compensates for inference latency by skipping stale actions:
        latency_steps = current_step - start_step, then only uses
        new_chunk[latency_steps:].

        Fusion is decided atomically under the lock: if the buffer still
        has remaining actions, the new chunk is blended with them.
        """
        new_chunk = np.array(new_chunk, dtype=np.float32)
        with self._lock:
            if self._buffer is None or self._is_cleared:
                self._is_cleared = False
                self._buffer = np.copy(new_chunk)
                self._available_cnt = len(self._buffer)
                return

            # Skip stale actions that correspond to steps already passed
            latency_steps = current_step - start_step
            if latency_steps >= len(new_chunk):
                # Promoted from debug to warning: a fully-discarded chunk means
                # inference was too slow for the configured infer_interval and
                # the policy is effectively running open-loop on stale buffer
                # actions. Frequent discards correlate with degraded behavior.
                logging.warning(
                    f"ActionBuffer: discarding stale chunk "
                    f"(latency_steps={latency_steps} >= chunk_len={len(new_chunk)})"
                )
                return  # entire chunk is stale

            valid_new = new_chunk[latency_steps:]

            # Guard: truncate to buffer size if inference returned oversized chunk
            if len(valid_new) > len(self._buffer):
                valid_new = valid_new[:len(self._buffer)]

            if self._available_cnt > 0:
                # Limit fusion to a configurable window: enough to prevent
                # physical jitter, short enough for policy responsiveness.
                # fusion_window=0 disables fusion (backward compat / infer_delay>0).
                num_to_fuse = min(self._fusion_window, self._available_cnt, len(valid_new))
                if num_to_fuse > 0 and self._fusion_type in ("linear", "exponential"):
                    for i in range(num_to_fuse):
                        if self._fusion_type == "linear":
                            w_old = 1.0 - (i + 1) / (num_to_fuse + 1)
                        else:
                            w_old = self._fusion_exp_decay ** (i + 1)
                        self._buffer[i] = self._buffer[i] * w_old + valid_new[i] * (1.0 - w_old)
                # Fill remaining new actions beyond the fused region
                if len(valid_new) > num_to_fuse:
                    self._buffer[num_to_fuse:len(valid_new)] = valid_new[num_to_fuse:]
            else:
                self._buffer = np.zeros_like(self._buffer)
                self._buffer[:len(valid_new)] = valid_new

            self._available_cnt = len(valid_new)

            # Gaussian smoothing AFTER fusion: smooths both model output
            # noise AND the fusion boundary jump in a single pass.
            if self._smooth_sigma > 0 and self._available_cnt > 1:
                self._buffer[:self._available_cnt] = _gaussian_smooth_1d(
                    self._buffer[:self._available_cnt], self._smooth_sigma
                )


# ---------------------------------------------------------------------------
# queue_put_replace_oldest
# ---------------------------------------------------------------------------

def queue_put_replace_oldest(
    q: queue.Queue,
    item: Any,
    logger: AsyncLogger | None = None,
) -> bool:
    """Put item into queue. If full, drop oldest item first.

    Returns True if an old item was replaced.
    """
    try:
        q.put_nowait(item)
        return False
    except queue.Full:
        try:
            old = q.get_nowait()
            if logger:
                step_id = old[4] if isinstance(old, tuple) and len(old) > 4 else "?"
                logger.log(f"Queue full, dropped old item (step_id={step_id})")
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            return True
        except queue.Full:
            if logger:
                logger.log("Warning: queue still full after removing old item")
            return False


# ---------------------------------------------------------------------------
# inference_worker
# ---------------------------------------------------------------------------

def inference_worker(
    policy_client: Any,
    input_queue: queue.Queue,
    output_queue: queue.Queue,
    stop_event: threading.Event,
    logger: AsyncLogger,
    worker_idle_event: threading.Event | None = None,
) -> None:
    """Background thread that runs policy inference.

    Reads (lang_prompt, obs, action_prefix, infer_delay, step_id) tuples
    from input_queue.  Puts (action_chunk, step_id) on output_queue.
    On exception, puts (None, step_id) to signal failure to main thread.
    """
    # Rolling-window stats for inference latency. Per-call line is already
    # logged below; this summary surfaces tail latency without needing to
    # post-process the logs.
    latency_window: list[float] = []
    error_count_window = 0
    summary_every_n = 30

    while not stop_event.is_set():
        try:
            item = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        if item is None:
            break

        if worker_idle_event:
            worker_idle_event.clear()

        lang_prompt, obs_dict, action_prefix, infer_delay, valid_len, step_id = item
        t0 = time.perf_counter()

        try:
            action_chunk = policy_client.get_action(
                obs_dict, action_prefix, infer_delay, valid_len, lang_prompt
            )
            dt_ms = (time.perf_counter() - t0) * 1e3
            latency_window.append(dt_ms)
            debug_print(f"inference_worker: step={step_id}, time={dt_ms:.1f}ms, chunk_shape={action_chunk.shape if hasattr(action_chunk, 'shape') else 'N/A'}")
            logger.log(
                f"inference_worker: step={step_id}, time={dt_ms:.1f}ms"
            )
            output_queue.put((action_chunk, step_id))
        except Exception as e:
            dt_ms = (time.perf_counter() - t0) * 1e3
            error_count_window += 1
            logging.warning(
                f"inference_worker: error at step={step_id}: {e} ({dt_ms:.1f}ms)"
            )
            debug_print(f"inference_worker ERROR: step={step_id}, error={e}, time={dt_ms:.1f}ms")
            logger.log(
                f"inference_worker: ERROR at step={step_id}: {e} ({dt_ms:.1f}ms)"
            )
            output_queue.put((None, step_id))
        finally:
            if worker_idle_event:
                worker_idle_event.set()

        if len(latency_window) >= summary_every_n:
            sw = sorted(latency_window)
            n = len(sw)
            mean = sum(sw) / n
            p50 = sw[n // 2]
            p99 = sw[min(n - 1, int(0.99 * n))]
            max_ = sw[-1]
            logger.log(
                f"inference_worker stats (last {n}): "
                f"mean={mean:.1f}ms p50={p50:.1f}ms p99={p99:.1f}ms max={max_:.1f}ms "
                f"errors={error_count_window}"
            )
            latency_window = []
            error_count_window = 0


# ---------------------------------------------------------------------------
# PolicyRuntime
# ---------------------------------------------------------------------------

class PolicyRuntime:
    """Full inference runtime: client + buffer + background thread.

    Implements ActionSource to plug into the control loop.
    """

    def __init__(
        self,
        policy_client: Any,
        robot: Any,
        lang_prompt: str,
        logger: AsyncLogger,
        *,
        inference_mode: str = "async",
        action_horizon: int = 50,
        fusion_type: str = "linear",
        fusion_exp_decay: float = 0.5,
        infer_interval: int = 1,
        default_infer_delay: int = 0,
        transition_steps: int = 10,
        effective_fps: int = 30,
        auto_infer_interval: bool = False,
        smooth_sigma: float = 0.0,
        fusion_window: int = 5,
        chunk_logger: ActionChunkLogger | None = None,
        inference_log_writer: Any = None,
        role: str = "",
    ):
        self.client = policy_client
        self.robot = robot
        self.logger = logger
        self.lang_prompt = lang_prompt
        self.chunk_logger = chunk_logger
        # Optional InferenceLogWriter — appended to on every successful chunk
        # receipt when ``inference.persist_inference_log`` is set. ``role`` is
        # only meaningful in self_play (one writer instance is shared across
        # per-role PolicyRuntimes); set at construction so each runtime tags
        # its rows correctly.
        self.inference_log_writer = inference_log_writer
        self.role = role
        # Set by the recorder before each episode starts so writer rows carry
        # the correct episode_index.
        self.current_episode_index: int = 0

        self.inference_mode = inference_mode
        self.infer_interval = infer_interval
        self.default_infer_delay = default_infer_delay
        # Scale transition_steps to preserve wall-clock duration at higher fps
        self.transition_steps = max(1, int(transition_steps * effective_fps / 30))
        self.transition_weight: float = 0.0
        self._effective_fps = effective_fps
        self._auto_infer_interval = auto_infer_interval
        self._last_stutter_ratio: float = 0.0

        # Determine dims from client
        self.robot_state_keys: list[str] = list(policy_client.robot_state_keys)
        self.action_dim: int = len(self.robot_state_keys)

        # Action buffer (thread-safe)
        self.action_buffer = ActionBuffer(
            robot_state_keys=self.robot_state_keys,
            action_dim=self.action_dim,
            action_horizon=action_horizon,
            fusion_type=fusion_type,
            fusion_exp_decay=fusion_exp_decay,
            smooth_sigma=smooth_sigma,
            fusion_window=fusion_window,
        )

        # Inference thread
        self.is_async = inference_mode == "async"
        self.input_queue: queue.Queue = queue.Queue(maxsize=1 if self.is_async else 0)
        self.output_queue: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        # step_id → (t_submit, prompt_at_submit, infer_delay_at_submit).
        # Used to compute end-to-end inference latency for the optional
        # InferenceLogWriter without changing the queue tuple shape (so
        # existing tests + workers stay compatible).
        self._pending_submits: dict[int, tuple[float, str, int]] = {}
        self._worker_idle = threading.Event()
        self._worker_idle.set()  # idle initially

        self._thread = threading.Thread(
            target=inference_worker,
            args=(policy_client, self.input_queue, self.output_queue, self.stop_event, logger, self._worker_idle),
            daemon=True,
        )
        self._thread.start()

        # Step counter
        self.global_step_id: int = 0
        self._consecutive_inference_failures: int = 0

        # Warmup
        try:
            self._warmup(lang_prompt, action_horizon)
        except Exception:
            self.cleanup()
            raise

    def _warmup(self, lang_prompt: str, action_horizon: int) -> None:
        self.logger.log("PolicyRuntime: warmup inference...")
        zero_prefix = np.zeros((action_horizon, self.action_dim), dtype=np.float32)
        obs = self.robot.get_observation()
        try:
            # Measure latency over multiple warmup calls
            # Extra call when auto-tuning: first call may be slow (cold start)
            n_warmup = 4 if self._auto_infer_interval else 1
            latencies: list[float] = []
            chunk = None
            for _ in range(n_warmup):
                t0 = time.perf_counter()
                chunk = self.client.get_action(obs, zero_prefix, 0, 0, lang_prompt)
                latencies.append(time.perf_counter() - t0)

            self.action_buffer.init_buffer(chunk)
            self.logger.log(f"PolicyRuntime: warmup done, chunk shape={chunk.shape}")

            # Debug: log first action from warmup
            first_action = self.action_buffer.get_next_action()
            if first_action:
                vals = list(first_action.values())[:5]
                debug_print(f"Warmup first action (first 5 values): {vals}")
                self.action_buffer.init_buffer(chunk)

            # Auto-adjust infer_interval based on measured latency
            if self._auto_infer_interval and self.is_async:
                # Discard first measurement (cold start bias)
                calibration_latencies = latencies[1:] if len(latencies) > 1 else latencies
                result = compute_optimal_infer_interval(
                    latencies=calibration_latencies,
                    effective_fps=self._effective_fps,
                    action_horizon=action_horizon,
                )
                old_I = self.infer_interval
                self.infer_interval = result["infer_interval"]
                auto_msg = (
                    f"Auto infer_interval: {old_I} → {result['infer_interval']} "
                    f"(T={result['latency_mean_ms']:.0f}±{result['latency_sigma_ms']:.0f}ms, "
                    f"L_mean={result['latency_steps_mean']}, "
                    f"L_3σ={result['latency_steps_3sigma']}, "
                    f"feasible={result['feasible']})"
                )
                self.logger.log(auto_msg)
                logging.info(auto_msg)
                if not result["feasible"]:
                    warn_msg = f"⚠️ {result.get('warning', '')}"
                    self.logger.log(warn_msg)
                    logging.warning(warn_msg)

        except Exception as e:
            self.logger.log(f"PolicyRuntime: warmup failed: {e}")
            raise

    def get_action(self, observation: dict, step_id: int) -> dict | None:
        """ActionSource protocol. Returns next action from buffer."""
        if self.is_async:
            return self._get_action_async(observation, step_id)
        else:
            return self._get_action_sync(observation, step_id)

    def _log_inference(self, raw_chunk: np.ndarray, start_step: int) -> None:
        """Match a received chunk with its earlier submission and append a row
        to the optional InferenceLogWriter. Also prunes pending entries older
        than ``start_step`` so dropped submissions don't accumulate."""
        # Always prune stale pending entries; this stays bounded even if no
        # writer is configured.
        stale = [k for k in self._pending_submits if k < start_step]
        for k in stale:
            self._pending_submits.pop(k, None)
        submit_info = self._pending_submits.pop(start_step, None)
        if self.inference_log_writer is None or submit_info is None:
            return
        t_submit, prompt_at_submit, delay_at_submit = submit_info
        try:
            self.inference_log_writer.append(
                episode_index=self.current_episode_index,
                step_id=start_step,
                t_submit=t_submit,
                t_complete=time.perf_counter(),
                action_chunk_raw=raw_chunk,
                prompt=prompt_at_submit,
                delay=delay_at_submit,
                role=self.role,
            )
        except Exception as e:  # pragma: no cover - log writer should never break inference
            logging.warning("InferenceLogWriter.append failed: %s", e)

    def _get_action_async(self, observation: dict, step_id: int) -> dict | None:
        submitted_delay = 0
        if step_id % self.infer_interval == 0:
            prefix, submitted_delay, valid_len = self.action_buffer.get_future_actions(self.default_infer_delay)
            avail = self.action_buffer.available_count
            debug_print(f"Submit inference: step={step_id}, delay={submitted_delay}, valid_len={valid_len}, buffer_avail={avail}")
            self._pending_submits[step_id] = (time.perf_counter(), self.lang_prompt, submitted_delay)
            queue_put_replace_oldest(
                self.input_queue,
                (self.lang_prompt, observation, prefix, submitted_delay, valid_len, step_id),
                self.logger,
            )

        # Check for new inference results
        try:
            new_chunk, start_step = self.output_queue.get_nowait()
            if new_chunk is not None:
                self._consecutive_inference_failures = 0
                self._log_inference(new_chunk, start_step)
                if self.chunk_logger is not None:
                    self.chunk_logger.log_chunk(new_chunk, start_step, step_id, self.lang_prompt)
                latency = step_id - start_step
                debug_print(f"Consuming result: start_step={start_step}, current_step={step_id}, latency={latency}, chunk_len={len(new_chunk)}")
                self.action_buffer.update_from_inference(
                    new_chunk, start_step, step_id,
                )
                debug_print(f"Buffer after update: available_cnt={self.action_buffer.available_count}")
            else:
                self._consecutive_inference_failures += 1
                debug_print(f"Got None result from worker for start_step={start_step}")
                if self._consecutive_inference_failures >= 3:
                    logging.warning(
                        f"PolicyRuntime: {self._consecutive_inference_failures} consecutive "
                        f"inference failures (last at step {step_id})"
                    )
        except queue.Empty:
            pass

        # If buffer is empty, busy-wait for inference result (up to 3s)
        if self.action_buffer.is_empty():
            debug_print(f"Buffer empty at step {step_id}, waiting for inference...")
            wait_start = time.perf_counter()
            while self.action_buffer.is_empty():
                time.sleep(0.001)
                try:
                    new_chunk, start_step = self.output_queue.get_nowait()
                    if new_chunk is not None:
                        self._consecutive_inference_failures = 0
                        self._log_inference(new_chunk, start_step)
                        if self.chunk_logger is not None:
                            self.chunk_logger.log_chunk(new_chunk, start_step, step_id, self.lang_prompt)
                        self.action_buffer.update_from_inference(
                            new_chunk, start_step, step_id,
                        )
                    else:
                        self._consecutive_inference_failures += 1
                except queue.Empty:
                    pass
                if time.perf_counter() - wait_start > 3.0:
                    logging.warning(
                        f"PolicyRuntime: inference timeout at step {step_id} "
                        f"(worker_alive={self._thread.is_alive()}, "
                        f"consecutive_failures={self._consecutive_inference_failures})"
                    )
                    self.logger.log(f"Warning: inference timeout at step {step_id}")
                    return None

        return self.action_buffer.get_next_action()

    def _get_action_sync(self, observation: dict, step_id: int) -> dict | None:
        if self.action_buffer.is_empty():
            zero_prefix = np.zeros(
                (self.action_buffer._action_horizon, self.action_dim),
                dtype=np.float32,
            )
            t0 = time.perf_counter()
            self._pending_submits[step_id] = (t0, self.lang_prompt, 0)
            self.input_queue.put(
                (self.lang_prompt, observation, zero_prefix, 0, 0, step_id)
            )
            try:
                result = self.output_queue.get(timeout=10.0)
            except queue.Empty:
                logging.error(f"PolicyRuntime sync: no inference response after 10s at step {step_id}")
                return None
            infer_time = time.perf_counter() - t0
            chunk, start_step = result
            if chunk is None:
                logging.warning(f"PolicyRuntime sync: inference failed at step {step_id}")
                return None
            self._log_inference(chunk, start_step)
            if self.chunk_logger is not None:
                self.chunk_logger.log_chunk(chunk, start_step, step_id, self.lang_prompt)
            self.action_buffer.init_buffer(chunk)

            # Compute and monitor stutter ratio
            H = self.action_buffer._action_horizon
            buffer_duration = H / self._effective_fps
            self._last_stutter_ratio = infer_time / (infer_time + buffer_duration)
            if self._last_stutter_ratio > 0.15:
                stutter_msg = (
                    f"Sync stutter={self._last_stutter_ratio*100:.1f}% "
                    f"(T={infer_time*1000:.0f}ms, buffer={buffer_duration*1000:.0f}ms). "
                    f"Consider switching to async mode."
                )
                self.logger.log(stutter_msg)
                logging.warning(stutter_msg)

        return self.action_buffer.get_next_action()

    def reinit(self, lang_prompt: str) -> None:
        """Clear queues and buffer, re-warmup with new prompt."""
        self.lang_prompt = lang_prompt
        self.clear_queues()
        # Wait for worker to finish any in-flight inference before
        # doing synchronous warmup (avoid concurrent websocket calls)
        if not self._worker_idle.wait(timeout=5.0):
            logging.warning("PolicyRuntime.reinit: worker did not become idle within 5s")
        self.clear_queues()  # drain any result that arrived during wait
        self.action_buffer.clear()
        # Cross-episode reinit invalidates every pending submit — those
        # chunks are stale by definition. Drop them so InferenceLogWriter
        # doesn't attribute their result to the wrong episode.
        self._pending_submits.clear()
        self._warmup(lang_prompt, self.action_buffer._action_horizon)

    def update_prompt(self, lang_prompt: str) -> None:
        """Change prompt without full reinit."""
        self.lang_prompt = lang_prompt

    def clear_queues(self) -> None:
        for q in [self.input_queue, self.output_queue]:
            while True:
                try:
                    q.get_nowait()
                except queue.Empty:
                    break

    def reset_for_resume(self) -> None:
        """Reset transition weight after human intervention ends."""
        self.transition_weight = 0.9

    def get_value_score(
        self, observation: dict, lang_prompt: str, timeout_s: float = 5.0
    ) -> float | None:
        """Get value score from value model. Returns None on failure."""
        if not hasattr(self.client, "get_value_score"):
            return None
        try:
            result = self.client.get_value_score(observation, lang=lang_prompt)
            if result is not None and "value" in result:
                val = float(result["value"])
                if np.isfinite(val):
                    return val
        except Exception as e:
            self.logger.log(f"PolicyRuntime: value score failed: {e}")
        return None

    def cleanup(self) -> None:
        """Stop inference thread and verify it terminated."""
        self.stop_event.set()
        try:
            self.input_queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=3.0)
        if self._thread.is_alive():
            logging.error("PolicyRuntime: inference thread did not stop after timeout")
