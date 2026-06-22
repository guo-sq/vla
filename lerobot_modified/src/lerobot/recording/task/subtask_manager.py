from collections import defaultdict

from lerobot.recording.utils.tts import log_say


class SubTaskManager:
    """
    子任务管理器，管理子任务计时、播报和索引跟踪。

    示例：durations = [10, 5, 20] 表示 3 个子任务，时长分别为 10s, 5s, 20s。
    """

    def __init__(
        self,
        durations: list[float] | None,
        is_inference_mode: bool,
        subtask_ind: list[int] | None = None,
        play_sounds: bool = True,
        enable_log_say: bool = True,
    ):
        self.durations = durations or []
        self.play_sounds = play_sounds
        self.enable_log_say = enable_log_say
        self.subtask_ind = subtask_ind
        if self.subtask_ind:
            assert len(self.subtask_ind) == len(self.durations), \
                "subtask_ind length must match durations length"
        else:
            self.subtask_ind = list(range(len(self.durations)))

        # 纯推理模式或无子任务时禁用
        if is_inference_mode or not self.durations:
            self.enabled = False
        else:
            self.enabled = all(d > 0 for d in self.durations)

        if self.enabled:
            self.timestamps = []
            cumsum = 0.0
            for d in self.durations:
                cumsum += d
                self.timestamps.append(cumsum)
            self.total_duration = self.timestamps[-1]
            self.current_index = -1
            self.announced = defaultdict(bool)
            self.finished_announced = False                
        else:
            self.timestamps = []
            self.total_duration = 0.0
            self.current_index = -1
            self.announced = defaultdict(bool)
            self.finished_announced = False

    def update(self, timestamp: float) -> int:
        if not self.enabled:
            return -1

        if timestamp >= self.total_duration:
            if not self.finished_announced:
                self.finished_announced = True
                self.current_index = -1
                log_say("结束所有步骤", play_sounds=self.play_sounds, blocking=True, enabled=self.enable_log_say)
            return -1

        new_index = 0
        for i, (ts, subtask_id) in enumerate(zip(self.timestamps, self.subtask_ind)):
            if timestamp < ts:
                new_index = subtask_id - 1
                break

        if not self.announced[new_index]:
            self.announced[new_index] = True
            self.current_index = new_index
            log_say(f"开始第{new_index + 1}个步骤", play_sounds=self.play_sounds, blocking=False, enabled=self.enable_log_say)

        return self.current_index

    def get_current_index(self) -> int:
        return self.current_index if self.enabled else -1

    def is_finished(self) -> bool:
        return self.finished_announced