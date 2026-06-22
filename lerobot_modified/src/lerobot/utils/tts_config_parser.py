import os
import logging
import yaml
import random

from lerobot.datasets.utils import (
    load_jsonlines,
)


def load_yaml_file(yaml_path: str):
    with open(yaml_path, "r", encoding="utf-8") as f:
        yaml_content = yaml.safe_load(f)
    return yaml_content


def parse_record_config(yaml_path, episode_max_time_s, record_task=None):
    """Parse recording config YAML and build sub-episode and frame-state schedules.

    Returns a dict with at least the keys:
      - sub_episode_time_s: list[float]
      - sub_episode_taskinds: list[int]
      - sub_episode_task_json: str (path)
      - frame_state_schedule: dict[int, list[dict]] (optional) where each dict has
          'state', 'start_s', 'end_s'

    If `record_task` is not provided, the first top-level key in the YAML is used.
    """
    record_cfg = load_yaml_file(yaml_path)
    if record_task is None:
        # pick first task key if not specified
        if not isinstance(record_cfg, dict) or len(record_cfg) == 0:
            raise ValueError(f"Empty or invalid record config file: {yaml_path}")
        record_task = next(iter(record_cfg))

    task_record_cfg = record_cfg[record_task]
    sub_episode_task_json = task_record_cfg["sub_episode_task_json"]
    assert os.path.exists(sub_episode_task_json), \
        f"sub_episode_task_json file {sub_episode_task_json} does not exist!"
    sub_episode_task_annos = load_jsonlines(sub_episode_task_json)
    sub_episode_task_annos = {
        item["abbrevation"]: item["subtask_index"] 
        for item in sorted(sub_episode_task_annos, key=lambda x: x["subtask_index"])
    }

    subtask_time_s = task_record_cfg["subtask_time_s"]
    # dur_set_total_time = 0
    # for key, dur in subtask_time_s.items():
    #     dur_set_total_time += dur
    #     assert dur_set_total_time <= episode_max_time_s, \
    #         f"pls check the subtask duration setting, must equal episode_time-{episode_max_time_s}"

    task_ind_duration = {}
    for key, value in subtask_time_s.items():
        assert key in sub_episode_task_annos, \
            f"{key} not in subtask abbrevations."
        task_ind_duration[sub_episode_task_annos[key]] = value

    # task_episode_time_s = [start_record_time_s]
    # for i in range(len(sub_episode_task_annos) - 1):
    #     dur_t = task_ind_duration[i]
    #     start_record_time_s = start_record_time_s + dur_t
    #     task_episode_time_s.append(start_record_time_s)
    is_loop_task_record = task_record_cfg.get("loop_task_record", False)
    task_record_mode = task_record_cfg["record_mode"]
    task_record_mode_cfg = task_record_cfg[task_record_mode]
    sub_episode_taskinds = task_record_mode_cfg.get(
        "sub_episode_taskinds", 
        list(range(len(sub_episode_task_annos)+1))[1:]
    )
    sub_episode_taskinds = sorted(set(sub_episode_taskinds))
    if len(sub_episode_taskinds) == 0:
        logging.warning(f"sub_episode_taskinds is empty in record config for task {record_task} mode {task_record_mode}!")
        return None
    
    start_record_time_s = task_record_mode_cfg.get("start_record_time_s", 0)
    assert start_record_time_s < episode_max_time_s, \
        f"start_record-{start_record_time_s} must small than episode_time-{episode_max_time_s}"
    sub_episode_time_s = [start_record_time_s]
    for i in range(len(sub_episode_taskinds) - 1):
        task_ind = sub_episode_taskinds[i] - 1
        start_record_time_s += task_ind_duration[task_ind]
        assert start_record_time_s <= episode_max_time_s, \
            f"pls set suitbale start_record time, subtask time larger than episode_time-{episode_max_time_s}"
        sub_episode_time_s.append(start_record_time_s)
    sub_episode_time_s = sorted(set(sub_episode_time_s))

    sub_episode_dur_time = []
    for i in range(len(sub_episode_taskinds)):
        task_ind = sub_episode_taskinds[i] - 1
        sub_episode_dur_time.append(task_ind_duration[task_ind])
    
    parse_res = {
        "sub_episode_time_s": sub_episode_time_s,
        "sub_episode_taskinds": sub_episode_taskinds,
        "sub_episode_dur_time": sub_episode_dur_time,
        "sub_episode_task_json": sub_episode_task_json,
        "frame_state_mapping": task_record_cfg["frame_state_mapping"],
        "is_loop_task_record": is_loop_task_record,
    }
    # 如果是循环录制模式，单次流程录制时间为要录制任务时间的和。start_record_time_s最好为0
    sub_task_record_time = sum(sub_episode_dur_time)
    if is_loop_task_record:
        num_loop = (episode_max_time_s - task_record_mode_cfg["start_record_time_s"]) // sub_task_record_time
        assert num_loop > 1, \
            f"episode_max_time must be a multiple of the subtask recording time. "
        parse_res["loop_task_record_cfgs"] = {
            "is_loop_task_record": is_loop_task_record,
            "num_loop": num_loop
        }
    # 考虑到子任务如果按规定时间已经结束，但录制还没有结束
    record_sub_task_till_episode_end = task_record_mode_cfg.get("record_sub_task_till_episode_end", False)
    last_state_dur = task_ind_duration[sub_episode_taskinds[-1] - 1]
    subtask_end_time = start_record_time_s + last_state_dur
    if not record_sub_task_till_episode_end:
        if not is_loop_task_record:
            episode_remaining_dur = episode_max_time_s - start_record_time_s - last_state_dur
        else:
            episode_remaining_dur = subtask_end_time - start_record_time_s - last_state_dur
        if episode_remaining_dur > 0:
            episode_remaining_state = -1 # invalid
            parse_res["episode_remaining_dur_cfg"] = {
                "episode_remaining_start_time": subtask_end_time,
                "episode_remaining_state": episode_remaining_state
            }
    else:
        if not is_loop_task_record:
            subtask_end_time = episode_max_time_s
    
    record_state_subtask = task_record_mode_cfg.get("record_state_subtask", [])
    if len(record_state_subtask) == 0:
        return parse_res

    is_subset = set(record_state_subtask).issubset(set(sub_episode_taskinds))
    if not is_subset:
        return parse_res

    frame_state_prob = task_record_mode_cfg.get("frame_state_prob")
    if not isinstance(frame_state_prob, dict) or len(frame_state_prob) == 0:
        raise ValueError("frame_state_prob must be a non-empty mapping state->probability in config.")

    # normalize frame_state_prob to a probability distribution
    total_p = sum(float(v) for v in frame_state_prob.values())
    if total_p <= 0:
        raise ValueError("Sum of frame_state_prob probabilities must be > 0")
    normalized_states = []
    normalized_probs = []
    for k, v in frame_state_prob.items():
        normalized_states.append(k)
        normalized_probs.append(float(v) / total_p)
    
    state_record_dur_s = task_record_mode_cfg["state_record_dur_s"]
    state_boundary_buffer_s = task_record_mode_cfg["state_boundary_buffer_s"]
    state_selection_mode = task_record_mode_cfg["state_selection_mode"]
    max_state_records_per_sub_episode = task_record_mode_cfg["max_state_records_per_sub_episode"]

    # Build schedule for eligible sub-episodes
    frame_state_schedule: dict[int, list[dict]] = {}

    # We treat sub_episode_time_s as the times at which a new sub-episode begins.
    # For scheduling durations we compute duration = next_time - cur_time when possible.
    times = sub_episode_time_s
    inds = sub_episode_taskinds
    # Map taskind -> index in times
    time_map = {inds[i]: i for i in range(len(inds))}

    for i in range(len(record_state_subtask)):
        taskind = record_state_subtask[i]
        state_record_dur_s[i].sort()
        min_state_record_dur_s = state_record_dur_s[i][0]
        max_state_record_dur_s = state_record_dur_s[i][1]
        idx = time_map[taskind]
        start_t = float(times[idx])
        if idx >= len(times) - 1:
            available_dur = subtask_end_time - float(times[idx])
            end_t = subtask_end_time
        else:
            end_t = float(times[idx + 1])
            available_dur = end_t - start_t
        if available_dur < min_state_record_dur_s:
            logging.warning(f"Available duration for frame state is <2s ({available_dur}s); skipping state recording.")
            continue

        # maximum number of non-overlapping state recordings in this stage (each at least 2s)
        max_possible = min(
            max_state_records_per_sub_episode[i], 
            int(available_dur // min_state_record_dur_s)
        )
        if max_possible <= 0:
            continue
        if max_possible == 1:
            n_rec = 1
        else:
            # choose number of recordings up to max_possible (respect config limit)
            n_rec = random.randint(1, max_possible)

        # enforce boundary buffer away from subtask start/end
        # `state_boundary_buffer_s` can be provided as:
        # - a single number: treated as seconds if >1 else proportion of sub-episode duration
        # - a two-element list/tuple: [start_buffer, end_buffer] (same interpretation per element)
        raw_buffer = state_boundary_buffer_s[i]
        if isinstance(raw_buffer, (list, tuple)) and len(raw_buffer) >= 2:
            start_buffer_s, end_buffer_s = float(raw_buffer[0]), float(raw_buffer[1])
        else:
            start_buffer_s = end_buffer_s = float(raw_buffer)

        # def to_seconds(value: float, duration: float) -> float:
        #     # If value > 1, assume seconds; otherwise treat as proportion of duration
        #     if value > 1.0:
        #         return value
        #     if value < 0.0:
        #         return 0.0
        #     return value * duration

        # # compute buffer in seconds using available_dur
        # start_buffer_s = to_seconds(start_raw, available_dur)
        # end_buffer_s = to_seconds(end_raw, available_dur)

        allowed_start = start_t + start_buffer_s

        # two possible selection modes: 'random_within' (default) or 'end_anchored'
        selection_mode = state_selection_mode[i]
        if selection_mode == "random_within":
            allowed_end = start_t + end_buffer_s
            if allowed_end - allowed_start < min_state_record_dur_s:
                logging.warning(
                    f"state within subtask{taskind} too small buffers start{start_buffer_s}s, end{end_buffer_s}s)!"
                )
                continue
        elif selection_mode == "end_anchored":
            allowed_end = end_t
        else:
            raise NotImplementedError

        # pick a state according to provided probabilities
        state = random.choices(normalized_states, weights=normalized_probs, k=1)[0]

        recordings: list[dict] = []
        if selection_mode == "end_anchored":
            total_needed_min = n_rec * min_state_record_dur_s
            if (allowed_end - allowed_start) < total_needed_min:
                # cannot fit
                logging.warning(
                    f"Not enough space in allowed region for {n_rec} end-anchored recordings in sub-episode {taskind}; skipping."
                )
            else:
                remaining_end = allowed_end
                for j in range(n_rec):
                    # remaining slots after this one
                    slots_left = n_rec - j - 1
                    max_dur_for_this = min(
                        max_state_record_dur_s,
                        remaining_end - allowed_start - slots_left * min_state_record_dur_s,
                    )
                    if max_dur_for_this < min_state_record_dur_s:
                        # cannot fit further
                        break
                    dur = random.uniform(min_state_record_dur_s, max_dur_for_this)
                    rec_end = remaining_end
                    rec_start = rec_end - dur
                    recordings.append({"state": state, "start_s": float(rec_start), "end_s": float(rec_end)})
                    # move remaining_end backward
                    remaining_end = rec_start
                # recordings produced are in reverse chronological order; sort
                recordings.sort(key=lambda r: r["start_s"])
        else:
            seg_len = (allowed_end - allowed_start) / n_rec
            for j in range(n_rec):
                seg_start = allowed_start + j * seg_len
                seg_end = seg_start + seg_len
                dur_upper = min(seg_len, max_state_record_dur_s)
                if dur_upper < min_state_record_dur_s:
                    # cannot fit minimum duration in this segment; skip this segment
                    continue
                dur = random.uniform(min_state_record_dur_s, dur_upper) 
                # start uniformly so that end <= seg_end
                max_start = seg_end - dur
                if max_start <= seg_start:
                    rec_start = seg_start
                else:
                    rec_start = random.uniform(seg_start, max_start)
                rec_end = rec_start + dur

                recordings.append({"state": state, "start_s": float(rec_start), "end_s": float(rec_end)})

        frame_state_schedule[taskind] = recordings
    parse_res["frame_state_schedule"] = frame_state_schedule

    return parse_res