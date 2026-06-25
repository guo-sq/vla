"""Seatbelt LoRA baseline plus offline difficulty-label downsampling."""

import dataclasses

from openpi.configs import cfg_baseline_seatbelt_lora_506 as _baseline

PILOT_REPO_IDS = [
    "seatbelt.both.hang.zhangyu.20260205.batch.1",
    "seatbelt.both.hang.zhangyu.20260205.batch.2",
    "seatbelt.both.hang.zhangyu.20260205.batch.3",
    "seatbelt.single.hang.zhangyu.20260206.batch.1",
    "seatbelt.single.hang.baichenglong.20260205.batch.5",
    "seatbelt.single.hang_move.baichenglong.20260212.batch.5",
    "seatbelt.single.insert_move.baichenglong.20260214.batch.1",
    "seatbelt.single.take_off_move.panjinlong.20260302.batch.1",
    "seatbelt.single.insert_move.zhaoshuai.20260305.batch.4",
]

_data = dataclasses.replace(
    _baseline.cfg.data,
    repo_id=PILOT_REPO_IDS,
    difficulty_label_file="meta/difficulty_labels.jsonl",
    difficulty_label_strict=False,
    frame_skip=1,
    lazy_load=True,
)

cfg = dataclasses.replace(
    _baseline.cfg,
    name="seatbelt_lora_difficulty_downsample_506",
    exp_name="seatbelt_lora_difficulty_downsample_506_exp",
    data=_data,
    checkpoint_base_dir="/root/workspaces/wujie_gsq/vla/eval_results/difficulty_downsample_506/checkpoints",
)
