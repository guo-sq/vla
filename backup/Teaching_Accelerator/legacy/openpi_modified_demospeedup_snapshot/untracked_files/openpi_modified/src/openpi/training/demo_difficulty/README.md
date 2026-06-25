# Demo Difficulty Downsampling

This folder contains the offline demo-difficulty prototype for seatbelt data:

- `compute_demo_difficulty_labels.py`: reads LeRobot parquet `action` data and writes per-frame difficulty labels.
- `sampling.py`: shared loader used by `AnyverseDataset` and the eager frame-attribute preprocessor.
- `cfg_seatbelt_lora_difficulty_downsample_506.py`: pilot training config.
- `pilot_repo_ids.txt`: the nine pilot repos that already have generated labels.

The label sidecar is written under each dataset repo:

```text
meta/difficulty_labels.jsonl
meta/difficulty_labels_summary.json
```

Generate labels for the pilot set:

```bash
cd /root/workspaces/wujie_gsq/vla/openpi_modified
./.venv/bin/python src/openpi/training/demo_difficulty/compute_demo_difficulty_labels.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file src/openpi/training/demo_difficulty/pilot_repo_ids.txt
```

Run the pilot training config in the normal training runtime:

```bash
cd /root/workspaces/wujie_gsq/vla/openpi_modified
uv run scripts/train.py --config src/openpi/training/demo_difficulty/cfg_seatbelt_lora_difficulty_downsample_506.py
```

The first estimator is deliberately simple: it scores each frame with
phase-conditioned action dispersion plus local direction changes and a small
acceleration guard. Raw speed defaults to weight 0, because fast, smooth,
consistent motion is often easy rather than hard. Low-score frames are
considered easy and are downsampled more aggressively.
