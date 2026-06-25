# Lightweight Action-Chunk BC Ensemble

This method trains a small behavior-cloning ensemble on the seatbelt
`both.hang.zhangyu` demonstrations and uses low ensemble disagreement as the
precision signal.

## Scope

- No images.
- No RL.
- No pi0.5 dependency.
- Reads only LeRobot parquet columns:
  `observation.state`, `observation.velocity`, `observation.current`, and
  `action`.
- Writes sidecar labels under this directory only.

The shared repo manifest is:

```text
../manifests/both_hang_zhangyu_repos.txt
```

## Data And Model

Each frame uses a 43-dim feature vector:

```text
[state(14), velocity(14), current(14), normalized_phase(1)]
```

The target is a future action chunk of shape `16 x 14`. Near the end of an
episode, missing future frames are padded with the final action and masked out
of the SmoothL1 loss.

The default ensemble has 5 independent MLPs:

- hidden size 256
- 3 hidden layers
- LayerNorm + SiLU
- AdamW, lr `1e-3`, weight decay `1e-4`
- batch size 1024
- 80 epochs

## Train

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/action_chunk_bc
/root/miniconda3/envs/openpi/bin/python scripts/train_bc_ensemble.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --checkpoint-dir checkpoints/both_hang_zhangyu
```

## Label

```bash
/root/miniconda3/envs/openpi/bin/python scripts/label_with_bc_ensemble.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --checkpoint-dir checkpoints/both_hang_zhangyu \
  --output-dir labels/both_hang_zhangyu \
  --report-dir reports/both_hang_zhangyu
```

## Debug Run

```bash
/root/miniconda3/envs/openpi/bin/python scripts/train_bc_ensemble.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --checkpoint-dir checkpoints/debug_both_hang \
  --max-episodes-per-repo 2 \
  --ensemble-size 2 \
  --epochs 3

/root/miniconda3/envs/openpi/bin/python scripts/label_with_bc_ensemble.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --checkpoint-dir checkpoints/debug_both_hang \
  --output-dir labels/debug_both_hang \
  --report-dir reports/debug_both_hang \
  --max-episodes-per-repo 2
```

## Outputs

```text
checkpoints/both_hang_zhangyu/seed_0.pt
checkpoints/both_hang_zhangyu/seed_1.pt
checkpoints/both_hang_zhangyu/seed_2.pt
checkpoints/both_hang_zhangyu/seed_3.pt
checkpoints/both_hang_zhangyu/seed_4.pt
labels/both_hang_zhangyu/bc_ensemble_labels.jsonl
labels/both_hang_zhangyu/summary.json
reports/both_hang_zhangyu/bc_ensemble_report.md
reports/both_hang_zhangyu/plots/*.svg
```

## Label Semantics

`ensemble_disagreement_score` is the ensemble variance over the predicted
future action chunk, robust-scaled to `0..1`.

`bc_precision_score = 1 - ensemble_disagreement_score`.

This keeps the DemoSpeedup-style interpretation: low disagreement/low entropy
means the proxy finds the frame more consistent and precision-like. High
disagreement is not called hard; it is treated as casual unless precision
overrides it.
