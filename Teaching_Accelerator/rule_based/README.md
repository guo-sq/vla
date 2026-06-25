# Rule-Based Teaching Accelerator

Lightweight rule-based labels for finding precision-heavy segments in the
seatbelt `both.hang.zhangyu` demonstrations.

## Scope

This version is action-only. It reads LeRobot parquet files, computes per-frame
`precision|neutral|casual` labels, and writes sidecar outputs under
`rule_based/`. It does not train a proxy policy and does not write new labels
back into the OSS dataset.

Selected repos are listed in the shared manifest:

```text
../manifests/both_hang_zhangyu_repos.txt
```

## Run

Use the OpenPI conda environment because the base environment does not include
`numpy` or `pyarrow`.

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/rule_based
/root/miniconda3/envs/openpi/bin/python scripts/compute_rule_labels.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --output-dir labels/both_hang_zhangyu \
  --report-dir reports/both_hang_zhangyu
```

For a quick smoke test:

```bash
/root/miniconda3/envs/openpi/bin/python scripts/compute_rule_labels.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --output-dir labels/debug_both_hang \
  --report-dir reports/debug_both_hang \
  --max-episodes-per-repo 2
```

## Outputs

```text
labels/both_hang_zhangyu/rule_labels.jsonl
labels/both_hang_zhangyu/summary.json
reports/both_hang_zhangyu/rule_label_report.md
```

Each episode record includes per-frame scores, labels, acceleration strides,
and merged `hard_spans` in both frame and second units.

## Rule Semantics

`precision` means a hard or fine-control segment worth preserving with a smaller
acceleration stride. It is not a high-entropy label. Following DemoSpeedup,
phase-conditioned action dispersion is treated as a `casualness` signal, and
inverse dispersion contributes to precision.

The default hard score combines:

- phase consistency: 0.35
- gripper event window: 0.30
- action direction turns: 0.15
- jerk: 0.15
- left/right coordination: 0.05

