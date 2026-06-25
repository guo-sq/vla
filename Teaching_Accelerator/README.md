# Teaching Accelerator

This workspace contains lightweight data-labeling experiments inspired by
DemoSpeedup for the seatbelt task.

## Layout

```text
manifests/
  both_hang_zhangyu_repos.txt
rule_based/
  Action-only heuristic precision labels and hard spans.
action_chunk_bc/
  Lightweight Action-Chunk BC Ensemble labels.
diffusion_proxy/
  Visual-proprio small diffusion proxy labels and rule fusion.
```

Both methods currently target the same four `both.hang.zhangyu` LeRobot repos
listed in `manifests/both_hang_zhangyu_repos.txt`.

## Methods

- `rule_based/`: no training; computes per-frame rule scores from 14-dim
  actions and writes `precision|neutral|casual` sidecars.
- `action_chunk_bc/`: trains a small ensemble of action-chunk BC MLPs from
  proprioception and uses low ensemble disagreement as a precision signal.
- `diffusion_proxy/`: trains out-of-fold conditional action diffusion proxies
  from visual embeddings plus proprioception and estimates action entropy by
  sampling future action chunks.

Use `/root/miniconda3/envs/openpi/bin/python` for both methods.

## Quick Entrypoints

Rule-based labels:

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/rule_based
/root/miniconda3/envs/openpi/bin/python scripts/compute_rule_labels.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --output-dir labels/both_hang_zhangyu \
  --report-dir reports/both_hang_zhangyu
```

BC ensemble train and label:

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/action_chunk_bc
/root/miniconda3/envs/openpi/bin/python scripts/train_bc_ensemble.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --checkpoint-dir checkpoints/both_hang_zhangyu

/root/miniconda3/envs/openpi/bin/python scripts/label_with_bc_ensemble.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --checkpoint-dir checkpoints/both_hang_zhangyu \
  --output-dir labels/both_hang_zhangyu \
  --report-dir reports/both_hang_zhangyu
```

Diffusion proxy starts with:

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/diffusion_proxy
/root/miniconda3/envs/openpi/bin/python scripts/extract_visual_embeddings.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --output-dir cache/vision_head
```
