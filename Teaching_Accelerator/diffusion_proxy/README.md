# Diffusion Proxy Teaching Accelerator

This method trains a small visual-proprio conditional action diffusion proxy
and uses out-of-fold sample dispersion as a proxy for action entropy.

## Scope

- Targets the shared `both.hang.zhangyu` repo manifest.
- Does not write into the original OSS dataset.
- Uses episode-level out-of-fold training.
- Uses frozen visual embeddings from `observation.images.head`.
- Outputs both pure diffusion labels and rule+diffusion fused labels.

## Pipeline

1. Extract visual embeddings.
2. Train OOF diffusion checkpoints.
3. Label held-out episodes with diffusion sample dispersion.
4. Fuse with `rule_based` labels for the recommended final sidecar.

## Commands

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/diffusion_proxy

/root/miniconda3/envs/openpi/bin/python scripts/extract_visual_embeddings.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --camera observation.images.head \
  --encoder resnet18 \
  --output-dir cache/vision_head

/root/miniconda3/envs/openpi/bin/python scripts/train_oof_diffusion_proxy.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --vision-cache cache/vision_head \
  --checkpoint-dir checkpoints/both_hang_zhangyu_oof \
  --folds 5 \
  --epochs 80

/root/miniconda3/envs/openpi/bin/python scripts/label_with_diffusion_proxy.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --vision-cache cache/vision_head \
  --checkpoint-dir checkpoints/both_hang_zhangyu_oof \
  --output-dir labels/both_hang_zhangyu \
  --report-dir reports/both_hang_zhangyu \
  --smoothing-half-window 8

/root/miniconda3/envs/openpi/bin/python scripts/fuse_rule_diffusion_labels.py \
  --rule-labels ../rule_based/labels/both_hang_zhangyu/rule_labels.jsonl \
  --diffusion-labels labels/both_hang_zhangyu/diffusion_labels.jsonl \
  --output labels/both_hang_zhangyu/diffusion_fused_labels.jsonl
```

## Debug

```bash
/root/miniconda3/envs/openpi/bin/python scripts/extract_visual_embeddings.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --output-dir cache/debug_vision_head \
  --max-episodes-per-repo 1 \
  --encoder grid

/root/miniconda3/envs/openpi/bin/python scripts/train_oof_diffusion_proxy.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --vision-cache cache/debug_vision_head \
  --checkpoint-dir checkpoints/debug_oof \
  --max-episodes-per-repo 1 \
  --encoder grid \
  --folds 2 \
  --epochs 2 \
  --hidden-dim 128 \
  --blocks 2 \
  --diffusion-steps 20 \
  --batch-size 512

/root/miniconda3/envs/openpi/bin/python scripts/label_with_diffusion_proxy.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --vision-cache cache/debug_vision_head \
  --checkpoint-dir checkpoints/debug_oof \
  --output-dir labels/debug_oof \
  --report-dir reports/debug_oof \
  --max-episodes-per-repo 1 \
  --encoder grid \
  --sampling-steps 5 \
  --samples-per-frame 4 \
  --smoothing-half-window 8
```

## Semantics

`diffusion_entropy_score` is robust-scaled sample variance across generated
future action chunks.

`diffusion_precision_score = 1 - diffusion_entropy_score`.

Pure diffusion labels keep DemoSpeedup-style semantics: low entropy means
precision-like. The recommended output is the fused label, which preserves
rule-based gripper/jerk/turn event evidence while using diffusion as a proxy
policy signal.

`label_with_diffusion_proxy.py` smooths raw per-frame entropy inside each
episode before robust scaling. The default `--smoothing-half-window 8` gives a
17-frame window at 30 FPS, reducing one-frame sampling noise without erasing
short manipulation events.
