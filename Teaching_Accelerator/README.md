# Teaching Accelerator

Teaching Accelerator is a lightweight workspace for optimizing seatbelt
demonstration data with DemoSpeedup-style ideas. The current goal is to find
fine-control or precision-heavy segments in successful demonstrations, then
write sidecar labels that can drive later data filtering or acceleration
experiments without modifying the original OSS datasets.

The project currently focuses on the `both.hang.zhangyu` seatbelt task.

## Data Scope

The shared repo manifest is:

```text
manifests/both_hang_zhangyu_repos.txt
```

It points to four LeRobot repos under:

```text
/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt
```

Current full-run coverage:

| item | value |
| --- | ---: |
| repos | 4 |
| episodes | 67 |
| frames | 69,676 |
| fps | 30 |
| action dim | 14 |

Episode count by repo:

| repo | episodes | frames |
| --- | ---: | ---: |
| `seatbelt.both.hang.zhangyu.20260205.batch.1` | 20 | 20,170 |
| `seatbelt.both.hang.zhangyu.20260205.batch.2` | 30 | 31,587 |
| `seatbelt.both.hang.zhangyu.20260205.batch.3` | 15 | 16,059 |
| `seatbelt.both.hang.zhangyu.20260205.batch.4` | 2 | 1,860 |

## Project Layout

```text
Teaching_Accelerator/
  README.md
  manifests/
    both_hang_zhangyu_repos.txt
  rule_based/
    Action-only heuristic labels.
  action_chunk_bc/
    Proprio-only action-chunk BC ensemble labels.
  diffusion_proxy/
    Visual-proprio small diffusion proxy labels and rule fusion.
```

Each method is self-contained. It has its own package, scripts, tests, labels,
reports, and README. All new labels are sidecar files inside this workspace;
the source dataset directories are not written to.

## Label Semantics

All methods write per-frame `precision|neutral|casual` labels and merged
`hard_spans`.

In this project, `precision` means "fine-control segment worth preserving with
a small acceleration stride." It does not mean high uncertainty. This follows
the DemoSpeedup intuition that consistent, low-entropy behavior is more likely
to be important and should be accelerated less aggressively.

Default acceleration stride:

| label | stride |
| --- | ---: |
| `precision` | 2 |
| `neutral` | 2 |
| `casual` | 4 |

Hard spans are derived from precision frames, then merged and padded:

| parameter | value |
| --- | ---: |
| merge gap | 10 frames |
| minimum span | 15 frames |
| padding | 8 frames each side |

## Methods

### 1. Rule-Based

Path:

```text
rule_based/
```

This is the simplest and most interpretable baseline. It reads only the
14-dimensional `action` sequence and computes per-frame rule scores.

Main signals:

| signal | meaning |
| --- | --- |
| phase consistency | low action dispersion in the same normalized task phase |
| gripper event | peaks around gripper dimensions 6 and 13 |
| turn score | fast action direction changes |
| jerk score | third-order action changes, useful for micro-corrections |
| coordination | simultaneous or coupled left/right arm changes |

Current full-run output:

| output | value |
| --- | ---: |
| precision frames | 13,564 |
| neutral frames | 33,628 |
| casual frames | 22,484 |
| hard spans | 354 |

Use this method when you want an explainable first pass or a sanity-check
reference for other proxy-policy labels.

### 2. Action-Chunk BC Ensemble

Path:

```text
action_chunk_bc/
```

This method trains five small behavior-cloning MLPs on proprioceptive features:

```text
[state(14), velocity(14), current(14), normalized_phase(1)]
```

Each model predicts a future `16 x 14` action chunk. The label signal is
ensemble disagreement:

```text
bc_precision_score = 1 - ensemble_disagreement_score
```

Current full-run output:

| output | value |
| --- | ---: |
| precision frames | 11,593 |
| neutral frames | 33,696 |
| casual frames | 24,387 |
| hard spans | 293 |

This version is useful as a lightweight proxy-policy baseline, but it tends to
miss some short, explicit manipulation events because deterministic MLP
agreement is not always aligned with human-visible precision.

### 3. Diffusion Proxy

Path:

```text
diffusion_proxy/
```

This method is the current strongest proxy-policy direction. It uses frozen
visual embeddings from `observation.images.head`, proprioception, normalized
phase, and a small conditional diffusion model to sample future action chunks.
It trains out-of-fold checkpoints, then labels each held-out episode with
sample dispersion:

```text
diffusion_precision_score = 1 - diffusion_entropy_score
```

Current pure diffusion full-run output:

| output | value |
| --- | ---: |
| precision frames | 16,681 |
| neutral frames | 28,608 |
| casual frames | 24,387 |
| hard spans | 204 |

The recommended output is the rule+diffusion fused label, which keeps the
rule-based event evidence while adding a proxy-policy entropy signal.

Current fused full-run output:

| output | value |
| --- | ---: |
| precision frames | 18,919 |
| neutral frames | 31,657 |
| casual frames | 19,100 |
| hard spans | 329 |

Recommended sidecar for the next iteration:

```text
diffusion_proxy/labels/both_hang_zhangyu/diffusion_fused_labels.jsonl
```

## Sidecar Record Shape

Each output JSONL has one record per episode. The exact score fields differ by
method, but every record contains the same core information:

```text
repo_id
episode_index
task
length
fps
label
acceleration_stride
hard_spans
```

Method-specific score arrays are frame-aligned and have length equal to the
episode length. Examples:

```text
hard_score
casualness_score
bc_precision_score
ensemble_disagreement_score
diffusion_precision_score
diffusion_entropy_score
fusion_precision_score
```

## Quick Commands

Use the OpenPI conda environment:

```bash
/root/miniconda3/envs/openpi/bin/python
```

### Rule-Based Labels

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/rule_based
/root/miniconda3/envs/openpi/bin/python scripts/compute_rule_labels.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --output-dir labels/both_hang_zhangyu \
  --report-dir reports/both_hang_zhangyu
```

### Action-Chunk BC Ensemble

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

### Diffusion Proxy And Fusion

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

## Tests

Run tests from each method directory so local packages are on `PYTHONPATH`:

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/rule_based
/root/miniconda3/envs/openpi/bin/python -m pytest tests

cd /root/workspaces/wujie_gsq/Teaching_Accelerator/action_chunk_bc
/root/miniconda3/envs/openpi/bin/python -m pytest tests

cd /root/workspaces/wujie_gsq/Teaching_Accelerator/diffusion_proxy
/root/miniconda3/envs/openpi/bin/python -m pytest tests
```

Last checked results:

| module | tests |
| --- | ---: |
| rule_based | 4 passed |
| action_chunk_bc | 4 passed |
| diffusion_proxy | 5 passed |

## Artifact Policy

Local generated artifacts can be large. The full local workspace includes
checkpoints and visual caches. The uploaded git copy keeps code, READMEs, tests,
labels, reports, plots, and sample previews, but excludes large regenerated
artifacts:

```text
action_chunk_bc/checkpoints/
diffusion_proxy/checkpoints/
diffusion_proxy/cache/
```

Keep original OSS data immutable. New labels should continue to be written as
sidecars under this workspace.
