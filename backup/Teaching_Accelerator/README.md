# Teaching Accelerator

This workspace contains the next iteration of the earlier `demo_difficulty`
prototype. The goal is to optimize robot teaching data for faster policy
execution, using DemoSpeedup's idea of precision-aware action-target
acceleration.

## Why This Exists

The previous prototype wrote `meta/difficulty_labels.jsonl` with
`easy/medium/hard` labels and `sample_weight`. That is useful for data
resampling, but it does not implement DemoSpeedup-style acceleration. Changing
which frames are sampled is different from changing the action chunk the model
learns to imitate.

This workspace fixes two semantics:

1. Phase-conditioned dispersion is a **casualness** signal, not a hard/difficulty
   signal. High dispersion means many actions appear plausible in the same task
   phase, which is closer to DemoSpeedup's high-entropy/casual interpretation.
2. Labels are now written for **action-target acceleration**. The output sidecar
   contains `precision|neutral|casual` and a per-frame `acceleration_stride`,
   instead of only a `sample_weight`.

## Current Rule-Based Pipeline

Generate labels:

```bash
cd /root/workspaces/wujie_gsq/vla/Teaching_Accelerator
python scripts/compute_teaching_labels.py \
  --root-dir /path/to/lerobot/repos \
  --repo-id your.repo.id
```

Default output per repo:

```text
meta/teaching_acceleration_labels.jsonl
meta/teaching_acceleration_labels_summary.json
```

Each episode record includes:

- `precision_score`: higher means preserve precision with smaller stride.
- `casualness_score`: higher means safe-to-accelerate candidate.
- `label`: `precision`, `neutral`, or `casual`.
- `acceleration_stride`: default `2`, `2`, `4` for precision/neutral/casual.

The helper `teaching_accelerator.action_targets.gather_accelerated_actions()`
builds a DemoSpeedup-style action target chunk from the labels.

## DemoSpeedup Proxy Policy Pattern

DemoSpeedup trains a proxy policy on the original, non-accelerated demos. The
proxy is not the final deployed policy; it is an entropy estimator.

For each observation, the proxy samples multiple action chunks:

- ACT samples through its CVAE latent prior.
- Diffusion Policy samples through different denoising noise seeds.

The samples are aggregated per timestep, KDE estimates conditional action
entropy, and HDBSCAN segments low-entropy precision regions from high-entropy
casual regions. The final accelerated policy is trained on original
observations paired with retimed action chunks.

## How This Should Connect To OpenPI

The old OpenPI integration changes `_sampler_indices`; that only changes which
start frames are sampled. The next integration should instead intercept action
query indices in `AnyverseDataset`:

```text
raw_idx + [0, 1, 2, ...]  ->  accelerated_indices(raw_idx)
```

Observation/state at `raw_idx` stays unchanged. Only the target action sequence
is retimed. Lazy and eager loading paths must use the same mapping, and
`action_mask`, `action_is_pad`, and segment continuity need to follow the new
indices.
