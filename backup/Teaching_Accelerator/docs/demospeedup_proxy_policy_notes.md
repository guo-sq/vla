# DemoSpeedup Proxy Policy Notes

This note records the implementation details we should copy or deliberately
avoid when building proxy-entropy labels for OpenPI/AnyVerse data.

## Common Pattern

DemoSpeedup uses the original, non-accelerated demonstrations to train a normal
BC policy first. This trained policy is the proxy entropy estimator.

The proxy is later run over every original demonstration timestep. It samples
multiple action chunks conditioned on the current observation, aggregates all
samples that cover the current frame, estimates entropy, and converts entropy
curves into precision/casual labels.

The final accelerated policy is trained separately.

## ALOHA Implementation

Entry point:

```text
/root/workspaces/wujie_gsq/DemoSpeedup/aloha/act/imitate_episodes.py
```

Proxy train command from the README:

```bash
python act/imitate_episodes.py \
  --task_name [TASK] \
  --ckpt_dir data/outputs/[ALGO]_ckpt/[TASK] \
  --policy_class [ALGO] \
  --kl_weight 10 \
  --chunk_size [Chunk_Size] \
  --hidden_dim 512 \
  --batch_size [Batch_Size] \
  --dim_feedforward 3200 \
  --num_epochs 16000 \
  --lr 1e-5 \
  --seed 0 \
  --temporal_agg
```

Data:

- Source demos are HDF5 episodes under `aloha/data/<task>/episode_*.hdf5`.
- Each episode contains `/observations/qpos`, `/observations/qvel`,
  `/observations/images/<camera>`, and `/action`.
- `load_data()` uses an 80/20 episode split, but normalization stats are computed
  over all episodes.
- Each training sample chooses a random `start_ts`, uses only the observation at
  that timestep, and uses the action suffix from `start_ts` onward padded to the
  episode length.

Proxy training details:

- ACT proxy: ResNet18, encoder layers 4, decoder layers 7, heads 8,
  hidden dim 512, feedforward dim 3200, KL weight 10.
- README recommends ACT `chunk_size=50`, batch size 8.
- README recommends DP batch size 40. The current code hard-codes the DP horizon
  around 24 despite README mentioning 48, so treat the README and code as
  slightly drifted.

Entropy labeling:

- `--label` loads `policy_last.ckpt` and `dataset_stats.pkl`; it does not train.
- ACT samples 10 chunks through latent sampling.
- DP tiles the same observation 10 times and samples through diffusion.
- With `--temporal_agg`, every timestep predicts a chunk, chunks are written into
  a `[T, T + K, N, action_dim]` buffer, and the current frame aggregates all
  predictions that cover it.
- Entropy is estimated with KDE over the aggregated samples.
- Per-episode entropy is normalized, concatenated with normalized time index,
  and clustered with HDBSCAN.
- Labels are written back into the original HDF5 as `/labels` for ACT or
  `/labels_dp` for DP.

Speedup training:

- The dataloader reads `/labels` or `/labels_dp`.
- `label=0` uses a low stride of 2.
- Continuous `label=1` uses a high stride of 4.
- The main speedup path is label-based action chunk retiming, not the older
  waypoint helper functions.

## RoboBase / Bigym Implementation

Entry points:

```text
/root/workspaces/wujie_gsq/DemoSpeedup/robobase/train.py
/root/workspaces/wujie_gsq/DemoSpeedup/robobase/label.py
```

Proxy train command from the README:

```bash
python3 train.py launch=dp_pixel_bigym env=bigym/sandwich_remove
```

Label command:

```bash
python3 label.py launch=dp_pixel_bigym env=bigym/sandwich_remove
```

Data:

- Bigym demos are loaded through `DemoStore`, not HDF5 files.
- `demos=-1` means use all demos, then failed demos are filtered.
- Demos are wrapped as `DemoEnv` and replayed into a replay buffer.
- Action/observation normalization is computed from demos; demo actions are
  rescaled into a tanh/min-max action space.

Training details:

- DP launch uses `num_pretrain_steps=100000`, `num_train_frames=0`,
  `batch_size=256`, `action_sequence=24`, `execution_length=24`, EMA on.
- ACT launch uses `action_sequence=25`, `execution_length=1`,
  `temporal_ensemble=true`.

Entropy labeling:

- `label.py` loads `snapshots/best_snapshot.pt` from the Hydra run dir.
- `Workspace.label()` calls `agent.sample()` for every demo timestep.
- RoboBase uses 50 proxy samples by default.
- It first uses KDE to choose a high-density teacher action chunk, then temporal
  aggregation to compute a teacher action for the current frame.
- It also computes KDE entropy over all samples that cover the current frame.
- Labels are saved as `../bigym_<task>/labels/labels_i.npy`.
- Teacher actions are saved as
  `../bigym_<task>/teacher_actions/teacher_actions_i.npy`.

Speedup training:

- `speedup=True` loads both labels and teacher actions into replay buffer.
- Batch sampling uses `episode[TEACHER_ACTION]`, not original `episode[ACTION]`.
- `downsample_action_with_labels()` applies low stride 2 and high stride 4, then
  pads to the configured action sequence length.

## Lessons For OpenPI / AnyVerse

- Do not train a proxy only to imitate final accelerated targets. The proxy must
  be trained on original demos so its uncertainty reflects the source data.
- Keep proxy entropy labels separate from rule labels. They can later be fused,
  but their failure modes differ.
- For pi0.5, proxy entropy can come from a different generative policy if action
  space, observation coverage, and normalization are aligned.
- Prefer episode-local entropy normalization and clustering/quantiles.
- If we use a proxy that is not pi0.5, store enough metadata in the sidecar:
  proxy model type, action representation, normalization stats source, sample
  count, and chunk horizon.
- For action acceleration, follow RoboBase's important trick: if the proxy's
  sampled high-density action is better than raw demo action, store
  `teacher_actions` separately and train accelerated chunks from that source.
