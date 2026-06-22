# Session configs

Everything an operator needs to start a recording session lives in one JSON file. Run with:

```bash
bash scripts/run_session.sh path/to/<your>.session.json
```

This directory ships two things:

```
sessions/
├── templates/        # blank-slate, one per mode, with inline _comment_* annotations
│   ├── record.json
│   ├── infer.json
│   ├── infer_record.json
│   └── self_play.json
├── examples/         # complete working configs for the seatbelt task
│   ├── seatbelt.record.json
│   ├── seatbelt.infer.json
│   ├── seatbelt.infer_record.json
│   ├── seatbelt.self_play.json
│   └── seatbelt.self_play_infer_only.json
└── README.md         # you are here
```

Pick whichever is closer to what you're doing:

- **Brand-new task** → copy `templates/<mode>.json`, replace every `FILL_ME_IN`, optionally delete the `_comment_*` keys once you've memorised the schema.
- **Adapting an existing task** → copy a matching `examples/seatbelt.<mode>.json`, change task_meta, prompts, cameras to fit your task.

## Pick a mode

| Mode | What runs | Saves data? | Needs `policy_server`? | Typical use |
|------|-----------|-------------|------------------------|-------------|
| `record` | Operator teleops the robot (leader / spacemouse / phone). | yes | no | Collect human demos for training. |
| `infer` | A trained policy drives the robot live. | no | yes | Demo, smoke-test, latency benchmark. |
| `infer_record` | Same as `infer`, but episodes are written to disk. | yes | yes | Evaluation runs; bootstrap next training round. |
| `self_play` | Two policies (builder + destroyer) alternate via a value model. | yes (unless `self_play.infer_only=true`) | yes (per role or top-level) | Scaled autonomous data collection with auto-reset. |

The mode is the **single source of truth** — set `collection_meta.mode` to one of the four strings above and the rest of the file's required-ness flips accordingly.

## Inline comments (`_comment_*`, `_doc`)

JSON has no native comment syntax, so the schema reserves any **key whose name starts with `_`**. The validator and every downstream consumer (recorder, preflight, session_to_args, task_spec parser) skips these keys, so they survive on disk without affecting behaviour.

Patterns used in `templates/`:

```jsonc
"task_spec": {
    "task_id": "FILL_ME_IN",
    "_comment_task_id": "Usually identical to task_meta.task_name.",
    "roles": {
        "_comment_roles": "self_play requires >=2 roles. Builder completes the task; destroyer disturbs the scene.",
        "operator": { ... }
    }
}
```

Three conventions in the templates:

- `"_doc": "..."` — one-line orientation comment at the top of a section.
- `"_comment_<field>": "..."` — sibling of `<field>`, explains valid values / typical use.
- `"_comment_<plural>": "..."` — sibling of a collection (e.g. `_comment_cameras`), explains the whole block.

Examples and dev-PC configs typically drop the comments once they're stable; templates keep them.

## Common gotchas

- **Camera indices** are per-machine. `lerobot-find-cameras opencv` lists what's connected; the integer that comes back is the `index_or_path` to put in your `hardware_meta.cameras`.
- **`task_spec.policy_server`** is required for `infer` / `infer_record` and (per role or top-level) for `self_play`. `record` mode must not have one.
- **`adversary_operator`** is informational only — set it to the name of a *distinct* second human alternating roles during self-play, or leave it as `""` when one person handles both roles (or when there's no adversary at all).
- **`data_root` placeholders** — `$HOME` and `$VAR` are expanded at write time, so `"$HOME/lerobot_data_collection"` works cross-machine.
- **Repo_id auto-naming** — `run_session.sh` derives `repo_id` as `<task>.<robot_type>.<robot_id>.<operator>.<TIMESTAMP>`. If you need a different layout (e.g. `<task>.<sub_task>.<operator>.<date>.batch.N`), pass `--dataset.repo_id=...` on the command line:
  ```bash
  bash scripts/run_session.sh foo.json \
      --dataset.repo_id=seatbelt/seatbelt.single.recover_2_left_move.yangjiaxu.20260413.batch.1
  ```

## Optional: per-inference debug log

Flip `inference.persist_inference_log` to `true` (only meaningful in `infer_record` and `self_play`) and every inference call is appended to `<dataset>/meta/inference_log.parquet`:

| column | meaning |
|---|---|
| `episode_index` | episode this call belongs to |
| `step_id` | global control-loop step when the chunk was applied |
| `t_submit` / `t_complete` | `time.perf_counter()` at submit / receipt |
| `latency_ms` | `(t_complete - t_submit) * 1000` |
| `action_chunk_raw` | flat `(H*D)` float32 list — the **pre-fusion** chunk |
| `action_horizon` / `action_dim` | `H`, `D` (for reshape) |
| `prompt` | post-template prompt that was actually sent |
| `delay` | RTC `submitted_delay` |
| `role` | `'operator'` for non-self-play, `'builder'` / `'destroyer'` for self-play |

Storage cost is ~600 KB/min for `H=50, D=32` — negligible next to video. Read back with:

```python
from lerobot.datasets.inference_log import read_inference_log
df = read_inference_log("/path/to/dataset")
print(df["latency_ms"].quantile([0.5, 0.99]))
chunks = df.attrs["action_chunks"]  # shape (N_inferences, H, D)
```

Rerecording an episode (Ctrl+Left) discards its in-flight inference rows so the persisted log stays consistent with the saved dataset.

## Where to dig deeper

- Schema source of truth: `src/lerobot/recording/task/session_config.py` — dataclasses + `SessionConfig.validate()`.
- Defaults for each block: `RECORDING_DEFAULTS`, `INFERENCE_DEFAULTS`, `INTERVENTION_DEFAULTS`, `SELF_PLAY_DEFAULTS`, `SUBTASK_DEFAULTS` in the same file.
- Task spec dataclasses (roles, success_when, reset, safety, recovery): `src/lerobot/recording/task/task_spec.py`.
- Preflight what-you-see-before-recording: `src/lerobot/recording/utils/preflight.py`.
- Runtime CLI splat (how a session JSON becomes argv): `src/lerobot/recording/utils/session_to_args.py`.
