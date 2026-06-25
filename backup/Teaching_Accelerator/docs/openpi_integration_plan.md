# OpenPI Integration Plan

This document describes the next integration step. It is not implemented in
`openpi_modified` yet.

## Current OpenPI State

The existing `demo_difficulty` path changes sampling only:

```text
difficulty label -> sample_weight -> AnyverseDataset._sampler_indices
```

That can reduce repeated/easy start frames, but it does not accelerate the
target action chunk and therefore does not reproduce DemoSpeedup's main effect.

## Desired State

Use teaching-acceleration labels to retime the target action chunk:

```text
raw observation frame t stays t
action target [t, t+1, t+2, ...] becomes [t, t+s0, t+s0+s1, ...]
```

where stride `s` comes from `precision|neutral|casual` labels.

## New Config Fields

Do not reuse `difficulty_label_file`; that name now means sampler weighting in
the old path. Add new fields instead:

```python
teaching_acceleration_label_file: str | None = None
teaching_acceleration_strict: bool = False
teaching_acceleration_action_keys: tuple[str, ...] = ("action",)
```

These belong on `DataConfig` and the relevant `DataConfigFactory`, especially
`Gr00tLerobotDataConfig`.

## AnyverseDataset Touch Points

1. In `MultiAnyverseDataset.__init__`, pass the new fields to each
   `AnyverseDataset`.
2. In `AnyverseDataset.__init__`, load
   `meta/teaching_acceleration_labels.jsonl` and cache labels by episode.
3. In eager mode, intercept action query indices. The current query is roughly:

   ```python
   raw_idx + delta
   ```

   Replace this for action keys only with:

   ```python
   accelerated_indices(raw_idx, horizon)
   ```

4. In lazy mode, mirror the same logic where query indices are manually built.
5. Recompute `action_is_pad`, `action_is_valid`, and `action_segment_id` from the
   accelerated indices.
6. Never cross episode boundaries. Clamp/pad at episode end.

## Tests Needed

- Same `raw_idx` returns the same observation/state.
- The action chunk changes from `[t, t+1, t+2]` to expected accelerated indices.
- Lazy and eager paths match.
- Episode tail pads correctly.
- Segment continuity still masks/pads action targets correctly.
- Old `difficulty_label_file` sampler-weight path keeps working independently.

## Sidecar Schema

Use the file produced by `scripts/compute_teaching_labels.py`:

```json
{
  "episode_index": 0,
  "label": ["precision", "neutral", "casual"],
  "acceleration_stride": [2, 2, 4],
  "precision_score": [0.9, 0.3, 0.1],
  "casualness_score": [0.1, 0.4, 0.8]
}
```

The integration may use either `label` plus config strides or
`acceleration_stride` directly. Prefer direct `acceleration_stride` for
reproducibility.
