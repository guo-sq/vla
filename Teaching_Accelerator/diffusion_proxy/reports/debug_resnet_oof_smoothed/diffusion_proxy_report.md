# Diffusion Proxy Label Report

- Episodes: 4
- Frames: 4083
- Hard spans: 35
- Label counts: precision=891, neutral=1763, casual=1429
- Checkpoints: `checkpoints/debug_resnet_oof`
- Output: `labels/debug_resnet_oof_smoothed/diffusion_labels.jsonl`

## Thresholds

```json
{
  "diffusion_precision_min": 0.7117845118045807,
  "diffusion_entropy_casual_min": 0.612971431016922,
  "static_speed_max": 0.018827836960554123,
  "precision_quantile": 0.75,
  "casual_quantile": 0.65,
  "static_speed_quantile": 0.1,
  "smoothing_half_window": 8,
  "min_span_frames": 15,
  "merge_gap_frames": 10,
  "span_padding_frames": 8
}
```

## Top Precision Spans

| repo_id | episode | start_s | end_s | duration_s | mean_diffusion_precision_score |
|---|---:|---:|---:|---:|---:|
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 29.067 | 30.967 | 1.900 | 0.799304 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 1.900 | 4.433 | 2.533 | 0.794692 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 23.033 | 24.900 | 1.867 | 0.770246 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 10.067 | 11.333 | 1.267 | 0.762521 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 32.333 | 33.867 | 1.533 | 0.749981 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 2.933 | 4.267 | 1.333 | 0.745016 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 9.467 | 11.300 | 1.833 | 0.743294 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 0.300 | 2.700 | 2.400 | 0.734099 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 26.500 | 27.933 | 1.433 | 0.729925 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 12.933 | 14.167 | 1.233 | 0.726761 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 15.733 | 17.100 | 1.367 | 0.720616 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 23.433 | 25.000 | 1.567 | 0.708213 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 26.800 | 28.733 | 1.933 | 0.706692 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 5.400 | 6.567 | 1.167 | 0.699105 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 6.667 | 9.200 | 2.533 | 0.698507 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 8.967 | 10.067 | 1.100 | 0.691054 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 5.167 | 7.067 | 1.900 | 0.689870 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 0.567 | 2.000 | 1.433 | 0.681061 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 11.800 | 12.833 | 1.033 | 0.680508 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 28.400 | 29.633 | 1.233 | 0.677022 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 10.200 | 11.233 | 1.033 | 0.675394 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 5.767 | 7.133 | 1.367 | 0.673894 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 22.700 | 23.733 | 1.033 | 0.672202 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 13.600 | 15.167 | 1.567 | 0.667051 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 18.667 | 19.833 | 1.167 | 0.666036 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 16.833 | 18.300 | 1.467 | 0.665761 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 1.233 | 2.367 | 1.133 | 0.654383 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 4.067 | 5.167 | 1.100 | 0.648692 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 28.433 | 29.667 | 1.233 | 0.639254 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 21.600 | 22.700 | 1.100 | 0.624884 |
