# Diffusion Proxy Label Report

- Episodes: 4
- Frames: 4083
- Hard spans: 29
- Label counts: precision=925, neutral=1729, casual=1429
- Checkpoints: `checkpoints/debug_oof`
- Output: `labels/debug_oof/diffusion_labels.jsonl`

## Thresholds

```json
{
  "diffusion_precision_min": 0.7118616998195648,
  "diffusion_entropy_casual_min": 0.6110175013542175,
  "static_speed_max": 0.018827836960554123,
  "precision_quantile": 0.75,
  "casual_quantile": 0.65,
  "static_speed_quantile": 0.1
}
```

## Top Precision Spans

| repo_id | episode | start_s | end_s | duration_s | mean_diffusion_precision_score |
|---|---:|---:|---:|---:|---:|
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 26.000 | 33.633 | 7.633 | 0.544312 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 4.133 | 7.767 | 3.633 | 0.532843 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 0.400 | 3.200 | 2.800 | 0.531822 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 0.933 | 8.300 | 7.367 | 0.531664 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 21.400 | 27.533 | 6.133 | 0.530826 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 16.867 | 33.400 | 16.533 | 0.529718 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 14.233 | 21.867 | 7.633 | 0.518978 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 27.467 | 29.233 | 1.767 | 0.514616 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 1.400 | 4.100 | 2.700 | 0.510364 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 12.333 | 14.500 | 2.167 | 0.508418 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 18.400 | 21.567 | 3.167 | 0.505800 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 8.700 | 10.167 | 1.467 | 0.503204 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 3.100 | 11.033 | 7.933 | 0.501409 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 11.667 | 13.133 | 1.467 | 0.501094 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 0.700 | 5.533 | 4.833 | 0.500002 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 13.100 | 18.267 | 5.167 | 0.499765 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 13.700 | 16.233 | 2.533 | 0.498537 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 22.733 | 24.067 | 1.333 | 0.496898 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 26.433 | 31.833 | 5.400 | 0.496369 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 10.500 | 12.500 | 2.000 | 0.490130 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 5.367 | 11.600 | 6.233 | 0.485013 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 19.100 | 21.433 | 2.333 | 0.484243 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 9.767 | 14.333 | 4.567 | 0.481272 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 14.500 | 16.167 | 1.667 | 0.479206 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 16.100 | 17.933 | 1.833 | 0.467778 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 21.767 | 22.967 | 1.200 | 0.453307 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 21.833 | 22.867 | 1.033 | 0.444655 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 24.000 | 26.600 | 2.600 | 0.418385 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 8.567 | 9.833 | 1.267 | 0.409433 |
