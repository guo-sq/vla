# Diffusion Proxy Label Report

- Episodes: 4
- Frames: 4083
- Hard spans: 39
- Label counts: precision=922, neutral=1732, casual=1429
- Checkpoints: `checkpoints/debug_resnet_oof`
- Output: `labels/debug_resnet_oof/diffusion_labels.jsonl`

## Thresholds

```json
{
  "diffusion_precision_min": 0.7224708795547485,
  "diffusion_entropy_casual_min": 0.5960064053535461,
  "static_speed_max": 0.018827836960554123,
  "precision_quantile": 0.75,
  "casual_quantile": 0.65,
  "static_speed_quantile": 0.1
}
```

## Top Precision Spans

| repo_id | episode | start_s | end_s | duration_s | mean_diffusion_precision_score |
|---|---:|---:|---:|---:|---:|
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 27.433 | 29.333 | 1.900 | 0.582361 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 1.100 | 6.167 | 5.067 | 0.555249 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 24.533 | 26.200 | 1.667 | 0.542798 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 25.633 | 29.467 | 3.833 | 0.542405 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 10.067 | 16.367 | 6.300 | 0.540722 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 20.467 | 24.567 | 4.100 | 0.540615 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 26.100 | 31.567 | 5.467 | 0.530512 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 0.667 | 3.267 | 2.600 | 0.529281 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 4.200 | 12.467 | 8.267 | 0.525557 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 26.167 | 28.700 | 2.533 | 0.521924 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 18.600 | 23.000 | 4.400 | 0.521848 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 15.033 | 22.100 | 7.067 | 0.521155 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 7.033 | 10.033 | 3.000 | 0.518010 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 29.667 | 32.900 | 3.233 | 0.517193 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 2.267 | 14.433 | 12.167 | 0.514414 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 31.300 | 33.300 | 2.000 | 0.513042 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 15.733 | 18.067 | 2.333 | 0.512088 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 23.233 | 25.467 | 2.233 | 0.509399 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 19.867 | 23.367 | 3.500 | 0.508799 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 0.533 | 4.567 | 4.033 | 0.504821 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 28.633 | 30.167 | 1.533 | 0.503844 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 25.233 | 27.567 | 2.333 | 0.503704 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 10.000 | 11.333 | 1.333 | 0.503424 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 6.033 | 9.667 | 3.633 | 0.501670 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 4.433 | 6.133 | 1.700 | 0.501138 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 11.200 | 15.167 | 3.967 | 0.494803 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 29.633 | 31.433 | 1.800 | 0.483961 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 16.000 | 19.433 | 3.433 | 0.481808 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 3.133 | 4.233 | 1.100 | 0.480794 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 22.667 | 24.833 | 2.167 | 0.480637 |
