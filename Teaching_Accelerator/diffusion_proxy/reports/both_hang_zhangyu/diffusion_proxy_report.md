# Diffusion Proxy Label Report

- Episodes: 67
- Frames: 69676
- Hard spans: 204
- Label counts: precision=16681, neutral=28608, casual=24387
- Checkpoints: `checkpoints/both_hang_zhangyu_oof`
- Output: `labels/both_hang_zhangyu/diffusion_labels.jsonl`

## Thresholds

```json
{
  "diffusion_precision_min": 0.8301121294498444,
  "diffusion_entropy_casual_min": 0.47998110204935074,
  "static_speed_max": 0.016281330958008766,
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
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 12 | 26.267 | 31.067 | 4.800 | 0.967314 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 15 | 10.700 | 15.933 | 5.233 | 0.966896 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 18 | 24.533 | 27.333 | 2.800 | 0.966435 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 21 | 24.267 | 29.167 | 4.900 | 0.963642 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 6 | 10.467 | 16.800 | 6.333 | 0.963021 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 1 | 10.233 | 18.367 | 8.133 | 0.962668 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 10 | 9.200 | 14.933 | 5.733 | 0.961221 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 7 | 24.567 | 30.633 | 6.067 | 0.959623 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 3 | 14.500 | 19.067 | 4.567 | 0.958607 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 3 | 21.733 | 29.467 | 7.733 | 0.958519 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 4 | 15.233 | 20.833 | 5.600 | 0.954337 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 27 | 9.533 | 16.200 | 6.667 | 0.953653 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 16 | 10.333 | 16.000 | 5.667 | 0.952450 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 29 | 11.800 | 16.667 | 4.867 | 0.952410 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 21.367 | 30.500 | 9.133 | 0.949818 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 6 | 16.733 | 20.333 | 3.600 | 0.949770 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 9 | 23.733 | 28.933 | 5.200 | 0.949253 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 3 | 21.567 | 25.400 | 3.833 | 0.948799 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 29 | 24.167 | 30.233 | 6.067 | 0.948262 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 12 | 23.800 | 25.933 | 2.133 | 0.948166 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 22 | 9.467 | 14.233 | 4.767 | 0.947889 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 5 | 22.767 | 26.800 | 4.033 | 0.944801 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 15 | 9.833 | 18.400 | 8.567 | 0.944483 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 9 | 14.467 | 19.467 | 5.000 | 0.944005 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 18 | 9.533 | 15.900 | 6.367 | 0.942635 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 21.967 | 25.700 | 3.733 | 0.940599 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 12 | 10.800 | 14.800 | 4.000 | 0.940280 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 2 | 26.967 | 30.800 | 3.833 | 0.939665 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 8 | 10.833 | 16.300 | 5.467 | 0.938857 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 6 | 21.967 | 27.667 | 5.700 | 0.937035 |
