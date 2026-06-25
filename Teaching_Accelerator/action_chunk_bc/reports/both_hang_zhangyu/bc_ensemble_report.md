# BC Ensemble Label Report

- Episodes: 67
- Frames: 69676
- FPS: [30]
- Hard spans: 293
- Label counts: precision=11593, neutral=33696, casual=24387
- Checkpoints: `checkpoints/both_hang_zhangyu`
- Output: `labels/both_hang_zhangyu/bc_ensemble_labels.jsonl`

## Thresholds

```json
{
  "bc_precision_min": 0.8973654359579086,
  "ensemble_disagreement_casual_min": 0.366026796400547,
  "static_speed_max": 0.016281330958008766,
  "precision_quantile": 0.75,
  "casual_quantile": 0.65,
  "static_speed_quantile": 0.1
}
```

## Top Precision Spans

| repo_id | episode | start_s | end_s | duration_s | mean_bc_precision_score |
|---|---:|---:|---:|---:|---:|
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 10 | 32.933 | 33.933 | 1.000 | 0.965296 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 3 | 31.433 | 32.500 | 1.067 | 0.961992 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 29 | 10.833 | 16.400 | 5.567 | 0.949069 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 29 | 24.267 | 29.267 | 5.000 | 0.940961 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 15 | 9.833 | 16.500 | 6.667 | 0.940910 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 5 | 17.967 | 20.267 | 2.300 | 0.940294 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 20 | 9.367 | 17.467 | 8.100 | 0.939885 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 18 | 25.133 | 27.333 | 2.200 | 0.938468 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 17 | 9.000 | 17.100 | 8.100 | 0.936469 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 12 | 24.200 | 25.933 | 1.733 | 0.936437 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 12 | 14.200 | 18.233 | 4.033 | 0.936263 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 2 | 11.700 | 18.300 | 6.600 | 0.931880 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 13 | 28.767 | 29.800 | 1.033 | 0.931116 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 27 | 23.400 | 27.633 | 4.233 | 0.930590 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 23 | 9.033 | 16.433 | 7.400 | 0.928474 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 8 | 10.267 | 16.100 | 5.833 | 0.927705 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 7 | 0.533 | 1.667 | 1.133 | 0.922462 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 6 | 23.133 | 26.000 | 2.867 | 0.921557 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 5 | 23.467 | 26.033 | 2.567 | 0.921494 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 18 | 36.600 | 38.167 | 1.567 | 0.921270 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 8 | 10.967 | 16.733 | 5.767 | 0.920429 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 7 | 26.200 | 29.367 | 3.167 | 0.919707 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 6 | 10.267 | 16.933 | 6.667 | 0.919205 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 21 | 25.300 | 28.300 | 3.000 | 0.918925 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 19 | 22.700 | 26.933 | 4.233 | 0.918342 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 18 | 25.933 | 27.967 | 2.033 | 0.917460 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 14 | 8.400 | 17.200 | 8.800 | 0.916863 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 10 | 8.833 | 13.700 | 4.867 | 0.916123 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 9 | 9.367 | 13.800 | 4.433 | 0.915795 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 8.367 | 16.267 | 7.900 | 0.915665 |
