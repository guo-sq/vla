# BC Ensemble Label Report

- Episodes: 8
- Frames: 8038
- FPS: [30]
- Hard spans: 36
- Label counts: precision=1370, neutral=3855, casual=2813
- Checkpoints: `checkpoints/debug_both_hang`
- Output: `labels/debug_both_hang/bc_ensemble_labels.jsonl`

## Thresholds

```json
{
  "bc_precision_min": 0.7616240680217743,
  "ensemble_disagreement_casual_min": 0.5248528808355332,
  "static_speed_max": 0.018452929332852364,
  "precision_quantile": 0.75,
  "casual_quantile": 0.65,
  "static_speed_quantile": 0.1
}
```

## Top Precision Spans

| repo_id | episode | start_s | end_s | duration_s | mean_bc_precision_score |
|---|---:|---:|---:|---:|---:|
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 30.933 | 33.967 | 3.033 | 0.889433 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 1 | 0.767 | 1.900 | 1.133 | 0.881681 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 0.300 | 1.433 | 1.133 | 0.873146 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 27.933 | 30.533 | 2.600 | 0.866366 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 1 | 0.267 | 1.500 | 1.233 | 0.864199 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 1 | 27.100 | 29.367 | 2.267 | 0.859573 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 31.133 | 33.867 | 2.733 | 0.858773 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 1 | 0.733 | 1.967 | 1.233 | 0.847028 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 0.567 | 2.467 | 1.900 | 0.846424 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 1 | 25.800 | 28.567 | 2.767 | 0.845912 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 31.100 | 33.433 | 2.333 | 0.842902 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 1 | 0.667 | 1.933 | 1.267 | 0.841676 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 1 | 31.567 | 34.433 | 2.867 | 0.833551 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 1 | 31.033 | 33.500 | 2.467 | 0.831170 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 16.267 | 18.733 | 2.467 | 0.829675 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 14.500 | 16.433 | 1.933 | 0.815215 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 8.967 | 16.433 | 7.467 | 0.757026 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 1 | 20.667 | 24.467 | 3.800 | 0.756309 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 1 | 3.467 | 4.967 | 1.500 | 0.751285 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 10.267 | 15.267 | 5.000 | 0.747815 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 0 | 20.933 | 24.100 | 3.167 | 0.746633 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 1 | 3.400 | 5.033 | 1.633 | 0.739868 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 10.233 | 14.267 | 4.033 | 0.738440 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 1 | 7.200 | 9.600 | 2.400 | 0.731761 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 3.167 | 5.367 | 2.200 | 0.730790 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 0 | 11.867 | 14.467 | 2.600 | 0.730732 |
| seatbelt.both.hang.zhangyu.20260205.batch.2 | 0 | 7.333 | 10.367 | 3.033 | 0.716662 |
| seatbelt.both.hang.zhangyu.20260205.batch.1 | 1 | 3.567 | 5.733 | 2.167 | 0.706867 |
| seatbelt.both.hang.zhangyu.20260205.batch.3 | 1 | 25.833 | 28.433 | 2.600 | 0.697835 |
| seatbelt.both.hang.zhangyu.20260205.batch.4 | 0 | 18.600 | 20.200 | 1.600 | 0.675308 |
