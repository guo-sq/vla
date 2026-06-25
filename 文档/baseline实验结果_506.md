# Seatbelt Baseline v1 实验结果

日期：2026-05-06

## 1. 当前状态

已创建 baseline 训练配置：

`openpi_modified/src/openpi/configs/cfg_baseline_seatbelt_lora_506.py`

已完成 baseline v1 训练与测试。

checkpoint：

`eval_results/baseline_506/checkpoints/seatbelt_lora_baseline_506/seatbelt_lora_baseline_506_exp/499`

checkpoint 大小约 `8.9G`。当前保存方式仍会保存完整 params 和 train_state，不只是 LoRA adapter。

训练设置：

| 项目 | 值 |
| --- | --- |
| base checkpoint | seatbelt `84999` |
| train repos | 15 |
| train steps | 500 |
| batch size | 1 |
| seed | 42 |
| LoRA | `gemma_2b_lora` + `gemma_300m_lora` |
| RECAP / value head / speed prompt | 均未启用 |

训练结果：

- 训练完成 500 step。
- 最终 checkpoint 保存到 step `499`。
- 未出现 OOM、NaN 或数据加载失败。
- 最后可见日志 `Step 490`：`action_loss=0.0009`，`grad_norm=0.0551`，`total_loss=0.0009`。

## 2. 结果汇总

评测设置：

- `num_batches=10`
- `batch_size=1`
- `sample_steps=4`
- `duration_dims=0-5,7-12`
- `duration_abs_tol=0.02`
- `duration_rel_tol=0.05`
- `duration_stable_steps=3`

指标说明：

- 表中的 `speedup factor` 是 `open-loop duration speedup factor`，计算方式为 `gt_mean_seconds / pred_mean_seconds`。
- `pred_mean_seconds` / `gt_mean_seconds` 不是实机完成时间，而是从固定长度 action chunk 中估计“该 chunk 何时连续接近自身末端动作”的帧数再除以 FPS。
- 因此该指标只表示模型输出动作 chunk 相比 GT chunk 是否更早收敛到自身末端动作，不能单独代表闭环成功率、真实任务完成时间或模型推理速度。

| split | repo | chunks | MSE | MAE | GT mean seconds | pred mean seconds | speedup factor | faster / same / slower |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| test_seen_task | `seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10` | 10 | 0.000904 | 0.011945 | 1.653 | 1.623 | 1.018 | 70% / 30% / 0% |
| test_seen_task | `seatbelt.single.hang.baichenglong.20260209.batch.15` | 10 | 0.017977 | 0.052404 | 1.633 | 1.623 | 1.006 | 30% / 70% / 0% |
| test_seen_task | `seatbelt.single.insert_move.baichenglong.20260225.batch.2` | 10 | 0.000765 | 0.010620 | 1.610 | 1.623 | 0.992 | 0% / 60% / 40% |
| test_seen_task | `seatbelt.single.take_off_move.panjinlong.20260303.batch.1` | 10 | 0.000141 | 0.002954 | 0.803 | 1.487 | 0.540 | 0% / 20% / 80% |
| test_unseen_task | `seatbelt.single.avoid_collision_1_move.zhaoshuai.20260306.batch.7` | 10 | 0.000964 | 0.015630 | 1.600 | 1.587 | 1.008 | 40% / 60% / 0% |
| test_unseen_task | `seatbelt.single.back_home.zhangruixng.20260312.batch.6` | 10 | 0.005056 | 0.022449 | 1.633 | 1.633 | 1.000 | 0% / 100% / 0% |
| test_unseen_task | `seatbelt.single.collision_recovery_0_move.haoshuailing.20260311.batch.12` | 10 | 0.031471 | 0.036055 | 1.330 | 1.603 | 0.830 | 30% / 30% / 40% |
| test_unseen_task | `seatbelt.single.fix_insert_move.zhaoshuai.20260305.batch.8` | 10 | 0.005451 | 0.021279 | 1.633 | 1.580 | 1.034 | 80% / 20% / 0% |

分组均值：

| split | avg MSE | avg MAE | avg speedup factor |
| --- | ---: | ---: | ---: |
| test_seen_task | 0.004947 | 0.019481 | 0.889 |
| test_unseen_task | 0.010736 | 0.023853 | 0.968 |

## 3. 运行记录

- 训练：已完成。
- 测试：已完成 `test_seen_task` 和 `test_unseen_task` 两组，共 8 个 repo。

输出目录：

- 测试结果：`eval_results/baseline_506/test_results`
- 可视化：`eval_results/baseline_506/vis`

每个 repo 均已输出：

- `test_all_preds.npy`
- `test_all_gts.npy`
- `duration_summary.json`
- `duration_per_chunk.csv`
- `pred_and_gt_*.png`

## 4. 初步解读

本 baseline 不应被理解为“已实现动作加速”的模型，而应作为后续方法的参照线。

现象：

- seen 任务中，`avoid_obstacle` 和 `hang` 略快，`insert_move` 略慢，`take_off_move` 明显慢于示教。
- unseen 任务中，`fix_insert_move` 和 `avoid_collision` 略快，`back_home` 基本持平，`collision_recovery` 慢于示教且 MSE 较高。
- 更准确地说，`take_off_move` 和 `collision_recovery` 是当前 open-loop duration proxy 下最明显的慢场景，后续 metadata-conditioned / DemoSpeedup / RECAP 方法应重点观察它们是否改善。

后续对比时，应优先观察：

- 平均 speedup factor 是否超过本 baseline。
- `take_off_move`、`collision_recovery` 是否不再明显变慢。
- MSE / MAE 是否没有明显恶化。
