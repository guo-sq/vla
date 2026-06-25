# LoRA 训练到测试链路验证

验证日期：2026-05-06

## 结论

已在当前 DSW 单张 NVIDIA L20 46GB 环境上跑通 LoRA smoke 流程：

1. 读取 seatbelt 数据集。
2. 加载已有 seatbelt pi0.5 checkpoint 的参数。
3. 初始化 LoRA 版本模型。
4. 训练 1 step。
5. 保存新 checkpoint。
6. 用 `test.py` 加载新 checkpoint 做 open-loop 推理测试。
7. 输出 MSE、MAE、动作耗时和 speedup 指标。

这说明：**L20 上可以先用 LoRA 打通训练到测试链路；全量训练 pi0.5 仍然需要 DLC 多卡环境。**

## LoRA 配置

配置文件：

`eval_results/lora_smoke/cfg_seatbelt_lora_smoke_l20.py`

关键设置：

| 项目 | 设置 |
| --- | --- |
| 基础 checkpoint | `/mnt/oss_models/models_deploy/2603/seatbelt/pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery/pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery_exp/84999/params` |
| norm stats | `84999/assets/20260312` |
| 训练 repo | `seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10` |
| LoRA 主模型 | `paligemma_variant="gemma_2b_lora"` |
| LoRA action expert | `action_expert_variant="gemma_300m_lora"` |
| 冻结策略 | `_model.get_freeze_filter()`，冻结非 LoRA 参数 |
| EMA | `ema_decay=None` |
| batch size | `1` |
| train steps | `1` |
| num workers | `0` |
| lazy load | `True` |

## 训练结果

训练命令：

```bash
cd /root/workspaces/wujie_gsq/vla/openpi_modified

LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH \
HF_HOME=/mnt/oss_data/tmp/hf_home \
HF_DATASETS_CACHE=/mnt/oss_data/tmp/hf_datasets_cache \
TMPDIR=/mnt/oss_data/tmp \
OPENPI_TMPDIR=/mnt/oss_data/tmp \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
WANDB_MODE=disabled \
uv run python scripts/train.py \
  --config /root/workspaces/wujie_gsq/vla/eval_results/lora_smoke/cfg_seatbelt_lora_smoke_l20.py
```

训练成功完成 1 step，关键输出：

```text
Step 0:
action_loss=0.0002
grad_norm=0.0230
total_loss=0.0002
```

训练阶段观察：

- 数据 lazy load 正常。
- 第一批 batch 正常：
  - image: `(1, 224, 224, 3)`
  - state: `(1, 32)`
  - action: `(1, 50, 32)`
- 基础权重成功从 84999 checkpoint 恢复。
- LoRA train state 成功初始化。
- 首次编译和 1 step 总耗时约 5 分钟多。
- checkpoint 保存成功。

输出 checkpoint：

`eval_results/lora_smoke/checkpoints/seatbelt_lora_smoke_l20/seatbelt_lora_smoke_l20_exp/0`

checkpoint 大小约 `8.4G`。当前保存方式仍会保存完整 params，而不只是 LoRA adapter，因此后续如果要长期跑 LoRA，可以考虑增加“只保存 LoRA 参数”的轻量保存逻辑。

## 测试结果

测试命令：

```bash
cd /root/workspaces/wujie_gsq/vla/openpi_modified

LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH \
HF_HOME=/mnt/oss_data/tmp/hf_home \
HF_DATASETS_CACHE=/mnt/oss_data/tmp/hf_datasets_cache \
TMPDIR=/mnt/oss_data/tmp \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python scripts/test.py \
  --ckpt_dir /root/workspaces/wujie_gsq/vla/eval_results/lora_smoke/checkpoints/seatbelt_lora_smoke_l20/seatbelt_lora_smoke_l20_exp/0 \
  --config_name /root/workspaces/wujie_gsq/vla/eval_results/lora_smoke/cfg_seatbelt_lora_smoke_l20.py \
  --dataset_root /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo_id seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10 \
  --num_batches 1 \
  --batch_size 1 \
  --sample_steps 1 \
  --num_workers 0 \
  --results_dir /root/workspaces/wujie_gsq/vla/eval_results/lora_smoke/test_results \
  --vis_dir /root/workspaces/wujie_gsq/vla/eval_results/lora_smoke/vis/lora_smoke_ \
  --duration_analysis 1 \
  --duration_dims 0-5,7-12 \
  --lazy_load_eval 1
```

测试成功完成 1 batch open-loop 推理。

| 指标 | 数值 |
| --- | ---: |
| pred shape | `(1, 50, 14)` |
| gt shape | `(1, 50, 14)` |
| MSE | `0.001726` |
| MAE | `0.017115` |
| sample_actions 耗时 | `17.893s` |
| 示教动作估计耗时 | `1.667s` |
| LoRA 输出估计耗时 | `1.267s` |
| speedup factor | `1.316` |
| faster / same / slower chunk rate | `100.00% / 0.00% / 0.00%` |

测试输出：

- `eval_results/lora_smoke/test_results/seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10/test_all_preds.npy`
- `eval_results/lora_smoke/test_results/seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10/test_all_gts.npy`
- `eval_results/lora_smoke/test_results/seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10/duration_summary.json`
- `eval_results/lora_smoke/test_results/seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10/duration_per_chunk.csv`
- `eval_results/lora_smoke/vis/lora_smoke_0506_1642_seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10/pred_and_gt_seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10.png`

## 结果解读

这次结果只能说明链路跑通，不能代表模型效果已经可靠。原因：

- 只训练了 1 step。
- 只测试了 1 batch。
- `sample_steps=1` 是为了快速验证流程，不是正式评测设置。
- speedup 统计只有 1 个 action chunk，`100% faster` 没有统计稳定性。

不过它已经证明：

- L20 上 LoRA 比全量训练可行。
- 训练保存出的 LoRA checkpoint 可以被 `test.py` 恢复。
- 测试阶段就是推理加指标计算，当前链路已闭环。

## 下一步建议

1. 在 L20 上把 LoRA smoke 扩大到小规模实验：
   - `num_train_steps=100~1000`
   - `batch_size=1` 或尝试 `2`
   - `sample_steps=4~10`
   - `num_batches=5~10`

2. 固定 2 到 3 个 seatbelt 测试集做横向比较：
   - `seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10`
   - `seatbelt.single.avoid_collision_1_move.zhaoshuai.20260306.batch.7`
   - `seatbelt.single.hang.baichenglong.20260205.batch.5`

3. 如果 LoRA 指标和轨迹可视化稳定，再提交 DLC 做正式多卡训练或更长 LoRA 训练。

4. 如果 LoRA checkpoint 频繁保存导致空间压力，可以优先实现“只保存 LoRA adapter 参数”的保存/加载路径。
