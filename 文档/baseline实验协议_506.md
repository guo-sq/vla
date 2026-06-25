# Seatbelt Baseline v1 实验协议

日期：2026-05-06

## 1. 实验目标

本实验用于建立后续动作加速训练的 baseline。

核心原则：

- 固定 base checkpoint。
- 固定训练集。
- 固定测试集。
- 固定评测脚本与参数。
- baseline 不加入 speed prompt、RECAP、value head 或额外加速目标。

后续 speed-conditioned LoRA、RECAP/value 等模型必须复用本协议，除非另起新版本协议。

## 2. Base Checkpoint

使用 seatbelt pi0.5 checkpoint：

`/mnt/oss_models/models_deploy/2603/seatbelt/pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery/pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery_exp/84999`

说明：

- `84999` 已经见过部分基础 seatbelt 数据。
- 本 baseline 是从已有 seatbelt 模型继续做多任务 LoRA 适配，不是从零训练。
- L20 只跑 LoRA，不做全量训练。

## 3. 训练配置

配置文件：

`openpi_modified/src/openpi/configs/cfg_baseline_seatbelt_lora_506.py`

关键设置：

| 项目 | 设置 |
| --- | --- |
| 模型 | `pi05=True` |
| LoRA 主模型 | `gemma_2b_lora` |
| LoRA action expert | `gemma_300m_lora` |
| base checkpoint | `84999/params` |
| batch size | `1` |
| train steps | `500` |
| seed | `42` |
| num workers | `0` |
| lazy load | `True` |
| save interval | `500` |
| keep period | `None` |
| RECAP | 关闭 |
| RL value head | 关闭 |
| speed prompt | 不使用 |

训练输出目录：

`eval_results/baseline_506/checkpoints`

## 4. 训练集

训练集共 15 个 repo，覆盖 4 类任务：

- `hang`
- `insert_move`
- `take_off_move`
- `avoid_obstacle`

具体 repo：

```text
seatbelt.single.hang.baichenglong.20260206.batch.10
seatbelt.single.hang.baichenglong.20260206.batch.11
seatbelt.single.hang.zhangyu.20260207.batch.1
seatbelt.single.hang.zhangyu.20260207.batch.2
seatbelt.single.insert_move.baichenglong.20260224.batch.1
seatbelt.single.insert_move.baichenglong.20260224.batch.2
seatbelt.single.insert_move.zhaoshuai.20260305.batch.1
seatbelt.single.insert_move.zhaoshuai.20260305.batch.2
seatbelt.single.take_off_move.panjinlong.20260302.batch.1
seatbelt.single.take_off_move.panjinlong.20260302.batch.2
seatbelt.single.take_off_move.haoshuailing.20260311.batch.9
seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.11
seatbelt.single.avoid_obstacle_1_move.baichenglong.20260226.batch.10
seatbelt.single.avoid_obstacle_2_move.baichenglong.20260225.batch.12
seatbelt.single.avoid_obstacle_3_move.baichenglong.20260226.batch.3
```

## 5. 测试集

### 5.1 test_seen_task

任务类型在训练集中出现过，但 repo/batch 没有放入本次 LoRA 训练集。

```text
seatbelt.single.hang.baichenglong.20260209.batch.15
seatbelt.single.insert_move.baichenglong.20260225.batch.2
seatbelt.single.take_off_move.panjinlong.20260303.batch.1
seatbelt.single.avoid_obstacle_0_move.baichenglong.20260225.batch.10
```

### 5.2 test_unseen_task

任务类型没有放入本次 LoRA 训练集，用于观察泛化。

```text
seatbelt.single.fix_insert_move.zhaoshuai.20260305.batch.8
seatbelt.single.avoid_collision_1_move.zhaoshuai.20260306.batch.7
seatbelt.single.collision_recovery_0_move.haoshuailing.20260311.batch.12
seatbelt.single.back_home.zhangruixng.20260312.batch.6
```

## 6. 评测参数

测试脚本：

`openpi_modified/scripts/test.py`

固定参数：

| 参数 | 值 |
| --- | --- |
| `sample_steps` | `4` |
| `num_batches` | `10` |
| `batch_size` | `1` |
| `num_workers` | `0` |
| `duration_analysis` | `1` |
| `duration_dims` | `0-5,7-12` |
| `duration_abs_tol` | `0.02` |
| `duration_rel_tol` | `0.05` |
| `duration_stable_steps` | `3` |
| `lazy_load_eval` | `1` |

每个测试 repo 输出：

- `test_all_preds.npy`
- `test_all_gts.npy`
- `duration_summary.json`
- `duration_per_chunk.csv`
- 轨迹可视化图

## 7. 训练命令

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
  --config src/openpi/configs/cfg_baseline_seatbelt_lora_506.py
```

## 8. 测试命令模板

```bash
cd /root/workspaces/wujie_gsq/vla/openpi_modified

LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH \
HF_HOME=/mnt/oss_data/tmp/hf_home \
HF_DATASETS_CACHE=/mnt/oss_data/tmp/hf_datasets_cache \
TMPDIR=/mnt/oss_data/tmp \
OPENPI_TMPDIR=/mnt/oss_data/tmp \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run python scripts/test.py \
  --ckpt_dir /root/workspaces/wujie_gsq/vla/eval_results/baseline_506/checkpoints/seatbelt_lora_baseline_506/seatbelt_lora_baseline_506_exp/499 \
  --config_name src/openpi/configs/cfg_baseline_seatbelt_lora_506.py \
  --dataset_root /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo_id <repo_id> \
  --num_batches 10 \
  --batch_size 1 \
  --sample_steps 4 \
  --num_workers 0 \
  --results_dir /root/workspaces/wujie_gsq/vla/eval_results/baseline_506/test_results/<split> \
  --vis_dir /root/workspaces/wujie_gsq/vla/eval_results/baseline_506/vis/<split>_ \
  --duration_analysis 1 \
  --duration_dims 0-5,7-12 \
  --duration_abs_tol 0.02 \
  --duration_rel_tol 0.05 \
  --duration_stable_steps 3 \
  --lazy_load_eval 1
```

