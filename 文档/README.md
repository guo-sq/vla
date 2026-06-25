# VLA 项目说明：以 π0.5 为主体，部分适配 π*0.6

当前项目的训练主体是 **π0.5 / OpenPI 改造链路**：核心策略配置仍加载 `pi05_base` 权重，并使用 `Pi0Config(pi05=True)` 做任务微调。项目同时引入了一部分 **π*0.6 / RECAP** 思路，主要体现在 value model、advantage 计算、indicator conditioning、自采集/干预数据处理等模块上。

本说明基于本地论文：

- `paper/pi_0.5.pdf`：`π0.5: a Vision-Language-Action Model with Open-World Generalization`
- `paper/pi0.6.pdf`：`π*0.6: a VLA That Learns From Experience`

## 总体判断

| 层级 | 当前项目状态 | 说明 |
| --- | --- | --- |
| π0.5 主策略 | 已落地，是当前主体 | seatbelt、pour_water、pick_place、fold_T 等配置都围绕 `Pi0Config(pi05=True)` / `pi05_base` 展开 |
| π0.5 高层子任务 + 低层动作 | 部分落地 | 有 subtask token、FAST action token、子任务录制标签、子任务条件动作等代码 |
| π0.5 异构数据共训练 | 工程化适配 | 通过多 repo_id、多任务数据配置、统一动作空间/14 维动作空间、frame preprocessors 实现 |
| π*0.6 value / advantage / RECAP | 部分落地 | 有 value head、value prediction、advantage/indicator 计算、indicator prompt conditioning |
| π*0.6 完整在线迭代闭环 | 部分落地 | self-play、intervention、value/advantage 工具有基础，但不是完整论文系统复刻 |
| π0.7 | 暂未落地 | 目前只有论文资料，没有 `pi07` / `pi0.7` 模型或配置实现 |

## π0.5 论文关键模块与项目对应

### 1. VLA 主体模型：视觉语言前缀 + 动作专家后缀

π0.5 论文主体是一个 Vision-Language-Action policy：输入图像、语言指令和状态，输出低层机器人动作。其关键结构是使用视觉语言模型处理多模态上下文，再由 action expert 预测连续动作。

项目对应：

| 论文模块 | 项目程序 | 说明 |
| --- | --- | --- |
| VLA 主模型 | `openpi_modified/src/openpi/models/pi0.py` | `Pi0` 类实现视觉 token、语言 token、动作 token/连续动作的主体前向逻辑 |
| π0.5 配置开关 | `openpi_modified/src/openpi/models/pi0_config.py` | `pi05=True` 启用 π0.5 行为；注释中说明 π0.5 的 discrete state input 和 adaRMSNorm 差异 |
| 图像编码 | `openpi_modified/src/openpi/models/pi0.py` | 使用 SigLIP 图像模块生成 image tokens |
| 语言/动作专家 | `openpi_modified/src/openpi/models/pi0.py` | PaliGemma/Gemma 模块与 action expert 组合，π0.5 下 action expert 使用 adaRMSNorm |
| 动作预测损失 | `openpi_modified/src/openpi/models/pi0.py`、`openpi_modified/scripts/train.py` | flow/action loss 在模型里计算，训练脚本汇总 action loss、subtask loss、FAST action loss |

### 2. π0.5 base 权重与任务微调

π0.5 论文强调先有通用 VLA，再在具体任务上进行适配。当前项目最明确的主链路就是在 `pi05_base` 上做 seatbelt 等任务微调。

项目对应：

| 任务/配置 | 项目程序 | 说明 |
| --- | --- | --- |
| seatbelt 14 维主训练 | `openpi_modified/src/openpi/configs/cfg_pi0.5_seatbelt_14_dim.py` | 当前重点配置；加载 `openpi-assets/checkpoints/pi05_base/params`，模型为 `Pi0Config(pi05=True, action_horizon=50)` |
| pour water 14 维 | `openpi_modified/src/openpi/configs/cfg_pi0.5_pour_water_14_dim.py` | 同样基于 `pi05_base`，并加入 pour water 数据和帧权重处理 |
| pick and place | `openpi_modified/src/openpi/configs/cfg_pi0.5_pick_and_place_14_dim.py` | π0.5 任务微调配置 |
| fold T | `openpi_modified/src/openpi/configs/cfg_pi0.5_fold_T_14_dim.py` | π0.5 任务微调配置 |
| 多公开数据集 | `openpi_modified/src/openpi/configs/cfg_opensource/` | 多个 `cfg_pi0.5_28_dim.*.py` 用于异构公开数据配置 |

### 3. 异构数据共训练与多机器人/多任务数据

π0.5 论文的核心之一是 co-training：混合不同机器人、不同任务、不同数据类型，使模型获得开放环境泛化能力。项目没有完整复刻论文的数据规模，但工程上保留了类似接口：多个 repo、多个任务、不同动作维度、不同数据预处理器。

项目对应：

| 论文模块 | 项目程序 | 说明 |
| --- | --- | --- |
| 多任务 repo 列表 | `openpi_modified/src/openpi/configs/dataset_config/` | seatbelt、pour_water 等任务的数据源定义 |
| 公开数据集配置 | `openpi_modified/src/openpi/configs/cfg_opensource/` | OpenX、DROID、RoboMIND、AgibotWorld 等数据配置 |
| 统一/对齐动作空间 | `Gr00tLerobotDataConfig` 相关配置 | 通过 `align_dim`、`target_action_dim`、`delta_action_mask_indices`、`unify_action_space` 控制动作空间 |
| 数据归一化 | `openpi_modified/scripts/compute_norm_stats.py`、`compute_norm_stats_fast.py` | 训练前计算 norm stats |
| 数据缓存 | `openpi_modified/src/openpi/training/shared_cache.py` | 共享缓存与多 worker 训练 IO 优化 |

### 4. 高层子任务预测与低层动作执行

π0.5 论文强调模型可以处理高层语义子任务，并用子任务辅助长程任务执行。项目里这一部分有两条实现线：训练侧的 subtask token / FAST token，采集侧的 subtask_index 标注。

项目对应：

| 论文模块 | 项目程序 | 说明 |
| --- | --- | --- |
| subtask token 配置 | `openpi_modified/src/openpi/models/pi0_config.py` | `pi05_subtask_fast`、`pi05_with_subtask`、`subtask_as_action_cond`、`infer_action_with_subtask` |
| subtask + FAST tokenizer | `openpi_modified/src/openpi/models/tokenizer.py` | `FASTTokenizerWithSubtask` 将 subtask 和 action token 编进 prompt 序列 |
| subtask/FAST 输入变换 | `openpi_modified/src/openpi/transforms.py` | `TokenizeFASTInputsWithSubtask`、`DecodeSubtaskFromTokens`、`ExtractFASTActionsWithSubtask` |
| subtask 训练损失 | `openpi_modified/src/openpi/models/pi0.py`、`openpi_modified/scripts/train.py` | 训练时把 `subtask_loss`、`fast_action_loss` 加入总 loss |
| pick_place 子任务条件配置 | `openpi_modified/src/openpi/configs/cfg_pi0.5_pick_and_place_14_dim_subtask_cond.py` | 启用 `pi05_subtask_fast=True`、`pi05_with_subtask=True`、`pi05_with_fast_action=True` |
| 录制子任务标签 | `lerobot_modified/src/lerobot/recording/task/subtask_manager.py` | 根据子任务时长维护当前 `subtask_index` |
| 数据帧写入 subtask_index | `lerobot_modified/src/lerobot/recording/runtime/control_loop.py` | 录制每帧时写入当前子任务索引 |
| YAML 子任务录制 | `lerobot_modified/src/lerobot/recording/record.py` | 支持 `subtask_config_path`、`record_task`，并复制 `subtask_annotations.jsonl` |

### 5. 指令、状态、图像到模型输入的转换

π0.5 依赖语言指令、视觉观测和状态输入。项目通过 transforms 和 tokenizer 将 LeRobot 数据转成模型输入。

项目对应：

| 论文模块 | 项目程序 | 说明 |
| --- | --- | --- |
| prompt/state tokenization | `openpi_modified/src/openpi/models/tokenizer.py` | `PaligemmaTokenizer` 将 task prompt 和离散化 state 编码为 token |
| 数据 transform 管线 | `openpi_modified/src/openpi/training/base_cfg.py` | 根据模型类型选择 PI0、PI05、PI05_SUBTASK_FAST 的输入输出 transform |
| prompt 来源 | `openpi_modified/src/openpi/transforms.py` | `PromptFromLeRobotTask`、`PromptFromEpisodeTask`、`InjectDefaultPrompt` 等 |
| 图像 resize/输入整理 | `openpi_modified/src/openpi/training/base_cfg.py` | transform 中包含 `ResizeImages`、`PadStatesAndActions` |

### 6. 项目额外工程改进：动作 chunk 与训练时 RTC

这部分不是 π0.5 论文本身的核心模块，但当前项目主配置已明显适配实时执行场景。

项目对应：

| 工程模块 | 项目程序 | 说明 |
| --- | --- | --- |
| action horizon | `openpi_modified/src/openpi/models/pi0_config.py`、各训练配置 | 默认/常用 `action_horizon=50` |
| 训练时 RTC | `openpi_modified/src/openpi/configs/cfg_pi0.5_seatbelt_14_dim.py` | `rtc_max_delay=15`，用于模拟/适配实时 chunk 执行延迟 |
| action prefix mask | `openpi_modified/scripts/train.py`、`openpi_modified/src/openpi/models/pi0.py` | 训练时只对有效 postfix action 计算 loss |
| open-loop 可视化 | `openpi_modified/scripts/test.py`、`scripts/test_unify.py` | 用于检查预测动作 chunk 和 GT 轨迹 |

## π*0.6 论文关键模块与项目对应

π*0.6 论文的核心是 RECAP：先用通用 VLA，再通过真实部署经验、奖励反馈、专家干预、value function 和 advantage-conditioned policy 让模型持续改进。当前项目适配的是其中一部分训练与数据处理机制，而不是完整复刻论文中的在线系统。

### 1. Value function / value head

π*0.6 使用 value function 估计动作/状态对任务完成的帮助。项目中已加入 value head 和 value model 配置。

项目对应：

| 论文模块 | 项目程序 | 说明 |
| --- | --- | --- |
| value head 开关 | `openpi_modified/src/openpi/models/pi0_config.py` | `enable_rl_value_head`、`value_bins`、`value_range`、`value_temperature` |
| value head 实现 | `openpi_modified/src/openpi/models/pi0.py` | `score_out_proj`、`attn_pool`、`score_mlp_*` 等 value 输出模块 |
| value 数据字段 | `openpi_modified/src/openpi/training/base_cfg.py` | `enable_rl_value_head` 时从数据中 repack `returns` |
| seatbelt value config | `openpi_modified/src/openpi/configs/cfg_pi06_train_distributional_value_model_t5gemma270M_seatbelt_data_0212_0304.py` | 使用 T5Gemma/Gemma 270M 配置训练 value model |
| bipiper clothes value config | `openpi_modified/src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_1215_0227_max3600.py` | π*0.6 风格 value model 配置 |
| value 推理/落盘 | `openpi_modified/scripts/compute_values.py` | 加载 value model，对数据集写出 value prediction |

### 2. Advantage 计算

π*0.6 的 RECAP 用 value function 计算 advantage，再把 advantage 转成 policy improvement 的条件信号。项目中有完整的 value prediction -> advantage -> indicator 工具。

项目对应：

| 论文模块 | 项目程序 | 说明 |
| --- | --- | --- |
| advantage 计算 | `openpi_modified/scripts/compute_advantages.py` | 支持 n-step / GAE、percentile threshold、clip、multi-dataset |
| indicator 生成 | `openpi_modified/scripts/compute_advantages.py` | 将 advantage 二值化/分桶，并保存为 indicators |
| advantage 参数配置 | `openpi_modified/src/openpi/configs/cfg_pi06_recap_indicator_bipiper_clothes_all_0322.py` | `ADVANTAGE_METHOD`、`ADVANTAGE_N_STEP`、`ADVANTAGE_PERCENTILE` 等 |

### 3. Advantage-conditioned policy / RECAP indicator conditioning

π*0.6 的关键是让策略根据 advantage 条件学习更优行为。项目中有两种痕迹：一种是模型内 optimality embedding，另一种是当前更明确使用的 prompt indicator 注入。

项目对应：

| 论文模块 | 项目程序 | 说明 |
| --- | --- | --- |
| RECAP optimality embedding | `openpi_modified/src/openpi/models/pi0_config.py`、`openpi_modified/src/openpi/models/pi0.py` | `enable_recap=True` 时使用 `optimality` embedding；目前不是主配置路径 |
| indicator prompt injection | `openpi_modified/src/openpi/transforms.py` | `AddAdvantageToPrompt` 向 task/prompt 注入 `Advantage: positive/negative` |
| indicator 数据预处理 | `openpi_modified/src/openpi/training/frame_attributes_preprocessors.py` | `IndicatorPreprocessor` 读取 indicators |
| RECAP indicator 配置 | `openpi_modified/src/openpi/configs/cfg_pi06_recap_indicator_bipiper_clothes_all_0322.py` | 注释明确写明 `compute_values -> compute_advantages -> this config` |

### 4. Demonstrations、rollouts、interventions、自我改进数据

π*0.6 使用 demonstrations、autonomous rollouts、human corrections/interventions 和 sparse rewards。项目侧有 intervention 字段和 self-play/推理录制链路，但需要具体任务数据和运行流程配合，属于部分适配。

项目对应：

| 论文模块 | 项目程序 | 说明 |
| --- | --- | --- |
| 人类干预标记 | `lerobot_modified/src/lerobot/recording/runtime/control_loop.py` | 写帧时保存 `is_human_intervention` |
| record / infer_record / self_play | `lerobot_modified/src/lerobot/recording/record.py` | 采集、推理采集、自博弈相关流程入口 |
| seatbelt self-play value 配置 | `openpi_modified/src/openpi/configs/cfg_pi06_seatbelt_value_selfplay_fixed.py` | 使用 scripted + self-play 数据训练 value |
| value/reward 诊断 | `openpi_modified/scripts/diagnose_gt_distribution.py`、`scripts/test_rl.py` | 用于检查 returns / advantage / value 分布 |

## 当前主训练链路

以 seatbelt 为例，当前主训练链路是：

```bash
cd /root/workspaces/wujie_gsq/vla/openpi_modified

uv run scripts/compute_norm_stats_fast.py \
  --config-name src/openpi/configs/cfg_pi0.5_seatbelt_14_dim.py

XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 uv run scripts/train.py \
  --config src/openpi/configs/cfg_pi0.5_seatbelt_14_dim.py
```

该链路的核心特征：

- `PRETRAINED_WEIGHT_PATH = "openpi-assets/checkpoints/pi05_base/params"`
- `model = Pi0Config(pi05=True, enable_rl_value_head=False, action_horizon=50)`
- `rtc_max_delay=15`
- 14 维动作空间，使用 delta joint actions
- 使用静止检测、有效帧裁剪、gripper 规则、sample weight 等帧属性预处理

## π*0.6 / RECAP 相关链路

项目中的 π*0.6 方向通常是三阶段：

```text
1. 训练或加载 value model
2. compute_values.py 对数据集生成 value prediction
3. compute_advantages.py 根据 value prediction 计算 advantage 和 indicator
4. 使用 RECAP indicator 配置继续训练策略
```

典型程序：

- `openpi_modified/scripts/compute_values.py`
- `openpi_modified/scripts/compute_advantages.py`
- `openpi_modified/src/openpi/configs/cfg_pi06_recap_indicator_bipiper_clothes_all_0322.py`
- `openpi_modified/src/openpi/configs/cfg_pi06_seatbelt_value_selfplay_fixed.py`

需要注意：这些文件名中出现 `pi06`，但很多策略训练本身仍然基于 `Pi0Config(pi05=True)` 或 π0.5 权重；`pi06` 在当前项目里更多表示 **RECAP/value/advantage 方向的实验配置**，不是完整替换成论文 π0.6/π*0.6 的官方完整系统。

## 尚未完整覆盖的论文能力

| 论文能力 | 当前项目状态 |
| --- | --- |
| π0.5 论文中的完整 web-scale 多模态数据共训练 | 只保留工程接口和部分数据配置，未见完整复刻 |
| π0.5 在新家庭长程移动操作的完整部署系统 | 当前项目更聚焦 AnyVerse/双臂任务训练与记录 |
| π*0.6 的完整在线 RECAP 闭环 | 有 value、advantage、indicator、自采集/干预基础，但未形成完整自动迭代系统 |
| π0.6 更大 backbone 的完整主策略替换 | 只在部分 value model/T5Gemma 配置中体现，主策略仍是 π0.5 |
| π0.7 steerable generalist foundation model | 暂未在代码中落地 |

## 目录速览

| 目录 | 作用 |
| --- | --- |
| `openpi_modified/` | OpenPI 改造主体：模型、训练、配置、测试、value/advantage/RECAP 工具 |
| `lerobot_modified/` | 机器人运行、数据录制、teleoperation、intervention、subtask_index 写入 |
| `paper/` | π0.5、π0.6、π0.7 及相关论文 |
| `训练模型.md` | 更细的训练链路、数据和命令说明 |
| `trajectory_speedup_readme.md` | 轨迹加速、RTC、部署加速相关说明 |

## 一句话总结

当前项目可以理解为：**以 π0.5/OpenPI 为主策略底座，围绕具体机器人任务做数据、动作空间、子任务和实时执行适配；同时吸收 π*0.6 的 value/advantage/RECAP 思路，用于自采集数据和经验反馈驱动的策略改进。**
