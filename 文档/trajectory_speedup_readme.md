# Trajectory Speedup Project README

这份文档面向当前工作区中的两个仓库：

- `openpi_modified`：训练侧
- `lerobot_modified`：部署侧

目标是回答三个问题：

1. 当前项目里，和“轨迹加速”相关的能力已经做到哪一步？
2. 哪些判断我认可，哪些地方需要修正？
3. 现在最值得推进的实验路径是什么？

## 一句话结论

你的整体判断我大体认可。

最重要的结论有三条：

- `openpi_modified` 里，RTC 训练机制已经实装，不是空白。
- `lerobot_modified` 里，部署侧加速框架也已经有不少现成能力，不是从零开始。
- 当前最现实的主线，不是“从原始慢轨迹自动生成快轨迹”，而是“基于现有数据和现有 RTC/异步执行框架，做训练与部署联动优化”。

同时有一个需要修正的点：

- 我在当前这份本地 `lerobot_modified` checkout 里，没有找到你提到的 `gym_aloha / gym_pusht / gym_xarm` 具体环境实现文件落在 `src/lerobot/envs/` 下；我只验证到了环境配置、factory、以及对外部包 `gym_aloha / gym_pusht / gym_xarm` 的依赖入口。因此，“有仿真环境配置支持”是对的，但“当前仓库本身已经内置那三个环境实现并且可直接接异步链路”这点，当前 checkout 里不能直接下结论。

## 1. 项目现状梳理

### 1.1 `openpi_modified` 已经具备的能力

#### 1.1.1 RTC 训练机制已经在模型里

核心位置：

- `openpi_modified/src/openpi/models/pi0.py`
- `openpi_modified/scripts/train.py`
- `openpi_modified/src/openpi/training/base_cfg.py`

已验证到的关键点：

- 训练时会随机采样 `delay`，并构造 prefix/postfix mask。
- 推理时会使用 `action_prefix + delay` 做 prefill。
- `rtc_max_delay` 已经是训练配置的一部分，并真实传入 `compute_loss(...)`。

这意味着：

- `Training-Time Action Conditioning for Efficient Real-Time Chunking` 这条路线，在训练机制层面已经不是“待实现”，而是“待系统实验和调参”。

#### 1.1.2 demo 质量条件化路径已经存在

核心位置：

- `openpi_modified/src/openpi/training/frame_attributes_preprocessors/optimality_preprocessor.py`
- `openpi_modified/src/openpi/training/anyverse_dataset.py`
- `openpi_modified/src/openpi/models/model.py`
- `openpi_modified/src/openpi/models/pi0.py`

已验证到的关键点：

- frame attributes 里有 `optimality` 字段。
- dataset 会把 `optimality` 读出来，喂到模型 observation。
- `pi0.py` 在 `enable_recap` 路径下，会以 70% 概率启用 condition on optimality。

这意味着：

- 项目里已经有一个“按 demo 质量或轨迹优劣做条件化训练”的通用框架。
- 它和 `DemoSpeedup` 的核心思想是同源的，只是当前实现更偏“条件化”，不是直接“自动分段 + 自动加速 + 重写数据集”。

#### 1.1.3 数据层已经留出了“加速轨迹”通路

核心位置：

- `openpi_modified/src/openpi/training/anyverse_dataset.py`
- `openpi_modified/src/openpi/training/base_cfg.py`

已验证到的关键点：

- `use_state_as_action` 开关已经存在。

这意味着：

- 如果未来你要把“加速后的状态轨迹”映射成训练目标，数据侧不是堵死的。
- 这更像是一个研究入口，而不是现成成品。

#### 1.1.4 已经有 RTC 相关配置和脚本

核心位置：

- `openpi_modified/scripts/policy_inference.py`
- `openpi_modified/src/openpi/configs/cfg_pi05_base_pack_socks_0106_0310_wo_pjl_trtc6.py`
- 其他带 `rtc_max_delay` 的 config

已验证到的关键点：

- `policy_inference.py` 里直接有 `infer_interval` 和 `infer_delay` 的离线实验逻辑。
- 多个 config 已经显式设置 `rtc_max_delay`。

这意味着：

- “RTC 已写好但没人用” 和 “RTC 已系统跑过消融” 目前还不能混为一谈。
- 现在最需要补的是实验结论，而不是基础设施。

### 1.2 `lerobot_modified` 已经具备的能力

#### 1.2.1 部署侧 `inference_speedup` 已经存在

核心位置：

- `lerobot_modified/src/lerobot/recording/record.py`
- `lerobot_modified/tests/test_inference_speedup.py`

已验证到的关键点：

- `inference_speedup`
- `fusion_window`
- `smooth_sigma`
- `log_action_chunks`

这意味着：

- 仓库已经支持通过相机帧复用和更高有效控制频率，去做“同一 chunk 更快消费”的执行策略。
- 这不是论文概念占位，而是已经进了运行时配置与测试。

#### 1.2.2 自动估计 re-plan 间隔已经存在

核心位置：

- `lerobot_modified/src/lerobot/recording/runtime/policy_runtime.py`

已验证到的关键点：

- `compute_optimal_infer_interval(...)` 已实现。
- 它使用均值、方差、3σ 和 `action_horizon` 关系来推断可行的 `infer_interval`。

这意味着：

- 运行时已经不只是“写死一个 infer_interval”，而是开始显式把推理延迟统计引入调度。
- 这和 `SAIL` / `VLASH` 关注的“延迟感知执行”方向非常贴近。

#### 1.2.3 ActionBuffer + fusion + smoothing 已经存在

核心位置：

- `lerobot_modified/src/lerobot/recording/runtime/policy_runtime.py`

已验证到的关键点：

- `ActionBuffer`
- `fusion_type`
- `fusion_window`
- `smooth_sigma`
- stale chunk 跳过

这意味着：

- chunk 切换抖动、重规划覆盖、旧 chunk 与新 chunk 的平滑过渡，这些你最关心的部署细节，已经有基础框架。
- 这块非常适合作为当前没有真机时的“高杠杆调参与消融区”。

#### 1.2.4 异步 server / client 通路已经存在

核心位置：

- `lerobot_modified/src/lerobot/scripts/server/policy_server.py`
- `lerobot_modified/src/lerobot/scripts/server/robot_client.py`
- `lerobot_modified/src/lerobot/scripts/server/configs.py`
- `lerobot_modified/docs/source/async.mdx`

已验证到的关键点：

- `actions_per_chunk`
- `chunk_size_threshold`
- `aggregate_fn_name`
- `weighted_average`

这意味着：

- 异步执行不是只停留在 `record_unified.py` 层，而是有独立 server/client 路线。
- 这条路线更适合做系统实验和可重复评测。

#### 1.2.5 chunk 日志已经存在

核心位置：

- `lerobot_modified/src/lerobot/recording/record.py`

已验证到的关键点：

- `log_action_chunks` 已进配置和运行逻辑。

这意味着：

- 你完全可以先做 open-loop 分析，不必一上来就上真机闭环。
- 这也是未来接 `VLASH` 风格离线分析最顺手的切口之一。

## 2. 当前没有的东西

### 2.1 当前仓库里没有现成的“原始轨迹自动加速数据管线”

我目前没有在这两个仓库里看到一条完整的现成流水线：

`原始慢速轨迹 -> 自动分段 -> 自动加速 -> 生成新的 speedup dataset`

这件事在当前代码里最多只存在“零件”：

- `optimality` 框架
- `valid_mask / sample_weight`
- `use_state_as_action`
- `replay / replay_and_record`
- `infer_record`

但它们没有被拼成一个 `DemoSpeedup` 式自动数据重写工具。

### 2.2 当前仓库里没有被我验证到的“仿真异步全链路”

我验证到了：

- 环境配置入口
- 对 `gym_aloha / gym_pusht / gym_xarm` 外部包的依赖检查
- 示例和文档里引用这些环境

但我没有在当前 checkout 里验证到：

- `src/lerobot/envs/` 下内置的对应环境实现文件
- 已经和你们自定义 async policy server / ActionBuffer / inference_speedup 直接打通的适配层

所以更稳妥的说法是：

- 当前仓库“支持接仿真环境”这件事大概率成立；
- 但“你现在可以零改动直接用它替代真机验证完整异步加速链路”这件事，当前不能直接下结论。

## 3. 对你三条路径判断的回应

### 3.1 原始数据 -> 优化数据

我的判断：

- 当前阶段不适合作为主线。

原因：

- 当前没有现成自动加速数据流水线。
- 你现在也没有一个明确可批处理的“原始慢速母数据”入口被暴露出来。

但这条路不是死路：

- `optimality_preprocessor`、`valid_mask`、`sample_weight` 这些都可以复用。
- 如果以后拿到原始慢轨迹，这条路的工程改造成本并不高。

### 3.2 优化数据 -> 训模

我的判断：

- 这是当前最好的主线。

补充：

- 这里不是“从头实现 RTC”，而是“先确认 RTC 到底有没有被实际跑起来并带来收益”。
- 重点应该放在：
  - RTC vs non-RTC 消融
  - `rtc_max_delay` 分布
  - prefix mask 的训练比例
  - `optimality` 条件化是否真的改善异步部署表现

### 3.3 机器人执行加速

我的判断：

- 完整闭环验证仍然高度依赖真机。
- 但很多推理侧参数，不一定要等真机才能开始研究。

更具体地说：

- `fusion_type`
- `fusion_window`
- `smooth_sigma`
- `infer_interval`
- `actions_per_chunk`
- `chunk_size_threshold`

这些参数的 open-loop 行为和部分 sim 行为，是可以先研究的。

因此，“完全没法在非真机条件下推进”这个说法太保守；但“可以不靠真机完整验证最终方案”也不对。

## 4. 我建议的优先级

### 优先级 1：先把 RTC 在 `openpi_modified` 里的实际状态摸清

这是我最认同你的地方。

原因：

- 代码几乎是现成的。
- 训练侧改动成本最低。
- 一旦有效，能直接影响部署侧异步执行表现。

建议先做的事：

- 找出已有 run 或 checkpoint 是否真的开了 `rtc_max_delay`
- 跑一组最小消融：
  - `rtc_max_delay = 0`
  - `rtc_max_delay > 0`
- 在固定 `infer_delay` 的部署评测里比 success rate、tracking error、chunk continuity

### 优先级 2：补一个“仿真/离线适配层”，把部署参数先扫起来

这件事不一定要一开始就接真机。

目标不是“替代真机”，而是：

- 让你持续有便宜的地方去调
  - `fusion_window`
  - `smooth_sigma`
  - `infer_interval`
  - `chunk_size_threshold`

如果后面确认 gym 环境能接起来，这条路就非常值。

如果一时接不上，也可以先用：

- `log_action_chunks`
- 离线回放
- open-loop metrics

把 chunk 质量分析先做起来。

### 优先级 3：把 DemoSpeedup 做成最小可行实验

之前更偏向“先做占位实现”，现在看应该前移一步：DemoSpeedup 已经是明确的数据侧执行加速方法，而项目里也已经有 baseline v1 LoRA，可以先做最小闭环。

建议第一版不要全量重写所有数据，而是：

- 使用 baseline v1 作为 entropy / uncertainty estimator。
- 先在 `take_off_move` 等当前 open-loop duration proxy 最慢的训练片段上做保守下采样。
- `collision_recovery` 当前是测试集慢点，若训练集中没有同类 repo，则先作为重点评测目标，而不是强行生成同类数据。
- 下采样倍率先控制在 `1.2x~1.5x`，避免破坏接触和插入等高精度阶段。
- `optimality_preprocessor` 和 frame attributes pipeline 可以承载 frame score / sample weight，但真正 DemoSpeedup 还需要额外的数据重写层，保证 image、state、action、timestamp、episode index 对齐。

这样可以先验证“数据侧压缩轨迹分布”是否真的改善 seatbelt，而不是只停留在预留接口。

### 优先级 4：激进执行加速最后做

我也认同这一点。

像：

- future-state-aware replanning
- 自适应速度调制
- 更强的 faster-than-demo execution

这些都更依赖真机和任务闭环。
当前资源下，ROI 确实不如先把训练侧和离线分析侧吃透。

## 5. 建议的最小实验路线

### 阶段 A：先验证 RTC 到底有没有价值

目标：

- 明确 `rtc_max_delay` 对异步部署是否真的有帮助。

最小对比：

- 同数据
- 同模型结构
- 同 action horizon
- 仅改 `rtc_max_delay`

输出指标建议：

- success rate
- task completion time
- inference latency
- chunk overlap continuity
- tracking error / action jump

### 阶段 B：把部署侧 chunk 参数扫起来

目标：

- 找到最稳的异步执行参数区间。

优先扫这些：

- `infer_interval`
- `fusion_window`
- `smooth_sigma`
- `actions_per_chunk`
- `chunk_size_threshold`

如果真机贵，就先用：

- chunk log
- open-loop replay
- sim/离线适配

### 阶段 C：并行推进 metadata conditioning 和 DemoSpeedup 数据化

目标：

- 让模型真正偏向“快而稳”的轨迹分布。

可做方向：

- 用 `Quality / Speed / Mistake` metadata 取代单一 speed prompt。
- 训练时做 metadata dropout，先对齐 pi0.7 的 CFG 训练分布；真正 CFG 推理后续再补。
- 基于段级别 entropy / uncertainty 做 DemoSpeedup-style 数据重写。
- 明确区分 open-loop duration proxy、真机完成时间和推理延迟三类指标。

## 6. 你现在最该盯的文件

### 训练侧

- `openpi_modified/src/openpi/models/pi0.py`
- `openpi_modified/scripts/train.py`
- `openpi_modified/src/openpi/training/base_cfg.py`
- `openpi_modified/src/openpi/training/anyverse_dataset.py`
- `openpi_modified/src/openpi/training/frame_attributes_preprocessors/optimality_preprocessor.py`
- `openpi_modified/scripts/policy_inference.py`

### 部署侧

- `lerobot_modified/src/lerobot/recording/record.py`
- `lerobot_modified/src/lerobot/recording/runtime/policy_runtime.py`
- `lerobot_modified/src/lerobot/scripts/server/policy_server.py`
- `lerobot_modified/src/lerobot/scripts/server/robot_client.py`
- `lerobot_modified/docs/source/async.mdx`

## 7. 我想和你进一步交流的几个点

### 7.1 你们现在到底有没有“明确的 RTC 训练 run”

这是现在最值得优先确认的事情。

如果已经有：

- 我建议直接做已有 checkpoint 的异步部署对比。

如果还没有：

- 那你最有价值的第一项工作，就是补一个最小 RTC 消融实验。

### 7.2 你们手里的 `_fast` 数据，到底是怎么来的

这很关键。

如果 `_fast` 是人工快示教：

- 那它更像 imitation target quality 的提升。

如果 `_fast` 是后处理压缩出来的：

- 那它更接近 `DemoSpeedup` 数据路线。

这会直接影响你后面论文叙事和实验设计。

### 7.3 你是更想做“训练侧论文点”，还是“部署侧系统点”

这两条都能做，但主线要尽早定。

如果偏训练侧：

- RTC + optimality + speedup data conditioning

如果偏部署侧：

- infer interval + fusion + smoothing + future-state-aware replanning

当前我更建议你先从训练侧切入，再用部署侧做结果放大和验证。

## 最终判断

我认可你的大方向判断，而且我会把它再压成一句更稳的版本：

当前项目并不是“轨迹加速还是空白”；相反，训练侧 RTC、部署侧异步 chunk 和推理加速框架都已经有了。真正缺的不是基础设施，而是系统实验、证据链，以及把这些能力组织成一条清晰研究路线的工作。
