# 测试说明

本文档说明当前仓库的测试分层、套件入口，以及不同方向同学如何给自己的模块补充测试。

## 1. 当前测试分层

仓库现在采用“两层组织”：

1. **源码就近测试**：和实现文件放在一起，适合模块级单测。
2. **tests 目录集中测试**：适合跨模块契约、集成回归、团队约定类测试。

### 目录约定

- `src/openpi/**/*_test.py`
  - 适合直接测试某个源码模块。
  - 例如模型、训练数据流、shared 工具、policy。
- `scripts/*_test.py`
  - 适合测试训练/评测脚本入口。
- `tests/unit/**`
  - 适合补充“按主题组织”的单测，例如 config / transform / rl / openloop。
- `tests/integration/**`
  - 适合集成测试，覆盖多模块联动。

## 2. pytest 自动归类规则

根目录的 `conftest.py` 会按路径自动打 marker，因此大多数情况下**不需要手写 `@pytest.mark.pretrain` 之类的标记**。

### 自动 marker 规则

#### 通用层

- `tests/unit/**`、`src/openpi/**`、`packages/openpi-client/**` → 自动打 `unit`
- `tests/integration/**`、`scripts/train_test.py` → 自动打 `integration`
- `tests/unit/config/**` → 额外打 `config`

#### 预训练相关

以下路径会自动打 `pretrain`：

- `src/openpi/models/**`
- `src/openpi/shared/**`
- `src/openpi/training/**`
- `src/openpi/transforms_test.py`
- `packages/openpi-client/**`
- `tests/unit/config/**`
- `tests/unit/transform/**`
- `tests/integration/**`

#### 后训练相关

以下路径会自动打 `posttrain`：

- `src/openpi/policies/**`
- `scripts/train_test.py`
- `tests/unit/openloop/**`

#### RL 相关

以下路径会自动打 `rl`：

- `tests/unit/rl/**`
- `scripts/test_rl.py`
- `scripts/train_rl.py`

### 手动 marker 什么时候加

以下 marker 仍建议手动加：

- `@pytest.mark.gpu`：必须 GPU 才能跑
- `@pytest.mark.slow`：本地默认不建议频繁跑
- `@pytest.mark.manual`：只能人工触发，不进默认套件

## 3. 套件入口

统一入口是 [tests/run_tests.sh](tests/run_tests.sh)。

### 常用命令

- `bash tests/run_tests.sh smoke`
- `bash tests/run_tests.sh unit`
- `bash tests/run_tests.sh config`
- `bash tests/run_tests.sh config-loss`
- `bash tests/run_tests.sh config-loss-drift`
- `bash tests/run_tests.sh config-loss-refresh`
- `bash tests/run_tests.sh integration`
- `bash tests/run_tests.sh pretrain`
- `bash tests/run_tests.sh posttrain`
- `bash tests/run_tests.sh rl`
- `bash tests/run_tests.sh ci`
- `bash tests/run_tests.sh full`

### 单独跑某个 test

当需要只验证一个文件、一个测试函数，或一个参数化 case 时，直接用 `pytest nodeid` 最稳妥。

#### 跑单个测试文件

- `uv run pytest src/openpi/transforms_test.py -q`

#### 跑单个测试函数

- `uv run pytest src/openpi/transforms_test.py::test_repack_transform -q`

#### 跑参数化 case

- `uv run pytest 'scripts/train_test.py::test_train[debug]' -q`

#### 结合套件筛选某个测试

如果希望仍沿用当前 marker/套件逻辑，可以直接给 `run_tests.sh` 追加 pytest 参数：

- `bash tests/run_tests.sh unit src/openpi/transforms_test.py::test_repack_transform`
- `bash tests/run_tests.sh posttrain 'scripts/train_test.py::test_train[debug]'`

#### 常用排查参数

- `-x`：遇到首个失败立即停止
- `-vv`：显示更详细的用例信息
- `-k 关键词`：按名称模糊筛选
- `-s`：打印 `print` 输出

### 各套件含义

- `smoke`：超快回归，提交前最低要求
- `unit`：模块单测
- `config`：配置加载、契约和配置回归测试
- `config-loss`：根目录 config 文件的保存基线对比检查
- `config-loss-drift`：只输出和保存基线不一致的顶层 config，适合 CI 日志排查
- `config-loss-refresh`：重新生成顶层 config 的 loss 基线
- `integration`：跨模块联动验证
- `pretrain`：预训练模型 / tokenizer / data pipeline / config
- `posttrain`：后训练 / 推理 / policy / 子任务评测
- `rl`：RL 配置和训练脚本相关
- `ci`：默认 CI 套件，附带覆盖率
- `full`：全量测试（排除 `manual`）

## 4. 公共模块如何新增测试

这里的“公共模块”主要指多个方向都会依赖的通用层，例如：

- `src/openpi/shared/**`
- `src/openpi/transforms.py`
- `src/openpi/training/data_loader.py`
- 通用 config 契约

### 推荐原则

1. **优先源码就近放测试**
   - 例如给 `src/openpi/shared/normalize.py` 加测试，应优先写在 `src/openpi/shared/normalize_test.py`。
2. **一个测试文件只测一个主题**
   - 不要把下载、归一化、图像处理混在同一个测试文件里。
3. **先测契约，再测细节**
   - 输入输出 shape
   - dtype
   - key 是否齐全
   - 异常分支是否正确报错
4. **尽量使用 fake data / fixture，避免外部依赖**
   - 不要默认依赖线上 Hugging Face、真实大数据集、外部服务。
5. **避免把慢测试混进 smoke**
   - 快速回归应控制在分钟级内。

### 示例：shared 模块

如果新增文件：

- `src/openpi/shared/foo.py`

建议同时新增：

- `src/openpi/shared/foo_test.py`

建议覆盖：

- 正常输入
- 边界输入
- 非法输入 / 报错分支
- shape / dtype / key 契约

### 示例：transform 模块

如果改动：

- `src/openpi/transforms.py`

建议：

- 直接补充 [src/openpi/transforms_test.py](src/openpi/transforms_test.py)
- 如果是跨配置/跨数据流行为，再补到 `tests/unit/transform/`

### 示例：config 契约

如果新增训练配置或修改配置拼装逻辑：

- 优先补到 `tests/unit/config/`
- 如果涉及多模块协同，再补 `tests/integration/test_integration.py`

### 顶层 config 的 loss 一致性测试

对于 `src/openpi/configs/` 根目录下的 `.py` 配置文件，现在统一使用：

- `tests/unit/config/test_top_level_config_loss_consistency.py`

这个测试会：

- 自动枚举 `src/openpi/configs/` 根目录下的配置文件（不含子目录）
- 逐个加载 `cfg`
- 使用固定 seed 构造测试模型与 fake obs/action
- 执行两次 `compute_loss`
- 将当前 loss 与 `tests/test_data/top_level_config_loss_baseline.json` 中保存的基线结果逐项对比
- 校验 loss 结果 shape 合法、值有限、且与保存基线完全一致

这份基线的主索引文件 `tests/test_data/top_level_config_loss_baseline.json` 只保存元信息和摘要字段，完整 loss 数组拆分存放在 `tests/test_data/top_level_config_loss_values/` 下的独立 `.npy` 文件中，这样主 JSON 的 diff 会更短。

当前环境下由于仓库根目录存在损坏的 `oss_data` 挂载点，`bash tests/run_tests.sh config-loss` 默认走独立检查脚本，而不是直接调用 pytest 收集，这样可以稳定执行同一套基线对比逻辑，同时仍然统一从 `tests/run_tests.sh` 入口触发。

主索引里会保存便于 review 的摘要字段：

- `sha256`
- `mean`
- `sum`
- `min`
- `max`

为了让测试保持单测级别可运行，Pi0 系列配置会在测试里自动替换为 `dummy` variant，只验证训练 loss 计算路径的一致性，不验证大模型规模下的数值基线。

如果当前版本需要刷新这份基线，可以执行：

- `bash tests/run_tests.sh config-loss-refresh`

如果只想在 CI 或本地快速看有哪些 config 漂移了，可以执行：

- `bash tests/run_tests.sh config-loss-drift`

## 5. 预训练同学如何新增 unittest

预训练方向通常覆盖：

- `src/openpi/models/**`
- `src/openpi/training/**`
- `src/openpi/shared/**`
- tokenizer / transform / config / 数据流

### 推荐落点

#### 模型级逻辑

放在源码旁：

- `src/openpi/models/model_test.py`
- `src/openpi/models/pi0_test.py`
- `src/openpi/models/tokenizer_test.py`

适合测试：

- 前向 shape
- loss 输出契约
- sample 接口
- tokenizer encode/decode
- checkpoint restore 后行为

#### 数据与配置契约

放在：

- `tests/unit/config/`
- `tests/unit/transform/`
- `tests/integration/`

适合测试：

- 配置可加载
- 配置字段组合是否合法
- 数据 transform 输出字段完整
- 归一化逻辑和 fallback 逻辑

### 预训练测试 checklist

- 是否可离线运行
- 是否使用 fake dataset 或最小样本
- 是否断言了 shape / dtype / key
- 是否覆盖异常分支
- 是否会误依赖不存在的 config 名

## 6. 后训练同学如何新增 unittest

后训练方向通常覆盖：

- `src/openpi/policies/**`
- `scripts/train_test.py`
- `tests/unit/openloop/**`

### 推荐落点

#### policy 级逻辑

放在源码旁：

- `src/openpi/policies/policy_test.py`

适合测试：

- policy 输入输出契约
- 不同 policy config 的构造
- 推理接口是否返回预期 action shape

#### 评测 / openloop / 子任务逻辑

放在：

- `tests/unit/openloop/`

适合测试：

- 指标计算
- 文本匹配/子任务匹配
- 多样本聚合逻辑

### 后训练测试 checklist

- 是否避免依赖真实 checkpoint
- 是否使用最小 mock / dummy tokenizer
- 是否把脚本入口测试和纯函数测试分开

## 7. RL 同学如何新增 unittest

RL 方向当前建议把“配置约束”和“训练脚本入口”分开。

### 推荐落点

#### RL 配置/契约

放在：

- `tests/unit/rl/`

适合测试：

- value head 配置合法性
- distributional 配置约束
- RL 特有超参组合是否合法

#### RL 脚本入口

放在：

- `scripts/test_rl.py`
- `scripts/train_rl.py`

适合测试：

- CLI/配置拼装
- 训练前初始化
- 关键 guard rail

### RL 测试 checklist

- 是否把纯配置校验写成快速单测
- 是否避免引入长时间 rollout
- GPU 必需场景是否显式打 `gpu`

## 8. 新增测试时的命名规范

### 文件名

统一使用以下之一：

- `test_xxx.py`
- `xxx_test.py`

当前仓库两种都会被 pytest 收集。

### 函数名

统一以 `test_` 开头，例如：

- `test_policy_returns_expected_action_shape`
- `test_tokenizer_handles_empty_prompt`

### fixture

公共 fixture 放在：

- 根目录 `conftest.py`：全局 marker / 环境行为
- [tests/conftest.py](tests/conftest.py)：tests 目录通用 fixture

如果某个 fixture 只服务于一个测试文件，就地写在该文件里即可，不要过度上提。

## 9. 推荐新增测试模板

### 模块旁单测模板

1. 准备最小输入
2. 调用目标函数/类
3. 断言：
   - shape
   - dtype
   - key
   - 数值/异常

### 集成测试模板

1. 准备最小 config
2. 替换外部依赖（如数据目录、repo、在线资源）
3. 跑完整链路的一小段
4. 断言最终契约而不是中间实现细节

## 10. 提交前建议

### 最低要求

公共模块改动提交前，至少运行：

- `bash tests/run_tests.sh smoke`
- `bash tests/run_tests.sh unit`

### 按方向补充

- 预训练改动：再跑 `bash tests/run_tests.sh pretrain`
- 后训练改动：再跑 `bash tests/run_tests.sh posttrain`
- RL 改动：再跑 `bash tests/run_tests.sh rl`
- 改动跨多模块：补跑 `bash tests/run_tests.sh integration`

### 不建议

- 把依赖远程网络的测试作为默认必跑测试
- 把超长训练过程塞进 unit
- 用字符串存在性检查替代真实行为断言

## 11. 一句话准则

- **公共层测试放源码旁，测稳定契约。**
- **团队方向测试按 pretrain / posttrain / rl 分层补齐。**
- **默认测试要快、可离线、可复现。**
