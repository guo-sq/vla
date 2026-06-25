# Diffusion Proxy 示教加速器

本文档说明 `diffusion_proxy` 的设计思路、数学定义、训练目标、推理打分和标签生成流程。该方法面向安全带 `both.hang.zhangyu` 双臂挂载示教数据，目标是在不修改原始 OSS 数据集的前提下，为每个 episode 生成 sidecar 形式的逐帧难度/精细度标签。

## 1. 方法目标

给定一组成功示教轨迹

$$
\mathcal{D}=\{\tau_i\}_{i=1}^{N},
\qquad
\tau_i=\{(o_t, s_t, v_t, c_t, a_t)\}_{t=0}^{T_i-1},
$$

其中：

- $o_t$ 是相机观测，当前实现使用 `observation.images.head`；
- $s_t\in\mathbb{R}^{14}$ 是 `observation.state`；
- $v_t\in\mathbb{R}^{14}$ 是 `observation.velocity`；
- $c_t\in\mathbb{R}^{14}$ 是 `observation.current`；
- $a_t\in\mathbb{R}^{14}$ 是机器人 action。

我们希望学习一个轻量 proxy policy，用它刻画条件分布

$$
p_\theta(A_t \mid x_t),
$$

其中 $x_t$ 是当前帧条件特征，$A_t$ 是从当前帧开始的未来 action chunk：

$$
A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}] \in \mathbb{R}^{H\times 14}.
$$

默认 horizon 为：

$$
H=16.
$$

episode 尾部不足 $H$ 帧时，用最后一帧 action padding，并用 mask 保证训练 loss 只在真实未来帧上计算。

核心假设沿用 DemoSpeedup 的语义：如果 proxy policy 在某个状态下生成的未来动作分布更集中，则说明示教在该 phase 上更一致、更可复现，通常对应更值得保留的精细操作段；反之，如果动作分布发散，则更像可加速的 casual 段。

因此本文使用：

$$
\text{precision}(t) \uparrow
\quad \Longleftrightarrow \quad
\text{entropy}(p_\theta(A_t\mid x_t)) \downarrow.
$$

注意：这里的 `precision` 不是“高不确定性”或“难以预测”，而是“低 entropy、模式一致、需要细致保留”的片段。

## 2. 数据范围

共享 manifest：

```text
../manifests/both_hang_zhangyu_repos.txt
```

当前正式处理的四个 repo 位于：

```text
/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt
```

repo 列表：

```text
seatbelt.both.hang.zhangyu.20260205.batch.1
seatbelt.both.hang.zhangyu.20260205.batch.2
seatbelt.both.hang.zhangyu.20260205.batch.3
seatbelt.both.hang.zhangyu.20260205.batch.4
```

正式配置覆盖：

| 项目 | 数值 |
| --- | ---: |
| episodes | 67 |
| frames | 69,676 |
| fps | 30 |
| action dim | 14 |
| camera | `observation.images.head` |
| visual encoder | `resnet18` |

## 3. 条件特征构造

每一帧的条件特征由三部分组成：

$$
x_t = \operatorname{concat}(p_t,\phi_t,z_t).
$$

其中 proprio-phase 特征为：

$$
p_t = [s_t, v_t, c_t]\in\mathbb{R}^{42},
\qquad
\phi_t=\frac{t}{T_i-1}\in[0,1].
$$

视觉特征 $z_t$ 来自冻结图像编码器：

$$
z_t = f_{\mathrm{vis}}(o_t).
$$

正式配置使用 ImageNet ResNet-18 的 avgpool 输出：

$$
z_t\in\mathbb{R}^{512}.
$$

因此正式条件维度为：

$$
\dim(x_t)=14+14+14+1+512=555.
$$

debug 配置也支持 `grid` 编码器。`grid` 编码器把图像缩放为 $8\times6$ RGB 网格，并拼接 RGB 均值、标准差和边缘统计，维度为 152。它便于快速冒烟测试，但正式结果使用 `resnet18`。

视觉特征会预先缓存到：

```text
cache/vision_head/
```

这样训练 diffusion proxy 时不需要重复解码视频。

## 4. 标准化

每个 out-of-fold 训练 fold 都只用训练 episode 估计标准化统计量：

$$
\mu_x,\sigma_x,\mu_a,\sigma_a.
$$

条件和目标 action chunk 分别标准化为：

$$
\tilde{x}_t = \frac{x_t-\mu_x}{\sigma_x},
\qquad
\tilde{A}_t = \frac{A_t-\mu_a}{\sigma_a}.
$$

这样可以避免 held-out episode 的统计信息泄漏到训练 fold 中。

## 5. 条件扩散模型

我们把未来 action chunk $\tilde{A}_t$ 看作扩散模型的干净样本：

$$
x_0 = \tilde{A}_t.
$$

前向加噪过程为 DDPM 标准形式：

$$
q(x_k\mid x_0)
=
\mathcal{N}\left(
\sqrt{\bar{\alpha}_k}x_0,\,
(1-\bar{\alpha}_k)I
\right),
$$

等价采样写作：

$$
x_k =
\sqrt{\bar{\alpha}_k}x_0
+
\sqrt{1-\bar{\alpha}_k}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,I).
$$

其中：

$$
\alpha_k=1-\beta_k,
\qquad
\bar{\alpha}_k=\prod_{j=1}^{k}\alpha_j.
$$

当前实现使用线性 beta schedule：

$$
\beta_1=10^{-4},
\qquad
\beta_K=2\times10^{-2},
\qquad
K=100.
$$

模型是一个小型条件 MLP denoiser：

$$
\epsilon_\theta(x_k,\tilde{x}_t,k)
\approx
\epsilon.
$$

网络结构：

| 参数 | 默认值 |
| --- | ---: |
| action horizon | 16 |
| action dim | 14 |
| hidden dim | 512 |
| residual blocks | 4 |
| timestep embedding dim | 128 |
| diffusion steps | 100 |

输入的 noisy action chunk 被 flatten 后映射到 hidden space；条件特征和 timestep sinusoidal embedding 分别投影到同一 hidden space，再在 residual blocks 中注入：

$$
h_0 = W_x\operatorname{vec}(x_k),
$$

$$
c = g_x(\tilde{x}_t) + g_k(\operatorname{emb}(k)),
$$

$$
h_{\ell+1}
=
h_\ell + F_\ell(\operatorname{LN}(h_\ell)+c).
$$

最终输出 reshape 回 $H\times14$ 的噪声预测。

## 6. 训练目标

对训练样本 $(\tilde{x}_t,\tilde{A}_t)$，随机采样扩散步：

$$
k\sim \operatorname{Uniform}\{0,\ldots,K-1\},
$$

并采样噪声：

$$
\epsilon\sim\mathcal{N}(0,I).
$$

模型训练目标是 masked noise MSE：

$$
\mathcal{L}(\theta)
=
\frac{
\sum_{h=0}^{H-1}m_{t,h}
\left\|
\epsilon_{\theta}(x_k,\tilde{x}_t,k)_h-\epsilon_h
\right\|_2^2
}{
\sum_{h=0}^{H-1}m_{t,h}\cdot d_a
},
$$

其中：

- $m_{t,h}\in\{0,1\}$ 是 action chunk mask；
- $d_a=14$ 是 action 维度；
- episode 尾部 padding 的 action 不参与 loss。

优化器配置：

| 参数 | 默认值 |
| --- | ---: |
| optimizer | AdamW |
| lr | $3\times10^{-4}$ |
| weight decay | $10^{-4}$ |
| batch size | 1024 |
| epochs | 80 |
| EMA decay | 0.995 |

checkpoint 中同时保存普通模型权重和 EMA 权重。推理默认使用 EMA 权重。

## 7. Episode-Level Out-of-Fold 训练

为了让 proxy 的 entropy 更接近“泛化到未见 episode 的不确定性”，当前实现使用 episode-level out-of-fold 训练。

将 episode key 集合划分为 $M$ 个 fold：

$$
\mathcal{E}
=
\mathcal{E}_1\cup\cdots\cup\mathcal{E}_M,
\qquad
\mathcal{E}_i\cap\mathcal{E}_j=\varnothing.
$$

第 $m$ 个模型在：

$$
\mathcal{E}\setminus\mathcal{E}_m
$$

上训练，只在 held-out fold $\mathcal{E}_m$ 上生成标签。正式配置使用：

$$
M=5.
$$

这样每个 episode 的 diffusion score 都来自没有训练过该 episode 的模型，降低“训练集记忆”导致 entropy 过低的问题。

## 8. 推理与 Entropy 估计

推理阶段，对每个 held-out 帧条件 $\tilde{x}_t$，用对应 fold 的模型重复采样 $S$ 个未来 action chunk：

$$
\hat{A}^{(1)}_t,\hat{A}^{(2)}_t,\ldots,\hat{A}^{(S)}_t
\sim
p_\theta(A_t\mid \tilde{x}_t).
$$

采样使用 DDIM 风格的确定性反推更新。实现中默认训练扩散步 $K=100$，推理可用较少采样步，例如 20 steps，以降低计算成本。

对同一帧，raw entropy 定义为样本方差在 horizon 和 action 维度上的均值：

$$
u_t
=
\frac{1}{H d_a}
\sum_{h=0}^{H-1}
\sum_{j=1}^{d_a}
\operatorname{Var}_{s=1}^{S}
\left[
\hat{A}^{(s)}_{t,h,j}
\right].
$$

同时记录 reconstruction error 作为诊断指标：

$$
r_t
=
\frac{1}{\sum_h m_{t,h}d_a}
\sum_{h,j}
m_{t,h}
\left(
\bar{A}_{t,h,j}-\tilde{A}_{t,h,j}
\right)^2,
$$

其中：

$$
\bar{A}_t=\frac{1}{S}\sum_{s=1}^{S}\hat{A}^{(s)}_t.
$$

当前标签主要使用 $u_t$，$r_t$ 只作为报告和排查模型质量的辅助信号。

## 9. 平滑与 Robust Scaling

采样方差会有逐帧噪声，因此先在每个 episode 内做滑动平均：

$$
\bar{u}_t
=
\frac{1}{2w+1}
\sum_{\Delta=-w}^{w}
u_{t+\Delta}.
$$

默认：

$$
w=8,
$$

即 30 FPS 下约 17 帧窗口，约 0.57 秒。

然后把所有 episode 的 $\bar{u}_t$ 拼接起来，做 robust unit scaling。设全局 5% 和 95% 分位数为：

$$
q_5=\operatorname{Quantile}_{0.05}(\bar{u}),
\qquad
q_{95}=\operatorname{Quantile}_{0.95}(\bar{u}).
$$

则：

$$
e_t
=
\operatorname{clip}
\left(
\frac{\bar{u}_t-q_5}{q_{95}-q_5},
0,1
\right).
$$

这里：

$$
e_t=\texttt{diffusion\_entropy\_score}.
$$

precision score 定义为：

$$
p_t = 1-e_t,
$$

即：

$$
p_t=\texttt{diffusion\_precision\_score}.
$$

## 10. 逐帧标签生成

先计算 action speed，用于排除明显静止段：

$$
v^a_t = \|a_t-a_{t-1}\|_2.
$$

同样经过 robust scaling 后，取全局 10% 分位数作为静止阈值：

$$
\gamma_{\mathrm{static}}
=
\operatorname{Quantile}_{0.10}(v^a).
$$

precision 阈值取全局 top 25%：

$$
\gamma_{\mathrm{precision}}
=
\operatorname{Quantile}_{0.75}(p).
$$

casual 阈值取 entropy top 35%：

$$
\gamma_{\mathrm{casual}}
=
\operatorname{Quantile}_{0.65}(e).
$$

逐帧标签为：

$$
y_t=
\begin{cases}
\texttt{precision}, &
p_t\ge \gamma_{\mathrm{precision}}
\ \land\
v^a_t>\gamma_{\mathrm{static}},
\\
\texttt{casual}, &
e_t\ge \gamma_{\mathrm{casual}}
\ \land\
\text{not precision},
\\
\texttt{neutral}, &
\text{otherwise}.
\end{cases}
$$

对应 acceleration stride：

| label | stride |
| --- | ---: |
| `precision` | 2 |
| `neutral` | 2 |
| `casual` | 4 |

因此，precision 段不会被过度加速，casual 段允许更大的时间下采样。

## 11. Hard Span 生成

`hard_spans` 由 precision mask 得到。设：

$$
b_t=\mathbb{1}[y_t=\texttt{precision}].
$$

实现步骤：

1. 找出所有连续 $b_t=1$ 的区间；
2. 若两个区间的间隔不超过 10 帧，则合并；
3. 合并后长度小于 15 帧的区间丢弃；
4. 对保留区间左右各 padding 8 帧；
5. 输出 frame index 和秒级时间戳。

每个 span 同时记录该区间内的平均 diffusion precision score：

```text
mean_diffusion_precision_score
```

## 12. 规则标签与 Diffusion 标签融合

纯 diffusion proxy 能捕捉“低 entropy、模式一致”的片段，但在短促夹爪事件、jerk 或方向突变附近可能不够敏感。因此推荐最终使用规则标签与 diffusion 标签融合后的 sidecar。

融合输入：

- 规则版的 `hard_score`；
- diffusion proxy 的 `diffusion_precision_score`；
- 规则版的事件分数：

$$
g_t=
\max(
\texttt{gripper\_event\_score}_t,
\texttt{jerk\_score}_t,
\texttt{turn\_score}_t
).
$$

融合前的 precision 原始分数：

$$
f^{\mathrm{raw}}_t
=
0.50\,h^{\mathrm{rule}}_t
+
0.30\,p^{\mathrm{diff}}_t
+
0.20\,g_t.
$$

其中：

- $h^{\mathrm{rule}}_t$ 是规则版 hard score；
- $p^{\mathrm{diff}}_t$ 是 diffusion precision score；
- $g_t$ 是显式操作事件信号。

融合前的 casualness 原始分数：

$$
c^{\mathrm{raw}}_t
=
0.5\,c^{\mathrm{rule}}_t
+
0.5\,e^{\mathrm{diff}}_t.
$$

两个原始分数都经过 robust unit scaling：

$$
f_t=\operatorname{RobustScale}(f^{\mathrm{raw}}_t),
\qquad
c_t=\operatorname{RobustScale}(c^{\mathrm{raw}}_t).
$$

融合后的 precision 阈值和 casual 阈值仍使用：

$$
\operatorname{Quantile}_{0.75}(f),
\qquad
\operatorname{Quantile}_{0.65}(c).
$$

此外，如果规则版标签已经是 `precision`，且事件分数位于该 episode 的 top 20%，则强制保留为 precision：

$$
y^{\mathrm{fusion}}_t=\texttt{precision}
\quad
\text{if}
\quad
y^{\mathrm{rule}}_t=\texttt{precision}
\land
g_t \ge \operatorname{Quantile}_{0.80}(g_{\tau}).
$$

这一项用于保护抓取、挂接、gripper 开合、双臂微调等短时操作证据。

当前推荐最终 sidecar：

```text
labels/both_hang_zhangyu/diffusion_fused_labels.jsonl
```

## 13. 当前正式结果

纯 diffusion proxy：

| 指标 | 数值 |
| --- | ---: |
| precision frames | 16,681 |
| neutral frames | 28,608 |
| casual frames | 24,387 |
| hard spans | 204 |

rule+diffusion fused：

| 指标 | 数值 |
| --- | ---: |
| precision frames | 18,919 |
| neutral frames | 31,657 |
| casual frames | 19,100 |
| hard spans | 329 |

从行为上看，fused 版本比纯 diffusion 更能保留短促精细操作段；纯 diffusion 更偏向连续、稳定、低 entropy 的片段。

## 14. 输出文件

正式输出：

```text
labels/both_hang_zhangyu/diffusion_labels.jsonl
labels/both_hang_zhangyu/summary.json
labels/both_hang_zhangyu/diffusion_fused_labels.jsonl
labels/both_hang_zhangyu/diffusion_fused_summary.json
reports/both_hang_zhangyu/diffusion_proxy_report.md
reports/both_hang_zhangyu/plots/*.svg
```

checkpoint 和视觉缓存：

```text
cache/vision_head/
checkpoints/both_hang_zhangyu_oof/
```

每条 JSONL record 对应一个 episode，核心字段包括：

```text
repo_id
episode_index
task
length
fps
diffusion_precision_score
diffusion_entropy_score
diffusion_reconstruction_error
label
acceleration_stride
hard_spans
```

fused record 额外包含：

```text
fusion_precision_score
fusion_casualness_score
fusion_reason
rule_label
diffusion_label
rule_hard_spans
diffusion_hard_spans
```

## 15. 运行命令

### 15.1 提取视觉特征

```bash
cd /root/workspaces/wujie_gsq/Teaching_Accelerator/diffusion_proxy

/root/miniconda3/envs/openpi/bin/python scripts/extract_visual_embeddings.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --camera observation.images.head \
  --encoder resnet18 \
  --output-dir cache/vision_head
```

### 15.2 训练 OOF diffusion proxy

```bash
/root/miniconda3/envs/openpi/bin/python scripts/train_oof_diffusion_proxy.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --vision-cache cache/vision_head \
  --checkpoint-dir checkpoints/both_hang_zhangyu_oof \
  --folds 5 \
  --epochs 80
```

### 15.3 用 diffusion entropy 生成标签

```bash
/root/miniconda3/envs/openpi/bin/python scripts/label_with_diffusion_proxy.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --vision-cache cache/vision_head \
  --checkpoint-dir checkpoints/both_hang_zhangyu_oof \
  --output-dir labels/both_hang_zhangyu \
  --report-dir reports/both_hang_zhangyu \
  --smoothing-half-window 8
```

### 15.4 融合 rule-based 标签

```bash
/root/miniconda3/envs/openpi/bin/python scripts/fuse_rule_diffusion_labels.py \
  --rule-labels ../rule_based/labels/both_hang_zhangyu/rule_labels.jsonl \
  --diffusion-labels labels/both_hang_zhangyu/diffusion_labels.jsonl \
  --output labels/both_hang_zhangyu/diffusion_fused_labels.jsonl
```

## 16. Debug 命令

快速调试可以使用 `grid` 视觉编码器、小模型、少量 episode：

```bash
/root/miniconda3/envs/openpi/bin/python scripts/extract_visual_embeddings.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --output-dir cache/debug_vision_head \
  --max-episodes-per-repo 1 \
  --encoder grid

/root/miniconda3/envs/openpi/bin/python scripts/train_oof_diffusion_proxy.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --vision-cache cache/debug_vision_head \
  --checkpoint-dir checkpoints/debug_oof \
  --max-episodes-per-repo 1 \
  --encoder grid \
  --folds 2 \
  --epochs 2 \
  --hidden-dim 128 \
  --blocks 2 \
  --diffusion-steps 20 \
  --batch-size 512

/root/miniconda3/envs/openpi/bin/python scripts/label_with_diffusion_proxy.py \
  --root-dir /mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt \
  --repo-list-file ../manifests/both_hang_zhangyu_repos.txt \
  --vision-cache cache/debug_vision_head \
  --checkpoint-dir checkpoints/debug_oof \
  --output-dir labels/debug_oof \
  --report-dir reports/debug_oof \
  --max-episodes-per-repo 1 \
  --encoder grid \
  --sampling-steps 5 \
  --samples-per-frame 4 \
  --smoothing-half-window 8
```

## 17. 设计取舍与局限

1. 该方法仍是 proxy policy，不是最终 pi0.5 action expert。它的价值在于给数据筛选和时间加速提供轻量、可解释的相对分数。

2. 低 entropy 不必然等于任务关键。某些重复、稳定但不关键的动作也可能低 entropy。因此正式建议使用 rule+diffusion fusion，而不是单独依赖 diffusion。

3. 高 entropy 不直接标为 hard。高 entropy 更可能表示示教模式分散、视觉条件不足、proxy 泛化差或动作存在多解；在当前语义中更接近 `casual`。

4. OOF 训练能缓解训练集记忆，但数据量仍然较小。若后续扩展到更多 seatbelt repo，应重新训练并重新标定全局 quantile。

5. 当前视觉只使用 head camera 的冻结 embedding，没有端到端训练视觉编码器。这降低了工程成本，也限制了对细小接触状态的感知能力。

## 18. 推荐使用方式

当前最推荐的候选标签是：

```text
labels/both_hang_zhangyu/diffusion_fused_labels.jsonl
```

如果目标是人工检查模型是否合理，优先查看：

```text
reports/both_hang_zhangyu/plots/
reports/both_hang_zhangyu/diffusion_proxy_report.md
```

如果目标是分析 proxy 本身质量，重点看：

```text
diffusion_entropy_score
diffusion_precision_score
diffusion_reconstruction_error
```

如果目标是构造下游训练数据，优先使用：

```text
label
acceleration_stride
hard_spans
```
