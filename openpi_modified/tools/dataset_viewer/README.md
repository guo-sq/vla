# Pour Water Frame Visualizer

三路相机帧（左/中/右）+ state/action 曲线同步浏览，用于快速定位关键阶段帧区间。


## 页面要点

- **按需解析**：选择 episode 后服务端按需抽帧、解析 state/action，并显示加载进度。
- **三视图**：左/中/右三路相机帧同步随 slider 切换；中间视图支持按钮或按 `R` 旋转 180°（会记住）。
- **曲线叠加**：每张小图可叠加 state（S）、action（A）以及 Δ（A−S，只有成功配对时才显示）。
- **数值读数**：每张小图右上角显示当前帧的 S/A/Δ 数值，并用红线标记当前帧位置。
- **配对逻辑**：按 index / 按名字 / 按相似度，把 state/action 画在同一张图或分开画（有啥画啥）。
- **预览**（页面最下方）：
  - **预览 repo_id**：每个 repo/dataset 展示第一个 episode 的三图拼接首帧，点击进入详情。
  - **预览当前 repo_id 的所有 episode**：按当前选择的 Dataset 分页预览其所有 episode 首帧。


## Step1: 指定数据集路径

- **推荐**：在 `visualize.py` 里配置 `DATASET_PATHS`（绝对路径列表）
- **也支持**：启动时用 `--dataset "<DATASET_PATH_OR_ROOT>"` 额外追加（可传单个数据集目录或包含多个数据集的根目录）


## Step2: Vscode运行，一般会自动映射端口

```bash
source .venv/bin/activate
cd tools/dataset_viewer
python -u visualize.py --port 8765
```

打开 `http://localhost:8765/viewer.html`



## 缓存

- `frames/<dataset_basename>/<episode>/<camera>/frame_000000.jpg`
- `thumbs_triptych/<dataset_basename>/<episode>.jpg`
