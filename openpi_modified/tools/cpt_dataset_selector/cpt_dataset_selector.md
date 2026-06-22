# CPT Dataset Selector：实现概览 / CLI / Web 界面

**可视化（流程图 + 表格）：** [cpt_dataset_selector.html](./cpt_dataset_selector.html)

面向 **Anyverse CPT / 公开 LeRobot 数据集** 的选题工具：从训练配置里的 `REPO_ID` 生成清单，扫描本机（或挂载根目录下）各 repo 的 **`meta/tasks.jsonl`**（及 `meta/info.json`）写入 **SQLite**，再通过 **FastAPI + 静态页** 按 **原子动作 / 物体类别 / 场景**（taxonomy）筛选数据集，并 **导出 `REPO_ID` 列表**。

代码根路径：`tools/cpt_dataset_selector/`。

---

## 1. 已实现模块（职责一览）

| 模块 | 作用 |
|------|------|
| `manifest.py` | 从一个或多个训练配置 `.py` 读取 `REPO_ID`，合并去重，写出 **`manifest.json`**（含 `repo_ids`、来源配置路径、可选 git 元数据）。默认配置：`src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.all_public_datasets.py`。 |
| `indexer.py` | 对 `ROOT_DIR/<repo_id>/` 扫描：`meta/tasks.jsonl` 解析入库；`meta/info.json` 取 `robot_type`、`total_episodes`、`duration_hours`（由 `total_frames/fps` 推导）。支持增量更新与 **`--full`** 全量重建。 |
| `parse_tasks.py` | 任务行解析、归一化文本（供匹配与 FTS）。 |
| `taxonomy.json` + `taxonomy_loader.py` | 结构化字段别名、各维选项 **id/label/synonyms**（与 RoboCOIN 参考说明字段一致）。 |
| `matcher.py` | `QueryFilter`：按 taxonomy 维做 **结构化字段 + 文本同义词** 匹配；可选 **`min_match_ratio`**；可选 **按数据集族名 `family`**（来自 `repo_id` 第二段）过滤。 |
| `cli.py` | 子命令 **`manifest` / `index` / `count`**。 |
| `server.py` | **FastAPI**：健康检查、数据集族列表、taxonomy、查询、单 repo 明细、**FTS 搜索**、导出 JSON/Python 列表；静态资源挂载 **`/static`**，**`/`** 返回单页 UI。 |
| `static/` | `index.html` + `app.js` + `styles.css`：筛选器、结果表、导出。 |

**说明：** `/api/search`、`/api/repo/{repo_id}` 已在服务端实现；**当前前端页面未调用**（无全文检索框、无单库详情面板）。

---

## 2. 使用流程（命令行）

在仓库根目录执行（需已安装项目依赖，如 `uv run` / 等价环境）。

### 2.1 生成清单 `manifest.json`

```bash
uv run python -m tools.cpt_dataset_selector manifest \
  --out tools/cpt_dataset_selector/data/manifest.json
```

- **`--config PATH1,PATH2,...`**（可选）：多个训练配置 **按顺序合并** `REPO_ID`，**先去重保留首次出现**。不传则仅用默认 `all_public_datasets` 配置。

### 2.2 统计合并后的 repo 数量

```bash
uv run python -m tools.cpt_dataset_selector count
# 同上可用 --config a.py,b.py
```

### 2.3 构建 / 更新索引数据库

```bash
# 增量（默认）
uv run python -m tools.cpt_dataset_selector index \
  --root-dir "${OPENPI_ROOT_DIR:-/mnt}" \
  --manifest tools/cpt_dataset_selector/data/manifest.json \
  --db tools/cpt_dataset_selector/data/index.sqlite3 \
  --workers 8

# 删除旧库全量重建
uv run python -m tools.cpt_dataset_selector index --full [同上其它参数]
```

- **`--root-dir`**：每个 `repo_id` 对应目录 **`root-dir/repo_id`**，且需存在可索引的 **`meta/tasks.jsonl`**（缺失则计入统计中的 missing/errors）。
- 环境变量习惯：文档/示例中常用 **`OPENPI_ROOT_DIR`**，与 `cli` 默认值一致（默认 `/mnt`）。

### 2.4 启动 Web 服务

```bash
uv run python -m tools.cpt_dataset_selector.server
```

- 默认 **`CPT_HOST=0.0.0.0`**，**`CPT_PORT=9897`**。
- **`CPT_INDEX_DB`**：SQLite 路径（默认 `tools/cpt_dataset_selector/data/index.sqlite3`）。
- **`CPT_TAXONOMY`**：自定义 taxonomy JSON；未设置或文件不存在则用内置默认。

浏览器访问：**`http://<host>:9897/`**。

---

## 3. Web 界面操作（当前静态页）

| 区域 | 操作 |
|------|------|
| **Datasets** | 下拉多选 **数据集族**（索引中的 `family`，与 `repo_id` 路径段一致）。**全选或不选 → 查询时不限数据集**；仅选部分则只在这些族内筛选。 |
| **Min task match** | 可选 **0–1**：在已选 taxonomy 条件下，隐藏 **`matched_tasks / task_count`** 低于该比例的 repo。 |
| **左侧 taxonomy** | 三类多选：**Atomic actions**、**Object categories**、**Scenes**（来自 `/api/taxonomy`）。 |
| **Apply filters** | `POST /api/query`，刷新结果表：repo_id、dataset、robot_type、duration(h)、tasks、matched、match%。 |
| **Clear filters** | 清空三类勾选、数据集恢复「全部」、清空 min match、清空结果与导出预览。 |
| **结果表** | 行首勾选；表头 **Select all** 全选当前页结果。 |
| **统计行** | Filtered repo 数、汇总 **Episodes** 与 **Duration (h)**、**Selected** 勾选数量。 |
| **Export** | **Export JSON** / **Export Python list**：请求 `/api/export`；预览区更新并 **下载** `cpt_repo_ids.json` 或 `cpt_repo_ids.py`。若未勾选任何行，则对 **当前结果列表全部 repo** 导出。 |

---

## 4. 数据流（端到端）

```text
训练配置 REPO_ID  ──► manifest 子命令 ──► data/manifest.json
                                              │
ROOT_DIR/<repo_id>/meta/tasks.jsonl  ◄─────────┘
        + meta/info.json
              │
              ▼
        index 子命令 ──► index.sqlite3 (repos + tasks, FTS)
              │
              ▼
        server (FastAPI) + static 页面
              │
              ▼
        taxonomy 筛选 + 可选族 / min_match_ratio
              │
              ▼
        导出 REPO_ID 列表（JSON / Python）
```

---

## 5. 单元测试

- `tests/unit/cpt_dataset_selector/test_matcher.py`
- `tests/unit/cpt_dataset_selector/test_parse_tasks.py`

---

## 6. 相关文件路径（速查）

| 用途 | 路径 |
|------|------|
| CLI 入口 | `python -m tools.cpt_dataset_selector` |
| 服务入口 | `python -m tools.cpt_dataset_selector.server` |
| 默认清单输出 | `tools/cpt_dataset_selector/data/manifest.json` |
| 默认索引库 | `tools/cpt_dataset_selector/data/index.sqlite3` |
| Taxonomy 数据 | `tools/cpt_dataset_selector/taxonomy.json` |
