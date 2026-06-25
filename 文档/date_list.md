# VLA 服务器数据与资料清单

整理时间：2026-04-27  
用途：记录当前服务器上对 VLA 训练、评估、部署有用的数据、模型、工具和文档入口。

## 存储位置概览

| 路径 | 类型 | 说明 |
| --- | --- | --- |
| `/root/workspaces/wujie_gsq/vla` | 项目代码 | 当前 VLA 工作区，包含 `openpi_modified` 和 `lerobot_modified` |
| `/mnt/oss_data` | 数据集 | OSS 挂载，主要存放 AnyVerse 自采数据和公开机器人数据集 |
| `/mnt/oss_models` | 模型 | OSS 挂载，存放 OpenPI base checkpoint、训练结果、部署模型、评估结果 |
| `/mnt/data` / `/mnt/workspace` | CPFS | 共享工作区和缓存，当前约 10T 已满 |

## 当前项目代码

| 路径 | 用途 |
| --- | --- |
| `/root/workspaces/wujie_gsq/vla/openpi_modified` | VLA 模型训练、norm stats、checkpoint、离线评估、policy server |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified` | 机器人侧采集、回放、推理执行、infer_record、自博弈、上传工具 |
| `/root/workspaces/wujie_gsq/vla/openpi_modified/src/openpi/configs` | 训练配置集合，包含 seatbelt、fold clothes、pour water、pack socks 等任务 |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/lerobot_example_config_files/task_specs` | 机器人任务配置，如 `seatbelt_*`、`fold_cloth_piper_*` |

重点配置文件：

- `/root/workspaces/wujie_gsq/vla/openpi_modified/src/openpi/configs/cfg_pi0.5_seatbelt_14_dim.py`
- `/root/workspaces/wujie_gsq/vla/openpi_modified/src/openpi/configs/cfg_pi0.5_fold_T_14_dim.py`
- `/root/workspaces/wujie_gsq/vla/openpi_modified/src/openpi/configs/cfg_pi05_base_pack_socks_0106_0310_wo_pjl_trtc6.py`
- `/root/workspaces/wujie_gsq/vla/openpi_modified/scripts.train.dlc.sh`
- `/root/workspaces/wujie_gsq/vla/openpi_modified/scripts.train.dsw.sh`
- `/root/workspaces/wujie_gsq/vla/lerobot_modified/src/lerobot/record_unified.py`

## AnyVerse 自采数据

### 已整理 LeRobot 数据

| 路径 | 数量/规模 | 内容 |
| --- | ---: | --- |
| `/mnt/oss_data/anyverse` | 约 154 个目录 | arxx5 抓鸭子、插管、fold towel、bipiper 象棋、grab tube 等早期任务 |
| `/mnt/oss_data/anyverse/bipiper_clothes` | 约 405 个目录 | bipiper 叠衣服、铺平、纠错、接管、policy rollout 数据 |
| `/mnt/oss_data/anyverse/bipiper_clothes49` | 约 62 个目录 | 另一批 fold clothes 数据 |
| `/mnt/oss_data/anyverse_pour_water_record` | 约 244 个目录 | bipiper 倒水任务 |
| `/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual` | 多任务汇总 | arxx5 双臂任务数据，按任务归档 |

`/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual` 下的任务规模：

| 任务目录 | 子目录数 | 说明 |
| --- | ---: | --- |
| `pack_socks` | 1180 | 袜子打包/翻袜/恢复等任务 |
| `seatbelt` | 1000 | 安全带悬挂、自博弈、恢复、避障、碰撞恢复等 |
| `fold_box` | 288 | 折盒子任务 |
| `insert_tube` | 174 | 插管/接管任务 |
| `string_bead` | 120 | 串珠任务 |
| `string_bead_v0` | 86 | 串珠早期版本 |
| `string_bead_test` | 83 | 串珠测试数据 |
| `invert_socks` | 65 | 翻袜子 |
| `fold_towel` | 37 | 叠毛巾 |
| `fold_shirt` | 24 | 叠衣服/衬衣 |
| `build_lego` | 22 | 搭积木 |
| `static_sort` | 5 | 静态分类/整理 |
| `build_pandora` | 2 | Pandora 任务 |
| `fold_box_right_order` | 1 | 折盒子顺序变体 |

### 原始/每日采集数据

| 路径 | 说明 |
| --- | --- |
| `/mnt/oss_data/anyverse_human_data_record_raw/arxx5_bimanual` | 每日采集数据，从 `20260116` 到 `20260427`，包含 `*_model_test`、debug 和 raw batch |

近期值得关注：

- `/mnt/oss_data/anyverse_human_data_record_raw/arxx5_bimanual/20260427`
  - `seatbelt`
  - `pack_socks`
  - `fold_clothes`
  - `string_bead`
  - `seatbelt.single.self_play_record.0205_0422_self_play_recovery.20260427.batch.5`
  - 多个 `seatbelt.single.infer_biao_*` batch
- `/mnt/oss_data/anyverse_human_data_record_raw/arxx5_bimanual/20260423` 到 `/mnt/oss_data/anyverse_human_data_record_raw/arxx5_bimanual/20260427`
  - 有连续的 seatbelt / string_bead / pack_socks 采集与测试数据

### 数据格式抽样结论

抽查的 LeRobot 数据大多是 `v2.1` 格式：

- `fps`: 30
- `action`: 14 维
- `observation.state`: 14 维
- 常见视频键：`observation.images.head`、`observation.images.left_wrist`、`observation.images.right_wrist`
- 常见机器人：
  - `arxx5_bimanual`
  - `bi_piper_follower`

抽样路径：

- `/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt/seatbelt.single.hang.baichenglong.20260205.batch.5`
  - `robot_type`: `arxx5_bimanual`
  - `total_episodes`: 12
  - `total_frames`: 4533
  - `fps`: 30
- `/mnt/oss_data/anyverse/bipiper_clothes/record.clothes.bipiper.v1215.1`
  - `robot_type`: `bi_piper_follower`
  - `total_episodes`: 11
  - `total_frames`: 34753
  - `fps`: 30
- `/mnt/oss_data/anyverse_pour_water_record/record.pourwater.bipiper.0120.1`
  - `robot_type`: `bi_piper_follower`
  - `total_episodes`: 30
  - `total_frames`: 61869
  - `fps`: 30
- `/mnt/oss_data/anyverse/record.arxx5_bimanual.right_arm_grab_toy_duck`
  - `robot_type`: `arxx5_bimanual`
  - `total_episodes`: 30
  - `total_frames`: 8997
  - `fps`: 30

## 公开机器人数据集镜像

这些数据适合用于 CPT、混训、开放场景预训练或动作空间参考。

| 路径 | 内容 |
| --- | --- |
| `/mnt/oss_data/IPEC-COMMUNITY` | 多个 OpenX/LeRobot 格式公开数据，如 Bridge、BC-Z、LIBERO、TACO、KUKA、Roboturk、NYU Franka |
| `/mnt/oss_data/lerobot` | ALOHA 静态/移动任务，如 `aloha_static_candy`、`aloha_static_towel` |
| `/mnt/oss_data/X-Humanoid` | RoboMIND2.0 多机器人版本及 `_lerobot` 格式 |
| `/mnt/oss_data/StarVLA` | RoboTwin Clean / Randomized |
| `/mnt/oss_data/droid_1.0.1_modelscope` | DROID 1.0.1 |
| `/mnt/oss_data/droid_1.0.1_modelscope.v2.1` | DROID 1.0.1 v2.1 版本 |
| `/mnt/oss_data/OpenGalaxea` | Galaxea Open World 数据 |
| `/mnt/oss_data/RoboChallenge` | RoboChallenge 数据 |
| `/mnt/oss_data/physical-intelligence/libero` | LIBERO 相关数据 |
| `/mnt/oss_data/robotics-diffusion-transformer` | RDT 相关数据 |
| `/mnt/oss_data/robocoin/RoboCOIN` | RoboCOIN 数据 |

## 模型与 checkpoint

### OpenPI 基础模型

| 路径 | 说明 |
| --- | --- |
| `/mnt/oss_models/openpi/openpi-assets/checkpoints/pi05_base` | pi0.5 base checkpoint |
| `/mnt/oss_models/openpi/openpi-assets/checkpoints/pi05_base.partial` | 含多机器人 norm stats，如 `arx`、`droid`、`franka`、`trossen`、`ur5e` |
| `/mnt/oss_models/openpi/openpi-assets/checkpoints/pi05_libero` | pi0.5 LIBERO checkpoint |
| `/mnt/oss_models/openpi/openpi-assets/checkpoints/pi05_libero.partial` | LIBERO 相关 partial 资产 |

### 已部署模型

| 路径 | 说明 |
| --- | --- |
| `/mnt/oss_models/models_deploy/2603/all_public_dataset/cfg_pi0.5_28_dim.all_public_datasets` | all public dataset 部署模型 |
| `/mnt/oss_models/models_deploy/2603/pack_socks/cfg_pi05_base_pack_socks_data_pure_recover_arx4_bs256_0318_more_upsample_recover` | pack socks 部署模型 |
| `/mnt/oss_models/models_deploy/2603/seatbelt/pi05_base_seatbelt_data_0205_0312_horizon_50_trtc_single_self_play_recovery` | seatbelt 部署模型 |

### AnyVerse pi0.5 / CPT 模型

| 路径 | 说明 |
| --- | --- |
| `/mnt/oss_models/pretrained_models/pi05_anyverse` | AnyVerse pi0.5 相关预训练、CPT、评估可视化 |
| `/mnt/oss_models/pretrained_models/pi05_anyverse/cfg_pi0.5_28_dim.all_public_datasets_exp_0216/99999` | all public dataset 早期重要 checkpoint |
| `/mnt/oss_models/pretrained_models/pi05_anyverse/cfg_pi0.5_28_dim.all_public_datasets_20260308_exp/50000` | 20260308 版本 checkpoint |
| `/mnt/oss_models/pretrained_models/pi05_anyverse/cfg_pi0.5_28_dim.all_public_datasets_20260308_exp/79999` | 20260308 版本 checkpoint |
| `/mnt/oss_models/pretrained_models/pi05_anyverse/cfg_pi0.5_28_dim.all_public_datasets_anyverse_speed_up_20260324/40000` | 20260324 speed up 版本 checkpoint |
| `/mnt/oss_models/pretrained_models/pi05_anyverse/cfg_pi0.5_28_dim.all_public_datasets_robomind2_20260330/19999` | RoboMIND2 混合版本 |
| `/mnt/oss_models/pretrained_models/pi05_anyverse/cpt` | AnyVerse CPT 相关结果 |

CPT 说明：

- `/mnt/oss_models/pretrained_models/pi05_anyverse/cpt/README.md`
  - `cfg_cfg_v1.0.0_28_dim.anyverse_20260320`：基于 pi05 训练全量 AnyVerse 数据集
  - `cfg_pi0.5_28_dim.anyverse_20260320`：基于 v1.0.0 训练全量 AnyVerse 数据集

模型对比报告：

- `/mnt/oss_models/pretrained_models/pi05_anyverse/cpt/compare_vis/compare_v1.0.0_vs_pi05/comparison_report.md`
  - 比较 `v1.0.0_anyverse_vis` 与 `pi05_anyverse`
  - 任务包括 `anyverse_pick_place`、`anyverse_insert_tube`、`anyverse_seatbelt`、`anyverse_pack_socks`、`anyverse_fold_box`、`anyverse_fold_clothes`、`anyverse_pour_water`
  - 抽样 MSE 中 `pi05_anyverse` 整体略优

### 任务模型

| 路径 | 说明 |
| --- | --- |
| `/mnt/oss_models/models_train/pick_water_clothes/pour_water` | pour water 训练结果 |
| `/mnt/oss_models/pretrained_models/pour_water` | pour water 预训练/微调相关 |
| `/mnt/oss_models/pretrained_models/xiangqi-cpt` | 象棋任务 CPT 相关 |

## 工具和资料

### 数据筛选与索引

| 路径 | 说明 |
| --- | --- |
| `/root/workspaces/wujie_gsq/vla/openpi_modified/tools/cpt_dataset_selector` | CPT 数据集筛选器 |
| `/root/workspaces/wujie_gsq/vla/openpi_modified/tools/cpt_dataset_selector/cpt_dataset_selector.md` | 使用说明 |
| `/root/workspaces/wujie_gsq/vla/openpi_modified/tools/cpt_dataset_selector/taxonomy.json` | 动作/物体/场景 taxonomy |

功能：

- 从训练配置读取 `REPO_ID`
- 扫描 `meta/tasks.jsonl` 和 `meta/info.json`
- 构建 SQLite 索引
- 通过 FastAPI + Web UI 按原子动作、物体类别、场景筛选数据集
- 导出 `REPO_ID` 列表

### LeRobot 数据上传系统

| 路径 | 说明 |
| --- | --- |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/upload_system` | 数据上传系统 |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/upload_system/README.md` | 使用说明 |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/upload_system/upload_config.yaml` | 上传配置 |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/upload_system/web_dashboard.py` | Web 控制面板 |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/upload_system/upload_daemon.py` | 上传守护进程 |

功能：

- 自动批次追踪
- 后台自动上传
- Web 可视化监控
- 手动触发、暂停、重试
- 扫描并上传未注册数据
- 修复 orphan batch、重算 duration

### 数据检查与修复

| 路径 | 说明 |
| --- | --- |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/wujie/lerobot_data_check.py` | LeRobot 数据完整性检查 |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/upload_system/auto_fix_dataset.py` | 数据自动修复 |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/upload_system/recalculate_durations.py` | 重新计算 episode duration |
| `/root/workspaces/wujie_gsq/vla/lerobot_modified/upload_system/recover_orphan_batches.py` | 恢复 orphan batch |

### OpenPI 文档

| 路径 | 说明 |
| --- | --- |
| `/root/workspaces/wujie_gsq/vla/openpi_modified/docs/remote_inference.md` | policy server / robot client 远程推理 |
| `/root/workspaces/wujie_gsq/vla/openpi_modified/docs/norm_stats.md` | normalization stats 说明 |
| `/root/workspaces/wujie_gsq/vla/openpi_modified/docs/docker.md` | Docker 说明 |
| `/root/workspaces/wujie_gsq/vla/openpi_modified/docs/ci.md` | CI / 测试说明 |

## 推荐优先级

### 如果目标是 seatbelt

优先查看：

1. `/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/seatbelt`
2. `/mnt/oss_data/anyverse_human_data_record_raw/arxx5_bimanual/20260427/seatbelt`
3. `/mnt/oss_models/models_deploy/2603/seatbelt`
4. `/root/workspaces/wujie_gsq/vla/openpi_modified/src/openpi/configs/cfg_pi0.5_seatbelt_14_dim.py`
5. `/root/workspaces/wujie_gsq/vla/lerobot_modified/lerobot_example_config_files/task_specs/seatbelt_self_play.json`

建议下一步：

- 为 seatbelt 生成 dataset manifest
- 统计每个 batch 的 `total_episodes`、`total_frames`、prompt、是否 self-play / infer / recovery / collision recovery
- 区分成功数据、接管数据、恢复数据、失败/负样本
- 对照当前训练配置确认哪些 batch 已进训练、哪些可增量加入

### 如果目标是 fold clothes

优先查看：

1. `/mnt/oss_data/anyverse/bipiper_clothes`
2. `/mnt/oss_data/anyverse/bipiper_clothes49`
3. `/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_shirt`
4. `/root/workspaces/wujie_gsq/vla/openpi_modified/src/openpi/configs/cfg_pi0.5_fold_T_14_dim.py`
5. `/root/workspaces/wujie_gsq/vla/lerobot_modified/lerobot_example_config_files/task_specs/fold_cloth_piper_self_play.json`

### 如果目标是 pack socks

优先查看：

1. `/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/pack_socks`
2. `/mnt/oss_data/anyverse_human_data_record_raw/arxx5_bimanual/20260427/pack_socks`
3. `/mnt/oss_models/models_deploy/2603/pack_socks`
4. `/root/workspaces/wujie_gsq/vla/openpi_modified/src/openpi/configs/cfg_pi05_base_pack_socks_0106_0310_wo_pjl_trtc6.py`

### 如果目标是 pour water

优先查看：

1. `/mnt/oss_data/anyverse_pour_water_record`
2. `/mnt/oss_models/models_train/pick_water_clothes/pour_water`
3. `/mnt/oss_models/pretrained_models/pour_water`

## 注意事项

- `/mnt/data` / `/mnt/workspace` 当前已满，训练前应避免把大数据复制到 CPFS。
- 大多数训练配置中的 `ROOT_DIR` 约定为 `/mnt`，repo id 通常拼成 `/mnt/<repo_id>`。
- `arxx5_bimanual` 与 `bi_piper_follower` 都是 14 维动作，但 joint 命名不同，混训时必须确认 action mapping / delta mask 一致。
- 当前常见 delta action mask 是 `[6, -1, 6, -1]`，对应双臂关节 + gripper 的处理方式。
- 训练、推理、机器人执行三处的 transform 和 action 维度必须保持一致。
