# LeRobot 数据上传系统

## 简介

为 LeRobot 数据采集设计的自动化上传系统，提供：

- 自动批次追踪
- 后台自动上传（支持立即响应）
- Web 可视化监控
- 手动控制（触发、暂停、重试）
- 扫描并上传未注册数据
- 并发控制（默认 2 个）、失败重试

## 快速开始

### 部署

```bash
cd /path/to/lerobot_modified
conda activate lerobot  # 或 source .venv/bin/activate
bash upload_system/deploy_upload_system.sh
```

### 访问 Web 界面

- 本地：http://localhost:5000
- 远程：通过 VSCode SSH 端口转发 5000

### 录制数据

按原方式录制，录制完成后自动记录并上传：

```bash
python -m lerobot.record_unified \
    --mode=record \
    --robot.type=arxx5_bimanual \
    --dataset.root=${data_root}/${today}/${task_name}/${repo_id} \
    --dataset.repo_id=${task_name}/${repo_id} \
    ...
```

## 目录结构

```
upload_system/
├── upload_daemon.py      # 上传守护进程
├── web_dashboard.py      # Web 控制面板
├── upload_data.sh        # 上传脚本
├── auto_fix_dataset.py   # 数据修复工具
├── upload_config.yaml    # 配置文件
├── deploy_upload_system.sh
├── uninstall_upload_system.sh
├── recover_orphan_batches.py   # 孤儿 batch 恢复
├── recalculate_durations_simple.py  # 重新计算时长
└── systemd/              # systemd 服务
```

## 常用命令

```bash
# 服务状态
sudo systemctl status lerobot-upload-daemon
sudo systemctl status lerobot-web-dashboard

# 重启
sudo systemctl restart lerobot-upload-daemon
sudo systemctl restart lerobot-web-dashboard

# 日志
tail -f upload_system_logs/upload_daemon.log
tail -f upload_system_logs/web_dashboard.log
tail -f upload_system_logs/uploads/<batch_id>.log

# 卸载
bash upload_system/uninstall_upload_system.sh
```

## 配置说明

编辑 `upload_system/upload_config.yaml`：

```yaml
upload:
  remote_ip: "服务器IP"
  remote_port: 1205
  remote_user: "root"
  remote_target_dir: "/mnt/oss/..."

daemon:
  check_interval_seconds: 60   # 轮询间隔
  max_concurrent_uploads: 2    # 最大并发

data:
  root: "~/lerobot_data_collection"   # matches scripts/run_session.sh default DATA_ROOT
```

## Web 界面功能

- **总览**：今日采集、已上传、待上传、有效数据时长、磁盘空间
- **Batch 列表**：按状态筛选，手动上传/暂停/重试
- **扫描并上传**：扫描未注册数据，批量选择后注册并上传（约 1 秒内开始上传）

## 时长统计

- 基于 `frame 数 / fps` 计算有效时长
- 今日有效数据时长：今日已上传 batch 的 `total_duration_min` 之和（小时）
- 历史数据时长有误时：`python3 upload_system/recalculate_durations_simple.py`

## 日志位置

| 类型       | 路径                          |
|------------|-------------------------------|
| 守护进程   | `upload_system_logs/upload_daemon.log`      |
| Web 服务   | `upload_system_logs/web_dashboard.log`     |
| 单次上传   | `upload_system_logs/uploads/<batch_id>.log`|

## 参见

- [故障排查与异常处理](TROUBLESHOOTING.md)
