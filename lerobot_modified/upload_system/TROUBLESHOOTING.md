# 故障排查与异常处理

## 三层防护机制

1. **信号处理**：Ctrl+C 时自动保存并注册已完成的 episodes
2. **恢复工具**：扫描未注册的「孤儿 batch」，交互式恢复
3. **守护进程检查**：启动时检查一致性，发现孤儿 batch 时提示运行恢复工具

## 孤儿 batch 恢复

**孤儿 batch**：数据在磁盘但未在 `batch_registry.json` 中注册。

### 常见场景

| 场景       | 是否需恢复 | 说明                     |
|------------|------------|--------------------------|
| Ctrl+C 中断 | 否         | 已自动注册，正常上传     |
| 程序崩溃   | 看日志     | 若提示运行恢复工具则需恢复 |
| 断电/死机  | 是         | 必须运行恢复工具         |

### 恢复命令

```bash
# 仅扫描（dry-run）
python upload_system/recover_orphan_batches.py --dry-run

# 交互式恢复（逐个决定）
python upload_system/recover_orphan_batches.py

# 自动恢复全部（慎用）
python upload_system/recover_orphan_batches.py --auto-recover

# 指定数据根目录
python upload_system/recover_orphan_batches.py --data-root /path/to/data
```

### 恢复后验证

```bash
curl http://localhost:5000/api/batches | jq '.[] | select(.status=="pending")'
# 或访问 Web 界面 http://localhost:5000
```

## 手动上传（扫描并上传）

用于上传已存在但未注册的 batch（历史数据、复制来的数据等）。

1. 打开 Web 界面 → 点击「扫描并上传」
2. 点击「开始扫描」→ 勾选要上传的 batch
3. 点击「上传选中的 Batch」→ 约 1 秒内开始上传

### 常见问题

**扫描不到数据**
- 检查 `upload_system/upload_config.yaml` 中 `data.root` 是否正确
- 确认目录含 `meta/info.json` 且结构为 `{data_root}/{tag}/{task_name}/{repo_id}`

**注册后不上传**
- 检查：`sudo systemctl status lerobot-upload-daemon`
- 查看日志：`tail -f upload_system_logs/upload_daemon.log`
- 当前最多 2 个并发，其余会排队

## 时长统计

- 时长基于 `frame 数 / fps` 计算
- 今日有效数据时长：只统计已上传完成（`status=completed`）的 batch

**历史数据时长有误时：**
```bash
python3 upload_system/recalculate_durations_simple.py
sudo systemctl restart lerobot-web-dashboard
```

## 数据损坏 Repair

```bash
python upload_system/auto_fix_dataset.py /path/to/batch --apply
```

## 守护进程一致性检查

启动时自动检查，发现孤儿 batch 时会 log 警告：

```
WARNING - Found orphan batch: ...
WARNING - Run recovery tool: python upload_system/recover_orphan_batches.py
```

## 最佳实践

- 录制时尽量用 Ctrl+C 优雅退出，避免 kill -9
- 定期：`python upload_system/recover_orphan_batches.py --dry-run`
- 使用 screen/tmux 运行录制，减少意外断开
