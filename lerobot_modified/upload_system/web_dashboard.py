#!/usr/bin/env python3
"""
Web可视化控制面板

提供Web界面用于监控数据采集和上传状态。
支持查看统计信息、batch列表、手动控制上传任务。
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil
import yaml
from flask import Flask, jsonify, render_template_string, request

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.common.data_tracker import BatchInfo, BatchTracker

Path("upload_system_logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("upload_system_logs/web_dashboard.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Flask应用
app = Flask(__name__)

# 全局配置
config = None
tracker = None


# ============================================================
# HTML模板
# ============================================================

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anyverse 数据上传监控</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .stat-card {
            border-left: 4px solid #007bff;
            margin-bottom: 1rem;
        }
        .stat-card.warning {
            border-left-color: #ffc107;
        }
        .stat-card.danger {
            border-left-color: #dc3545;
        }
        .progress-container {
            margin: 10px 0;
        }
        .badge-status {
            font-size: 0.9rem;
            padding: 0.4rem 0.6rem;
        }
        .refresh-time {
            color: #6c757d;
            font-size: 0.85rem;
        }
        .disk-warning {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .disk-critical {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .sticky-top {
            position: sticky;
            top: 0;
            z-index: 10;
        }
        .gap-2 {
            gap: 0.5rem;
        }
        .toast-msg {
            position: fixed;
            top: 60px;
            right: 20px;
            z-index: 9999;
            padding: 0.6rem 1.2rem;
            border-radius: 6px;
            color: #fff;
            font-size: 0.9rem;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .toast-msg.show { opacity: 1; }
        .toast-msg.success { background: #198754; }
        .toast-msg.error { background: #dc3545; }
    </style>
</head>
<body>
    <nav class="navbar navbar-dark bg-dark">
        <div class="container-fluid">
            <span class="navbar-brand mb-0 h1">Anyverse 数据上传监控</span>
            <span class="text-white refresh-time">最后更新: <span id="lastUpdate">--</span></span>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- 磁盘空间警告 -->
        <div id="diskAlert" style="display: none;"></div>

        <!-- 统计卡片 -->
        <div class="row mb-4">
            <div class="col">
                <div class="card stat-card">
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">今日采集数据条数</h6>
                        <h3 class="card-title" id="todayBatches">--</h3>
                        <p class="card-text text-muted"><span id="todayBatchesText">--</span> batches / <span id="todayEpisodes">--</span> episodes</p>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card stat-card">
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">今日采集数据大小</h6>
                        <h3 class="card-title" id="todaySize">--</h3>
                        <p class="card-text text-muted">GB</p>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card stat-card">
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">今日有效数据时长</h6>
                        <h3 class="card-title" id="todayDuration">--</h3>
                        <p class="card-text text-muted">小时</p>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card stat-card">
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">已上传数据</h6>
                        <h3 class="card-title" id="uploadedSize">--</h3>
                        <p class="card-text text-muted"><span id="completedBatches">--</span> batches</p>
                    </div>
                </div>
            </div>
            <div class="col">
                <div class="card stat-card warning">
                    <div class="card-body">
                        <h6 class="card-subtitle mb-2 text-muted">待上传数据</h6>
                        <h3 class="card-title" id="pendingBatches">--</h3>
                        <p class="card-text text-muted"><span id="uploadingBatches">--</span> 上传中</p>
                    </div>
                </div>
            </div>
        </div>

        <!-- 正在上传的任务 -->
        <div class="card mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">正在上传的任务</h5>
            </div>
            <div class="card-body" id="uploadingTasks">
                <p class="text-muted">暂无上传任务</p>
            </div>
        </div>

        <!-- Batch列表 -->
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0">Batch 列表</h5>
                <div class="d-flex gap-2">
                    <button class="btn btn-success btn-sm" id="scanUploadBtn">
                        <i class="bi bi-cloud-upload"></i> 扫描并上传
                    </button>
                    <input type="search" class="form-control form-control-sm" id="batchSearch"
                           placeholder="搜索 batch / tag" style="width: 200px;">
                    <select class="form-select form-select-sm" id="statusFilter" style="width: 150px;">
                        <option value="">所有状态</option>
                        <option value="pending">待上传</option>
                        <option value="uploading">上传中</option>
                        <option value="completed">已完成</option>
                        <option value="failed">失败</option>
                    </select>
                </div>
            </div>
            <div class="card-body">
                <div class="d-flex justify-content-between align-items-center mb-2">
                    <small class="text-muted" id="batchListCount">--</small>
                    <select class="form-select form-select-sm" id="batchPageSize" style="width: 110px;">
                        <option value="50">每页 50</option>
                        <option value="200">每页 200</option>
                        <option value="500">每页 500</option>
                        <option value="0" selected>全部</option>
                    </select>
                </div>
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Batch ID</th>
                                <th>Tag</th>
                                <th>Episodes</th>
                                <th>采集时长（h）</th>
                                <th>有效时长（h）</th>
                                <th>大小 (GB)</th>
                                <th>开始录制时间</th>
                                <th>结束录制时间</th>
                                <th>开始上传时间</th>
                                <th>结束上传时间</th>
                                <th>状态</th>
                                <th>操作</th>
                            </tr>
                        </thead>
                        <tbody id="batchList">
                            <tr>
                                <td colspan="12" class="text-center">加载中...</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- 扫描并上传模态框 -->
    <div class="modal fade" id="scanUploadModal" tabindex="-1">
        <div class="modal-dialog modal-xl" style="max-width: 95%;">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">扫描并上传未注册的Batch</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <div id="scanStatus" class="mb-3">
                        <button class="btn btn-primary" id="startScanBtn">开始扫描</button>
                        <span id="scanLoadingText" style="display:none;" class="ms-2">
                            <span class="spinner-border spinner-border-sm"></span> 扫描中...
                        </span>
                    </div>
                    <div id="scanResults" style="display:none;">
                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center">
                                <span>找到 <strong id="unregisteredCount">0</strong> 个未注册的batch</span>
                                <div>
                                    <button class="btn btn-sm btn-outline-primary" id="selectAllBtn">全选</button>
                                    <button class="btn btn-sm btn-outline-secondary" id="deselectAllBtn">取消全选</button>
                                </div>
                            </div>
                        </div>
                        <div class="table-responsive" style="max-height: 500px; overflow-y: auto;">
                            <table class="table table-sm table-hover" style="white-space: nowrap;">
                                <thead class="table-light sticky-top">
                                    <tr>
                                        <th width="40"><input type="checkbox" id="selectAllCheckbox"></th>
                                        <th>Tag</th>
                                        <th>任务</th>
                                        <th style="min-width: 400px;">Repo ID</th>
                                        <th>Episodes</th>
                                        <th>大小(GB)</th>
                                        <th>创建时间</th>
                                    </tr>
                                </thead>
                                <tbody id="unregisteredList">
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                    <button type="button" class="btn btn-success" id="uploadSelectedBtn" disabled>上传选中的Batch</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.0/font/bootstrap-icons.css">
    <script>
        let currentStatusFilter = '';
        let unregisteredBatches = [];
        let selectedBatchPaths = new Set();

        // 刷新统计信息
        async function refreshStatistics() {
            try {
                const response = await fetch('/api/statistics');
                const stats = await response.json();
                
                // 今日采集数据条数卡片
                document.getElementById('todayBatches').textContent = stats.today_batches;
                document.getElementById('todayBatchesText').textContent = stats.today_batches;
                document.getElementById('todayEpisodes').textContent = stats.today_episodes;
                
                // 今日采集数据大小卡片
                document.getElementById('todaySize').textContent = stats.today_size_gb.toFixed(2);
                
                // 今日有效数据时长卡片
                document.getElementById('todayDuration').textContent = stats.today_duration_hours.toFixed(2);
                document.getElementById('uploadedSize').textContent = stats.total_uploaded_gb.toFixed(2) + ' GB';
                document.getElementById('completedBatches').textContent = stats.status_counts.completed || 0;
                document.getElementById('pendingBatches').textContent = stats.status_counts.pending || 0;
                document.getElementById('uploadingBatches').textContent = stats.status_counts.uploading || 0;
                
                document.getElementById('lastUpdate').textContent = new Date().toLocaleString('zh-CN');
            } catch (error) {
                console.error('Failed to refresh statistics:', error);
            }
        }

        // 刷新磁盘空间信息
        async function refreshDiskSpace() {
            try {
                const response = await fetch('/api/health');
                const health = await response.json();
                const disk = health.disk_space;
                
                const alertDiv = document.getElementById('diskAlert');
                if (disk.free_gb < 10) {
                    alertDiv.className = 'disk-critical';
                    alertDiv.innerHTML = `<strong>⚠️ 磁盘空间严重不足!</strong> 剩余: ${disk.free_gb.toFixed(2)} GB / ${disk.total_gb.toFixed(2)} GB (${disk.percent_used.toFixed(1)}% 已使用)`;
                    alertDiv.style.display = 'block';
                } else if (disk.free_gb < 100) {
                    alertDiv.className = 'disk-warning';
                    alertDiv.innerHTML = `<strong>注意:</strong> 磁盘空间较低，剩余: ${disk.free_gb.toFixed(2)} GB / ${disk.total_gb.toFixed(2)} GB (${disk.percent_used.toFixed(1)}% 已使用)`;
                    alertDiv.style.display = 'block';
                } else {
                    alertDiv.style.display = 'none';
                }
            } catch (error) {
                console.error('Failed to refresh disk space:', error);
            }
        }

        // 上传速度计算状态
        let _uploadSnapshots = {};

        function formatSpeed(bytesPerSec) {
            if (bytesPerSec < 1024) return bytesPerSec.toFixed(0) + ' B/s';
            if (bytesPerSec < 1024 * 1024) return (bytesPerSec / 1024).toFixed(1) + ' KB/s';
            if (bytesPerSec < 1024 * 1024 * 1024) return (bytesPerSec / (1024 * 1024)).toFixed(2) + ' MB/s';
            return (bytesPerSec / (1024 * 1024 * 1024)).toFixed(2) + ' GB/s';
        }

        // 刷新正在上传的任务
        async function refreshUploadingTasks() {
            try {
                const response = await fetch('/api/batches?status=uploading');
                const batches = await response.json();
                const now = Date.now();

                const container = document.getElementById('uploadingTasks');
                if (batches.length === 0) {
                    container.innerHTML = '<p class="text-muted">暂无上传任务</p>';
                    _uploadSnapshots = {};
                } else {
                    container.innerHTML = batches.map(batch => {
                        const id = batch.batch_id;
                        const totalBytes = batch.total_size_gb * 1024 * 1024 * 1024;
                        const uploadedBytes = totalBytes * batch.upload_progress_percent / 100;
                        let speedText = '--';
                        const prev = _uploadSnapshots[id];
                        if (prev && now > prev.time) {
                            const dt = (now - prev.time) / 1000;
                            const db = uploadedBytes - prev.bytes;
                            if (db > 0 && dt > 0) speedText = formatSpeed(db / dt);
                        }
                        _uploadSnapshots[id] = { time: now, bytes: uploadedBytes };
                        return `
                        <div class="mb-3">
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <strong>${id}</strong>
                                <span>${batch.upload_progress_percent}%</span>
                            </div>
                            <div class="progress">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     role="progressbar" 
                                     style="width: ${batch.upload_progress_percent}%"
                                     aria-valuenow="${batch.upload_progress_percent}" 
                                     aria-valuemin="0" 
                                     aria-valuemax="100">
                                </div>
                            </div>
                            <small class="text-muted">${batch.num_episodes} episodes | ${batch.total_size_gb.toFixed(2)} GB | ${speedText}</small>
                        </div>`;
                    }).join('');
                }
            } catch (error) {
                console.error('Failed to refresh uploading tasks:', error);
            }
        }

        // 刷新batch列表
        async function refreshBatchList() {
            try {
                const url = currentStatusFilter 
                    ? `/api/batches?status=${currentStatusFilter}` 
                    : '/api/batches';
                const response = await fetch(url);
                const batches = await response.json();
                
                const tbody = document.getElementById('batchList');
                const fmt = (s) => s ? new Date(s).toLocaleString('zh-CN') : '--';
                const hours = (min) => min != null ? (min / 60).toFixed(2) : '--';
                const collectHours = (start, end) => {
                    if (!start || !end) return '--';
                    return ((new Date(end) - new Date(start)) / 3600000).toFixed(2);
                };
                const search = (document.getElementById('batchSearch')?.value || '').trim().toLowerCase();
                if (search) {
                    batches = batches.filter(b =>
                        (b.batch_id || '').toLowerCase().includes(search) ||
                        (b.tag || '').toLowerCase().includes(search) ||
                        (b.task_name || '').toLowerCase().includes(search)
                    );
                }
                const pageSize = parseInt(document.getElementById('batchPageSize')?.value ?? '0', 10);
                const visible = pageSize > 0 ? batches.slice(0, pageSize) : batches;
                const countEl = document.getElementById('batchListCount');
                if (countEl) {
                    countEl.textContent = pageSize > 0 && batches.length > pageSize
                        ? `显示 ${visible.length} / ${batches.length}`
                        : `共 ${batches.length}`;
                }
                if (visible.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="12" class="text-center text-muted">暂无数据</td></tr>';
                } else {
                    tbody.innerHTML = visible.map(batch => {
                        const statusBadge = getStatusBadge(batch);
                        const actions = getActionButtons(batch);
                        return `
                            <tr>
                                <td><small>${batch.batch_id}</small></td>
                                <td>${batch.tag}</td>
                                <td>${batch.num_episodes}</td>
                                <td>${collectHours(batch.recorded_at, batch.recorded_end_at)}</td>
                                <td>${hours(batch.total_duration_min)}</td>
                                <td>${batch.total_size_gb.toFixed(2)}</td>
                                <td><small>${fmt(batch.recorded_at)}</small></td>
                                <td><small>${fmt(batch.recorded_end_at)}</small></td>
                                <td><small>${fmt(batch.upload_started_at)}</small></td>
                                <td><small>${fmt(batch.upload_completed_at)}</small></td>
                                <td>${statusBadge}</td>
                                <td>${actions}</td>
                            </tr>
                        `;
                    }).join('');
                }
            } catch (error) {
                console.error('Failed to refresh batch list:', error);
            }
        }

        // 获取状态徽章
        function getStatusBadge(batch) {
            const status = batch.status;
            if (status === 'failed') {
                const batchKey = `${batch.robot_type}/${batch.task_name}/${batch.tag}/${batch.batch_id}`;
                return `<span class="badge bg-danger badge-status" style="cursor:pointer" data-batch-key="${batchKey}" onclick="showFailLog(this.dataset.batchKey)">失败</span>`;
            }
            const badges = {
                'pending': '<span class="badge bg-secondary badge-status">待上传</span>',
                'uploading': '<span class="badge bg-primary badge-status">上传中</span>',
                'completed': '<span class="badge bg-success badge-status">已完成</span>',
                'paused': '<span class="badge bg-warning badge-status">已暂停</span>'
            };
            return badges[status] || status;
        }

        // 获取操作按钮
        function getActionButtons(batch) {
            const batchKey = `${batch.robot_type}/${batch.task_name}/${batch.tag}/${batch.batch_id}`;
            if (batch.status === 'pending') {
                return `<button class="btn btn-sm btn-primary" onclick="startUpload('${batchKey}')">上传</button>`;
            } else if (batch.status === 'uploading') {
                return `<button class="btn btn-sm btn-warning" onclick="pauseUpload('${batchKey}')">暂停</button>`;
            } else if (batch.status === 'failed') {
                return `<button class="btn btn-sm btn-info" onclick="retryUpload('${batchKey}')">重试</button>`;
            } else if (batch.status === 'paused') {
                return `<button class="btn btn-sm btn-success" onclick="retryUpload('${batchKey}')">继续</button>`;
            }
            return '';
        }

        // 非阻塞提示
        function showToast(msg, type) {
            let el = document.getElementById('toastMsg');
            if (!el) {
                el = document.createElement('div');
                el.id = 'toastMsg';
                el.className = 'toast-msg';
                document.body.appendChild(el);
            }
            el.textContent = msg;
            el.className = 'toast-msg ' + type + ' show';
            clearTimeout(el._timer);
            el._timer = setTimeout(() => el.classList.remove('show'), 2500);
        }

        // 手动触发上传
        async function startUpload(batchKey) {
            try {
                const response = await fetch(`/api/upload/start/${encodeURIComponent(batchKey)}`, { method: 'POST' });
                const result = await response.json();
                showToast(result.message, result.success ? 'success' : 'error');
                refreshAll();
            } catch (error) {
                showToast('操作失败: ' + error.message, 'error');
            }
        }

        // 暂停上传
        async function pauseUpload(batchKey) {
            try {
                const response = await fetch(`/api/upload/pause/${encodeURIComponent(batchKey)}`, { method: 'POST' });
                const result = await response.json();
                showToast(result.message, result.success ? 'success' : 'error');
                refreshAll();
            } catch (error) {
                showToast('操作失败: ' + error.message, 'error');
            }
        }

        // 重试上传
        async function retryUpload(batchKey) {
            try {
                const response = await fetch(`/api/upload/retry/${encodeURIComponent(batchKey)}`, { method: 'POST' });
                const result = await response.json();
                showToast(result.message, result.success ? 'success' : 'error');
                refreshAll();
            } catch (error) {
                showToast('操作失败: ' + error.message, 'error');
            }
        }

        // 刷新所有数据
        function refreshAll() {
            refreshStatistics();
            refreshDiskSpace();
            refreshUploadingTasks();
            refreshBatchList();
        }

        // 状态过滤器
        document.getElementById('statusFilter').addEventListener('change', (e) => {
            currentStatusFilter = e.target.value;
            refreshBatchList();
        });

        // 客户端搜索 / 分页大小
        document.getElementById('batchPageSize').addEventListener('change', refreshBatchList);
        let batchSearchTimer = null;
        document.getElementById('batchSearch').addEventListener('input', () => {
            clearTimeout(batchSearchTimer);
            batchSearchTimer = setTimeout(refreshBatchList, 200);
        });

        // 初始加载
        refreshAll();

        // 定时刷新（每5秒）
        setInterval(refreshAll, 5000);

        // ========== 扫描并上传功能 ==========
        
        const scanUploadModal = new bootstrap.Modal(document.getElementById('scanUploadModal'));
        
        // 打开扫描模态框
        document.getElementById('scanUploadBtn').addEventListener('click', () => {
            // 重置状态
            document.getElementById('scanResults').style.display = 'none';
            document.getElementById('startScanBtn').style.display = 'inline-block';
            document.getElementById('scanLoadingText').style.display = 'none';
            document.getElementById('uploadSelectedBtn').disabled = true;
            selectedBatchPaths.clear();
            unregisteredBatches = [];
            
            scanUploadModal.show();
        });
        
        // 开始扫描
        document.getElementById('startScanBtn').addEventListener('click', async () => {
            document.getElementById('startScanBtn').style.display = 'none';
            document.getElementById('scanLoadingText').style.display = 'inline';
            
            try {
                const response = await fetch('/api/scan_unregistered');
                const result = await response.json();
                
                if (result.success) {
                    unregisteredBatches = result.batches;
                    document.getElementById('unregisteredCount').textContent = result.count;
                    
                    // 渲染列表
                    const tbody = document.getElementById('unregisteredList');
                    tbody.innerHTML = '';
                    
                    if (result.count === 0) {
                        tbody.innerHTML = '<tr><td colspan="7" class="text-center text-muted">没有找到未注册的batch</td></tr>';
                    } else {
                        result.batches.forEach((batch, index) => {
                            const row = document.createElement('tr');
                            row.innerHTML = `
                                <td><input type="checkbox" class="batch-checkbox" data-index="${index}" data-path="${batch.path}"></td>
                                <td>${batch.tag}</td>
                                <td>${batch.task_name}</td>
                                <td><small>${batch.repo_id}</small></td>
                                <td>${batch.num_episodes}</td>
                                <td>${batch.size_gb}</td>
                                <td><small>${new Date(batch.created_at).toLocaleString('zh-CN')}</small></td>
                            `;
                            tbody.appendChild(row);
                        });
                        
                        // 添加checkbox事件监听
                        document.querySelectorAll('.batch-checkbox').forEach(checkbox => {
                            checkbox.addEventListener('change', updateSelectedBatches);
                        });
                    }
                    
                    document.getElementById('scanResults').style.display = 'block';
                } else {
                    alert('扫描失败: ' + result.error);
                    document.getElementById('startScanBtn').style.display = 'inline-block';
                }
            } catch (error) {
                alert('扫描失败: ' + error.message);
                document.getElementById('startScanBtn').style.display = 'inline-block';
            } finally {
                document.getElementById('scanLoadingText').style.display = 'none';
            }
        });
        
        // 更新选中的batch
        function updateSelectedBatches() {
            selectedBatchPaths.clear();
            document.querySelectorAll('.batch-checkbox:checked').forEach(checkbox => {
                selectedBatchPaths.add(checkbox.dataset.path);
            });
            
            document.getElementById('uploadSelectedBtn').disabled = selectedBatchPaths.size === 0;
            
            // 同步全选checkbox状态
            const allCheckboxes = document.querySelectorAll('.batch-checkbox');
            const checkedCount = document.querySelectorAll('.batch-checkbox:checked').length;
            const selectAllCheckbox = document.getElementById('selectAllCheckbox');
            selectAllCheckbox.checked = allCheckboxes.length > 0 && checkedCount === allCheckboxes.length;
            selectAllCheckbox.indeterminate = checkedCount > 0 && checkedCount < allCheckboxes.length;
        }
        
        // 全选
        document.getElementById('selectAllBtn').addEventListener('click', () => {
            document.querySelectorAll('.batch-checkbox').forEach(checkbox => {
                checkbox.checked = true;
            });
            updateSelectedBatches();
        });
        
        document.getElementById('selectAllCheckbox').addEventListener('change', (e) => {
            document.querySelectorAll('.batch-checkbox').forEach(checkbox => {
                checkbox.checked = e.target.checked;
            });
            updateSelectedBatches();
        });
        
        // 取消全选
        document.getElementById('deselectAllBtn').addEventListener('click', () => {
            document.querySelectorAll('.batch-checkbox').forEach(checkbox => {
                checkbox.checked = false;
            });
            updateSelectedBatches();
        });
        
        // 上传选中的batch
        document.getElementById('uploadSelectedBtn').addEventListener('click', async () => {
            if (selectedBatchPaths.size === 0) {
                alert('请先选择要上传的batch');
                return;
            }
            
            const btn = document.getElementById('uploadSelectedBtn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> 注册中...';
            
            try {
                const response = await fetch('/api/register_and_upload', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        batch_paths: Array.from(selectedBatchPaths)
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    // 成功后直接关闭modal并刷新，不显示弹窗
                    scanUploadModal.hide();
                    refreshAll();
                } else {
                    alert('操作失败: ' + result.error);
                }
            } catch (error) {
                alert('操作失败: ' + error.message);
            } finally {
                btn.disabled = false;
                btn.innerHTML = '上传选中的Batch';
            }
        });

        // 显示失败日志
        async function showFailLog(batchKey) {
            const logContent = document.getElementById('failLogContent');
            logContent.textContent = '加载中...';
            document.getElementById('failLogTitle').textContent = '上传失败日志 - ' + batchKey.split('/').pop();
            const modal = new bootstrap.Modal(document.getElementById('failLogModal'));
            modal.show();

            try {
                const response = await fetch(`/api/upload/log/${encodeURIComponent(batchKey)}`);
                const result = await response.json();
                if (result.success) {
                    logContent.textContent = result.log;
                } else {
                    logContent.textContent = result.log || result.error || '日志不可用';
                }
            } catch (error) {
                logContent.textContent = '无法加载日志: ' + error.message;
            }
        }
    </script>

    <!-- 失败日志模态框 -->
    <div class="modal fade" id="failLogModal" tabindex="-1">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="failLogTitle">上传失败日志</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal"></button>
                </div>
                <div class="modal-body">
                    <pre id="failLogContent" style="max-height: 500px; overflow-y: auto; background: #f8f9fa; padding: 1rem; border-radius: 4px; white-space: pre-wrap; word-break: break-all; font-size: 0.85rem;"></pre>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">关闭</button>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
"""


# ============================================================
# API路由
# ============================================================


@app.route("/")
def index():
    """首页"""
    return render_template_string(INDEX_TEMPLATE)


@app.route("/api/statistics")
def api_statistics():
    """获取统计信息"""
    try:
        stats = tracker.get_statistics()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/batches")
def api_batches():
    """获取batch列表"""
    try:
        status_filter = request.args.get("status")
        tag_filter = request.args.get("tag")

        batches = tracker.get_all_batches(status_filter=status_filter, tag_filter=tag_filter)

        # 按录制时间倒序排序
        batches.sort(key=lambda x: x.recorded_at, reverse=True)

        # 转换为字典列表
        batch_dicts = [batch.to_dict() for batch in batches]

        return jsonify(batch_dicts)
    except Exception as e:
        logger.error(f"Failed to get batches: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/batch/<path:batch_key>")
def api_batch_detail(batch_key: str):
    """获取单个batch详情"""
    try:
        batch_info = tracker.get_batch_info(batch_key)
        if batch_info is None:
            return jsonify({"error": "Batch not found"}), 404

        return jsonify(batch_info.to_dict())
    except Exception as e:
        logger.error(f"Failed to get batch detail: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload/start/<path:batch_key>", methods=["POST"])
def api_upload_start(batch_key: str):
    """手动触发上传"""
    try:
        batch_info = tracker.get_batch_info(batch_key)
        logger.info(f"api_upload_start: batch_info: {batch_info}, batch_key: {batch_key}")

        if batch_info is None:
            return jsonify({"success": False, "message": "Batch not found"}), 404

        if batch_info.status not in ["pending", "failed"]:
            return jsonify({"success": False, "message": f"Cannot start upload, status is {batch_info.status}"}), 400

        # 更新状态为pending（守护进程会自动拾取）
        tracker.update_upload_status(batch_key, "pending")

        # 触发守护进程立即检查
        try:
            Path("upload_system_logs").mkdir(exist_ok=True)
            (Path("upload_system_logs") / "upload_trigger").touch()
        except Exception:
            pass

        logger.info(f"Manual upload triggered: {batch_key}")
        return jsonify({"success": True, "message": "上传已加入队列"})
    except Exception as e:
        logger.error(f"Failed to start upload: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/upload/pause/<path:batch_key>", methods=["POST"])
def api_upload_pause(batch_key: str):
    """暂停上传（需要守护进程支持）"""
    try:
        batch_info = tracker.get_batch_info(batch_key)
        if batch_info is None:
            return jsonify({"success": False, "message": "Batch not found"}), 404

        if batch_info.status != "uploading":
            return jsonify({"success": False, "message": "Batch is not uploading"}), 400

        # 更新状态为paused
        tracker.update_upload_status(batch_key, "paused", error="Paused by user")

        logger.info(f"Upload paused: {batch_key}")
        return jsonify({"success": True, "message": "上传已暂停"})
    except Exception as e:
        logger.error(f"Failed to pause upload: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/upload/retry/<path:batch_key>", methods=["POST"])
def api_upload_retry(batch_key: str):
    """重试失败的上传"""
    try:
        batch_info = tracker.get_batch_info(batch_key)
        if batch_info is None:
            return jsonify({"success": False, "message": "Batch not found"}), 404

        if batch_info.status not in ["failed", "paused"]:
            return jsonify({"success": False, "message": f"Cannot retry, status is {batch_info.status}"}), 400

        # 重置为pending状态
        tracker.update_upload_status(batch_key, "pending", progress=0, error=None)

        # 触发守护进程立即检查
        try:
            Path("upload_system_logs").mkdir(exist_ok=True)
            (Path("upload_system_logs") / "upload_trigger").touch()
        except Exception as e:
            logger.error(f"Failed to create upload trigger file: {e}", exc_info=True)

        logger.info(f"Upload retry triggered: {batch_key}")
        return jsonify({"success": True, "message": "已重新加入上传队列"})
    except Exception as e:
        logger.error(f"Failed to retry upload: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/upload/cancel/<path:batch_key>", methods=["POST"])
def api_upload_cancel(batch_key: str):
    """取消上传"""
    try:
        batch_info = tracker.get_batch_info(batch_key)
        if batch_info is None:
            return jsonify({"success": False, "message": "Batch not found"}), 404

        # 更新状态为paused
        tracker.update_upload_status(batch_key, "paused", error="Cancelled by user")

        logger.info(f"Upload cancelled: {batch_key}")
        return jsonify({"success": True, "message": "上传已取消"})
    except Exception as e:
        logger.error(f"Failed to cancel upload: {e}", exc_info=True)
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/upload/log/<path:batch_key>")
def api_upload_log(batch_key: str):
    """获取batch的上传日志"""
    try:
        batch_info = tracker.get_batch_info(batch_key)
        if batch_info is None:
            return jsonify({"success": False, "error": "Batch not found"}), 404

        log_file = Path("upload_system_logs/uploads") / f"{batch_info.batch_id}.log"
        if not log_file.exists():
            return jsonify({
                "success": False,
                "error": "日志文件不存在",
                "log": f"错误信息: {batch_info.upload_error or '未知错误'}\n\n(日志文件 {log_file} 不存在)"
            })

        log_content = log_file.read_text(encoding="utf-8", errors="replace")
        if len(log_content) > 100000:
            log_content = "...(日志过长，只显示最后部分)...\n\n" + log_content[-100000:]

        return jsonify({"success": True, "log": log_content})
    except Exception as e:
        logger.error(f"Failed to read upload log: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/scan_unregistered")
def api_scan_unregistered():
    """扫描未注册的batch"""
    try:
        data_root = Path(config["data"]["root"])
        if not data_root.exists():
            return jsonify({"error": "Data root not found"}), 404

        # 获取已注册的batch路径
        registered_batches = tracker.get_all_batches()
        registered_paths = {Path(batch.local_path) for batch in registered_batches}

        # 扫描所有可能的batch目录
        unregistered = []
        
        # 遍历 data_root 下的所有目录，查找包含 meta/info.json 的目录
        for tag_dir in data_root.iterdir():
            if not tag_dir.is_dir() or tag_dir.name.startswith('.'):
                continue
            
            for task_dir in tag_dir.iterdir():
                if not task_dir.is_dir() or task_dir.name.startswith('.'):
                    continue
                
                for repo_dir in task_dir.iterdir():
                    if not repo_dir.is_dir() or repo_dir.name.startswith('.'):
                        continue
                    
                    info_file = repo_dir / "meta" / "info.json"
                    if info_file.exists() and repo_dir not in registered_paths:
                        import time
                        dir_mtime = repo_dir.stat().st_mtime
                        if (time.time() - dir_mtime) < 100:
                            continue
                        total_size = sum(f.stat().st_size for f in repo_dir.rglob('*') if f.is_file())
                        size_gb = total_size / (1024**3)
                        
                        num_episodes = 0
                        data_dir = repo_dir / "data"
                        if data_dir.exists():
                            num_episodes = len(list(data_dir.rglob("episode_*.parquet")))
                        
                        created_at = datetime.fromtimestamp(repo_dir.stat().st_ctime).isoformat()
                        
                        unregistered.append({
                            "path": str(repo_dir.absolute()),
                            "tag": tag_dir.name,
                            "task_name": task_dir.name,
                            "repo_id": repo_dir.name,
                            "num_episodes": num_episodes,
                            "size_gb": round(size_gb, 2),
                            "created_at": created_at,
                        })
        
        # 按创建时间倒序排序
        unregistered.sort(key=lambda x: x["created_at"], reverse=True)
        
        return jsonify({
            "success": True,
            "count": len(unregistered),
            "batches": unregistered
        })
    except Exception as e:
        logger.error(f"Failed to scan unregistered batches: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/register_and_upload", methods=["POST"])
def api_register_and_upload():
    """批量注册并上传batch"""
    try:
        data = request.get_json()
        batch_paths = data.get("batch_paths", [])
        
        if not batch_paths:
            return jsonify({"success": False, "message": "No batches selected"}), 400
        
        results = []
        for batch_path in batch_paths:
            try:
                path = Path(batch_path)
                if not path.exists():
                    results.append({"path": batch_path, "success": False, "error": "Path not found"})
                    continue
                
                # 从路径提取信息
                # 路径格式: {data_root}/{tag}/{task_name}/{repo_id}
                parts = path.parts
                repo_id = parts[-1]
                task_name = parts[-2]
                
                # 读取info.json获取更多信息
                info_file = path / "meta" / "info.json"
                if not info_file.exists():
                    results.append({"path": batch_path, "success": False, "error": "Invalid dataset: meta/info.json not found"})
                    continue
                
                import json
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                robot_type = info.get("robot_type", "unknown")
                robot_id = info.get("robot_id", robot_type)  # 使用robot_type作为robot_id的fallback
                
                # 计算统计信息（在data/chunk-*/目录下查找episode_*.parquet文件）
                num_episodes = 0
                data_dir = path / "data"
                if data_dir.exists():
                    num_episodes = len(list(data_dir.rglob("episode_*.parquet")))
                
                # 估算录制时间
                created_time = path.stat().st_ctime
                modified_time = path.stat().st_mtime
                duration_s = modified_time - created_time
                
                # 注册batch
                batch_info = tracker.record_batch(
                    robot_type=robot_type,
                    robot_id=robot_id,
                    repo_id=repo_id,
                    task_name=task_name,
                    dataset_path=path,  # batch的完整路径
                    num_episodes=num_episodes,
                    start_time=created_time,
                    end_time=modified_time
                )
                
                results.append({
                    "path": batch_path,
                    "success": True,
                    "batch_id": batch_info.batch_id,
                    "message": "Registered and queued for upload"
                })
                
                logger.info(f"Registered batch: {batch_info.batch_id}")
                
            except Exception as e:
                logger.error(f"Failed to register batch {batch_path}: {e}", exc_info=True)
                results.append({"path": batch_path, "success": False, "error": str(e)})
        
        success_count = sum(1 for r in results if r["success"])

        # 触发守护进程立即检查，避免等待轮询周期
        if success_count > 0:
            try:
                Path("upload_system_logs").mkdir(exist_ok=True)
                (Path("upload_system_logs") / "upload_trigger").touch()
            except Exception:
                pass

        return jsonify({
            "success": True,
            "message": f"Registered {success_count}/{len(batch_paths)} batches",
            "results": results
        })
    except Exception as e:
        logger.error(f"Failed to register and upload batches: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/health")
def api_health():
    """健康检查和系统信息"""
    try:
        # 获取磁盘空间信息
        disk_usage = shutil.disk_usage(config["data"]["root"])
        disk_info = {
            "total_gb": disk_usage.total / (1024**3),
            "used_gb": disk_usage.used / (1024**3),
            "free_gb": disk_usage.free / (1024**3),
            "percent_used": (disk_usage.used / disk_usage.total) * 100,
        }

        # 获取内存信息
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent_used": memory.percent,
        }

        return jsonify(
            {
                "status": "healthy",
                "disk_space": disk_info,
                "memory": memory_info,
                "timestamp": datetime.now().isoformat(),
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({"status": "unhealthy", "error": str(e)}), 500


def load_config(config_path: str) -> Dict:
    """加载配置文件。

    ``data.root`` 中的 ``~`` / ``$VAR`` 占位符在加载时一次性展开，避免后续
    ``Path(config["data"]["root"])`` 把字面量 ``~`` 当成普通目录名。
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        data_section = cfg.get("data") if isinstance(cfg, dict) else None
        if isinstance(data_section, dict) and data_section.get("root"):
            data_section["root"] = os.path.expandvars(
                os.path.expanduser(str(data_section["root"]))
            )
        return cfg
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}，请确保 upload_config.yaml 已正确放置。"
        )
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def main():
    """主入口"""
    global config, tracker

    parser = argparse.ArgumentParser(description="LeRobot Web Dashboard")
    parser.add_argument(
        "--config",
        type=str,
        default="upload_system/upload_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host to bind (overrides config)",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind (overrides config)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    args = parser.parse_args()

    # 确保日志目录存在
    Path("upload_system_logs").mkdir(exist_ok=True)

    # 加载配置
    config = load_config(args.config)

    # 初始化追踪器
    tracker = BatchTracker(config["data"]["root"], config["data"]["registry_filename"])

    # 获取Web配置
    host = args.host or config.get("web", {}).get("host", "0.0.0.0")
    port = args.port or config.get("web", {}).get("port", 5000)

    logger.info(f"Starting web dashboard on {host}:{port}")

    # 启动Flask应用
    app.run(host=host, port=port, debug=args.debug)


if __name__ == "__main__":
    main()
