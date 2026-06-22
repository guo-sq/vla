#!/bin/bash

# Find more details on: https://qcn4se3qlil6.feishu.cn/wiki/JKyqw9ieWi1SHZk18g7cjVPBn8o

# ================= 配置区域 =================
REMOTE_IP="39.96.194.254"
REMOTE_PORT="1314"
CUR_IP="$(hostname -I | awk '{print $1}')"
WAITING_LIST="/mnt/workspace/zengqi/auto_download_ckpt_infos/waiting_download.json"
HISTORY_LIST="/mnt/workspace/zengqi/auto_download_ckpt_infos/history.json"
# 本地存放 checkpoints 的根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_WORKSPACE_ROOT="${SCRIPT_DIR}/checkpoints"
echo "🏠 本地存放 checkpoints 根目录: $LOCAL_WORKSPACE_ROOT"

# ===========================================
# 1. 设置 SSH 连接复用
SSH_SOCKET="/tmp/rsync_socket_$$"
# echo "🔌 正在建立持久化 SSH 连接 (加速后续传输)..."
ssh -p $REMOTE_PORT -M -S $SSH_SOCKET -f -N -o ControlPersist=60 root@$REMOTE_IP
if [ $? -ne 0 ]; then
    echo "❌ 无法连接到服务器，请检查网络或密码。"
    exit 1
fi

# 定义清理函数
cleanup() {
    ssh -p $REMOTE_PORT -S $SSH_SOCKET -O exit root@$REMOTE_IP 2>/dev/null
    echo "🔌 SSH 连接已关闭。"
}
trap cleanup EXIT

import json
import os
import sys

waiting_json = os.environ.get("WAITING_JSON", "")
cur_ip = os.environ.get("CUR_IP", "")
try:
    data = json.loads(waiting_json)
except json.JSONDecodeError:
    print("", end="")
    sys.exit(0)

for item in data:
    if str(item.get("ip", "")).strip() == cur_ip:
        path = str(item.get("path", "")).strip()
        if path:
            print(path)
PY
}

# ================= 定时轮询 WAITING_LIST =================
while true; do
    echo -e "\n=============================================="
    echo "📥 $(TZ=Asia/Shanghai date '+%Y-%m-%d %H:%M:%S') 读取 WAITING_LIST: $WAITING_LIST"
    WAITING_JSON=$(ssh -p $REMOTE_PORT -S $SSH_SOCKET root@$REMOTE_IP "cat $WAITING_LIST")
    if [ $? -ne 0 ] || [ -z "$WAITING_JSON" ]; then
        echo "❌ 读取 WAITING_LIST 失败或为空，10 分钟后重试。"
        sleep 600
        continue
    fi
    # echo "$WAITING_JSON"

    MATCHED_PATHS=$(get_matched_paths "$WAITING_JSON")

    if [ -z "$MATCHED_PATHS" ]; then
        echo "❌ WAITING_LIST 中未找到与本机 IP 匹配的条目: $CUR_IP, 10 分钟后自动重试"
        sleep 600
        continue
    fi

    echo "✅ 匹配到本机 IP ($CUR_IP) 的 ckpt:"
    echo -e "$MATCHED_PATHS \n"

    mapfile -t MATCHED_PATHS_ARRAY <<< "$MATCHED_PATHS"

    # ================= 处理单个路径 =================
    # 记录总开始时间

    for REMOTE_ABS_PATH in "${MATCHED_PATHS_ARRAY[@]}"; do
        [ -z "$REMOTE_ABS_PATH" ] && continue
        SINGLE_START_TIME=$(date +%s)

        # 1. 路径解析
        RELATIVE_PATH="${REMOTE_ABS_PATH#*/checkpoints/}"
        RELATIVE_PATH="${RELATIVE_PATH%/}"
        REMOTE_ABS_PATH="${REMOTE_ABS_PATH%/}"

        # 2. 检查 ckpt 是否存在（本轮若不存在则跳过）
        echo "📂 远程待下载路径: $REMOTE_ABS_PATH"
        ssh -p $REMOTE_PORT -S $SSH_SOCKET root@$REMOTE_IP "test -d '$REMOTE_ABS_PATH'"
        if [ $? -ne 0 ]; then
            echo "⏳ 未找到 ckpt，本轮跳过。"
            continue
        fi
        echo "✅ 检测到 ckpt，开始传输。"

        # 3. 构建本地目标路径
        LOCAL_TARGET_PATH="${LOCAL_WORKSPACE_ROOT}/${RELATIVE_PATH}"
        echo "📂 本地下载路径: $LOCAL_TARGET_PATH"
        if [ -d "$LOCAL_TARGET_PATH" ]; then
            rm -r "$LOCAL_TARGET_PATH"
        fi
        mkdir -p "$LOCAL_TARGET_PATH"

        # 4. 执行 Rsync
        echo "🚀 开始传输 params 和 assets..."
        rsync -avzP \
            -e "ssh -p $REMOTE_PORT -S $SSH_SOCKET" \
            --timeout=20 \
            "root@$REMOTE_IP:${REMOTE_ABS_PATH}/params" \
            "root@$REMOTE_IP:${REMOTE_ABS_PATH}/assets" \
            "$LOCAL_TARGET_PATH"


        # 5.立即捕获 rsync 的退出状态码
        RSYNC_RET=$?
        if [ $RSYNC_RET -eq 0 ]; then
            echo "✅ 传输成功！"
            move_to_history "$REMOTE_ABS_PATH"
        else
            echo "❌ 传输失败！"
        fi

        # 6. 记录下载时间并计算耗时
        SINGLE_END_TIME=$(date +%s)
        SINGLE_DURATION=$((SINGLE_END_TIME - SINGLE_START_TIME))
        SINGLE_MINUTES=$(awk "BEGIN {printf \"%.2f\", $SINGLE_DURATION/60}")
        echo "⏱️ 本次耗时: ${SINGLE_MINUTES} 分钟"
    done

    echo "⏳ 10 分钟后开始下一轮 WAITING_LIST 轮询..."
    sleep 600
done
