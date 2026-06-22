#!/bin/bash

# ================= 配置区域 =================
# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/upload_config.yaml"

# Helper: 从 upload_config.yaml 中读取值（使用 python 解析 YAML）
_read_yaml() {
    local key="$1"
    local python_bin
    python_bin=$(_find_python)
    "$python_bin" -c "
import yaml, sys
with open('${CONFIG_FILE}') as f:
    cfg = yaml.safe_load(f)
keys = '${key}'.split('.')
v = cfg
for k in keys:
    v = v[k]
print(v)
" 2>/dev/null
}

# Helper: 查找可用的 python（优先 conda lerobot 环境）
_find_python() {
    # 1. 已激活的 conda/venv 环境
    if [ -n "$CONDA_DEFAULT_ENV" ] || [ -n "$VIRTUAL_ENV" ]; then
        which python 2>/dev/null && return
    fi
    # 2. conda lerobot 环境
    if command -v conda &> /dev/null; then
        local lerobot_python
        lerobot_python=$(conda run -n lerobot which python 2>/dev/null)
        if [ -n "$lerobot_python" ]; then
            echo "$lerobot_python"
            return
        fi
    fi
    # 3. 系统 python3
    which python3 2>/dev/null && return
    which python 2>/dev/null
}

PYTHON_BIN=$(_find_python)
if [ -z "$PYTHON_BIN" ]; then
    echo "❌ 错误: 找不到可用的 Python 解释器"
    exit 1
fi

# 从环境变量读取，如果未设置则从 upload_config.yaml 读取
REMOTE_IP="${REMOTE_IP:-$(_read_yaml upload.remote_ip)}"
REMOTE_PORT="${REMOTE_PORT:-$(_read_yaml upload.remote_port)}"
REMOTE_USER="${REMOTE_USER:-$(_read_yaml upload.remote_user)}"
DEFAULT_REMOTE_TARGET_DIR="${DEFAULT_REMOTE_TARGET_DIR:-$(_read_yaml upload.remote_target_dir)}"

# 校验必填字段
if [ -z "$REMOTE_IP" ] || [ -z "$REMOTE_PORT" ] || [ -z "$REMOTE_USER" ] || [ -z "$DEFAULT_REMOTE_TARGET_DIR" ]; then
    echo "❌ 错误: 缺少必要的上传配置"
    echo "   请确保环境变量已设置或 upload_config.yaml 配置正确"
    echo "   需要: REMOTE_IP, REMOTE_PORT, REMOTE_USER, DEFAULT_REMOTE_TARGET_DIR"
    exit 1
fi
# ===========================================

# 1. 检查是否有输入参数
if [ $# -lt 1 ]; then
    echo "❌ 使用方法: $0 <本地文件或文件夹路径> [远程目标目录]"
    echo "📝 示例: $0 /home/anyverse/data/batch1"
    echo "📝 示例: $0 /home/anyverse/data/batch1 /remote/path/20260210/task/"
    exit 1
fi

# 第一个参数是本地路径
LOCAL_PATH="$1"

# 第二个参数是可选的远程目标目录
if [ $# -ge 2 ]; then
    REMOTE_TARGET_DIR="$2"
else
    REMOTE_TARGET_DIR="$DEFAULT_REMOTE_TARGET_DIR"
fi

# 2. 建立 SSH 持久化连接 (解决网络抖动核心)
SSH_SOCKET="/tmp/rsync_upload_socket_$$"
echo "🔌 正在建立持久化 SSH 连接..."

# 后台启动主连接
ssh -p $REMOTE_PORT -M -S $SSH_SOCKET -f -N -o ControlPersist=60 $REMOTE_USER@$REMOTE_IP
if [ $? -ne 0 ]; then
    echo "❌ 无法连接到服务器，请检查网络。"
    exit 1
fi

# 清理函数
cleanup() {
    ssh -p $REMOTE_PORT -S $SSH_SOCKET -O exit $REMOTE_USER@$REMOTE_IP 2>/dev/null
    echo "🔌 连接已关闭。"
}
trap cleanup EXIT

# 3. 预先检查远程目录是否存在
# 这一步是为了防止 rsync 因为父目录不存在而报错
echo "🔍 检查远程目标目录..."
ssh -p $REMOTE_PORT -S $SSH_SOCKET $REMOTE_USER@$REMOTE_IP "mkdir -p $REMOTE_TARGET_DIR"

# 检查本地文件是否存在
if [ ! -e "$LOCAL_PATH" ]; then
    echo "❌ 错误: 本地路径 '$LOCAL_PATH' 不存在"
    exit 1
fi

echo -e "\n=============================================="
echo "📦 正在上传: $LOCAL_PATH"
echo "📍 目标路径: $REMOTE_TARGET_DIR"

if [ -d "$LOCAL_PATH" ]; then
    # 调用auto_fix_dataset.py（使用正确的python路径）
    "$PYTHON_BIN" "$SCRIPT_DIR/auto_fix_dataset.py" "$LOCAL_PATH" --apply || {
        echo "❌ 数据修复失败，上传终止"
        exit 1
    }
fi

# 4. 执行 Rsync 上传
# 注意：REMOTE_TARGET_DIR 结尾有 /，这会将本地文件夹放入该目录下
# --timeout=30: 上传通常更慢，超时时间设稍微长一点
# --info=progress2: 显示整体进度而不是每个文件的进度
rsync -avz --info=progress2 --timeout=30 \
    -e "ssh -p $REMOTE_PORT -S $SSH_SOCKET" \
    "$LOCAL_PATH" \
    "$REMOTE_USER@$REMOTE_IP:$REMOTE_TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "✅ 上传成功！"
    exit 0
else
    echo "❌ 上传失败！"
    exit 1
fi
