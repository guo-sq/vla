#!/bin/bash
# 单机模拟 2 节点分布式训练（仅用于本地测试，模拟 DLC 多节点环境）
# 用法: bash scripts.train.dlc.local_multinode_test.sh <CONFIG_NAME> <TAG> <YOUR_USER_DIR> <YOUR_WANDB_API_KEY>
# 示例: bash scripts.train.dlc.local_multinode_test.sh src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.rdt_local.py 0208 jy 77a6f87ffb15b774adba94a59a7d0687344ff8da

# 注意：不使用 set -euo pipefail，因为后台进程的失败需要手动处理

echo ""
echo "bash script start (local multinode simulation)"

# 检查参数数量
if [ $# -ne 4 ]; then
    echo "错误: 脚本需要4个参数"
    echo "用法: $0 <CONFIG_NAME> <TAG> <YOUR_USER_DIR> <YOUR_WANDB_API_KEY>"
    echo "示例: $0 src/openpi/configs/cfg_opensource/cfg_pi0.5_28_dim.rdt_local.py 0208 jy 77a6f87ffb15b774adba94a59a7d0687344ff8da"
    exit 1
fi

CONFIG_NAME="$1"
TAG="$2"
YOUR_USER_DIR="$3"
YOUR_WANDB_API_KEY="$4"

# 单机模拟多节点时，通过 loopback 通信，不依赖 eth0
export NCCL_SOCKET_IFNAME=lo
export NCCL_DEBUG=INFO
export XLA_PYTHON_CLIENT_PREALLOCATE=true

export OPENPI_HF_LOAD_NUM_PROC=1
export OPENPI_DISABLE_HF_ARROW_CACHE=1

echo "Installing uv..."
pip install uv -i https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

PROJECT_DIR=/mnt/workspace/${YOUR_USER_DIR}/openpi_modified
cd $PROJECT_DIR || {
    echo "错误: 无法进入目录 $PROJECT_DIR"
    exit 1
}

# 检查虚拟环境是否存在
if [ ! -f ".venv/bin/activate" ]; then
    echo "错误: 虚拟环境 .venv 不存在"
    exit 1
fi

source .venv/bin/activate

export WANDB_API_KEY=${YOUR_WANDB_API_KEY}
uv run wandb login

export WANDB_MODE=online
export WANDB_RUN_NAME="${CONFIG_NAME}.exp.${TAG}"

echo "Config Name: $CONFIG_NAME"
echo "TAG: $TAG"
echo "WandB Run Name: $WANDB_RUN_NAME"

echo ""
echo "Computing normalize statistics (single process, no distributed needed)..."
uv run scripts/compute_norm_stats_fast.py --config-name $CONFIG_NAME

echo ""
echo "Launching 2-node simulation on single machine..."
echo "  Rank 0 -> GPU 0, log: /tmp/rank0_${TAG}.log"
echo "  Rank 1 -> GPU 1, log: /tmp/rank1_${TAG}.log"

# Rank 0: GPU 0，作为 coordinator（RANK=0），先启动绑定端口
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=0 \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py --config $CONFIG_NAME > /tmp/rank0_${TAG}.log 2>&1 &
PID0=$!
echo "Rank 0 started (PID=$PID0)"

# 等待 1 秒让 Rank 0 先完成端口绑定
sleep 1

# Rank 1: GPU 1，作为 worker（RANK=1），连接 Rank 0 的 coordinator
MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 WORLD_SIZE=2 RANK=1 \
CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py --config $CONFIG_NAME > /tmp/rank1_${TAG}.log 2>&1 &
PID1=$!
echo "Rank 1 started (PID=$PID1)"

echo ""
echo "Both ranks running. Waiting for completion..."
echo "You can monitor progress with:"
echo "  tail -f /tmp/rank0_${TAG}.log"
echo "  tail -f /tmp/rank1_${TAG}.log"

# 等待两个进程结束，任意一个失败则整体失败
FAILED=0
wait $PID0
EXIT0=$?
if [ $EXIT0 -ne 0 ]; then
    echo "Rank 0 failed (exit code $EXIT0)! See /tmp/rank0_${TAG}.log"
    kill $PID1 2>/dev/null || true
    FAILED=1
fi

wait $PID1
EXIT1=$?
if [ $EXIT1 -ne 0 ] && [ $FAILED -eq 0 ]; then
    echo "Rank 1 failed (exit code $EXIT1)! See /tmp/rank1_${TAG}.log"
    FAILED=1
fi

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "Both ranks finished successfully."
    echo "Check logs: /tmp/rank0_${TAG}.log  /tmp/rank1_${TAG}.log"
else
    echo ""
    echo "Distributed training simulation failed."
    exit 1
fi
