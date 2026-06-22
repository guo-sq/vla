#!/bin/bash

export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
# export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_PREALLOCATE=true

# 多个数据集 + 共享缓存锁文件 + spawn worker，默认 1024 远远不够
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

# 启用严格的错误检查
set -euo pipefail

echo ""
echo "bash script start"

# 检查参数数量
if [ $# -lt 4 ] || [ $# -gt 5 ]; then
    echo "错误: 脚本需要 4 或 5 个参数"
    echo "用法: $0 <CONFIG_NAME> <TAG> <YOUR_USER_DIR> <YOUR_WANDB_API_KEY> [MODE=il|rl]"
    echo "示例:"
    echo "  $0 src/openpi/configs/cfg_pi0.5_14_dim_example.py 0208 heyuan <wandb_key>        # 默认 IL"
    echo "  $0 src/openpi/configs/cfg_pi0.5_14_dim_example.py 0208 heyuan <wandb_key> rl     # RL"
    exit 1
fi

CONFIG_NAME="$1"
TAG="$2"
YOUR_USER_DIR="$3"
YOUR_WANDB_API_KEY="$4"
MODE="${5:-il}"

case "$MODE" in
    il|rl) ;;
    *)
        echo "错误: MODE 必须是 il 或 rl，当前为: $MODE"
        exit 1
        ;;
esac

export OPENPI_HF_LOAD_NUM_PROC=1
export OPENPI_DISABLE_HF_ARROW_CACHE=1
export HF_ENDPOINT=https://hf-mirror.com

# export WANDB_API_KEY=77a6f87ffb15b774adba94a59a7d0687344ff8da
# uv run wandb login
# export WANDB_MODE=online

echo "Installing uv..."

pip install uv -i https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

# 用脚本所在目录作为 PROJECT_DIR，DLC 任务跑脚本实际所在的 worktree，
# 而不是某个写死的全局路径。YOUR_USER_DIR 参数保留只是为了兼容已提交的 DLC user_command。
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR" || {
    echo "错误: 无法进入目录 $PROJECT_DIR"
    exit 1
}
echo "PROJECT_DIR: $PROJECT_DIR"

# 设置 HuggingFace 缓存目录
# 重要：分布式训练时，每个 worker 必须使用独立的缓存目录，避免文件竞争
# RANK 环境变量由分布式训练框架设置（0, 1, 2, ...）
RANK_ID="${RANK:-0}"
HF_CACHE_ROOT="/tmp/.hf_cache"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/rank_${RANK_ID}"
export HF_HOME="${HF_CACHE_ROOT}/rank_${RANK_ID}"
# 注意：不要设置 HF_DATASETS_OFFLINE，因为需要加载本地 parquet 文件

mkdir -p "${HF_DATASETS_CACHE}"

echo "RANK: ${RANK_ID}, HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

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

# 多节点训练时，仅 rank 0 计算 norm stats，其他 rank 等待文件就绪
WORLD_SIZE="${WORLD_SIZE:-1}"
if [ "${WORLD_SIZE}" -gt 1 ]; then
    # 多节点模式
    if [ "${RANK_ID}" = "0" ]; then
        echo ""
        echo "Computing normalize statistics (rank 0 only)..."
        uv run scripts/compute_norm_stats_fast.py --config-name $CONFIG_NAME
    else
        # 非 rank 0：等待 rank 0 执行完 compute_norm_stats 并写出 norm_stats.json
        # get_config 会打印大量日志，用 tail -1 只取最后一行路径
        NORM_STATS_PATH=$(uv run scripts/compute_norm_stats_fast.py --config-name $CONFIG_NAME --print-output-path 2>/dev/null | tail -1)
        WAITED=0
        echo ""
        echo "Rank ${RANK_ID}: waiting for rank 0 to finish computing norm_stats..."
        echo "  (norm_stats.json path: ${NORM_STATS_PATH})"
        while [ ! -f "${NORM_STATS_PATH}" ]; do
            sleep 10
            WAITED=$((WAITED + 10))
            echo "Rank ${RANK_ID}: waiting... (${WAITED}s elapsed)"
        done
        echo "Rank ${RANK_ID}: norm_stats.json 已就绪，继续训练"
    fi
else
    # 单节点模式
    echo ""
    echo "Computing normalize statistics..."
    uv run scripts/compute_norm_stats_fast.py --config-name $CONFIG_NAME
fi

# RL per_task 策略下,train/val 必须读取同一份预计算的 rl_norm_stats.json
# 以避免 task_to_norm_length 漂移。script 对非 per_task / 无 value_net_cfg
# 的场景是 early-return no-op,所以所有 RL 训练都可以安全无脑调用。失败时让
# 整个 DLC 任务 fail-fast,避免训练启动后才在 strict gate 里报错。
#
# 多节点下必须有 wait barrier:rank 0 才跑 compute_rl_norm_stats,
# 非 rank 0 在跑训练前必须等 rank 0 完成,否则会撞上 base_cfg._load_rl_norm_stats
# 找不到 file → strict gate raise → 全集群 crash。norm_stats.json 已有同样的
# barrier (上面 95-115 行),这里复用同模式:rank 0 在共享 cpfs 写一个 marker
# file,非 rank 0 wait 这个 marker 出现再继续。
# CONFIG_NAME includes a path (e.g. "src/openpi/configs/cfg_xxx.py"), so using it
# verbatim in the marker filename would require mkdir -p of the embedded
# subdirs before `touch`. Strip to the basename for a flat, collision-free key.
RL_NORM_READY_MARKER="${PROJECT_DIR}/.rl_norm_stats.$(basename "${CONFIG_NAME}").ready"
if [ "$MODE" = "rl" ]; then
    if [ "${WORLD_SIZE}" -gt 1 ] && [ "${RANK_ID}" != "0" ]; then
        echo ""
        echo "Rank ${RANK_ID}: waiting for rank 0 to finish computing RL norm stats..."
        echo "  (marker: ${RL_NORM_READY_MARKER})"
        WAITED=0
        while [ ! -f "${RL_NORM_READY_MARKER}" ]; do
            sleep 10
            WAITED=$((WAITED + 10))
            echo "Rank ${RANK_ID}: waiting for rl_norm_stats... (${WAITED}s elapsed)"
            if [ "${WAITED}" -gt 1800 ]; then
                echo "Rank ${RANK_ID}: timeout (>30 min) waiting for rl_norm_stats marker; aborting"
                exit 1
            fi
        done
        echo "Rank ${RANK_ID}: rl_norm_stats marker found, continuing"
    else
        echo ""
        echo "Computing RL norm stats..."
        rm -f "${RL_NORM_READY_MARKER}"
        if uv run scripts/compute_rl_norm_stats.py --config $CONFIG_NAME; then
            touch "${RL_NORM_READY_MARKER}"
            echo "Rank 0: rl_norm_stats marker written: ${RL_NORM_READY_MARKER}"
        else
            echo "Rank 0: compute_rl_norm_stats FAILED — not writing marker; non-rank 0 will time out"
            exit 1
        fi
    fi
fi

echo ""
echo "Start training (mode: $MODE)..."

if [ "$MODE" = "il" ]; then
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py --config $CONFIG_NAME
else
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_rl.py --config $CONFIG_NAME
fi
