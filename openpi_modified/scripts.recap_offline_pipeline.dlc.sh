#!/bin/bash
#
# RECAP 离线流水线 DLC 统一脚本
#
# 支持两种运行模式:
#   value  — compute_values → train_rl.py (value model 二阶段训练)
#   recap  — compute_values → compute_advantages → train.py (RECAP indicator 训练)
#
# 用法:
#   bash scripts.recap_offline_pipeline.dlc.sh \
#       --stage <value|recap>                              \
#       --value-config <value_model_config>                \
#       --value-checkpoint <path_to_checkpoint>            \
#       --tag <experiment_tag>                             \
#       --user-dir <your_user_dir>                         \
#       --wandb-key <your_wandb_api_key>                   \
#       [--recap-config <recap_indicator_config>]          \   # recap stage 必需
#       [--threshold-mode <per_dataset|global|per_task>]   \   # 默认 per_task
#       [--batch-size-values <N>]                              # compute_values batch size, 默认 1024
#       [--start-from <value|advantage|norm|train>]   \   # 指定起始阶段（recap 模式下），默认从头开始
#
# 示例:
#   # Task 1: Value model 二阶段训练 (234 datasets)
#   bash scripts.recap_offline_pipeline.dlc.sh \
#       --stage value \
#       --value-config src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_all_0322_max3600.py \
#       --value-checkpoint /mnt/workspace/tianwanxin/dev/openpi_fix_episode_mapping/checkpoints/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_1215_0227_max3600/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_1215_0227_max3600.exp.0314/10000 \
#       --tag 0323 --user-dir tianwanxin/dev --wandb-key YOUR_KEY
#
#   # Task 2: RECAP indicator 训练 (1215_0227 datasets)
#   bash scripts.recap_offline_pipeline.dlc.sh \
#       --stage recap \
#       --value-config src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_1215_0227_max3600.py \
#       --value-checkpoint /mnt/workspace/tianwanxin/dev/openpi_fix_episode_mapping/checkpoints/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_fast_mode_max3600/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_fast_mode_max3600.exp.0314/10000 \
#       --recap-config src/openpi/configs/cfg_pi06_recap_indicator_bipiper_clothes_1215_0227.py \
#       --tag 0330 --user-dir tianwanxin/dev --wandb-key YOUR_KEY
#
#   # Task 3: RECAP indicator 训练 (all_0322 datasets, 从 train 开始跳过预处理)
#   bash scripts.recap_offline_pipeline.dlc.sh \
#       --stage recap \
#       --value-config src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_1215_0227_max3600.py \
#       --value-checkpoint /mnt/workspace/tianwanxin/dev/openpi_fix_episode_mapping/checkpoints/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_1215_0227_max3600/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_1215_0227_max3600.exp.0314/10000 \
#       --recap-config src/openpi/configs/cfg_pi06_recap_indicator_bipiper_clothes_all_0322.py \
#       --tag 0330 --user-dir tianwanxin/dev --wandb-key YOUR_KEY \
#       --start-from train

export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO
export XLA_PYTHON_CLIENT_PREALLOCATE=true

ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn) 2>/dev/null || true

set -euo pipefail

# ===========================================================================
# 参数解析
# ===========================================================================
STAGE=""
VALUE_CONFIG=""
RECAP_CONFIG=""
VALUE_CHECKPOINT=""
TAG=""
USER_DIR=""
WANDB_KEY=""
THRESHOLD_MODE="per_task"
BATCH_SIZE_VALUES=1024
SUFFIX=""
VALUE_SUFFIX=""
START_FROM=""  # value|advantage|norm|train (recap 模式下默认从头开始)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)             STAGE="$2";             shift 2 ;;
        --value-config)      VALUE_CONFIG="$2";      shift 2 ;;
        --recap-config)      RECAP_CONFIG="$2";      shift 2 ;;
        --value-checkpoint)  VALUE_CHECKPOINT="$2";  shift 2 ;;
        --tag)               TAG="$2";               shift 2 ;;
        --user-dir)          USER_DIR="$2";          shift 2 ;;
        --wandb-key)         WANDB_KEY="$2";         shift 2 ;;
        --threshold-mode)    THRESHOLD_MODE="$2";    shift 2 ;;
        --batch-size-values) BATCH_SIZE_VALUES="$2"; shift 2 ;;
        --suffix)            SUFFIX="$2";            shift 2 ;;
        --value-suffix)      VALUE_SUFFIX="$2";      shift 2 ;;
        --start-from)        START_FROM="$2";        shift 2 ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# Validate --start-from early so typos (e.g. "values") fail loudly
# instead of silently falling through to "run everything from scratch".
case "$START_FROM" in
    ""|value|advantage|norm|train) ;;
    *)
        echo "错误: --start-from 必须是 value|advantage|norm|train 之一，当前为: $START_FROM"
        exit 1
        ;;
esac

# ===========================================================================
# 参数验证
# ===========================================================================
if [[ -z "$STAGE" || -z "$VALUE_CONFIG" || -z "$VALUE_CHECKPOINT" || -z "$TAG" || -z "$USER_DIR" || -z "$WANDB_KEY" ]]; then
    echo "错误: 缺少必需参数"
    echo "必需: --stage, --value-config, --value-checkpoint, --tag, --user-dir, --wandb-key"
    exit 1
fi

if [[ "$STAGE" != "value" && "$STAGE" != "recap" ]]; then
    echo "错误: --stage 必须是 value 或 recap"
    exit 1
fi

if [[ "$STAGE" == "recap" && -z "$RECAP_CONFIG" ]]; then
    echo "错误: recap 模式必须指定 --recap-config"
    exit 1
fi

echo ""
echo "=========================================="
echo "RECAP Offline Pipeline (DLC)"
echo "=========================================="
echo "Stage:             $STAGE"
echo "Value Config:      $VALUE_CONFIG"
echo "RECAP Config:      ${RECAP_CONFIG:-N/A}"
echo "Value Checkpoint:  $VALUE_CHECKPOINT"
echo "Threshold Mode:    $THRESHOLD_MODE"
echo "Batch Size Values: $BATCH_SIZE_VALUES"
echo "Suffix:            ${SUFFIX:-N/A}"
echo "Value Suffix:      ${VALUE_SUFFIX:-N/A}"
echo "TAG:               $TAG"
echo "=========================================="

# ===========================================================================
# 环境准备
# ===========================================================================
export OPENPI_HF_LOAD_NUM_PROC=1
export OPENPI_DISABLE_HF_ARROW_CACHE=1

echo "Installing uv..."
pip install uv -i https://mirrors.cloud.aliyuncs.com/pypi/simple --trusted-host mirrors.cloud.aliyuncs.com

PROJECT_DIR=/mnt/workspace/${USER_DIR}/openpi_modified
cd "$PROJECT_DIR" || { echo "错误: 无法进入 $PROJECT_DIR"; exit 1; }

RANK_ID="${RANK:-0}"
HF_CACHE_ROOT="/tmp/.hf_cache"
export HF_DATASETS_CACHE="${HF_CACHE_ROOT}/rank_${RANK_ID}"
export HF_HOME="${HF_CACHE_ROOT}/rank_${RANK_ID}"
mkdir -p "${HF_DATASETS_CACHE}"
echo "RANK: ${RANK_ID}, HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"

if [ ! -f ".venv/bin/activate" ]; then
    echo "错误: 虚拟环境 .venv 不存在"
    exit 1
fi
source .venv/bin/activate

export WANDB_API_KEY="$WANDB_KEY"
uv run wandb login
export WANDB_MODE=online

WORLD_SIZE="${WORLD_SIZE:-1}"
IS_RANK_0=false
[[ "${RANK_ID}" == "0" ]] && IS_RANK_0=true

# ===========================================================================
# 通用函数
# ===========================================================================

compute_norm_stats() {
    local config="$1"
    if [ "${WORLD_SIZE}" -gt 1 ]; then
        if [ "$IS_RANK_0" = true ]; then
            echo "Computing normalize statistics (rank 0)..."
            uv run scripts/compute_norm_stats_fast.py --config-name "$config"
        else
            local norm_path
            norm_path=$(uv run scripts/compute_norm_stats_fast.py --config-name "$config" --print-output-path 2>/dev/null | tail -1)
            local waited=0
            echo "Rank ${RANK_ID}: 等待 norm_stats..."
            while [ ! -f "${norm_path}" ]; do
                sleep 10
                waited=$((waited + 10))
                echo "Rank ${RANK_ID}: 等待中... (${waited}s)"
            done
            echo "Rank ${RANK_ID}: norm_stats 已就绪"
        fi
    else
        echo "Computing normalize statistics..."
        uv run scripts/compute_norm_stats_fast.py --config-name "$config"
    fi
}

wait_for_file() {
    local marker="$1"
    local timeout="${2:-7200}"
    local waited=0
    echo "Rank ${RANK_ID}: 等待 $marker ..."
    while [ ! -f "$marker" ]; do
        sleep 30
        waited=$((waited + 30))
        echo "Rank ${RANK_ID}: 等待中... (${waited}s)"
        if [ $waited -ge $timeout ]; then
            echo "Rank ${RANK_ID}: 超时 (${timeout}s)，退出"
            exit 1
        fi
    done
    echo "Rank ${RANK_ID}: $marker 已就绪"
}

# ===========================================================================
# Stage: value — compute_values → train_rl.py
# ===========================================================================
run_value() {
    echo ""
    echo "=========================================="
    echo "Value Model Stage 2 Training"
    echo "  Value config:     $VALUE_CONFIG"
    echo "  Value checkpoint: $VALUE_CHECKPOINT"
    echo "=========================================="

    export WANDB_RUN_NAME="${VALUE_CONFIG}.exp.${TAG}"

    if [ "$IS_RANK_0" = true ]; then
        # Step 1: compute_values
        echo ""
        echo "[Value - Step 1/3] Computing values..."
        NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
        [ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
        echo "Detected $NUM_GPUS GPUs for value inference"
        SUFFIX_ARGS=""
        [[ -n "$SUFFIX" ]] && SUFFIX_ARGS="--suffix $SUFFIX"
        # When --recap-config is provided, use its repo_id list for value computation
        DATA_CONFIG_ARGS=""
        [[ -n "$RECAP_CONFIG" ]] && DATA_CONFIG_ARGS="--data_config_name $RECAP_CONFIG"
        JAXTYPING_DISABLE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/compute_values.py \
            --config_name "$VALUE_CONFIG" \
            --ckpt_dir "$VALUE_CHECKPOINT" \
            --batch_size "$BATCH_SIZE_VALUES" \
            --num_gpus "$NUM_GPUS" \
            --resume \
            $SUFFIX_ARGS $DATA_CONFIG_ARGS
        echo "Values 已保存"

        # Step 2: compute_norm_stats — MUST use $VALUE_CONFIG so norm_stats.json lands
        # in the value config's assets dir (which train_rl.py below will read).
        # $RECAP_CONFIG is only used above to expand the repo_id list for compute_values;
        # its assets dir is different and using it here would silently starve value training.
        echo ""
        echo "[Value - Step 2/3] Computing normalize statistics..."
        uv run scripts/compute_norm_stats_fast.py --config-name "$VALUE_CONFIG"
        echo "Norm stats 已计算"

        touch "/tmp/.recap_value_preprocess_done_${TAG}"
    else
        wait_for_file "/tmp/.recap_value_preprocess_done_${TAG}" 7200
    fi

    # Step 3: train value model
    echo ""
    echo "[Value - Step 3/3] Training value model..."
    JAXTYPING_DISABLE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_rl.py --config "$VALUE_CONFIG"

    echo "Value stage 完成!"
}

# ===========================================================================
# Stage: recap — compute_values → compute_advantages → train.py
# ===========================================================================
run_recap() {
    echo ""
    echo "=========================================="
    echo "RECAP Indicator Pipeline"
    echo "  Value config:     $VALUE_CONFIG"
    echo "  Value checkpoint: $VALUE_CHECKPOINT"
    echo "  RECAP config:     $RECAP_CONFIG"
    echo "  Threshold mode:   $THRESHOLD_MODE"
    echo "  Start from:      ${START_FROM:-从头开始}"
    echo "=========================================="

    export WANDB_RUN_NAME="${RECAP_CONFIG}.exp.${TAG}"

    # Decide which preprocess steps to run.
    # Semantics: --start-from X means "start from stage X and run every stage after it".
    #   ""|value   → value + advantage + norm + train   (default = full pipeline)
    #   advantage  → advantage + norm + train
    #   norm       → norm + train
    #   train      → only train (assumes all preprocess outputs already on disk)
    case "$START_FROM" in
        ""|value)
            RUN_STEP_VALUE=true
            RUN_STEP_ADVANTAGE=true
            RUN_STEP_NORM=true
            ;;
        advantage)
            RUN_STEP_VALUE=false
            RUN_STEP_ADVANTAGE=true
            RUN_STEP_NORM=true
            ;;
        norm)
            RUN_STEP_VALUE=false
            RUN_STEP_ADVANTAGE=false
            RUN_STEP_NORM=true
            ;;
        train)
            RUN_STEP_VALUE=false
            RUN_STEP_ADVANTAGE=false
            RUN_STEP_NORM=false
            ;;
    esac

    if [ "$IS_RANK_0" = true ]; then
        SUFFIX_ARGS=""
        [[ -n "$SUFFIX" ]] && SUFFIX_ARGS="--suffix $SUFFIX"
        VALUE_SUFFIX_ARGS=""
        [[ -n "$VALUE_SUFFIX" ]] && VALUE_SUFFIX_ARGS="--value-suffix $VALUE_SUFFIX"

        if [ "$RUN_STEP_VALUE" = true ]; then
            echo ""
            echo "[RECAP - Step 1/4] Computing values..."
            NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
            [ "$NUM_GPUS" -lt 1 ] && NUM_GPUS=1
            echo "Detected $NUM_GPUS GPUs for value inference"
            JAXTYPING_DISABLE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run python scripts/compute_values.py \
                --config_name "$VALUE_CONFIG" \
                --ckpt_dir "$VALUE_CHECKPOINT" \
                --data_config_name "$RECAP_CONFIG" \
                --batch_size "$BATCH_SIZE_VALUES" \
                --num_gpus "$NUM_GPUS" \
                --resume \
                $SUFFIX_ARGS
            echo "Values 已保存"
        else
            echo ""
            echo "[RECAP - Step 1/4] SKIP compute_values (start-from: ${START_FROM})"
        fi

        if [ "$RUN_STEP_ADVANTAGE" = true ]; then
            echo ""
            echo "[RECAP - Step 2/4] Computing advantages & indicators (threshold_mode=$THRESHOLD_MODE)..."
            uv run python scripts/compute_advantages.py \
                --config-name "$RECAP_CONFIG" \
                --threshold-mode "$THRESHOLD_MODE" \
                $SUFFIX_ARGS $VALUE_SUFFIX_ARGS
            echo "Advantages & indicators 已保存"
        else
            echo ""
            echo "[RECAP - Step 2/4] SKIP compute_advantages (start-from: ${START_FROM})"
        fi

        if [ "$RUN_STEP_NORM" = true ]; then
            echo ""
            echo "[RECAP - Step 3/4] Computing normalize statistics..."
            uv run scripts/compute_norm_stats_fast.py --config-name "$RECAP_CONFIG"
            echo "Norm stats 已计算"
        else
            echo ""
            echo "[RECAP - Step 3/4] SKIP compute_norm_stats (start-from: ${START_FROM})"
        fi

        # Touch marker whenever rank 0 ran at least one preprocess step, so that
        # non-rank-0 workers blocked on wait_for_file can proceed into Step 4.
        # When --start-from train skips all of them, non-rank-0 also skips the wait.
        if [ "$RUN_STEP_VALUE" = true ] || [ "$RUN_STEP_ADVANTAGE" = true ] || [ "$RUN_STEP_NORM" = true ]; then
            touch "/tmp/.recap_indicator_preprocess_done_${TAG}"
        fi
    else
        # Non-rank-0 waits for the preprocess marker whenever rank 0 runs any preprocess
        # step. Only --start-from train lets everyone skip straight to training.
        if [ "$START_FROM" != "train" ]; then
            wait_for_file "/tmp/.recap_indicator_preprocess_done_${TAG}" 7200
        else
            echo ""
            echo "[RECAP - Rank ${RANK_ID}: skip preprocess wait (start-from: train)]"
        fi
    fi

    # Step 4: RECAP indicator training
    echo ""
    echo "[RECAP - Step 4/4] RECAP Indicator Training..."
    JAXTYPING_DISABLE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py --config "$RECAP_CONFIG"

    echo "RECAP stage 完成!"
}

# ===========================================================================
# 主流程
# ===========================================================================

if [[ "$STAGE" == "value" ]]; then
    run_value
elif [[ "$STAGE" == "recap" ]]; then
    run_recap
fi

echo ""
echo "=========================================="
echo "Pipeline 完成! (stage=$STAGE)"
echo "=========================================="
