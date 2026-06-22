#!/usr/bin/env bash
set -uo pipefail

# ===========================================================================
# Unified value-model evaluation / labelling dispatcher.
#
# Usage:
#   bash tools/value_model_tools/auto_test_value_model.sh <MODE>
#
#   MODE = "label"  -> label failure data (stage1, saves parquet)
#          "test"   -> test value model  (stage2, visualisation only)
#
# If MODE is omitted, defaults to "test".
# ===========================================================================

MODE="${1:-test}"

LOG_DIR="logs_auto_value_model_${MODE}"
mkdir -p "${LOG_DIR}"

# --- Segmented value-model flag ---
# 开启后把 --segmented 传给 test_rl.py:
#   * dataset 下 meta/segment_values.json 存在 → 画 segment 基的真 GT
#   * 不存在 → 跳过 GT 绘制和 MSE(避免画误导性的 fallback 线)
# 默认关闭。只在测 segmented value 模型时手动改为 1。
SEGMENTED=0

# --- 基础配置（按 MODE 区分）---
DATASET_ROOT="/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/"

if [ "${MODE}" = "label" ]; then
    REPO_IDs=(
        "fold_box_scratch.bad2-8.10s.20260310.batch.1"
        "fold_box_scratch.bad2-8.10s.20260310.batch.2"
        # "fold_box_scratch.bad2-8.10s.20260310.batch.3"
        # "fold_box_scratch.bad2-8.10s.20260311.batch.0"
        # "fold_box_scratch.bad2-8.10s.20260311.batch.1"
        # "fold_box_scratch.bad2-8.10s.20260311.batch.2"
        # "fold_box_scratch.bad2-8.10s.20260311.batch.3"
        # "fold_box_scratch.bad2-8.10s.20260311.batch.4"
        # "fold_box_scratch.bad23.10s.20260311.batch.1"
        # "fold_box_scratch.bad23.10s.20260311.batch.2"
        # "fold_box_scratch.bad31.10s.20260311.batch.1"
        # "fold_box_scratch.bad31.10s.20260311.batch.2"
        # "fold_box_scratch.bad37.10s.20260311.batch.1"
        # "fold_box_scratch.bad37.10s.20260311.batch.2"
    )
    CKPT_DIR="checkpoints/pi05_base_finetune_box_value_stage1/pi05_base_finetune_box_value_0311_only_right_exp.0311_1730/5000"
    CONFIG_NAME="src/openpi/configs/cfg_pi05_base_finetune_box_value_stage1.py"
    VIS_PREFIX="label_failure"
    EXTRA_ARGS="--enable_save_parquet"
elif [ "${MODE}" = "test" ]; then
    REPO_IDs=(
        "fold_box_scratch_infer.all.6000s.20260311.batch.1"
        # "fold_box_scratch_infer.all.6000s.20260311.batch.2"
        # "fold_box_scratch_infer.all.6000s.20260311.batch.3"
    )
    CKPT_DIR="checkpoints/pi05_base_finetune_box_value_stage2/pi05_base_finetune_box_value_0311_good_bad_exp.0312_0000/5000"
    CONFIG_NAME="src/openpi/configs/cfg_pi05_base_finetune_box_value_stage2.py"
    VIS_PREFIX="test_value_model"
    EXTRA_ARGS=""
else
    echo "ERROR: unknown MODE '${MODE}'. Use 'label' or 'test'." >&2
    exit 1
fi

if [ "${SEGMENTED}" = "1" ]; then
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    echo ">>> SEGMENTED=1: appending --segmented to test_rl.py"
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    EXTRA_ARGS="${EXTRA_ARGS} --segmented"
fi

echo "=== MODE=${MODE} | VIS_PREFIX=${VIS_PREFIX} | CKPT_DIR=${CKPT_DIR} ==="

# --- 组合成任务列表 ---
JOBS=()
for repo_id in "${REPO_IDs[@]}"; do
    JOBS+=(
        "python scripts/test_rl.py \
            --dataset_root ${DATASET_ROOT} \
            --repo_id ${repo_id} \
            --ckpt_dir ${CKPT_DIR} \
            --vis_prefix ${VIS_PREFIX} \
            --config_name ${CONFIG_NAME} ${EXTRA_ARGS}"
    )
done

retry_limit=1

i=0
for cmd in "${JOBS[@]}"; do
    i=$((i + 1))

    # try to extract a repo_id from the command for nice log filenames
    repo_id=$(echo "${cmd}" | sed -n "s/.*--repo_id[[:space:]]\+\([^ ]\+\).*/\1/p")
    if [ -z "${repo_id}" ]; then
        name="job_${i}"
    else
        # sanitize repo_id to a filesystem-friendly name
        name=$(echo "${repo_id}" | tr '/ ' '__')
    fi

    logfile="${LOG_DIR}/${name}.log"

    attempt=0
    success=1
    while [ ${attempt} -le ${retry_limit} ]; do
        attempt=$((attempt + 1))
        echo "[${i}/${#JOBS[@]}] Attempt ${attempt}: running -> ${cmd}"
        echo "--- CMD: ${cmd} (attempt ${attempt}) ---" >>"${logfile}"
        # Run the command and append output to logfile
        bash -lc "${cmd}" >>"${logfile}" 2>&1 || rc=$?
        if [ "${rc:-0}" -eq 0 ]; then
            echo "OK: ${name} (attempt ${attempt})"
            success=0
            break
        else
            echo "WARN: ${name} failed (attempt ${attempt}), rc=${rc}" >&2
        if [ ${attempt} -le ${retry_limit} ]; then
                echo "Retrying ${name} after 2s..."
                sleep 2
            fi
        fi
    done

    if [ ${success} -ne 0 ]; then
        echo "FAILED: ${name} after ${attempt} attempts (see ${logfile})" >&2
    fi

    # small delay between jobs
    sleep 0.5
done

echo "All jobs processed. Logs in ${LOG_DIR}/"
