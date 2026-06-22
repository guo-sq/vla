#!/usr/bin/env bash
set -uo pipefail

# ===========================================================================
# Policy model open-loop evaluation dispatcher.
#
# Usage:
#   bash auto_test_policy_model.sh
#
# Uses scripts/test.py to run open-loop eval on specified datasets.
# ===========================================================================

LOG_DIR="logs_auto_test_policy_model"
mkdir -p "${LOG_DIR}"

# --- 基础配置 ---
DATASET_ROOT="/mnt/oss_data/anyverse_human_data_record/arxx5_bimanual/fold_box/close_the_flap/"
# CKPT_DIR="checkpoints/pi05_base_finetune_box_recap_pt_0421_close_noinfer/pi05_base_finetune_box_recap_pt_0421_close_noinfer_exp/20000/"
# CONFIG_NAME="src/openpi/configs/cfg_pi05_base_finetune_box_recap_pt_0421_close_noinfer.py"
CKPT_DIR="checkpoints/pi05_base_finetune_box_recap_pt_0421_close_noinfer/pi05_base_finetune_box_recap_pt_0421_close_noinfer_exp/29999/"
CONFIG_NAME="src/openpi/configs/cfg_pi05_base_finetune_box_recap_pt_0421_close_noinfer.py"
BATCH_SIZE=128
SAMPLE_STEPS=10
VIS_GAP=50

# --- 评测数据集及对应 num_batches ---
# 格式: "repo_id|num_batches"
# episode 0 帧数 -> num_batches = ceil(frames / batch_size)
#   close_the_flap.Stuff.85s.20260407.batch.5: ep0=1740 frames -> 14 batches
#   close_the_flap.recover14_2.Any.16s.20260409.batch.1: ep0=479 frames -> 4 batches
EVAL_ITEMS=(
    # "close_the_flap/total_steps/close_the_flap.Stuff.85s.20260407.batch.5|14"
    # "second_half/close_the_flap.second_half.screwdriver.zhongyou.22s.20260421.batch.1|6"
    "infer/raw_infer/close_the_box_infer.origin.pi05_base_finetune_box_recap_pt_0420_close.1w9.6000s.20260421.batch.5|7"
)

echo "=== Policy Model Eval | CKPT_DIR=${CKPT_DIR} ==="

# --- 组合成任务列表 ---
JOBS=()
for item in "${EVAL_ITEMS[@]}"; do
    repo_id="${item%%|*}"
    num_batches="${item##*|}"
    JOBS+=(
        "PYTHONPATH=src:third_party/lerobot XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
        python scripts/test.py \
            --ckpt_dir ${CKPT_DIR} \
            --config_name ${CONFIG_NAME} \
            --dataset_root ${DATASET_ROOT} \
            --repo_id ${repo_id} \
            --num_batches ${num_batches} \
            --batch_size ${BATCH_SIZE} \
            --sample_steps ${SAMPLE_STEPS} \
            --vis_gap ${VIS_GAP} \
            --num_workers 0"
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
        rc=0
        bash -lc "${cmd}" >>"${logfile}" 2>&1 || rc=$?
        if [ ${rc} -eq 0 ]; then
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
