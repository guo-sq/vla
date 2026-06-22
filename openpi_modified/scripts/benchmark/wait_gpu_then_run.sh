#!/bin/bash
# Wait for GPU 0 to have > THRESHOLD_MB free, then launch sparse benchmark on all clothes repos.
#
# Runs the repo list in chunks of BATCH_CHUNK_SIZE (default 60) so a crash mid-run
# only loses the current batch, not all 2-4 hours of work. Each batch writes to
# ``${OUTPUT_DIR}/batch_N/``; a post-run merge step can combine them.
#
# Usage:
#   bash scripts/benchmark/wait_gpu_then_run.sh [threshold_mb] [output_dir]
#
# Environment:
#   BATCH_CHUNK_SIZE=60    Override how many repos per run_benchmark.py invocation.
#   REPO_LIST_FILE=...     Override the repo list path.
#
# Intended for the Step 7.7 local production run (DLC quota exhausted).

set -uo pipefail

THRESHOLD_MB=${1:-22000}   # Default: 22GB, smoke-test-proven minimum with ~1GB safety margin
OUTPUT_DIR=${2:-test_results/benchmark/clothes_v0409_sparse/fast_mode_max3600}
BATCH_CHUNK_SIZE=${BATCH_CHUNK_SIZE:-60}
POLL_SECONDS=60

CKPT_DIR="/mnt/workspace/tianwanxin/dev/openpi_modified/checkpoints/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_fast_mode_max3600/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_fast_mode_max3600.exp.0314/10000"
CONFIG="src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_fast_mode_max3600.py"
DATASET_ROOT="/mnt/oss_data/anyverse/bipiper_clothes"

CLOTHES_PROMPT="You are a two-armed piper robot with a total of three perspectives. Your task is to fold a T-shirt. First, look for the collar of the T-shirt. If you can't see it, pick up the T-shirt and let it fall naturally until the collar is visible. Then, grab the collar to lay the T-shirt flat. Next, simultaneously grab the collar and the bottom hem to fold it in half and lay it flat. Finally, lay it flat with the collar facing down, then fold it up and place it in the fixed position on the right."

REPO_LIST_FILE=${REPO_LIST_FILE:-scripts/benchmark/clothes_all_repos.txt}

echo "[wait_gpu] Threshold: ${THRESHOLD_MB} MB free on GPU 0"
echo "[wait_gpu] Output dir: ${OUTPUT_DIR}"
echo "[wait_gpu] Repo list:  ${REPO_LIST_FILE}"
echo "[wait_gpu] Batch size: ${BATCH_CHUNK_SIZE} repos per run_benchmark.py invocation"

if [[ ! -f "${REPO_LIST_FILE}" ]]; then
    echo "[wait_gpu] ERROR: repo list ${REPO_LIST_FILE} not found"
    exit 1
fi

mapfile -t REPO_IDS < <(grep -v '^#' "${REPO_LIST_FILE}" | grep -v '^$')
TOTAL_REPOS=${#REPO_IDS[@]}
NUM_BATCHES=$(( (TOTAL_REPOS + BATCH_CHUNK_SIZE - 1) / BATCH_CHUNK_SIZE ))
echo "[wait_gpu] Loaded ${TOTAL_REPOS} repos → ${NUM_BATCHES} batches"

# Wait for GPU to free up.
while true; do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
    TS=$(date '+%H:%M:%S')
    if [[ "${FREE_MB}" -ge "${THRESHOLD_MB}" ]]; then
        echo "[wait_gpu] ${TS} GPU free ${FREE_MB} MB ≥ ${THRESHOLD_MB} MB → launching"
        break
    fi
    echo "[wait_gpu] ${TS} GPU free ${FREE_MB} MB < ${THRESHOLD_MB} MB, waiting ${POLL_SECONDS}s..."
    sleep "${POLL_SECONDS}"
done

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH=src:packages/openpi-client/src:.
export PYTHONUNBUFFERED=1
export JAXTYPING_DISABLE=1

# Run each batch of BATCH_CHUNK_SIZE repos as a separate run_benchmark.py invocation.
# A crash in batch N only loses batch N — batches 0..N-1 stay on disk.
FAILED_BATCHES=()
for (( batch_idx=0; batch_idx<NUM_BATCHES; batch_idx++ )); do
    start=$(( batch_idx * BATCH_CHUNK_SIZE ))
    end=$(( start + BATCH_CHUNK_SIZE ))
    if [[ "${end}" -gt "${TOTAL_REPOS}" ]]; then
        end=${TOTAL_REPOS}
    fi
    batch_repos=("${REPO_IDS[@]:start:end-start}")
    batch_out="${OUTPUT_DIR}/batch_$(printf '%03d' "${batch_idx}")"

    echo ""
    echo "[wait_gpu] ========================================================"
    echo "[wait_gpu] Batch $((batch_idx + 1))/${NUM_BATCHES}: repos ${start}..$((end - 1)) → ${batch_out}"
    echo "[wait_gpu] $(date '+%H:%M:%S')"
    echo "[wait_gpu] ========================================================"

    if python scripts/benchmark/run_benchmark.py \
        --ckpt_dir "${CKPT_DIR}" \
        --config_name "${CONFIG}" \
        --dataset_root "${DATASET_ROOT}" \
        --repo_ids "${batch_repos[@]}" \
        --output_dir "${batch_out}" \
        --batch_size 64 \
        --sparse_mode \
        --override_prompt "${CLOTHES_PROMPT}"; then
        echo "[wait_gpu] Batch $((batch_idx + 1))/${NUM_BATCHES} OK"
    else
        exit_code=$?
        echo "[wait_gpu] Batch $((batch_idx + 1))/${NUM_BATCHES} FAILED (exit ${exit_code}), continuing"
        FAILED_BATCHES+=("${batch_idx}")
    fi
done

echo ""
echo "[wait_gpu] ========================================================"
echo "[wait_gpu] Done at $(date '+%H:%M:%S')"
echo "[wait_gpu] Completed ${NUM_BATCHES} batch(es), ${#FAILED_BATCHES[@]} failed"
if [[ "${#FAILED_BATCHES[@]}" -gt 0 ]]; then
    echo "[wait_gpu] Failed batches: ${FAILED_BATCHES[*]}"
    exit 1
fi
