#!/bin/bash
# Run 3 remaining value models on the v0409 split repos (union of fold + flatten)
# for the final 4-model Separation Score comparison.
#
# Why this script exists:
# - fast_mode_max3600 has already been scored on all 305 repos (and thus
#   automatically covers the 97 v0409 split repos) from the original local
#   Step 7.7 run. No need to re-run it.
# - per_task_p90 / 1215_0227_max3600 / stage2_all_0322 still need sparse
#   head_pred/tail_pred scores on the 97 split repos to complete the Step 5
#   "4 baseline 模型 Separation Score 对比表" deliverable.
#
# Why fold prompt is used for both fold and flatten eval:
# - ALL 4 models were trained with the identical long fold prompt as the
#   `default_prompt` in their training configs (verified in
#   cfg_pi06_value_model_*.py). There is no flatten-specific training prompt.
# - Scoring flatten repos under the fold prompt is therefore in-distribution;
#   using a made-up flatten prompt would be OOD and the resulting Separation
#   Scores would conflate model quality with prompt-sensitivity noise.
# - The 1D tail_pred analysis (4c-1) already validated that models produce
#   distinct distributions for flatten_success vs fold_success under this
#   single prompt, so the prompt is sufficient to separate the two tasks.
#
# Usage:
#   bash scripts/benchmark/run_v0409_multi_model_eval.sh
#
# GPU memory: capped at 50% via XLA_PYTHON_CLIENT_MEM_FRACTION (~49 GB on H20).
# Wall time estimate: ~40-50 min per model × 3 models ≈ 2-2.5 h serial.
# Crash resilience: each (model) run is a separate invocation; a crash in
# model N only loses model N, and re-running the script resumes from the first
# missing episode_details.json.

set -uo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_BASE="test_results/benchmark/clothes_v0409_multi_model"
SPLIT_DIR="test_results/split/clothes_v0409"
DATASET_ROOT="/mnt/oss_data/anyverse/bipiper_clothes"

# Same fold prompt as every training config's default_prompt and the original
# Step 7.7 local run — ensures in-distribution scoring for both fold and
# flatten repos.
FOLD_PROMPT="You are a two-armed piper robot with a total of three perspectives. Your task is to fold a T-shirt. First, look for the collar of the T-shirt. If you can't see it, pick up the T-shirt and let it fall naturally until the collar is visible. Then, grab the collar to lay the T-shirt flat. Next, simultaneously grab the collar and the bottom hem to fold it in half and lay it flat. Finally, lay it flat with the collar facing down, then fold it up and place it in the fixed position on the right."

# 3 models remaining (fast_mode_max3600 already done).
declare -A CKPTS
CKPTS[per_task_p90]="/mnt/workspace/tianwanxin/dev/checkpoints_backup_value_exp/pi06_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_1215_0227_per_task_p90/pi06_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_1215_0227_per_task_p90_exp/9999"
CKPTS[1215_0227_max3600]="/mnt/workspace/tianwanxin/dev/openpi_modified/checkpoints/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_1215_0227_max3600/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_1215_0227_max3600.exp.0314/10000"
CKPTS[stage2_all_0322]="/mnt/workspace/tianwanxin/dev/openpi_modified/checkpoints/pi06_value_stage2_t5gemma270M_bin1_bs1024_max3600_all_bipiper_clothes/pi06_value_stage2_t5gemma270M_bin1_bs1024_max3600_all_bipiper_clothes.exp.0318/10000"

declare -A CONFIGS
CONFIGS[per_task_p90]="src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_1215_0227_per_task_p90.py"
CONFIGS[1215_0227_max3600]="src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_1215_0227_max3600.py"
CONFIGS[stage2_all_0322]="src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_all_0322_stage2.py"

MODEL_ORDER=(per_task_p90 1215_0227_max3600 stage2_all_0322)

GPU_THRESHOLD_MB=${GPU_THRESHOLD_MB:-48000}  # Require ~48 GB free (we'll cap to 50%)
POLL_SECONDS=60

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

cd "$(dirname "$0")/../.."

FOLD_LIST="${SPLIT_DIR}/fold/repo_list.txt"
FLATTEN_LIST="${SPLIT_DIR}/flatten/repo_list.txt"

if [[ ! -f "${FOLD_LIST}" || ! -f "${FLATTEN_LIST}" ]]; then
    echo "[eval] ERROR: split repo lists not found."
    echo "[eval]   expected: ${FOLD_LIST}"
    echo "[eval]   expected: ${FLATTEN_LIST}"
    echo "[eval] Run scripts/benchmark/post_sparse_pipeline.sh first."
    exit 1
fi

mkdir -p "${OUTPUT_BASE}"
UNION_LIST="${OUTPUT_BASE}/v0409_union_repos.txt"
sort -u "${FOLD_LIST}" "${FLATTEN_LIST}" > "${UNION_LIST}"
UNION_COUNT=$(wc -l < "${UNION_LIST}")

echo "[eval] ============================================"
echo "[eval] v0409 multi-model Separation Score runner"
echo "[eval] ============================================"
echo "[eval] Union repo list: ${UNION_LIST} (${UNION_COUNT} repos)"
echo "[eval] Fold split:      ${FOLD_LIST} ($(wc -l < "${FOLD_LIST}") repos)"
echo "[eval] Flatten split:   ${FLATTEN_LIST} ($(wc -l < "${FLATTEN_LIST}") repos)"
echo "[eval] Output base:     ${OUTPUT_BASE}"
echo "[eval] Models to run:   ${MODEL_ORDER[*]}"
echo "[eval] GPU cap:         XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 (~49 GB)"
echo "[eval] ============================================"

mapfile -t REPO_IDS < "${UNION_LIST}"

export PYTHONPATH=src:packages/openpi-client/src:.
export PYTHONUNBUFFERED=1
export JAXTYPING_DISABLE=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# ---------------------------------------------------------------------------
# GPU wait loop (reused from wait_gpu_then_run.sh)
# ---------------------------------------------------------------------------

wait_for_gpu() {
    while true; do
        FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
        TS=$(date '+%H:%M:%S')
        if [[ "${FREE_MB}" -ge "${GPU_THRESHOLD_MB}" ]]; then
            echo "[eval] ${TS} GPU free ${FREE_MB} MB >= ${GPU_THRESHOLD_MB} MB → proceeding"
            return 0
        fi
        echo "[eval] ${TS} GPU free ${FREE_MB} MB < ${GPU_THRESHOLD_MB} MB, waiting ${POLL_SECONDS}s..."
        sleep "${POLL_SECONDS}"
    done
}

# ---------------------------------------------------------------------------
# Run each model
# ---------------------------------------------------------------------------

FAILED_MODELS=()
SKIPPED_MODELS=()

for model_name in "${MODEL_ORDER[@]}"; do
    ckpt="${CKPTS[$model_name]}"
    config="${CONFIGS[$model_name]}"
    model_out="${OUTPUT_BASE}/${model_name}"
    done_marker="${model_out}/metrics/episode_details.json"

    echo ""
    echo "[eval] ========================================================"
    echo "[eval] Model: ${model_name}"
    echo "[eval] Ckpt:  ${ckpt}"
    echo "[eval] Cfg:   ${config}"
    echo "[eval] Out:   ${model_out}"
    echo "[eval] ========================================================"

    if [[ -f "${done_marker}" ]]; then
        echo "[eval]   SKIP: ${done_marker} already exists"
        SKIPPED_MODELS+=("${model_name}")
        continue
    fi

    if [[ ! -d "${ckpt}" ]]; then
        echo "[eval]   ERROR: checkpoint dir not found, skipping model"
        FAILED_MODELS+=("${model_name}:no_ckpt")
        continue
    fi

    wait_for_gpu

    echo "[eval]   Launching at $(date '+%H:%M:%S')"
    if python scripts/benchmark/run_benchmark.py \
        --ckpt_dir "${ckpt}" \
        --config_name "${config}" \
        --dataset_root "${DATASET_ROOT}" \
        --repo_ids "${REPO_IDS[@]}" \
        --output_dir "${model_out}" \
        --batch_size 64 \
        --sparse_mode \
        --override_prompt "${FOLD_PROMPT}"; then
        echo "[eval]   OK at $(date '+%H:%M:%S')"
    else
        exit_code=$?
        echo "[eval]   FAILED (exit ${exit_code}) at $(date '+%H:%M:%S')"
        FAILED_MODELS+=("${model_name}:exit_${exit_code}")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo ""
echo "[eval] ========================================================"
echo "[eval] Done at $(date '+%H:%M:%S')"
echo "[eval] ========================================================"
echo "[eval] Models attempted: ${#MODEL_ORDER[@]}"
echo "[eval] Skipped (already done): ${#SKIPPED_MODELS[@]} -> ${SKIPPED_MODELS[*]:-none}"
echo "[eval] Failed: ${#FAILED_MODELS[@]} -> ${FAILED_MODELS[*]:-none}"
echo ""
echo "[eval] Next step:"
echo "[eval]   Compute per-model Separation Score on fold + flatten splits."
echo "[eval]   Each model's head_pred/tail_pred live under:"
echo "[eval]     ${OUTPUT_BASE}/<model>/metrics/episode_details.json"
echo "[eval]   fast_mode_max3600 scores are already in:"
echo "[eval]     test_results/benchmark/clothes_v0409_sparse/fast_mode_max3600/metrics/episode_details.json"

if [[ "${#FAILED_MODELS[@]}" -gt 0 ]]; then
    exit 1
fi
