#!/bin/bash

set -u

WORKDIR="/mnt/workspace/zys_dev/CodingBranch/openpi_modified"
CONFIG_PATH="src/openpi/configs/cfg_pi0.5_pick_and_place_14_dim_subtask_cond_twostage.py"
DATASET_ROOT="anyverse_pickAndplace_record_tmp"
CHECK_INTERVAL_SECONDS=60
GPU_MEMORY_THRESHOLD_MIB=1000
GPU_UTIL_THRESHOLD_PERCENT=10
REPO_ID="record.pick.place.scheme2.bipiper.v0112.1,record.pick.place.scheme2.otherbackgrounds.bipiper.v0112.8,record.pick.place.pickfromcontainer.bipiper.v0210.10,record.pick.place.pushotherout.bipiper.v0225.7"

CKPT_DIRS=(
    "checkpoints/pi05_pickplace_14_dim_subtask_fda_cond_180token_50k_1st/pi05_pickplace_14_dim_subtask_fda_cond_180token_50k_1st_exp/49999"
    "checkpoints/pi05_pickplace_14_dim_subtask_fda_cond_180token_50k_2ndstage/pi05_pickplace_14_dim_subtask_fda_cond_180token_50k_2ndstage_exp/49999"
    "checkpoints/pi05_pickplace_14_dim_subtask_fda_cond_180token_60k_2nd/pi05_pickplace_14_dim_subtask_fda_cond_180token_60k_2nd_exp/59999"
    "checkpoints/pi05_pickplace_14_dim_subtask_fda_cond_180token_80k_2ndstage/pi05_pickplace_14_dim_subtask_fda_cond_180token_80k_2ndstage_exp/79999"
)

if [ "${#CKPT_DIRS[@]}" -eq 0 ]; then
    echo "error: CKPT_DIRS is empty" >&2
    exit 1
fi

cd "$WORKDIR" || exit 1

MASTER_LOG_PATH="${1:-logs/pi05_multi_eval_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$MASTER_LOG_PATH")"

update_config_for_model() {
    local ckpt_dir="$1"
    local asset_id="$2"

    python - "$CONFIG_PATH" "$ckpt_dir" "$asset_id" <<'PY'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
ckpt_dir = Path(sys.argv[2])
asset_id = sys.argv[3]
exp_name = ckpt_dir.parent.name
if not exp_name.endswith("_exp"):
    raise SystemExit(f"Unexpected experiment directory name: {exp_name}")
task_name = exp_name.removesuffix("_exp")

text = config_path.read_text()
for pattern, replacement, label in [
    (r'^TASK_NAME = ".*"$', f'TASK_NAME = "{task_name}"', "TASK_NAME"),
    (r'^EXP_NAME = TASK_NAME \+ "_exp"$', 'EXP_NAME = TASK_NAME + "_exp"', "EXP_NAME"),
    (r'^ASSET_ID = ".*"$', f'ASSET_ID = "{asset_id}"', "ASSET_ID"),
]:
    text, replacements = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if replacements != 1:
        raise SystemExit(f"Failed to update {label} in {config_path}")
config_path.write_text(text)
PY
}

resolve_asset_id() {
    local ckpt_dir="$1"

    python - "$ckpt_dir" <<'PY'
from pathlib import Path
import sys

assets_dir = Path(sys.argv[1]) / "assets"
if not assets_dir.is_dir():
    raise SystemExit(f"Assets directory not found: {assets_dir}")

asset_dirs = sorted(path.name for path in assets_dir.iterdir() if path.is_dir())
if not asset_dirs:
    raise SystemExit(f"No asset subdirectories found in: {assets_dir}")

print(asset_dirs[0])
PY
}

wait_for_gpu() {
    echo "[$(date '+%F %T')] waiting_for_gpu" | tee -a "$MASTER_LOG_PATH"

    while true; do
        IFS=, read -r mem util <<< "$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"

        if [ "$mem" -lt "$GPU_MEMORY_THRESHOLD_MIB" ] && [ "$util" -lt "$GPU_UTIL_THRESHOLD_PERCENT" ]; then
            echo "[$(date '+%F %T')] gpu_ready_mem=${mem}MiB_util=${util}%" | tee -a "$MASTER_LOG_PATH"
            return 0
        fi

        echo "[$(date '+%F %T')] gpu_busy_mem=${mem}MiB_util=${util}%" | tee -a "$MASTER_LOG_PATH"
        sleep "$CHECK_INTERVAL_SECONDS"
    done
}

run_eval() {
    local ckpt_dir="$1"
    local asset_id
    local model_name
    local model_log_path
    local exit_code

    model_name="$(basename "$(dirname "$ckpt_dir")")_$(basename "$ckpt_dir")"
    model_log_path="logs/${model_name}.log"
    asset_id="$(resolve_asset_id "$ckpt_dir")"

    echo "[$(date '+%F %T')] start_model ckpt_dir=${ckpt_dir} asset_id=${asset_id}" | tee -a "$MASTER_LOG_PATH"
    update_config_for_model "$ckpt_dir" "$asset_id"
    echo "[$(date '+%F %T')] config_updated ckpt_dir=${ckpt_dir} asset_id=${asset_id}" | tee -a "$MASTER_LOG_PATH"

    export HF_ENDPOINT="https://hf-mirror.com"

    uv run scripts/test.py \
        --ckpt_dir "$ckpt_dir" \
        --dataset_root "$DATASET_ROOT" \
        --config_name "$CONFIG_PATH" \
        --num_batches 64 \
        --batch_size 64 \
        --sample_steps 1 \
        --repo_id "$REPO_ID" 2>&1 | tee "$model_log_path" | tee -a "$MASTER_LOG_PATH"

    exit_code=${PIPESTATUS[0]}

    echo "[$(date '+%F %T')] finish_model ckpt_dir=${ckpt_dir} exit_code=${exit_code} log=${model_log_path}" | tee -a "$MASTER_LOG_PATH"
    return "$exit_code"
}

wait_for_gpu

overall_exit_code=0
for ckpt_dir in "${CKPT_DIRS[@]}"; do
    if ! run_eval "$ckpt_dir"; then
        overall_exit_code=1
    fi
done

echo "[$(date '+%F %T')] all_models_finished exit_code=${overall_exit_code}" | tee -a "$MASTER_LOG_PATH"
exit "$overall_exit_code"
