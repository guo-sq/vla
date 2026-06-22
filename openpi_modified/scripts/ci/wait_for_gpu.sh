#!/usr/bin/env bash

set -euo pipefail

gpu_wait_timeout_s="${OPENPI_GPU_WAIT_TIMEOUT_S:-1800}"
gpu_wait_poll_interval_s="${OPENPI_GPU_WAIT_POLL_INTERVAL_S:-15}"
gpu_min_free_mb="${OPENPI_GPU_MIN_FREE_MB:-76000}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; skipping GPU availability check."
    exit 0
fi

start_ts="$(date +%s)"

while true; do
    mapfile -t gpu_lines < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    mapfile -t process_lines < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)

    active_process_count=0
    for process_line in "${process_lines[@]}"; do
        process_pid="${process_line//[[:space:]]/}"
        if [[ -n "$process_pid" ]]; then
            active_process_count=$((active_process_count + 1))
        fi
    done

    has_enough_free_memory=0
    for gpu_line in "${gpu_lines[@]}"; do
        free_mb="${gpu_line##*,}"
        free_mb="${free_mb//[[:space:]]/}"
        if [[ "$free_mb" =~ ^[0-9]+$ ]] && (( free_mb >= gpu_min_free_mb )); then
            has_enough_free_memory=1
            break
        fi
    done

    if (( active_process_count == 0 && has_enough_free_memory == 1 )); then
        nvidia-smi --query-gpu=timestamp,index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
        exit 0
    fi

    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= gpu_wait_timeout_s )); then
        echo "Timed out waiting for a free GPU: active_processes=${active_process_count}, required_free_mb=${gpu_min_free_mb}" >&2
        nvidia-smi || true
        exit 1
    fi

    echo "Waiting for GPU availability: active_processes=${active_process_count}, required_free_mb=${gpu_min_free_mb}" >&2
    nvidia-smi --query-gpu=timestamp,index,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
    sleep "$gpu_wait_poll_interval_s"
done