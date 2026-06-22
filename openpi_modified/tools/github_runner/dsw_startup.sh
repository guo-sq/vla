#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/.runner-state/runner.env}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Runner config not found: $CONFIG_PATH" >&2
    exit 2
fi

STATE_DIR="$ROOT_DIR/.runner-state"
config_state_dir="$(awk -F= '/^RUNNER_STATE_DIR=/{print substr($0, index($0, $2))}' "$CONFIG_PATH" | tail -n 1 | tr -d '"' | tr -d "'")"
if [[ -n "$config_state_dir" ]]; then
    STATE_DIR="$config_state_dir"
fi

BOOT_LOG_DIR="$STATE_DIR/logs"
BOOT_LOG_PATH="$BOOT_LOG_DIR/dsw_startup.log"
WATCH_LOG_PATH="$BOOT_LOG_DIR/watchdog.log"

mkdir -p "$BOOT_LOG_DIR"

"$PYTHON_BIN" "$ROOT_DIR/tools/github_runner/runner_manager.py" --config "$CONFIG_PATH" ensure-running >> "$BOOT_LOG_PATH" 2>&1

if [[ -f "$STATE_DIR/watchdog.pid" ]]; then
    old_pid="$(cat "$STATE_DIR/watchdog.pid")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        exit 0
    fi
fi

nohup "$PYTHON_BIN" "$ROOT_DIR/tools/github_runner/runner_manager.py" --config "$CONFIG_PATH" watch >> "$WATCH_LOG_PATH" 2>&1 &
echo $! > "$STATE_DIR/watchdog.pid"