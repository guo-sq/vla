#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

heartbeat_interval() {
    local configured="${OPENPI_TEST_HEARTBEAT_INTERVAL:-60}"
    if [[ "$configured" =~ ^[0-9]+$ ]] && (( configured > 0 )); then
        printf '%s\n' "$configured"
    else
        printf '60\n'
    fi
}

run_with_heartbeat() {
    local interval
    interval="$(heartbeat_interval)"

    echo "[start] suite=${SUITE} heartbeat_interval=${interval}s command=$*"

    "$@" &
    local command_pid=$!
    local start_ts
    start_ts="$(date +%s)"

    while kill -0 "$command_pid" 2>/dev/null; do
        sleep "$interval"
        if kill -0 "$command_pid" 2>/dev/null; then
            local now_ts elapsed
            now_ts="$(date +%s)"
            elapsed=$((now_ts - start_ts))
            echo "[heartbeat] suite=${SUITE} still running after ${elapsed}s"
        fi
    done

    wait "$command_pid"
}

run_pytest() {
    if command -v uv >/dev/null 2>&1; then
        run_with_heartbeat uv run pytest "$@"
    else
        run_with_heartbeat python -m pytest "$@"
    fi
}

run_python() {
    if command -v uv >/dev/null 2>&1; then
        run_with_heartbeat uv run python "$@"
    else
        run_with_heartbeat python "$@"
    fi
}

usage() {
    cat <<'EOF'
Usage: tests/run_tests.sh [suite] [extra pytest args...]

Suites:
  smoke        超快回归测试
  unit         单元测试
    config       配置与契约测试
    config-loss  顶层 config 训练 loss 基线对比检查
    config-loss-drift  顶层 config 漂移报告脚本
    config-loss-refresh  刷新顶层 config loss 基线文件
  openloop-eval  开环评估测试
  integration  集成测试
  pretrain     预训练相关测试
  posttrain    后训练/推理相关测试
  rl           RL 相关测试
  ci           CI 默认测试（含覆盖率）
  full         全量测试（排除 manual）

示例:
  tests/run_tests.sh smoke
  tests/run_tests.sh pretrain -k tokenizer
  tests/run_tests.sh ci
EOF
}

SUITE="${1:-ci}"
if [[ $# -gt 0 ]]; then
    shift
fi

COMMON_EXPR="not manual"
COMMON_ARGS=("-q")

case "$SUITE" in
    smoke)
        run_pytest "${COMMON_ARGS[@]}" -m "smoke and ${COMMON_EXPR}" "$@"
        ;;
    unit)
        run_pytest "${COMMON_ARGS[@]}" -m "unit and ${COMMON_EXPR}" "$@"
        ;;
    config)
        run_pytest "${COMMON_ARGS[@]}" -m "config and ${COMMON_EXPR}" "$@"
        ;;
    config-loss)
        run_python tests/check_top_level_config_loss_drift.py "$@"
        ;;
    config-loss-drift)
        run_python tests/check_top_level_config_loss_drift.py "$@"
        ;;
    config-loss-refresh)
        run_python tests/generate_top_level_config_loss_baseline.py "$@"
        ;;
    openloop-eval)
        run_python scripts/run_openloop_eval.py "$@"
        ;;
    integration)
        run_pytest "${COMMON_ARGS[@]}" -m "integration and ${COMMON_EXPR}" "$@"
        ;;
    pretrain)
        run_pytest "${COMMON_ARGS[@]}" -m "pretrain and ${COMMON_EXPR}" "$@"
        ;;
    posttrain)
        run_pytest "${COMMON_ARGS[@]}" -m "posttrain and ${COMMON_EXPR}" "$@"
        ;;
    rl)
        run_pytest "${COMMON_ARGS[@]}" -m "rl and ${COMMON_EXPR}" "$@"
        ;;
    ci)
        run_pytest "${COMMON_ARGS[@]}" --cov=openpi --cov=openpi_client --cov-report=term-missing -m "$COMMON_EXPR" "$@"
        ;;
    full)
        run_pytest "${COMMON_ARGS[@]}" -m "${COMMON_EXPR} and not smoke" "$@"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "Unknown suite: $SUITE" >&2
        usage >&2
        exit 2
        ;;
esac
