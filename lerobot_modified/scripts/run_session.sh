#!/usr/bin/env bash
# Single entry point for every recording session.
#
# Usage:
#   bash scripts/run_session.sh <session.json> [extra --flag=value ...]
#
# The session config bundles everything that used to be split across
#   - lerobot_example_config_files/collection_infos/<task>/<rig>.json
#   - lerobot_example_config_files/task_specs/<task>/<mode>.json
#   - scripts/<rig>/<task>.sh   (NUM_EPISODES, EPISODE_TIME_S, ...)
#
# What this script does:
#   1. validate session JSON via lerobot.recording.utils.session_to_args
#   2. derive REPO_ID / DATASET_ROOT from data_root + task_name + timestamp
#      (operator can override either via env var)
#   3. chmod /dev/video* for every camera the session declares
#   4. run camera + session preflight popups
#   5. invoke `python -m lerobot.recording.record` with all session-derived
#      flags, plus any extra flags the operator passed after the JSON path
#   6. filter the noisy ARX SDK banner unless QUIET_BANNER=0
#
# Env-overridable knobs (rare; almost everything lives in the JSON now):
#   DATA_ROOT          override session.recording.data_root
#   REPO_ID            override default repo_id (repo_id is path-suffix-only)
#   DATASET_ROOT       override default dataset.root (full path)
#   TODAY              YYYYMMDD date stamp used in default dataset.root
#   TIMESTAMP          full timestamp used in default dataset.root
#   QUIET_BANNER       set to 0 to disable ARX banner filtering

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$#" -lt 1 ]; then
    echo "Usage: bash $0 <session.json> [extra --flag=value ...]" >&2
    exit 2
fi

SESSION_PATH="$1"
shift

if [ ! -f "$SESSION_PATH" ]; then
    echo "session config not found: $SESSION_PATH" >&2
    exit 2
fi
SESSION_PATH="$(readlink -f "$SESSION_PATH")"

# ---------------------------------------------------------------------------
# 1. Validate JSON + read fields we need on the shell side
# ---------------------------------------------------------------------------
# Run from REPO_ROOT so `python -m lerobot...` resolves with the editable
# install. Subshells keep cwd from leaking back to the rest of the script.
field() {
    ( cd "$REPO_ROOT" && \
      python -m lerobot.recording.utils.session_to_args field "$SESSION_PATH" "$1" )
}

# A first call also runs full validation; bail loudly if it fails.
TASK_NAME=$(field "task_meta.task_name") || {
    echo "session config validation failed (see error above)" >&2
    exit 1
}

# Pull the rest of the fields the batch suffix uses. ``field`` ran full
# validation on the first call, so these can't fail with a stale registry.
ROBOT_TYPE=$(field "robot.type")
ROBOT_ID=$(field "robot.id")
OPERATOR_RAW=$(field "collection_meta.operator_name")

# Sanitize free-form fields for use in a filesystem path. Lowercase, then
# replace any run of non-alnum characters with a single underscore so an
# operator name like ``Da Tengfei`` or ``张三`` becomes ``da_tengfei`` /
# ``___`` (or whatever the locale-appropriate transliteration produced
# before sanitize). Empty values fall back to ``unknown`` so we never
# generate a path with ``..`` runs.
sanitize() {
    local s="${1:-}"
    s=$(echo -n "$s" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//')
    if [ -z "$s" ]; then
        echo "unknown"
    else
        echo "$s"
    fi
}
ROBOT_TYPE_SANE=$(sanitize "$ROBOT_TYPE")
ROBOT_ID_SANE=$(sanitize "$ROBOT_ID")
OPERATOR_SANE=$(sanitize "$OPERATOR_RAW")

# Default data paths follow the same shape the per-task scripts used so old
# downstream tools (BatchTracker, on-disk path conventions, the upload
# pipeline) keep working unchanged. The batch suffix carries enough
# operator + hardware metadata that an operator can identify a batch from
# its directory name alone, without opening meta/info.json.
#
#     <task>.<robot_type>.<robot_id>.<operator>.<timestamp>
#     e.g. pack_socks.arxx5_bimanual.5.datengfei.20260428_192943
: "${TODAY:=$(date +%Y%m%d)}"
: "${TIMESTAMP:=$(date +%Y%m%d_%H%M%S)}"
BATCH_NAME="${TASK_NAME}.${ROBOT_TYPE_SANE}.${ROBOT_ID_SANE}.${OPERATOR_SANE}.${TIMESTAMP}"

# data_root: env var > session.recording.data_root > legacy default.
if [ -z "${DATA_ROOT:-}" ]; then
    SESSION_DATA_ROOT=$(field "recording.data_root" 2>/dev/null || true)
    if [ -n "$SESSION_DATA_ROOT" ]; then
        DATA_ROOT="$(eval echo "$SESSION_DATA_ROOT")"
    else
        DATA_ROOT="$HOME/lerobot_data_collection"
    fi
fi
: "${REPO_ID:=${TASK_NAME}/${BATCH_NAME}}"
: "${DATASET_ROOT:=${DATA_ROOT}/${TODAY}/${TASK_NAME}/${BATCH_NAME}}"

# ---------------------------------------------------------------------------
# 2. Hardware preflight: chmod each /dev/video* the session declares
# ---------------------------------------------------------------------------
mapfile -t VIDEO_DEVS < <(
    cd "$REPO_ROOT" && \
    python -m lerobot.recording.utils.session_to_args video_devs "$SESSION_PATH"
)
for dev in "${VIDEO_DEVS[@]}"; do
    if [ -e "$dev" ]; then
        sudo chmod 777 "$dev"
    fi
done

# ---------------------------------------------------------------------------
# 3. Operator preflight: camera grid + session summary popup
# ---------------------------------------------------------------------------
( cd "$REPO_ROOT" && \
  python -m lerobot.recording.utils.preflight cameras "$SESSION_PATH" )
( cd "$REPO_ROOT" && \
  python -m lerobot.recording.utils.preflight session  "$SESSION_PATH" )

# ---------------------------------------------------------------------------
# 4. Build the recorder argv from the session config
# ---------------------------------------------------------------------------
mapfile -t SESSION_ARGS < <(
    cd "$REPO_ROOT" && \
    python -m lerobot.recording.utils.session_to_args args "$SESSION_PATH"
)

# ---------------------------------------------------------------------------
# 5. Invoke the recorder
# ---------------------------------------------------------------------------
cd "$REPO_ROOT"
# ``python -u`` disables stdout/stderr block buffering. Without it, when we
# pipe the recorder through ``grep --line-buffered`` below, Python switches
# sys.stdout to block-buffered mode and the per-tick episode progress line
# (and any other ``print(...)``) only appears when the buffer fills or the
# process exits — operators saw a frozen terminal until Ctrl+Right
# happened to flush via the logging module.
RECORDER_CMD=(
    python -u -m lerobot.recording.record
    "${SESSION_ARGS[@]}"
    "--dataset.root=$DATASET_ROOT"
    "--dataset.repo_id=$REPO_ID"
    "$@"
)

if [ "${QUIET_BANNER:-1}" = "0" ]; then
    "${RECORDER_CMD[@]}"
else
    # stdbuf + grep --line-buffered keep the rest of the output streaming
    # in real time. set -o pipefail (above) propagates the Python exit code
    # through the pipe.
    stdbuf -oL "${RECORDER_CMD[@]}" 2>&1 \
        | grep --line-buffered -v 'ARX方舟无限'
fi
