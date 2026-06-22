#!/usr/bin/env bash

set -euo pipefail

GITLAB_INTERNAL_IP="${GITLAB_INTERNAL_IP:-106.38.41.82}"
GITLAB_WEB_PORT="${GITLAB_WEB_PORT:-30443}"
GITLAB_SSH_PORT="${GITLAB_SSH_PORT:-30022}"
GITLAB_HOSTNAME="${GITLAB_HOSTNAME:-gitlab.anyverse.work}"
GITLAB_RUNNER_URL="${GITLAB_RUNNER_URL:-https://${GITLAB_HOSTNAME}:${GITLAB_WEB_PORT}}"
GITLAB_RUNNER_NAME="${GITLAB_RUNNER_NAME:-openpi-dsw-runner}"
GITLAB_RUNNER_TAG_LIST="${GITLAB_RUNNER_TAG_LIST:-self-hosted,linux,x64}"
GITLAB_RUNNER_EXECUTOR="${GITLAB_RUNNER_EXECUTOR:-shell}"
GITLAB_RUNNER_CONFIG="${GITLAB_RUNNER_CONFIG:-/mnt/workspace/heyuan/ci_cd/openpi_modified/.gitlab-runner/config.toml}"
GITLAB_RUNNER_WORKDIR="${GITLAB_RUNNER_WORKDIR:-/mnt/workspace/heyuan/ci_cd/openpi_modified/.gitlab-runner/workdir}"
GITLAB_RUNNER_CERT_DIR="${GITLAB_RUNNER_CERT_DIR:-/mnt/workspace/heyuan/ci_cd/openpi_modified/.gitlab-runner/certs}"
GITLAB_RUNNER_TLS_CA_FILE="${GITLAB_RUNNER_TLS_CA_FILE:-${GITLAB_RUNNER_CERT_DIR}/${GITLAB_HOSTNAME}.crt}"
GITLAB_RUNNER_CLONE_URL="${GITLAB_RUNNER_CLONE_URL:-${GITLAB_RUNNER_URL}}"
GITLAB_RUNNER_LOG_FILE="${GITLAB_RUNNER_LOG_FILE:-/mnt/workspace/heyuan/ci_cd/openpi_modified/.gitlab-runner/runner.log}"
GITLAB_RUNNER_TOKEN="${GITLAB_RUNNER_TOKEN:-}"

if [[ -z "$GITLAB_RUNNER_TOKEN" ]]; then
    echo "GITLAB_RUNNER_TOKEN is required." >&2
    exit 2
fi

if [[ $EUID -ne 0 ]]; then
    echo "Run this script as root on the DSW container." >&2
    exit 2
fi

install_runner_binary() {
    if command -v gitlab-runner >/dev/null 2>&1; then
        return
    fi
    curl -L --fail --output /usr/local/bin/gitlab-runner \
        https://gitlab-runner-downloads.s3.amazonaws.com/latest/binaries/gitlab-runner-linux-amd64
    chmod +x /usr/local/bin/gitlab-runner
}

ensure_gitlab_host_mapping() {
    local tmp_hosts
    tmp_hosts="$(mktemp)"
    awk -v host="$GITLAB_HOSTNAME" '$2 != host { print }' /etc/hosts > "$tmp_hosts"
    echo "${GITLAB_INTERNAL_IP} ${GITLAB_HOSTNAME}" >> "$tmp_hosts"
    cat "$tmp_hosts" > /etc/hosts
    rm -f "$tmp_hosts"
}

write_gitlab_cert() {
    mkdir -p "$GITLAB_RUNNER_CERT_DIR"
    python3 - <<'PY'
import os
import socket
import ssl
from pathlib import Path

host = os.environ["GITLAB_HOSTNAME"]
ip = os.environ["GITLAB_INTERNAL_IP"]
port = int(os.environ["GITLAB_WEB_PORT"])
path = Path(os.environ["GITLAB_RUNNER_TLS_CA_FILE"])

ctx = ssl._create_unverified_context()
with ctx.wrap_socket(socket.socket(), server_hostname=host) as sock:
    sock.settimeout(8)
    sock.connect((ip, port))
    der = sock.getpeercert(True)

path.write_text(ssl.DER_cert_to_PEM_cert(der), encoding="utf-8")
PY
}

register_runner() {
    mkdir -p "$(dirname "$GITLAB_RUNNER_CONFIG")" "$GITLAB_RUNNER_WORKDIR"
    gitlab-runner register \
        --non-interactive \
        --config "$GITLAB_RUNNER_CONFIG" \
        --url "$GITLAB_RUNNER_URL" \
        --token "$GITLAB_RUNNER_TOKEN" \
        --executor "$GITLAB_RUNNER_EXECUTOR" \
        --name "$GITLAB_RUNNER_NAME" \
        --tag-list "$GITLAB_RUNNER_TAG_LIST" \
        --run-untagged=false \
        --locked=false \
        --access-level=not_protected \
        --tls-ca-file "$GITLAB_RUNNER_TLS_CA_FILE" \
        --clone-url "$GITLAB_RUNNER_CLONE_URL"
}

start_runner() {
    pkill -f "gitlab-runner run --config ${GITLAB_RUNNER_CONFIG}" 2>/dev/null || true
    nohup gitlab-runner run \
        --config "$GITLAB_RUNNER_CONFIG" \
        --working-directory "$GITLAB_RUNNER_WORKDIR" \
        >> "$GITLAB_RUNNER_LOG_FILE" 2>&1 &
    echo $! > "${GITLAB_RUNNER_WORKDIR}/gitlab-runner.pid"
}

install_runner_binary
ensure_gitlab_host_mapping
write_gitlab_cert
register_runner
start_runner

cat <<EOF
Runner configured.
URL: ${GITLAB_RUNNER_URL}
Tags: ${GITLAB_RUNNER_TAG_LIST}
Config: ${GITLAB_RUNNER_CONFIG}
Log: ${GITLAB_RUNNER_LOG_FILE}

For manual git operations on the DSW container, use:
export GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -p ${GITLAB_SSH_PORT}"
EOF