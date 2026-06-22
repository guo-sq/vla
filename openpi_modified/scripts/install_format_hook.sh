#!/usr/bin/env bash
# Install the git pre-commit hook that runs scripts/format_staged_python.sh via pre-commit.
# Requires: uv on PATH.
#
# Usage: from repo root: bash scripts/install_format_hook.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

exec uvx --from pre-commit==4.1.0 pre-commit install
