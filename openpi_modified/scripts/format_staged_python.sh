#!/usr/bin/env bash
# Black + Ruff, aligned with .github/workflows/ci.yml (black-check / ruff-check).
# Requires: uv (https://github.com/astral-sh/uv) on PATH for uvx.
#
# Usage:
#   - Pre-commit: receives file paths from pre-commit as arguments.
#   - Manual (staged only): run with no arguments from repo root; formats
#     staged *.py under src/, tests/, scripts/, packages/.
#   - Manual (explicit files): pass paths as arguments.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

readonly BLACK_VERSION="24.10.0"
readonly RUFF_VERSION="0.8.6"

if [[ $# -gt 0 ]]; then
  files=("$@")
else
  mapfile -t files < <(
    git diff --cached --name-only --diff-filter=ACM 2>/dev/null | grep -E '^(src|tests|scripts|packages)/.*\.py$' || true
  )
fi

if [[ ${#files[@]} -eq 0 ]]; then
  exit 0
fi

# Black: same as ci.yml (no --force-exclude; explicit paths are already scoped).
uvx --from "black==${BLACK_VERSION}" black "${files[@]}"
# Ruff: match ci.yml ruff-check (--force-exclude honors pyproject extend-exclude).
uvx --from "ruff==${RUFF_VERSION}" ruff check --fix --force-exclude "${files[@]}"
