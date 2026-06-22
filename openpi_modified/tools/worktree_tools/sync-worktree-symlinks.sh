#!/usr/bin/env bash
#
# sync-worktree-symlinks.sh
#
# Restore gitignored runtime resources from the main repository into a
# newly created git worktree:
#   - openpi-assets  (symlink in main repo → shared model weights)
#   - assets         (symlink in main repo → shared norm_stats)
#   - .venv          (real dir in main repo → full uv venv install)
#
# Without this, a fresh worktree is missing model weights / norm_stats
# and has an empty uv-bootstrapped venv with no dependencies installed,
# so training / eval / even basic imports like `flax.nnx` fail.
#
# Usage:
#   sync-worktree-symlinks.sh <worktree-path>
#
# The main repository is resolved from the target worktree's git metadata
# (git-common-dir), so this script works regardless of whether it is run
# from the main-repo copy or from a copy committed inside a worktree.

set -euo pipefail

# Size threshold (KB) for detecting a bootstrap-stub .venv. A populated
# openpi venv is multi-GB; an empty uv-bootstrapped stub is under 1 MB.
# 100 MB gives a comfortable safety margin without risking deletion of
# a venv that actually has packages installed.
STUB_VENV_MAX_KB=$((100 * 1024))

if [ $# -ne 1 ]; then
    echo "Usage: $(basename "$0") <worktree-path>" >&2
    exit 1
fi

WORKTREE=$(cd "$1" && pwd)

if ! git -C "$WORKTREE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Error: '$WORKTREE' is not a git worktree" >&2
    exit 1
fi

GIT_COMMON_DIR=$(git -C "$WORKTREE" rev-parse --path-format=absolute --git-common-dir)
MAIN_REPO=$(dirname "$GIT_COMMON_DIR")

if [ "$WORKTREE" = "$MAIN_REPO" ]; then
    echo "Error: target is the main repository itself, not a worktree" >&2
    exit 1
fi

echo "main repo:  $MAIN_REPO"
echo "worktree:   $WORKTREE"
echo

LINKS=(openpi-assets assets .venv)

for link in "${LINKS[@]}"; do
    src="$MAIN_REPO/$link"
    dst="$WORKTREE/$link"

    if [ ! -e "$src" ] && [ ! -L "$src" ]; then
        echo "[skip] $link: missing in main repo"
        continue
    fi

    if [ -L "$src" ]; then
        # Main repo entry is itself a symlink (e.g. openpi-assets, assets).
        # Clone the same target into the worktree.
        target=$(readlink "$src")

        if [ -L "$dst" ] || [ -e "$dst" ]; then
            echo "[skip] $link: already exists in worktree"
            continue
        fi

        ln -s "$target" "$dst"
        echo "[ok]   $link -> $target (cloned main symlink)"
    else
        # Main repo entry is a real directory / file (e.g. .venv).
        # Symlink the worktree entry to the main-repo path.

        if [ -L "$dst" ]; then
            echo "[skip] $link: already a symlink in worktree"
            continue
        fi

        if [ -e "$dst" ]; then
            # Only replace if the existing dir looks like an empty
            # bootstrap stub (small). Anything larger means the user
            # has installed real content and we refuse to touch it.
            size_kb=$(du -sk "$dst" 2>/dev/null | awk '{print $1}')
            if [ -n "$size_kb" ] && [ "$size_kb" -lt "$STUB_VENV_MAX_KB" ]; then
                rm -rf "$dst"
                ln -s "$src" "$dst"
                echo "[ok]   $link -> $src (replaced ${size_kb}KB bootstrap stub)"
            else
                echo "[skip] $link: worktree has a ${size_kb:-?}KB real directory, not replacing"
            fi
        else
            ln -s "$src" "$dst"
            echo "[ok]   $link -> $src (symlinked to main)"
        fi
    fi
done
