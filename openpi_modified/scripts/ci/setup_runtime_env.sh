#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "This script must be sourced, not executed directly." >&2
    exit 2
fi

runtime_root="${1:-}"
shared_home="${2:-}"

if [[ -z "$runtime_root" ]]; then
    echo "runtime_root is required" >&2
    return 2
fi

if [[ -z "$shared_home" ]]; then
    echo "shared_home is required" >&2
    return 2
fi

export OPENPI_RUNTIME_ROOT="$runtime_root"
export HOME="$OPENPI_RUNTIME_ROOT/home"
export OPENPI_DATA_HOME="$OPENPI_RUNTIME_ROOT/openpi-cache"
runtime_home_cache_dir="$HOME/.cache"
runtime_openpi_cache_dir="$runtime_home_cache_dir/openpi"
export TMPDIR="$OPENPI_RUNTIME_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export TEMPDIR="$TMPDIR"
export TEST_TMPDIR="$TMPDIR"
export OPENPI_TMPDIR="$TMPDIR"
export XDG_CACHE_HOME="$OPENPI_RUNTIME_ROOT/xdg-cache"
export JAX_COMPILATION_CACHE_DIR="$OPENPI_RUNTIME_ROOT/jax-cache"
export CUDA_CACHE_PATH="$OPENPI_RUNTIME_ROOT/cuda-cache"
export TRITON_CACHE_DIR="$OPENPI_RUNTIME_ROOT/triton-cache"

mkdir -p \
    "$HOME" \
    "$runtime_home_cache_dir" \
    "$TMPDIR" \
    "$XDG_CACHE_HOME/tmp" \
    "$JAX_COMPILATION_CACHE_DIR" \
    "$CUDA_CACHE_PATH" \
    "$TRITON_CACHE_DIR" \
    "$OPENPI_DATA_HOME/big_vision"

ln -sfn "$OPENPI_DATA_HOME" "$runtime_openpi_cache_dir"

cp -f \
    "$shared_home/.cache/openpi/big_vision/paligemma_tokenizer.model" \
    "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"

# Reuse the shared runner Hugging Face cache under the sandbox HOME so FASTTokenizer
# tests can load physical-intelligence/fast without hitting the network.
if [[ -d "$shared_home/.cache/huggingface" ]]; then
    mkdir -p "$HOME/.cache"
    ln -sfn "$shared_home/.cache/huggingface" "$HOME/.cache/huggingface"
fi