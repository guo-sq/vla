#!/bin/bash

trap 'kill -INT $PID' TERM INT

if [ -f ".path" ]; then
    export PATH=`cat .path`
    echo ".path=${PATH}"
fi

current_home="${HOME:-}"
runner_home="$(getent passwd "$(id -u)" | cut -d: -f6)"

if [ -z "$runner_home" ] || [ ! -d "$runner_home" ] || [ ! -w "$runner_home" ]; then
    runner_home="$PWD/.runner-home"
fi

if [ -z "$current_home" ] || [ "$current_home" = "/root" ] || [ ! -w "$current_home" ]; then
    export HOME="$runner_home"
fi

mkdir -p "$HOME"

cache_home="${XDG_CACHE_HOME:-$HOME/.cache}"
if [ ! -d "$cache_home" ] || [ ! -w "$cache_home" ]; then
    cache_home="$PWD/.runner-cache"
fi

export XDG_CACHE_HOME="$cache_home"
mkdir -p "$XDG_CACHE_HOME"

tmp_root="${TMPDIR:-$XDG_CACHE_HOME/tmp}"
mkdir -p "$tmp_root"
export TMPDIR="$tmp_root"
export TMP="$tmp_root"
export TEMP="$tmp_root"
export TEMPDIR="$tmp_root"
export TEST_TMPDIR="${TEST_TMPDIR:-$tmp_root}"

nodever="node20"

./externals/$nodever/bin/node ./bin/RunnerService.js &
PID=$!
wait $PID
trap - TERM INT
wait $PID