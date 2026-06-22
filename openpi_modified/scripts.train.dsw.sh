
#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com

CONFIG_NAME="$1"

source .venv/bin/activate

echo "Config Name: $CONFIG_NAME"

echo "Computing normalize statistics..."
uv run scripts/compute_norm_stats_fast.py --config-name $CONFIG_NAME

echo ""
echo "Start training..."
XLA_PYTHON_CLIENT_MEM_FRACTION=0.98 uv run scripts/train.py --config $CONFIG_NAME