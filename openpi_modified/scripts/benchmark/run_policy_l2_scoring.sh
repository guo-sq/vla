#!/bin/bash
# L2 打分：用 fast_mode 模型对 policy + 新 self_play + record_infer 数据跑 benchmark
# 只跑 1 个模型，只需 1 GPU，关闭可视化以加速
# 输出 tail_pred 用于 success/failure 推断

set -e

# 使用 openpi_modified 的现有 venv（避免 DLC 容器中 uv 编译 av 等依赖失败）
PYTHON="/mnt/workspace/tianwanxin/dev/openpi_modified/.venv/bin/python"
export PYTHONPATH="/mnt/workspace/tianwanxin/dev/openpi_clothes_data_audit/src:/mnt/workspace/tianwanxin/dev/openpi_clothes_data_audit/packages/openpi-client/src:/mnt/workspace/tianwanxin/dev/openpi_clothes_data_audit"
cd /mnt/workspace/tianwanxin/dev/openpi_clothes_data_audit
export PYTHONUNBUFFERED=1

BATCH_SIZE=${1:-64}

# Model: fast_mode (Sep=0.815, best performer)
CKPT="/mnt/workspace/tianwanxin/dev/openpi_modified/checkpoints/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_fast_mode_max3600/pi06_train_distributional_value_model_bipiper_clothes_t5gemma270M_bin1_bs1024_fast_mode_max3600.exp.0314/10000"
CONFIG="src/openpi/configs/cfg_pi06_value_model_bipiper_clothes_fast_mode_max3600.py"

CLOTHES_PROMPT="You are a two-armed piper robot with a total of three perspectives. Your task is to fold a T-shirt. First, look for the collar of the T-shirt. If you can't see it, pick up the T-shirt and let it fall naturally until the collar is visible. Then, grab the collar to lay the T-shirt flat. Next, simultaneously grab the collar and the bottom hem to fold it in half and lay it flat. Finally, lay it flat with the collar facing down, then fold it up and place it in the fixed position on the right."

# === Part 1: Policy repos (workspace path, 98 repos) ===
DATASET_ROOT_WS="/mnt/workspace/shared/3090_14_26/cache_huggingface/lerobot/heyuan1993/bipiper_clothes"
POLICY_REPO_IDS=(
    record.clothes.bipiper.v0112.policy.1
    record.clothes.bipiper.v0112.policy.2
    record.clothes.bipiper.v0112.policy.3
    record.clothes.bipiper.v0112.policy.4
    record.clothes.bipiper.v0112.policy.5
    record.clothes.bipiper.v0116.policy.1
    record.clothes.bipiper.v0116.policy.2
    record.clothes.bipiper.v0119.policy.1
    record.clothes.bipiper.v0119.policy.2
    record.clothes.bipiper.v0119.policy.3
    record.clothes.bipiper.v0119.policy.4
    record.clothes.bipiper.v0120.policy.1
    record.clothes.bipiper.v0120.policy.2
    record.clothes.bipiper.v0120.policy.3
    record.clothes.bipiper.v0120.policy.4
    record.clothes.bipiper.v0121.policy.1
    record.clothes.bipiper.v0121.policy.10
    record.clothes.bipiper.v0121.policy.11
    record.clothes.bipiper.v0121.policy.12
    record.clothes.bipiper.v0121.policy.2
    record.clothes.bipiper.v0121.policy.3
    record.clothes.bipiper.v0121.policy.4
    record.clothes.bipiper.v0121.policy.5
    record.clothes.bipiper.v0121.policy.6
    record.clothes.bipiper.v0121.policy.7
    record.clothes.bipiper.v0121.policy.8
    record.clothes.bipiper.v0121.policy.9
    record.clothes.bipiper.v0122.policy.1
    record.clothes.bipiper.v0122.policy.2
    record.clothes.bipiper.v0122.policy.3
    record.clothes.bipiper.v0122.policy.4
    record.clothes.bipiper.v0122.policy.5
    record.clothes.bipiper.v0122.policy.6
    record.clothes.bipiper.v0122.policy.7
    record.clothes.bipiper.v0122.policy.8
    record.clothes.bipiper.v0123.policy.1
    record.clothes.bipiper.v0123.policy.2
    record.clothes.bipiper.v0123.policy.3
    record.clothes.bipiper.v0123.policy.4
    record.clothes.bipiper.v0123.policy.5
    record.clothes.bipiper.v0123.policy.6
    record.clothes.bipiper.v0123.policy.7
    record.clothes.bipiper.v0123.policy.8
    record.clothes.bipiper.v0126.policy.1
    record.clothes.bipiper.v0126.policy.10
    record.clothes.bipiper.v0126.policy.2
    record.clothes.bipiper.v0126.policy.3
    record.clothes.bipiper.v0126.policy.4
    record.clothes.bipiper.v0126.policy.5
    record.clothes.bipiper.v0126.policy.6
    record.clothes.bipiper.v0126.policy.7
    record.clothes.bipiper.v0126.policy.8
    record.clothes.bipiper.v0126.policy.9
    record.clothes.bipiper.v0127.policy.1
    record.clothes.bipiper.v0127.policy.2
    record.clothes.bipiper.v0127.policy.3
    record.clothes.bipiper.v0127.policy.4
    record.clothes.bipiper.v0127.policy.5
    record.clothes.bipiper.v0127.policy.6
    record.clothes.bipiper.v0127.policy.7
    record.clothes.bipiper.v0127.policy.8
    record.clothes.bipiper.v0127.policy.9
    record.clothes.bipiper.v0128.policy.1
    record.clothes.bipiper.v0128.policy.2
    record.clothes.bipiper.v0128.policy.3
    record.clothes.bipiper.v0128.policy.4
    record.clothes.bipiper.v0128.policy.5
    record.clothes.bipiper.v0128.policy.6
    record.clothes.bipiper.v0128.policy.7
    record.clothes.bipiper.v0128.policy.8
    record.clothes.bipiper.v0129.policy.1
    record.clothes.bipiper.v0129.policy.10
    record.clothes.bipiper.v0129.policy.11
    record.clothes.bipiper.v0129.policy.12
    record.clothes.bipiper.v0129.policy.14
    record.clothes.bipiper.v0129.policy.2
    record.clothes.bipiper.v0129.policy.3
    record.clothes.bipiper.v0129.policy.4
    record.clothes.bipiper.v0129.policy.5
    record.clothes.bipiper.v0129.policy.6
    record.clothes.bipiper.v0129.policy.7
    record.clothes.bipiper.v0129.policy.8
    record.clothes.bipiper.v0129.policy.9
    record.clothes.bipiper.v0130.policy.1
    record.clothes.bipiper.v0130.policy.2
    record.clothes.bipiper.v0130.policy.3
    record.clothes.bipiper.v0130.policy.4
    record.clothes.bipiper.v0130.policy.5
    record.clothes.bipiper.v0130.policy.6
    record.clothes.bipiper.v0203.policy.1
    record.clothes.bipiper.v0311.policy.pi05_rpt_fast_mode_0123_0309_sp_ftb0_3c_0206_exp0309.1
    record.clothes.bipiper.v0311.policy.pi05_rpt_fast_mode_0123_0309_sp_ftb0_3c_0206_exp0309.2
    record.clothes.bipiper.v0311.policy.pi05_rpt_fast_mode_0123_0309_sp_ftb0_3c_0206_exp0309.3
    record.clothes.bipiper.v0311.policy.pi05_rpt_fast_mode_0123_0309_sp_ftb0_3c_0206_exp0309.4
    record.clothes.bipiper.v0311.policy.pi05_rpt_fast_mode_0123_0309_sp_ftb0_3c_0206_exp0309.5
    record.clothes.bipiper.v0311.policy.pi05_rpt_fast_mode_0123_0309_sp_ftb0_3c_0206_exp0309.6
    record.clothes.bipiper.v0311.policy.pi05_rpt_fast_mode_0123_0309_sp_ftb0_3c_0206_exp0309.7
)

OUTPUT_POLICY="test_results/data_audit/l2_scoring/policy_fast_mode"
echo "============================================"
echo "L2 Scoring — Policy repos"
echo "============================================"
echo "Model: fast_mode_max3600"
echo "Policy repos: ${#POLICY_REPO_IDS[@]}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output: ${OUTPUT_POLICY}"
echo "============================================"

${PYTHON} scripts/benchmark/run_benchmark.py \
    --ckpt_dir "${CKPT}" \
    --config_name "${CONFIG}" \
    --dataset_root "${DATASET_ROOT_WS}" \
    --repo_ids "${POLICY_REPO_IDS[@]}" \
    --output_dir "${OUTPUT_POLICY}" \
    --batch_size "${BATCH_SIZE}" \
    --max_vis_per_quadrant 0 \
    --override_prompt "${CLOTHES_PROMPT}"

echo ""
echo "=== Part 1 Done: Policy repos ==="

# === Part 2: New self_play v0402 + record_infer (OSS path) ===
DATASET_ROOT_OSS="/mnt/oss_data/anyverse/bipiper_clothes"
NEW_REPO_IDS=(
    # v0402 self_play (all with meta)
    self_play.clothes.bipiper.v0402.batch.1
    self_play.clothes.bipiper.v0402.batch.2
    self_play.clothes.bipiper.v0402.batch.3
    self_play.clothes.bipiper.v0402.batch.4
    self_play.clothes.bipiper.v0402.batch.5
    self_play.clothes.bipiper.v0402.batch.6
    self_play.clothes.bipiper.v0402.batch.7
    self_play.clothes.bipiper.v0402.batch.8
    self_play.clothes.bipiper.v0402.batch.9
    self_play.clothes.bipiper.v0402.batch.10
    # v0407 self_play (with meta)
    self_play.clothes.bipiper.v0407.batch.1
    self_play.clothes.bipiper.v0407.batch.4
    self_play.clothes.bipiper.v0407.batch.5
    # v0408 self_play (newly uploaded, most with meta)
    self_play.clothes.bipiper.v0408.batch.2
    self_play.clothes.bipiper.v0408.batch.3
    self_play.clothes.bipiper.v0408.batch.4
    self_play.clothes.bipiper.v0408.batch.5
    self_play.clothes.bipiper.v0408.batch.6
    self_play.clothes.bipiper.v0408.batch.7
    self_play.clothes.bipiper.v0408.batch.8
    self_play.clothes.bipiper.v0408.batch.9
    self_play.clothes.bipiper.v0408.batch.10
    self_play.clothes.bipiper.v0408.batch.11
    self_play.clothes.bipiper.v0408.batch.12
    # record_infer
    record_infer.clothes.bipiper.v0401.1
    record_infer.clothes.bipiper.v0401.2
    record_infer.clothes.bipiper.v0401.3
    record_infer.clothes.bipiper.v0401.4
    record_infer.clothes.bipiper.v0401.5
)

OUTPUT_NEW="test_results/data_audit/l2_scoring/new_data_fast_mode"
echo ""
echo "============================================"
echo "L2 Scoring — New self_play v0402 + record_infer"
echo "============================================"
echo "Repos: ${#NEW_REPO_IDS[@]}"
echo "Output: ${OUTPUT_NEW}"
echo "============================================"

${PYTHON} scripts/benchmark/run_benchmark.py \
    --ckpt_dir "${CKPT}" \
    --config_name "${CONFIG}" \
    --dataset_root "${DATASET_ROOT_OSS}" \
    --repo_ids "${NEW_REPO_IDS[@]}" \
    --output_dir "${OUTPUT_NEW}" \
    --batch_size "${BATCH_SIZE}" \
    --max_vis_per_quadrant 0 \
    --override_prompt "${CLOTHES_PROMPT}"

echo ""
echo "=== Part 2 Done: New data ==="
echo ""
echo "=== All L2 Scoring Complete ==="
