#!/bin/bash
# Post-sparse analysis pipeline: turn a completed chunked sparse benchmark run
# into a HEAD_PRED_RANGES prior for episode_classifier_2d.py.
#
# Usage:
#   bash scripts/benchmark/post_sparse_pipeline.sh [output_dir] [gt_labels_path]
#
# Steps:
#   1. merge_batches.py          → combine batch_NNN outputs
#   2. merge_ground_truth_labels → fuse self_play_label_qc + flatten_classification
#                                  (skipped if ground_truth_labels.json exists)
#   3. fill_head_pred_ranges     → compute p5/p25/p50/p75/p95 per category
#   4. episode_classifier_2d     → apply the learned ranges and classify every
#                                  episode (head_pred_ranges.json overrides the
#                                  module-level HEAD_PRED_RANGES constant, no
#                                  source-code edits required)
#   5. build_benchmark_v0409     → construct fold+flatten multi-task manifests
#
# The final Separation Score comparison across 4 models still needs to be run
# separately (one model at a time while the DLC GPU quota recovers).

set -euo pipefail

OUTPUT_DIR=${1:-test_results/benchmark/clothes_v0409_sparse/fast_mode_max3600}
GT_LABELS=${2:-test_results/data_audit/ground_truth_labels.json}

cd "$(dirname "$0")/../.."

if [[ ! -d "$OUTPUT_DIR" ]]; then
    echo "[post-sparse] ERROR: output dir not found: $OUTPUT_DIR"
    echo "[post-sparse] Did the sparse benchmark finish? Check wait_gpu_then_run.sh log."
    exit 1
fi

export PYTHONPATH=src:packages/openpi-client/src:.
export JAXTYPING_DISABLE=1

echo "[post-sparse] ============================================"
echo "[post-sparse] Post-sparse analysis pipeline"
echo "[post-sparse] Output dir: $OUTPUT_DIR"
echo "[post-sparse] GT labels:  $GT_LABELS"
echo "[post-sparse] ============================================"

# ---------------------------------------------------------------------------
echo ""
echo "[post-sparse] [1/3] Merging batch outputs..."
python scripts/benchmark/merge_batches.py --output-dir "$OUTPUT_DIR"

MERGED_DETAILS="$OUTPUT_DIR/metrics/episode_details.json"
MERGED_SKIPPED="$OUTPUT_DIR/skipped_repos.json"

if [[ ! -s "$MERGED_DETAILS" ]]; then
    echo "[post-sparse] ERROR: $MERGED_DETAILS is empty or missing after merge"
    exit 1
fi

# ---------------------------------------------------------------------------
echo ""
echo "[post-sparse] [2/3] Ensuring ground-truth labels exist..."
if [[ ! -f "$GT_LABELS" ]]; then
    python scripts/benchmark/merge_ground_truth_labels.py \
        --output "$GT_LABELS"
else
    echo "[post-sparse]   $GT_LABELS already exists, skipping regeneration"
fi

# ---------------------------------------------------------------------------
echo ""
echo "[post-sparse] [3/5] Computing head_pred ranges per category..."
HEAD_PRED_RANGES="test_results/data_audit/head_pred_ranges.json"
python scripts/benchmark/fill_head_pred_ranges.py \
    --episode-details "$MERGED_DETAILS" \
    --ground-truth-labels "$GT_LABELS" \
    --output "$HEAD_PRED_RANGES" \
    --lower-percentile 1.0 --upper-percentile 99.0 \
    --bootstrap-from-1d

# ---------------------------------------------------------------------------
echo ""
echo "[post-sparse] [4/5] Classifying episodes with 2D dispatch..."
EPISODE_CLASSIFICATION="test_results/data_audit/episode_classification.json"
python scripts/benchmark/episode_classifier_2d.py \
    --details "$MERGED_DETAILS" \
    --output "$EPISODE_CLASSIFICATION" \
    --head-pred-ranges "$HEAD_PRED_RANGES"

# ---------------------------------------------------------------------------
echo ""
echo "[post-sparse] [5/5] Building v0409 multi-task benchmark manifests..."
BENCHMARK_SPLIT_DIR="test_results/split/clothes_v0409"
python scripts/benchmark/build_benchmark_v0409.py \
    --episode-classification "$EPISODE_CLASSIFICATION" \
    --flatten-classification test_results/data_audit/flatten_classification.json \
    --exclusion-list test_results/data_audit/exclusion_list.json \
    --output-dir "$BENCHMARK_SPLIT_DIR" \
    --seed 42

# ---------------------------------------------------------------------------
echo ""
echo "[post-sparse] ============================================"
echo "[post-sparse] Done at $(date '+%H:%M:%S')"
echo "[post-sparse] ============================================"
echo ""
echo "Artifacts:"
echo "  - $MERGED_DETAILS"
echo "  - $MERGED_SKIPPED"
echo "  - $GT_LABELS"
echo "  - $HEAD_PRED_RANGES"
echo "  - $EPISODE_CLASSIFICATION"
echo "  - $BENCHMARK_SPLIT_DIR/"
echo ""
echo "Next step: run the fold + flatten benchmarks on additional value models"
echo "(per_task_p90 / 1215_0227_max3600 / stage2_all_0322) once GPU is free,"
echo "then compare their Separation Scores against fast_mode_max3600."
