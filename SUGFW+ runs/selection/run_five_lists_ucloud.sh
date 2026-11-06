#!/usr/bin/env bash
set -Eeuo pipefail

# UCloud Batch Processing script. Run from the same persistent directory used
# by initialize_ucloud.sh.
WORK_ROOT="${WORK_ROOT:-/work}"
WORK_ROOT="$(cd "$WORK_ROOT" && pwd)"
REPO_DIR="$WORK_ROOT/repos/SUGFW-plus"
DATA_DIR="$REPO_DIR/data/Promise12"
CHECKPOINT="$WORK_ROOT/checkpoints/sam_vit_b_01ec64.pth"
ENV_PYTHON="$WORK_ROOT/envs/sugfw/bin/python"
GENERATOR="$WORK_ROOT/generate_and_compare_sugfw_lists.py"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT_DIR="$WORK_ROOT/results/selection_stability/run_$STAMP"

mkdir -p "$WORK_ROOT/logs"
if [[ ! -x "$ENV_PYTHON" || ! -f "$GENERATOR" ]]; then
  echo "ERROR: Initialization is incomplete. Run initialize_ucloud.sh first." >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: No NVIDIA GPU is visible in this UCloud job." >&2
  exit 3
fi
nvidia-smi

LOG_FILE="$WORK_ROOT/logs/selection_lists_$STAMP.log"
"$ENV_PYTHON" "$GENERATOR" \
  --upstream-uc-sam "$REPO_DIR/UC-SAM" \
  --train-csv "$DATA_DIR/Foreslices/splits/train.csv" \
  --image-root "$DATA_DIR/Foreslices/image" \
  --path-base "$REPO_DIR/UC-SAM" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" 2>&1 | tee "$LOG_FILE"
