#!/usr/bin/env bash
set -Eeuo pipefail

WORK_ROOT="${WORK_ROOT:-/work/ATDL3}"
TOOLS_DIR="${TOOLS_DIR:-$WORK_ROOT/finetuning_tools}"
PYTHON="$WORK_ROOT/envs/sugfw/bin/python"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_DIR="${RUN_DIR:-$WORK_ROOT/results/selection_stability/fine_tuning/run_$RUN_STAMP}"
TRAIN_SEEDS="${TRAIN_SEEDS:-1233,1234}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [[ ! -x "$PYTHON" || ! -f "$TOOLS_DIR/run_five_list_finetuning.py" ]]; then
  echo "ERROR: Run initialize_finetuning_ucloud.sh first." >&2
  exit 2
fi
mkdir -p "$WORK_ROOT/logs" "$RUN_DIR"
LOG_FILE="$WORK_ROOT/logs/fine_tuning_$(basename "$RUN_DIR").log"

echo "Fine-tuning run directory: $RUN_DIR"
echo "Log file: $LOG_FILE"
echo "One visible GPU; batch size: $BATCH_SIZE; training seeds per list: $TRAIN_SEEDS; epochs per run: $EPOCHS"
echo "If the browser terminal disconnects, reconnect to this same UCloud job and inspect the log."

ARGS=(
  --work-root "$WORK_ROOT"
  --run-dir "$RUN_DIR"
  --training-seeds "$TRAIN_SEEDS"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --n-gpu 1
)
if [[ -n "${SELECTION_RUN_DIR:-}" ]]; then
  ARGS+=(--selection-run-dir "$SELECTION_RUN_DIR")
fi

"$PYTHON" "$TOOLS_DIR/run_five_list_finetuning.py" "${ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
