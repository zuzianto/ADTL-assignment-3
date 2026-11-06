#!/usr/bin/env bash
set -Eeuo pipefail

# Lightweight preflight for the already initialized SUGFW+ UCloud workspace.
# This deliberately reuses the Python environment, clone, data, checkpoint,
# and selection caches created by the first bundle.
WORK_ROOT="${WORK_ROOT:-/work/ATDL3}"
TOOLS_DIR="${TOOLS_DIR:-$WORK_ROOT/finetuning_tools}"
PYTHON="$WORK_ROOT/envs/sugfw/bin/python"
REPO_DIR="$WORK_ROOT/repos/SUGFW-plus"
DATA_ROOT="$REPO_DIR/data/Promise12/Foreslices"
CHECKPOINT="$WORK_ROOT/checkpoints/sam_vit_b_01ec64.pth"
PREPARE_DATA="$TOOLS_DIR/prepare_promise12_eval_data_ucloud.sh"

echo "Fine-tuning preflight"
echo "WORK_ROOT=$WORK_ROOT"
for path in "$PYTHON" "$REPO_DIR/MA-SAM-un/train.py" "$REPO_DIR/MA-SAM-un/inference.py" \
            "$DATA_ROOT/splits/train.csv" "$DATA_ROOT/splits/valid.csv" \
            "$DATA_ROOT/splits/test.csv" "$CHECKPOINT" \
            "$TOOLS_DIR/run_five_list_finetuning.py" "$PREPARE_DATA"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: Required file is missing: $path" >&2
    exit 2
  fi
done

"$PYTHON" - "$DATA_ROOT/splits" "$REPO_DIR/MA-SAM-un" "$PREPARE_DATA" <<'PY'
from pathlib import Path
import sys
import pandas as pd

split_dir = Path(sys.argv[1])
base = Path(sys.argv[2])
prepare_script = Path(sys.argv[3])
expected = {"train.csv": 960, "valid.csv": None, "test.csv": None}
case_sets = {}
for name, expected_rows in expected.items():
    path = split_dir / name
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SystemExit(
            f"ERROR: {path} is empty. Run `bash {prepare_script}` "
            "to download the official PROMISE12 validation/test data and regenerate the splits."
        ) from exc
    required = {"image_pth", "mask_pth"}
    if frame.empty or not required.issubset(frame.columns):
        raise SystemExit(f"ERROR: {path} has no usable labeled slices; required columns: {sorted(required)}")
    if expected_rows is not None and len(frame) != expected_rows:
        raise SystemExit(f"ERROR: {name} has {len(frame)} rows; expected {expected_rows} to preserve the selection pool.")
    missing = []
    for column in ("image_pth", "mask_pth"):
        for value in frame[column]:
            source = Path(str(value))
            source = source if source.is_absolute() else base / source
            if not source.is_file():
                missing.append(str(source))
                if len(missing) == 3:
                    break
        if missing:
            break
    if missing:
        raise SystemExit(
            f"ERROR: {name} references missing image/mask files, e.g. {missing[0]}. "
            "Run prepare_promise12_eval_data_ucloud.sh, then this preflight again."
        )
    case_sets[name] = {Path(str(value)).name.split("_Slice")[0] for value in frame["image_pth"]}
    print(f"{name}: {len(frame)} labeled slices across {len(case_sets[name])} cases")

if case_sets["train.csv"] & case_sets["valid.csv"] or case_sets["train.csv"] & case_sets["test.csv"] or case_sets["valid.csv"] & case_sets["test.csv"]:
    raise SystemExit("ERROR: train/valid/test case splits overlap.")
print("PROMISE12 splits are non-empty, labeled, present on disk, and case-disjoint.")
PY

"$PYTHON" - <<'PY'
import importlib
import torch

for name in ("numpy", "pandas", "matplotlib", "scipy", "SimpleITK", "medpy"):
    importlib.import_module(name)
try:
    import cv2
except ImportError as exc:
    if "libGL.so.1" in str(exc):
        raise SystemExit(
            "ERROR: OpenCV cannot load libGL.so.1 in this headless UCloud image. "
            "Replace the GUI wheel with the matching headless wheel:\n"
            "  /work/ATDL3/envs/sugfw/bin/python -m pip uninstall -y opencv-python\n"
            "  /work/ATDL3/envs/sugfw/bin/python -m pip install --no-deps opencv-python-headless==4.7.0.72\n"
            "Then verify with: /work/ATDL3/envs/sugfw/bin/python -c 'import cv2; print(cv2.__version__)'"
        ) from exc
    raise
print("Python:", __import__("sys").version.split()[0])
print("PyTorch:", torch.__version__)
print("OpenCV:", cv2.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: PyTorch cannot see a CUDA GPU in this job.")
print("Visible GPUs:", torch.cuda.device_count())
PY

command -v nvidia-smi >/dev/null 2>&1 || {
  echo "ERROR: nvidia-smi is not available in this job." >&2
  exit 3
}
nvidia-smi
echo "Preflight complete. No packages, data, or upstream files were changed."
