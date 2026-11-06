#!/usr/bin/env bash
set -Eeuo pipefail

# UCloud Initialization script. Run from the persistent directory selected
# for the job; environment, data, caches, and outputs remain under this path.
WORK_ROOT="${WORK_ROOT:-/work}"
WORK_ROOT="$(cd "$WORK_ROOT" && pwd)"
cd "$WORK_ROOT"
if [[ ! -w "$WORK_ROOT" ]]; then
  echo "ERROR: The persistent UCloud folder must be mounted and writable at $WORK_ROOT." >&2
  exit 2
fi

if [[ "$WORK_ROOT" == *image* || "$WORK_ROOT" == *splits* ]]; then
  echo "ERROR: Choose a persistent directory whose path does not contain 'image' or 'splits'." >&2
  exit 2
fi

for tool in nvidia-smi git wget unzip md5sum; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: Required command is missing: $tool" >&2
    exit 4
  fi
done
nvidia-smi

ENV_DIR="$WORK_ROOT/envs/sugfw"
MINIFORGE_DIR="$WORK_ROOT/tools/miniforge3"
mkdir -p "$WORK_ROOT/downloads" "$WORK_ROOT/tools" "$WORK_ROOT/envs"

# SUGFW+ specifies Python 3.10. Some UCloud Terminal images only provide
# Python 3.12, so bootstrap a persistent Miniforge environment when needed.
PYTHON_BIN=""
if [[ -x "$ENV_DIR/bin/python" ]]; then
  ENV_VERSION="$("$ENV_DIR/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  if [[ "$ENV_VERSION" != "3.10" ]]; then
    echo "ERROR: $ENV_DIR already exists with Python $ENV_VERSION; expected Python 3.10." >&2
    exit 3
  fi
  PYTHON_BIN="$ENV_DIR/bin/python"
else
  for candidate in python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      version="$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      if [[ "$version" == "3.10" ]]; then
        PYTHON_BIN="$(command -v "$candidate")"
        break
      fi
    fi
  done
  if [[ -n "$PYTHON_BIN" ]]; then
    "$PYTHON_BIN" -m venv "$ENV_DIR"
  else
    MINIFORGE_INSTALLER="$WORK_ROOT/downloads/Miniforge3-Linux-x86_64.sh"
    if [[ ! -x "$MINIFORGE_DIR/bin/conda" ]]; then
      wget --continue --output-document "$MINIFORGE_INSTALLER" \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
      bash "$MINIFORGE_INSTALLER" -b -p "$MINIFORGE_DIR"
    fi
    "$MINIFORGE_DIR/bin/conda" create --yes --prefix "$ENV_DIR" \
      --channel conda-forge python=3.10 pip
  fi
  PYTHON_BIN="$ENV_DIR/bin/python"
fi

BUNDLE="$WORK_ROOT/sugfw_ucloud_bundle.zip"
if [[ -f "$BUNDLE" && ! -f "$WORK_ROOT/generate_and_compare_sugfw_lists.py" ]]; then
  unzip -oq "$BUNDLE" -d "$WORK_ROOT"
fi
if [[ ! -f "$WORK_ROOT/generate_and_compare_sugfw_lists.py" ]]; then
  echo "ERROR: Upload sugfw_ucloud_bundle.zip into the selected persistent directory." >&2
  exit 5
fi

REPO_DIR="$WORK_ROOT/repos/SUGFW-plus"
RAW_DIR="$REPO_DIR/data/Promise12/raw_data"
DATA_DIR="$REPO_DIR/data/Promise12"
CHECKPOINT_DIR="$WORK_ROOT/checkpoints"
TRAIN_CSV="$DATA_DIR/Foreslices/splits/train.csv"
UPSTREAM_COMMIT="9b7776f33f6d581665a2d91ca9abbe628a9d9eaa"
ZIP_PATH="$WORK_ROOT/downloads/training_data.zip"
ZIP_MD5="f6d23994117c989daf07e5291edd0aea"
ZIP_URL="https://zenodo.org/records/8026660/files/training_data.zip?download=1"
CHECKPOINT="$CHECKPOINT_DIR/sam_vit_b_01ec64.pth"
CHECKPOINT_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
mkdir -p "$WORK_ROOT/repos" "$WORK_ROOT/downloads" "$WORK_ROOT/checkpoints" \
  "$WORK_ROOT/envs" "$WORK_ROOT/results/selection_stability" "$WORK_ROOT/logs"

if [[ -f "$ENV_DIR/.sugfw_ready" && -s "$TRAIN_CSV" && -s "$CHECKPOINT" ]]; then
  echo "Persistent SUGFW+ environment and data already prepared under $WORK_ROOT."
  exit 0
fi

if [[ ! -d "$REPO_DIR/.git" ]]; then
  git clone https://github.com/HiLab-git/SUGFW-plus.git "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --tags origin
git -C "$REPO_DIR" checkout "$UPSTREAM_COMMIT"

if [[ ! -f "$ZIP_PATH" ]]; then
  wget --continue --output-document "$ZIP_PATH" "$ZIP_URL"
fi
echo "$ZIP_MD5  $ZIP_PATH" | md5sum --check --status || {
  echo "ERROR: PROMISE12 training archive checksum mismatch." >&2
  exit 6
}

mkdir -p "$RAW_DIR/training_data" "$RAW_DIR/livechallenge_test_data" "$RAW_DIR/test_data"
if ! find "$RAW_DIR/training_data" -maxdepth 1 -type f -name '*.mhd' -print -quit | grep -q .; then
  STAGE_DIR="$(mktemp -d "$WORK_ROOT/promise12-stage.XXXXXX")"
  unzip -q -o "$ZIP_PATH" -d "$STAGE_DIR"
  find "$STAGE_DIR" -type f \( -iname '*.mhd' -o -iname '*.raw' \) \
    -exec cp -f {} "$RAW_DIR/training_data/" \;
  rm -rf "$STAGE_DIR"
fi
MHD_COUNT="$(find "$RAW_DIR/training_data" -maxdepth 1 -type f -name '*.mhd' | wc -l)"
if [[ "$MHD_COUNT" -lt 50 ]]; then
  echo "ERROR: Expected at least 50 PROMISE12 training .mhd files; found $MHD_COUNT." >&2
  exit 7
fi

if [[ ! -s "$CHECKPOINT" ]]; then
  wget --continue --output-document "$CHECKPOINT" "$CHECKPOINT_URL"
fi
"$ENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
# PyTorch's current stable PyPI wheels use CUDA 13, which supports Blackwell/B200.
"$ENV_DIR/bin/python" -m pip install torch torchvision torchaudio
"$ENV_DIR/bin/python" -m pip install -e "$REPO_DIR"

if [[ ! -s "$TRAIN_CSV" ]] || \
  ! find "$DATA_DIR/Foreslices/image" -maxdepth 1 -type f -name '*.pkl' -print -quit 2>/dev/null | grep -q .; then
  (
    cd "$REPO_DIR/preprocessing"
    "$ENV_DIR/bin/python" Promise12_pre.py
    "$ENV_DIR/bin/python" Promise12_csv.py
  )
fi
if [[ ! -s "$TRAIN_CSV" ]]; then
  echo "ERROR: Training split was not created: $TRAIN_CSV" >&2
  exit 8
fi
TRAIN_ROWS="$("$ENV_DIR/bin/python" -c 'import pandas as pd,sys; print(len(pd.read_csv(sys.argv[1])))' "$TRAIN_CSV")"
echo "PROMISE12 training slices in train.csv: $TRAIN_ROWS"
if [[ "$TRAIN_ROWS" != "960" ]]; then
  echo "STOP: This experiment expects 960 training slices and selects 28 (3%)." >&2
  echo "Check preprocessing and the split before running list generation." >&2
  exit 9
fi
"$ENV_DIR/bin/python" -c 'import torch; print("PyTorch", torch.__version__, "CUDA", torch.version.cuda, "available", torch.cuda.is_available())'
if ! "$ENV_DIR/bin/python" -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "ERROR: PyTorch cannot access CUDA. Do not launch the experiment." >&2
  exit 10
fi
touch "$ENV_DIR/.sugfw_ready"
echo "Initialization complete. Persistent work root: $WORK_ROOT"
