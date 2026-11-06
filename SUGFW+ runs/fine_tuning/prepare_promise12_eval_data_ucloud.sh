#!/usr/bin/env bash
set -Eeuo pipefail

# Complete the official PROMISE12 splits in the existing UCloud workspace.
# The earlier selection setup downloaded only training_data.zip; the upstream
# preprocessor also needs the public live-challenge and test archives to build
# non-empty validation/test CSVs.
WORK_ROOT="${WORK_ROOT:-/work/ATDL3}"
REPO_DIR="$WORK_ROOT/repos/SUGFW-plus"
RAW_ROOT="$REPO_DIR/data/Promise12/raw_data"
DATA_ROOT="$REPO_DIR/data/Promise12/Foreslices"
PYTHON="$WORK_ROOT/envs/sugfw/bin/python"
DOWNLOADS="$WORK_ROOT/downloads"

LIVE_URL="https://zenodo.org/records/8026660/files/livechallenge_test_data.zip?download=1"
LIVE_MD5="08f69b5586b9cf516a7102839f05c7f3"
TEST_URL="https://zenodo.org/records/8026660/files/test_data.zip?download=1"
TEST_MD5="823ef560f4a54083348e58d67403e4bb"

for path in "$PYTHON" "$REPO_DIR/preprocessing/Promise12_pre.py" \
            "$REPO_DIR/preprocessing/Promise12_csv.py" \
            "$RAW_ROOT/training_data" "$WORK_ROOT/finetuning_tools/run_five_list_finetuning.py"; do
  if [[ ! -e "$path" ]]; then
    echo "ERROR: Required existing SUGFW+ path is missing: $path" >&2
    exit 2
  fi
done
for tool in wget unzip md5sum; do
  command -v "$tool" >/dev/null 2>&1 || {
    echo "ERROR: Required command is missing: $tool" >&2
    exit 3
  }
done

mkdir -p "$DOWNLOADS" "$RAW_ROOT/livechallenge_test_data" "$RAW_ROOT/test_data"

download_checked() {
  local name="$1" url="$2" expected_md5="$3"
  local archive="$DOWNLOADS/$name"
  local partial="$archive.download"
  if [[ -s "$archive" ]] && echo "$expected_md5  $archive" | md5sum --check --status; then
    echo "Verified cached archive: $archive"
    return
  fi
  echo "Downloading $name from the PROMISE12 Zenodo record..."
  # Re-download to a temporary file rather than append an unverifiable partial.
  wget --output-document "$partial" "$url"
  if ! echo "$expected_md5  $partial" | md5sum --check --status; then
    echo "ERROR: Checksum mismatch for $name; archive was not extracted." >&2
    exit 4
  fi
  mv -f "$partial" "$archive"
  echo "Verified $archive"
}

archive_has_expected_cases() {
  local archive="$1" expected="$2" name="$3"
  local images labels raw_files missing_raw=0 header
  images="$(find "$archive" -type f -iname '*.mhd' ! -iname '*_segmentation.mhd' | wc -l)"
  labels="$(find "$archive" -type f -iname '*_segmentation.mhd' | wc -l)"
  raw_files="$(find "$archive" -type f -iname '*.raw' | wc -l)"
  while IFS= read -r header; do
    [[ -n "$header" ]] || continue
    if [[ ! -f "${header%.mhd}.raw" ]]; then
      missing_raw=$((missing_raw + 1))
    fi
  done < <(find "$archive" -type f -iname '*.mhd')
  if [[ "$images" -ne "$expected" || "$labels" -ne "$expected" || \
        "$raw_files" -ne "$((expected * 2))" || "$missing_raw" -ne 0 ]]; then
    echo "ERROR: $name must contain $expected image/mask pairs with their .raw files; found $images images, $labels masks, $raw_files raw files, and $missing_raw missing .raw companions." >&2
    echo "No files from this archive were copied into raw_data." >&2
    exit 5
  fi
}

count_pairs() {
  local dir="$1"
  local images labels
  images="$(find "$dir" -maxdepth 1 -type f -iname '*.mhd' ! -iname '*_segmentation.mhd' | wc -l)"
  labels="$(find "$dir" -maxdepth 1 -type f -iname '*_segmentation.mhd' | wc -l)"
  echo "$images $labels"
}

validate_raw_pairs() {
  local dir="$1" expected="$2" name="$3"
  local images labels raw_files
  images="$(find "$dir" -maxdepth 1 -type f -iname '*.mhd' ! -iname '*_segmentation.mhd' | wc -l)"
  labels="$(find "$dir" -maxdepth 1 -type f -iname '*_segmentation.mhd' | wc -l)"
  raw_files="$(find "$dir" -maxdepth 1 -type f -iname '*.raw' | wc -l)"
  if [[ "$images" -ne "$expected" || "$labels" -ne "$expected" || \
        "$raw_files" -ne "$((expected * 2))" ]]; then
    echo "ERROR: $name in raw_data is incomplete: expected $expected images, $expected masks, and $((expected * 2)) .raw files; found $images, $labels, and $raw_files." >&2
    exit 7
  fi
  while IFS= read -r header; do
    [[ -n "$header" ]] || continue
    if [[ ! -f "${header%.mhd}.raw" ]]; then
      echo "ERROR: Missing raw-data companion for $header" >&2
      exit 7
    fi
  done < <(find "$dir" -maxdepth 1 -type f -iname '*.mhd')
}

read -r live_images live_labels < <(count_pairs "$RAW_ROOT/livechallenge_test_data")
if [[ "$live_images" -ne 20 || "$live_labels" -ne 20 ]]; then
  download_checked livechallenge_test_data.zip "$LIVE_URL" "$LIVE_MD5"
  STAGE_DIR="$(mktemp -d "$DOWNLOADS/promise12_eval.XXXXXX")"
  case "$STAGE_DIR" in
    "$DOWNLOADS"/promise12_eval.*) ;;
    *) echo "ERROR: Unsafe temporary extraction path: $STAGE_DIR" >&2; exit 6 ;;
  esac
  unzip -q "$DOWNLOADS/livechallenge_test_data.zip" -d "$STAGE_DIR/livechallenge"
  archive_has_expected_cases "$STAGE_DIR/livechallenge" 20 "livechallenge_test_data.zip"
  find "$STAGE_DIR/livechallenge" -type f \( -iname '*.mhd' -o -iname '*.raw' \) \
    -exec cp -f {} "$RAW_ROOT/livechallenge_test_data/" \;
  rm -rf -- "$STAGE_DIR"
else
  echo "livechallenge_test_data already contains 20 image/segmentation pairs."
fi

read -r test_images test_labels < <(count_pairs "$RAW_ROOT/test_data")
if [[ "$test_images" -ne 30 || "$test_labels" -ne 30 ]]; then
  download_checked test_data.zip "$TEST_URL" "$TEST_MD5"
  STAGE_DIR="$(mktemp -d "$DOWNLOADS/promise12_eval.XXXXXX")"
  case "$STAGE_DIR" in
    "$DOWNLOADS"/promise12_eval.*) ;;
    *) echo "ERROR: Unsafe temporary extraction path: $STAGE_DIR" >&2; exit 6 ;;
  esac
  unzip -q "$DOWNLOADS/test_data.zip" -d "$STAGE_DIR/test"
  archive_has_expected_cases "$STAGE_DIR/test" 30 "test_data.zip"
  find "$STAGE_DIR/test" -type f \( -iname '*.mhd' -o -iname '*.raw' \) \
    -exec cp -f {} "$RAW_ROOT/test_data/" \;
  rm -rf -- "$STAGE_DIR"
else
  echo "test_data already contains 30 image/segmentation pairs."
fi

validate_raw_pairs "$RAW_ROOT/livechallenge_test_data" 20 livechallenge_test_data
validate_raw_pairs "$RAW_ROOT/test_data" 30 test_data

echo "Running the upstream Promise12 preprocessing and CSV generation..."
(
  cd "$REPO_DIR/preprocessing"
  "$PYTHON" Promise12_pre.py
  "$PYTHON" Promise12_csv.py
)

"$PYTHON" - "$DATA_ROOT/splits" <<'PY'
from pathlib import Path
import pandas as pd
import sys

split_dir = Path(sys.argv[1])
expected = {"train.csv": 960, "valid.csv": None, "test.csv": None}
counts = {}
for name in expected:
    path = split_dir / name
    if not path.is_file():
        raise SystemExit(f"ERROR: preprocessing did not create {path}")
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise SystemExit(f"ERROR: {path} is empty; no labeled slices were produced") from exc
    if frame.empty or not {"image_pth", "mask_pth"}.issubset(frame.columns):
        raise SystemExit(f"ERROR: {path} must contain non-empty image_pth and mask_pth columns")
    counts[name] = len(frame)
if counts["train.csv"] != expected["train.csv"]:
    raise SystemExit(f"ERROR: train.csv changed from the expected 960 rows to {counts['train.csv']}")
for name in ("valid.csv", "test.csv"):
    frame = pd.read_csv(split_dir / name)
    cases = {Path(p).name.split("_Slice")[0] for p in frame["image_pth"]}
    if not cases:
        raise SystemExit(f"ERROR: {name} has no cases")
    print(f"{name}: {counts[name]} slices across {len(cases)} cases ({min(cases)}–{max(cases)})")
print(f"train.csv: {counts['train.csv']} slices (original selection pool retained)")
PY

echo "PROMISE12 evaluation splits are ready. Run initialize_finetuning_ucloud.sh, then the fine-tuning runner."
