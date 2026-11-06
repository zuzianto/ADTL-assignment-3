#!/usr/bin/env python3
"""Fine-tune and evaluate SUGFW+ models across lists and training seeds.

The crossed comparison uses the same training seeds and validation/test data
for all five lists. It stages files with symlinks without altering PROMISE12
data or selection-run caches. Outputs and per-run status are written
incrementally to make completed runs reusable if UCloud interrupts the job.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LABELS = ("reference", "rerun_1", "rerun_2", "rerun_3", "rerun_4")
UPSTREAM_COMMIT = "9b7776f33f6d581665a2d91ca9abbe628a9d9eaa"
INFERENCE_RESIZE_MARKER = "# ATDL3 evaluation alignment fix"
INFERENCE_ORIGINAL_BLOCK = '''        if image.shape[1] != args.img_size:
            zoom_factor = (
                1.0,
                args.img_size / image.shape[1],
                args.img_size / image.shape[2],
                1.0,
            )
            image = zoom(image, zoom_factor, order=3)
'''
INFERENCE_FIXED_BLOCK = '''        if image.shape[1] != args.img_size:
            zoom_factor = (
                1.0,
                args.img_size / image.shape[1],
                args.img_size / image.shape[2],
                1.0,
            )
            image = zoom(image, zoom_factor, order=3)
        # ATDL3 evaluation alignment fix: resize labels with nearest-neighbor
        # interpolation when the evaluation image is resized.
        if label.shape[1] != args.img_size or label.shape[2] != args.img_size:
            label_zoom_factor = (
                1.0,
                args.img_size / label.shape[1],
                args.img_size / label.shape[2],
                1.0,
            )
            label = zoom(label, label_zoom_factor, order=0)
'''
HD95_ORIGINAL_LINE = "        HD95 = hd95(preds, label, voxelspacing=spacing) if cal_hd95 else 10000\n"
HD95_SAFE_BLOCK = '''        if cal_hd95:
            if not np.any(preds) and not np.any(label):
                HD95 = 0.0
            elif not np.any(preds) or not np.any(label):
                # Undefined if exactly one mask is empty; retain Dice and
                # report this HD95 case as NaN rather than aborting the run.
                HD95 = float("nan")
            else:
                HD95 = hd95(preds, label, voxelspacing=spacing)
        else:
            HD95 = 10000
'''
CASE_METRIC_RE = re.compile(
    r"(Case\d{3})\s*-\s*Dice:\s*([0-9]*\.?[0-9]+),\s*HD95:\s*(nan|[0-9]*\.?[0-9]+)"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_path(value: Any, base: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def read_split(csv_path: Path, base: Path) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Required split is missing: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError as exc:
        preparation_script = Path(__file__).resolve().parent / "prepare_promise12_eval_data_ucloud.sh"
        raise ValueError(
            f"Split is empty: {csv_path}. Run {preparation_script} to download the official "
            "PROMISE12 evaluation data and regenerate the validation/test CSVs, then rerun preflight."
        ) from exc
    required = {"image_pth", "mask_pth"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{csv_path} must contain {sorted(required)}")
    if frame.empty:
        raise ValueError(f"Split is empty: {csv_path}")
    rows: list[dict[str, str]] = []
    for index, row in frame.iterrows():
        image = resolve_path(row["image_pth"], base)
        mask = resolve_path(row["mask_pth"], base)
        if not image.is_file():
            raise FileNotFoundError(f"Missing image in {csv_path}, row {index + 2}: {image}")
        if not mask.is_file():
            raise FileNotFoundError(f"Missing mask in {csv_path}, row {index + 2}: {mask}")
        rows.append({"image_pth": str(image), "mask_pth": str(mask)})
    result = pd.DataFrame(rows)
    if result["image_pth"].duplicated().any():
        raise ValueError(f"Duplicate image paths in {csv_path}")
    return result


def relative_symlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve(strict=False) == source.resolve(strict=False):
            return
        raise FileExistsError(f"Existing symlink points elsewhere: {target}")
    if target.exists():
        if target.resolve(strict=False) == source.resolve(strict=False):
            return
        raise FileExistsError(f"Refusing to overwrite existing file: {target}")
    target.symlink_to(source.resolve(strict=True))


def find_selection_run(work_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        candidate = explicit.expanduser().resolve(strict=True)
        validate_selection_run(candidate)
        return candidate
    root = work_root / "results" / "selection_stability"
    candidates = sorted(
        (path.parent for path in root.glob("run_*/run_manifest.json")),
        key=lambda path: path.name,
        reverse=True,
    )
    errors: list[str] = []
    for candidate in candidates:
        try:
            validate_selection_run(candidate)
            return candidate
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{candidate}: {exc}")
    details = "\n".join(errors[:5])
    raise FileNotFoundError(
        "No complete five-list selection run was found under "
        f"{root}. Set --selection-run-dir explicitly.\n{details}"
    )


def validate_selection_run(path: Path) -> None:
    manifest_path = path / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No run_manifest.json in {path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid selection manifest: {manifest_path}") from exc
    list_dir = path / "lists"
    for label in LABELS:
        list_path = list_dir / f"{label}.csv"
        cache_dir = path / "cache" / label
        uncertainty_dir = cache_dir / "uncertainty"
        if not list_path.is_file():
            raise FileNotFoundError(f"Missing list: {list_path}")
        frame = pd.read_csv(list_path)
        if len(frame) != 28 or "image_pth" not in frame.columns:
            raise ValueError(f"{list_path} must have 28 rows and image_pth")
        if frame["image_pth"].duplicated().any():
            raise ValueError(f"Duplicate selections in {list_path}")
        if not (cache_dir / "image").is_dir() or not uncertainty_dir.is_dir():
            raise FileNotFoundError(f"Missing cache directories for {label}: {cache_dir}")
        map_count = len(list(uncertainty_dir.glob("*_uncertainty_10.pkl")))
        if map_count < 960:
            raise ValueError(
                f"Expected at least 960 cached uncertainty maps for {label}; found {map_count}"
            )
    expected = manifest.get("settings", {}).get("expected_budget")
    if expected is not None and int(expected) != 28:
        raise ValueError(f"Selection manifest budget is {expected}, expected 28")


def patch_inference_alignment(inference_path: Path) -> dict[str, str]:
    source = inference_path.read_text(encoding="utf-8")
    if INFERENCE_RESIZE_MARKER in source:
        return {"status": "already_patched", "path": str(inference_path)}
    if INFERENCE_ORIGINAL_BLOCK not in source:
        raise RuntimeError(
            "Could not identify the expected upstream inference resize block; "
            "no upstream source was modified. Inspect inference.py manually."
        )
    backup_path = inference_path.with_name("inference.py.atdl3_backup")
    if not backup_path.exists():
        shutil.copy2(inference_path, backup_path)
    patched = source.replace(INFERENCE_ORIGINAL_BLOCK, INFERENCE_FIXED_BLOCK, 1)
    if HD95_ORIGINAL_LINE not in patched:
        raise RuntimeError(
            "Could not identify the expected upstream binary HD95 line; "
            "the backup was saved but inference.py was not changed."
        )
    patched = patched.replace(HD95_ORIGINAL_LINE, HD95_SAFE_BLOCK, 1)
    inference_path.write_text(patched, encoding="utf-8")
    return {
        "status": "patched_label_resize",
        "path": str(inference_path),
        "backup": str(backup_path),
    }


def common_average_uncertainty(
    uncertainty_dirs: list[Path], output_path: Path, expected_count_per_dir: int
) -> dict[str, Any]:
    maps: list[Path] = []
    counts_by_dir: dict[str, int] = {}
    for directory in uncertainty_dirs:
        directory_maps = sorted(directory.glob("*_uncertainty_10.pkl"))
        counts_by_dir[str(directory)] = len(directory_maps)
        if len(directory_maps) != expected_count_per_dir:
            raise ValueError(
                f"Expected {expected_count_per_dir} cached maps; found "
                f"{len(directory_maps)} in {directory}"
            )
        maps.extend(directory_maps)
    total: np.ndarray | None = None
    for index, path in enumerate(maps, start=1):
        with path.open("rb") as stream:
            array = np.asarray(pickle.load(stream), dtype=np.float64)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D uncertainty map in {path}; got {array.shape}")
        if total is None:
            total = np.zeros_like(array, dtype=np.float64)
        if array.shape != total.shape:
            raise ValueError(f"Uncertainty map shape mismatch at {path}: {array.shape}")
        total += array
        if index % 100 == 0 or index == len(maps):
            print(f"Preparing shared uncertainty prior: {index}/{len(maps)} maps", flush=True)
    assert total is not None
    mean_map = (total / len(maps)).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(mean_map, stream)
    return {
        "path": str(output_path),
        "source": "mean of cached uncertainty maps from all five selection runs, training pool only",
        "number_of_maps": len(maps),
        "maps_per_selection_run": counts_by_dir,
        "shape": list(mean_map.shape),
        "dtype": str(mean_map.dtype),
    }


def build_train_aliases(
    *,
    label: str,
    selected_csv: Path,
    train_pool: pd.DataFrame,
    selection_run: Path,
    arm_dir: Path,
    common_avg_path: Path,
) -> tuple[Path, int]:
    cache_root = selection_run / "cache" / label
    cached_images = cache_root / "image"
    cached_uncertainty = cache_root / "uncertainty"
    pool_by_name = {
        Path(row.image_pth).name: row.mask_pth
        for row in train_pool.itertuples(index=False)
    }
    selected = pd.read_csv(selected_csv)
    if "image_pth" not in selected.columns or len(selected) != 28:
        raise ValueError(f"Invalid 28-slice selection file: {selected_csv}")
    train_root = arm_dir / "train_data"
    rows: list[dict[str, str]] = []
    for value in selected["image_pth"]:
        name = Path(str(value)).name
        if name not in pool_by_name:
            raise ValueError(f"Selected slice is not in train.csv: {name}")
        image_target = cached_images / name
        map_source = cached_uncertainty / name.replace(
            ".pkl", "_uncertainty_10.pkl"
        )
        if not image_target.exists():
            raise FileNotFoundError(f"Cached training image link is missing: {image_target}")
        if not map_source.is_file():
            raise FileNotFoundError(f"Cached selected-slice uncertainty map is missing: {map_source}")
        mask_source = Path(pool_by_name[name])
        image_alias = train_root / "image" / name
        mask_alias = train_root / "label" / name
        map_alias = train_root / "uncertainty" / map_source.name
        relative_symlink(image_target, image_alias)
        relative_symlink(mask_source, mask_alias)
        relative_symlink(map_source, map_alias)
        rows.append({"image_pth": str(image_alias), "mask_pth": str(mask_alias)})
    avg_alias = train_root / "uncertainty" / "uncertainty_10_avg.pkl"
    relative_symlink(common_avg_path, avg_alias)
    split_path = train_root / "splits" / "train.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(split_path, index=False)
    return split_path, len(rows)


def build_eval_root(
    *,
    eval_root: Path,
    valid_pool: pd.DataFrame,
    test_pool: pd.DataFrame,
    common_avg_path: Path,
) -> dict[str, int]:
    eval_root.mkdir(parents=True, exist_ok=True)
    for split_name, frame in (("valid", valid_pool), ("test", test_pool)):
        rows: list[dict[str, str]] = []
        for row in frame.itertuples(index=False):
            image_source = Path(row.image_pth)
            mask_source = Path(row.mask_pth)
            name = image_source.name
            image_alias = eval_root / "image" / name
            mask_alias = eval_root / "label" / name
            relative_symlink(image_source, image_alias)
            relative_symlink(mask_source, mask_alias)
            rows.append({"image_pth": str(image_alias), "mask_pth": str(mask_alias)})
        split_path = eval_root / "splits" / f"{split_name}.csv"
        split_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(split_path, index=False)
    avg_alias = eval_root / "uncertainty" / "uncertainty_10_avg.pkl"
    relative_symlink(common_avg_path, avg_alias)
    return {"valid_slices": len(valid_pool), "test_slices": len(test_pool)}


def run_logged(command: list[str], cwd: Path, log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        print("COMMAND:", " ".join(command), file=stream, flush=True)
        print("WORKING_DIRECTORY:", cwd, file=stream, flush=True)
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        try:
            log_tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-60:]
        except OSError:
            log_tail = ["Could not read the subprocess log."]
        detail = "\n".join(log_tail)
        raise RuntimeError(
            f"Subprocess exited with status {result.returncode}: {' '.join(command)}\n"
            f"Full log: {log_path}\n"
            f"Last log lines:\n{detail}"
        )


def bootstrap_mean_ci(
    values: np.ndarray, seed: int, repetitions: int = 10_000
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        estimates[index] = rng.choice(values, size=values.size, replace=True).mean()
    lower, upper = np.percentile(estimates, [2.5, 97.5])
    return float(lower), float(upper)


def parse_case_metrics(log_path: Path) -> pd.DataFrame:
    if not log_path.is_file():
        raise FileNotFoundError(f"Test inference log is missing: {log_path}")
    text = log_path.read_text(encoding="utf-8", errors="replace")
    records: dict[str, tuple[float, float]] = {}
    for case, dice_text, hd95_text in CASE_METRIC_RE.findall(text):
        records[case] = (float(dice_text), float(hd95_text))
    if not records:
        raise ValueError(
            f"No per-case Dice/HD95 metrics were parsed from {log_path}. "
            "Inspect the upstream inference log before rerunning."
        )
    return pd.DataFrame(
        [
            {"case": case, "dice": values[0], "hd95_voxels": values[1]}
            for case, values in sorted(records.items())
        ]
    )


def summarize_seed(
    label: str, frame: pd.DataFrame, train_seed: int, n_train: int, seed: int
) -> dict[str, Any]:
    dice = frame["dice"].to_numpy(dtype=np.float64)
    hd95 = frame["hd95_voxels"].to_numpy(dtype=np.float64)
    valid_hd95 = hd95[np.isfinite(hd95)]
    dice_ci = bootstrap_mean_ci(dice, seed)
    hd_ci = bootstrap_mean_ci(valid_hd95, seed + 17)
    return {
        "selection_set": label,
        "training_seed": int(train_seed),
        "n_training_slices": int(n_train),
        "n_test_cases": int(len(frame)),
        "mean_test_dice": float(np.mean(dice)),
        "mean_test_dice_percent": float(np.mean(dice) * 100.0),
        "sd_test_dice": float(np.std(dice, ddof=1)) if len(dice) > 1 else 0.0,
        "sd_test_dice_percentage_points": (
            float(np.std(dice, ddof=1) * 100.0) if len(dice) > 1 else 0.0
        ),
        "dice_bootstrap_95ci_low": dice_ci[0],
        "dice_bootstrap_95ci_high": dice_ci[1],
        "dice_bootstrap_95ci_low_percent": dice_ci[0] * 100.0,
        "dice_bootstrap_95ci_high_percent": dice_ci[1] * 100.0,
        "n_hd95_valid_cases": int(len(valid_hd95)),
        "n_hd95_undefined_cases": int(len(hd95) - len(valid_hd95)),
        "mean_hd95_voxels": float(np.mean(valid_hd95)) if len(valid_hd95) else float("nan"),
        "sd_hd95_voxels": float(np.std(valid_hd95, ddof=1)) if len(valid_hd95) > 1 else 0.0,
        "hd95_bootstrap_95ci_low": hd_ci[0],
        "hd95_bootstrap_95ci_high": hd_ci[1],
    }


def summarize_lists(
    per_case: pd.DataFrame, by_seed: pd.DataFrame, seeds: list[int], bootstrap_seed: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for list_index, label in enumerate(LABELS):
        subset = per_case.loc[per_case["selection_set"] == label]
        case_means = subset.groupby("case", as_index=False).agg(
            dice=("dice", "mean"),
            hd95_voxels=("hd95_voxels", "mean"),
            n_slices=("n_slices", "first"),
        )
        case_dice = case_means["dice"].to_numpy(dtype=np.float64)
        valid_hd95 = case_means["hd95_voxels"].to_numpy(dtype=np.float64)
        valid_hd95 = valid_hd95[np.isfinite(valid_hd95)]
        seed_rows = by_seed.loc[by_seed["selection_set"] == label]
        seed_means = seed_rows["mean_test_dice"].to_numpy(dtype=np.float64)
        seed_ci = bootstrap_mean_ci(case_dice, bootstrap_seed + list_index)
        hd_ci = bootstrap_mean_ci(valid_hd95, bootstrap_seed + 100 + list_index)
        within_case_seed_sd = subset.groupby("case")["dice"].std(ddof=1)
        rows.append({
            "selection_set": label,
            "training_seeds": ",".join(map(str, seeds)),
            "n_training_seeds": len(seed_rows),
            "n_training_slices": int(seed_rows["n_training_slices"].min()),
            "n_test_cases": len(case_means),
            "mean_test_dice": float(np.mean(case_dice)),
            "mean_test_dice_percent": float(np.mean(case_dice) * 100.0),
            "sd_test_dice_across_cases": float(np.std(case_dice, ddof=1)) if len(case_dice) > 1 else 0.0,
            "dice_bootstrap_95ci_low": seed_ci[0],
            "dice_bootstrap_95ci_high": seed_ci[1],
            "dice_bootstrap_95ci_low_percent": seed_ci[0] * 100.0,
            "dice_bootstrap_95ci_high_percent": seed_ci[1] * 100.0,
            "mean_seed_specific_dice": float(np.mean(seed_means)),
            "sd_across_training_seed_means": float(np.std(seed_means, ddof=1)) if len(seed_means) > 1 else 0.0,
            "mean_within_case_sd_across_seeds": float(within_case_seed_sd.mean()),
            "n_hd95_valid_cases": int(len(valid_hd95)),
            "n_hd95_undefined_cases": int(len(case_means) - len(valid_hd95)),
            "mean_hd95_voxels": float(np.mean(valid_hd95)) if len(valid_hd95) else float("nan"),
            "sd_hd95_voxels_across_cases": float(np.std(valid_hd95, ddof=1)) if len(valid_hd95) > 1 else 0.0,
            "hd95_bootstrap_95ci_low": hd_ci[0],
            "hd95_bootstrap_95ci_high": hd_ci[1],
        })
    return pd.DataFrame(rows)


def plot_results(
    per_case: pd.DataFrame, summaries: pd.DataFrame, by_seed: pd.DataFrame, output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [label for label in LABELS if label in set(per_case["selection_set"])]
    case_average = per_case.groupby(["selection_set", "case"], as_index=False)["dice"].mean()
    samples = [
        case_average.loc[case_average["selection_set"] == label, "dice"].to_numpy()
        for label in labels
    ]
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.boxplot(samples, positions=np.arange(1, len(labels) + 1), widths=0.52,
               showfliers=False, patch_artist=True,
               boxprops={"facecolor": "#d9e7f5", "edgecolor": "#345b7e"},
               medianprops={"color": "#a63603", "linewidth": 1.5})
    rng = np.random.default_rng(20261103)
    for position, label, values in zip(range(1, len(labels) + 1), labels, samples):
        jitter = rng.uniform(-0.09, 0.09, size=len(values))
        ax.scatter(position + jitter, values, s=18, alpha=0.55, color="#314f6f")
    summary_by_label = summaries.set_index("selection_set")
    means = np.array([summary_by_label.loc[label, "mean_test_dice"] for label in labels])
    lower = np.array([summary_by_label.loc[label, "dice_bootstrap_95ci_low"] for label in labels])
    upper = np.array([summary_by_label.loc[label, "dice_bootstrap_95ci_high"] for label in labels])
    ax.errorbar(
        np.arange(1, len(labels) + 1), means,
        yerr=np.vstack((means - lower, upper - means)),
        fmt="D", color="#b23a48", capsize=4, markersize=5,
        label="Mean Dice with 95% bootstrap CI",
    )
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set_ylabel("Per-volume test Dice")
    ax.set_title("Test performance across the five SUGFW+ selection lists")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "dice_by_selection_set.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    seed_values = sorted(by_seed["training_seed"].unique())
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(seed_values), 2)))
    seed_color = {seed: colors[i] for i, seed in enumerate(seed_values)}
    x_positions = np.arange(1, len(labels) + 1)
    for position, label in zip(x_positions, labels):
        arm = by_seed.loc[by_seed["selection_set"] == label].sort_values("training_seed")
        arm_x = np.linspace(position - 0.11, position + 0.11, len(arm))
        means = arm["mean_test_dice"].to_numpy()
        ax.plot(arm_x, means, color="#687786", linewidth=1.0, zorder=1)
        for x, (_, row) in zip(arm_x, arm.iterrows()):
            ax.scatter(
                x, row["mean_test_dice"], s=70, color=seed_color[row["training_seed"]],
                edgecolor="white", linewidth=0.7, zorder=2,
                label=f"Training seed {row['training_seed']}" if label == labels[0] else None,
            )
    ax.set_xticks(x_positions, labels)
    ax.set_ylabel("Mean per-volume test Dice")
    ax.set_title("Training-seed variation within each selection list")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "dice_by_selection_and_seed.png", dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=Path("/work/ATDL3"))
    parser.add_argument("--selection-run-dir", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-seeds", type=str, default="1233,1234")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-gpu", type=int, default=1)
    parser.add_argument("--expected-pool-size", type=int, default=960)
    parser.add_argument("--expected-budget", type=int, default=28)
    args = parser.parse_args()
    try:
        training_seeds = [int(value.strip()) for value in args.training_seeds.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("--training-seeds must be comma-separated integers, e.g. 1233,1234") from exc
    if not training_seeds or len(training_seeds) != len(set(training_seeds)):
        raise ValueError("Provide at least one unique training seed")

    work_root = args.work_root.expanduser().resolve(strict=True)
    repo_dir = work_root / "repos" / "SUGFW-plus"
    ma_dir = repo_dir / "MA-SAM-un"
    python = work_root / "envs" / "sugfw" / "bin" / "python"
    data_root = repo_dir / "data" / "Promise12" / "Foreslices"
    checkpoint = work_root / "checkpoints" / "sam_vit_b_01ec64.pth"
    run_dir = args.run_dir.expanduser().resolve(strict=False)
    selection_run = find_selection_run(work_root, args.selection_run_dir)
    if not python.is_file() or not (ma_dir / "train.py").is_file():
        raise FileNotFoundError("The initialized SUGFW+ environment or checkout is missing")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint is missing: {checkpoint}")
    if args.epochs < 1 or args.batch_size < 1 or args.n_gpu != 1:
        raise ValueError("Use positive epochs/batch size and n_gpu=1 for this UCloud bundle")
    if run_dir.exists() and any(run_dir.iterdir()):
        previous_manifest = run_dir / "experiment_manifest.json"
        if not previous_manifest.is_file():
            raise FileExistsError(
                f"Run output is not empty and has no experiment manifest: {run_dir}. "
                "Choose a fresh --run-dir."
            )
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        previous_training = previous.get("training", {})
        if (
            previous.get("selection_run_dir") != str(selection_run)
            or previous_training.get("seeds") != training_seeds
            or previous_training.get("epochs") != args.epochs
            or previous_training.get("batch_size") != args.batch_size
        ):
            raise ValueError(
                f"Existing run {run_dir} uses different inputs/settings. "
                "Choose a fresh --run-dir rather than mixing experiments."
            )
        print(f"Resuming matching fine-tuning run: {run_dir}", flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    train_pool = read_split(data_root / "splits" / "train.csv", ma_dir)
    valid_pool = read_split(data_root / "splits" / "valid.csv", ma_dir)
    test_pool = read_split(data_root / "splits" / "test.csv", ma_dir)
    if len(train_pool) != args.expected_pool_size:
        raise ValueError(
            f"train.csv has {len(train_pool)} rows, expected {args.expected_pool_size}"
        )
    if int(len(train_pool) * 0.03) != args.expected_budget:
        raise ValueError("Expected 3% of the training pool to equal the 28-slice budget")

    selection_manifest = json.loads(
        (selection_run / "run_manifest.json").read_text(encoding="utf-8")
    )
    inference_path = ma_dir / "inference.py"
    inference_patch = patch_inference_alignment(inference_path)
    average_path = run_dir / "shared_uncertainty" / "uncertainty_10_avg.pkl"
    average_manifest = common_average_uncertainty(
        [selection_run / "cache" / label / "uncertainty" for label in LABELS],
        average_path,
        expected_count_per_dir=args.expected_pool_size,
    )
    eval_root = run_dir / "shared_evaluation_data"
    eval_counts = build_eval_root(
        eval_root=eval_root,
        valid_pool=valid_pool,
        test_pool=test_pool,
        common_avg_path=average_path,
    )

    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    manifest: dict[str, Any] = {
        "started_utc": utc_now(),
        "work_root": str(work_root),
        "selection_run_dir": str(selection_run),
        "selection_run_id": selection_run.name,
        "selection_provenance": selection_manifest.get("runs", []),
        "upstream_commit_expected": UPSTREAM_COMMIT,
        "upstream_commit_actual": git_commit,
        "data_splits": {
            "training_pool_slices": len(train_pool),
            "selected_budget_per_list": args.expected_budget,
            "validation_slices": eval_counts["valid_slices"],
            "test_slices": eval_counts["test_slices"],
            "test_cases": int(test_pool["image_pth"].map(lambda x: Path(x).name.split("_Slice")[0]).nunique()),
        },
        "training": {
            "seeds": training_seeds,
            "number_of_runs": len(LABELS) * len(training_seeds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "n_gpu": args.n_gpu,
            "img_size": 512,
            "base_lr": 0.0008,
            "vit_name": "vit_b_new",
            "rank": 32,
            "model_scale": 1.0,
            "checkpoint_selection": "best validation Dice at epochs 100, 200, or 300",
        },
        "shared_uncertainty_prior": average_manifest,
        "upstream_inference_patch": inference_patch,
        "interpretation": (
            "The same training seeds are used for all five selection lists. "
            "The crossed list-by-seed design supports descriptive comparison of "
            "between-list and within-list training-seed variation. With only two "
            "training seeds, the training variance estimate remains imprecise. "
            "Bootstrap CIs are over held-out test cases, not training seeds."
        ),
        "arms": {},
    }
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(f"Selection run: {selection_run}", flush=True)
    print(f"Fine-tuning outputs: {run_dir}", flush=True)
    print(f"Train pool: {len(train_pool)}; budget/list: {args.expected_budget}", flush=True)
    print(f"Validation/test rows: {len(valid_pool)}/{len(test_pool)}", flush=True)
    print(f"Training seeds: {training_seeds}; total runs: {len(LABELS) * len(training_seeds)}", flush=True)
    print(f"Epochs per run: {args.epochs}; batch: {args.batch_size}", flush=True)

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env["PYTHONUNBUFFERED"] = "1"
    all_case_frames: list[pd.DataFrame] = []
    seed_summary_rows: list[dict[str, Any]] = []
    run_index = 0
    total_runs = len(LABELS) * len(training_seeds)
    for arm_index, label in enumerate(LABELS):
        print(f"\n===== Selection list [{arm_index + 1}/5]: {label} =====", flush=True)
        arm_dir = run_dir / label
        arm_dir.mkdir(parents=True, exist_ok=True)
        train_csv, n_train = build_train_aliases(
            label=label,
            selected_csv=selection_run / "lists" / f"{label}.csv",
            train_pool=train_pool,
            selection_run=selection_run,
            arm_dir=arm_dir,
            common_avg_path=average_path,
        )
        manifest["arms"][label] = {
            "training_csv": str(train_csv),
            "n_training_slices": n_train,
            "seeds": {},
        }
        for seed_index, training_seed in enumerate(training_seeds):
            run_index += 1
            seed_dir = arm_dir / f"seed_{training_seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            status_path = seed_dir / "status.json"
            status: dict[str, Any] = {
                "selection_set": label,
                "training_seed": training_seed,
                "started_utc": utc_now(),
                "status": "preparing",
            }
            write_json(status_path, status)
            model_dir = seed_dir / "model"
            checkpoint_path = model_dir / "train_best_model.pth"
            train_complete_marker = seed_dir / "training_complete.json"
            train_log = seed_dir / "logs" / "train.log"
            test_log = seed_dir / "logs" / "test.log"
            per_case_path = seed_dir / "test_metrics.csv"
            model_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n===== Run [{run_index}/{total_runs}]: {label}, training seed {training_seed} =====",
                flush=True,
            )
            if not train_complete_marker.is_file():
                if checkpoint_path.exists():
                    raise RuntimeError(
                        f"Found an unmarked checkpoint for {label}, seed {training_seed}: "
                        f"{checkpoint_path}. Training may have been interrupted; use a fresh "
                        "run directory rather than evaluating a partial model."
                    )
                command = [
                    str(python), str(ma_dir / "train.py"),
                    "-b", str(args.batch_size),
                    "--n_gpu", str(args.n_gpu),
                    "--root_path", str(eval_root),
                    "--csv_path", str(train_csv),
                    "-o", str(model_dir),
                    "--seed", str(training_seed),
                    "--train_max_epochs", str(args.epochs),
                    "--deterministic", "1",
                    "--img_size", "512",
                    "--num_classes", "2",
                    "--ckpt", str(checkpoint),
                    "--vit_name", "vit_b_new",
                    "--rank", "32",
                    "--scale", "1.0",
                ]
                status.update({"status": "training", "train_started_utc": utc_now()})
                write_json(status_path, status)
                run_logged(command, seed_dir, train_log, env)
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(
                        f"Upstream trainer finished without expected checkpoint: {checkpoint_path}"
                    )
                write_json(train_complete_marker, {
                    "completed_utc": utc_now(),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_rule": "best validation Dice at upstream validation checkpoints",
                })
            else:
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"Completed checkpoint is missing: {checkpoint_path}")
                print(f"Training already complete for {label}, seed {training_seed}; reusing checkpoint.", flush=True)

            if per_case_path.is_file():
                case_metrics = pd.read_csv(per_case_path)
                print(f"Test metrics already complete for {label}, seed {training_seed}; reusing them.", flush=True)
            else:
                command = [
                    str(python), str(ma_dir / "inference.py"),
                    "--adapt_ckpt", str(checkpoint_path),
                    "--root_path", str(eval_root),
                    "--stage", "test",
                    "--num_classes", "2",
                    "--img_size", "512",
                    "--seed", str(training_seed),
                    "--deterministic", "1",
                    "--ckpt", str(checkpoint),
                    "--vit_name", "vit_b_new",
                    "--rank", "32",
                    "--scale", "1.0",
                    "--cal_hd95",
                ]
                status.update({"status": "testing", "test_started_utc": utc_now()})
                write_json(status_path, status)
                run_logged(command, seed_dir, test_log, env)
                case_metrics = parse_case_metrics(test_log)
                expected_cases = set(
                    test_pool["image_pth"].map(lambda value: Path(value).name.split("_Slice")[0])
                )
                found_cases = set(case_metrics["case"])
                if expected_cases != found_cases:
                    raise ValueError(
                        f"Test case mismatch for {label}, seed {training_seed}. "
                        f"Missing={sorted(expected_cases-found_cases)}; "
                        f"unexpected={sorted(found_cases-expected_cases)}"
                    )
                slices_per_case = test_pool["image_pth"].map(
                    lambda value: Path(value).name.split("_Slice")[0]
                ).value_counts()
                case_metrics["n_slices"] = case_metrics["case"].map(slices_per_case).astype(int)
                case_metrics.to_csv(per_case_path, index=False)

            case_metrics["selection_set"] = label
            case_metrics["training_seed"] = training_seed
            if "n_slices" not in case_metrics:
                slices_per_case = test_pool["image_pth"].map(
                    lambda value: Path(value).name.split("_Slice")[0]
                ).value_counts()
                case_metrics["n_slices"] = case_metrics["case"].map(slices_per_case).astype(int)
            case_metrics.to_csv(per_case_path, index=False)
            all_case_frames.append(case_metrics)
            seed_summary = summarize_seed(
                label, case_metrics, training_seed, n_train,
                training_seed + arm_index * 100 + seed_index,
            )
            seed_summary_rows.append(seed_summary)
            status.update({
                "status": "complete",
                "completed_utc": utc_now(),
                "summary": seed_summary,
            })
            write_json(status_path, status)
            manifest["arms"][label]["seeds"][str(training_seed)] = {
                "status": "complete",
                "checkpoint": str(checkpoint_path),
                "test_metrics_csv": str(per_case_path),
                "summary": seed_summary,
            }
            write_json(run_dir / "experiment_manifest.json", manifest)
            print(
                f"{label}, seed {training_seed}: Dice {seed_summary['mean_test_dice']:.4f} "
                f"(patient bootstrap 95% CI "
                f"{seed_summary['dice_bootstrap_95ci_low']:.4f}–"
                f"{seed_summary['dice_bootstrap_95ci_high']:.4f})",
                flush=True,
            )

    per_case = pd.concat(all_case_frames, ignore_index=True)
    by_seed = pd.DataFrame(seed_summary_rows)
    summaries = summarize_lists(per_case, by_seed, training_seeds, training_seeds[0])
    per_case.to_csv(run_dir / "per_case_metrics.csv", index=False)
    by_seed.to_csv(run_dir / "summary_by_seed.csv", index=False)
    summaries.to_csv(run_dir / "summary.csv", index=False)
    reference = per_case.loc[
        per_case["selection_set"] == "reference", ["case", "training_seed", "dice"]
    ].rename(columns={"dice": "reference_dice"})
    paired = per_case.merge(reference, on=["case", "training_seed"], how="left")
    paired["dice_difference_vs_reference"] = paired["dice"] - paired["reference_dice"]
    paired = paired.loc[paired["selection_set"] != "reference"].copy()
    paired_seed_rows: list[dict[str, Any]] = []
    for index, label in enumerate(LABELS[1:], start=1):
        for seed_index, training_seed in enumerate(training_seeds):
            values = paired.loc[
                (paired["selection_set"] == label)
                & (paired["training_seed"] == training_seed),
                "dice_difference_vs_reference",
            ].to_numpy()
            lower, upper = bootstrap_mean_ci(values, training_seed + 1000 + index)
            paired_seed_rows.append({
                "selection_set": label,
                "training_seed": training_seed,
                "n_paired_test_cases": len(values),
                "mean_dice_difference_vs_reference": float(np.mean(values)),
                "mean_dice_difference_percentage_points": float(np.mean(values) * 100.0),
                "paired_bootstrap_95ci_low": lower,
                "paired_bootstrap_95ci_high": upper,
                "paired_bootstrap_95ci_low_percentage_points": lower * 100.0,
                "paired_bootstrap_95ci_high_percentage_points": upper * 100.0,
            })
    paired_by_seed = pd.DataFrame(paired_seed_rows)
    paired_by_seed.to_csv(run_dir / "paired_vs_reference_by_seed.csv", index=False)
    paired_case_means = paired.groupby(["selection_set", "case"], as_index=False)[
        "dice_difference_vs_reference"
    ].mean()
    paired_rows: list[dict[str, Any]] = []
    for index, label in enumerate(LABELS[1:], start=1):
        values = paired_case_means.loc[
            paired_case_means["selection_set"] == label,
            "dice_difference_vs_reference",
        ].to_numpy()
        lower, upper = bootstrap_mean_ci(values, training_seeds[0] + 2000 + index)
        paired_rows.append({
            "selection_set": label,
            "training_seeds": ",".join(map(str, training_seeds)),
            "n_paired_test_cases": len(values),
            "mean_dice_difference_vs_reference": float(np.mean(values)),
            "mean_dice_difference_percentage_points": float(np.mean(values) * 100.0),
            "paired_bootstrap_95ci_low": lower,
            "paired_bootstrap_95ci_high": upper,
            "paired_bootstrap_95ci_low_percentage_points": lower * 100.0,
            "paired_bootstrap_95ci_high_percentage_points": upper * 100.0,
        })
    pd.DataFrame(paired_rows).to_csv(run_dir / "paired_vs_reference.csv", index=False)
    plot_results(per_case, summaries, by_seed, run_dir / "figures")

    overall = {
        "completed_utc": utc_now(),
        "status": "complete",
        "mean_of_five_list_mean_dice": float(summaries["mean_test_dice"].mean()),
        "sd_across_five_list_mean_dice": float(summaries["mean_test_dice"].std(ddof=1)),
        "n_selection_lists": len(summaries),
        "training_seeds": training_seeds,
        "mean_within_list_sd_across_training_seeds": float(summaries["sd_across_training_seed_means"].mean()),
        "warning": (
            "Between-list and training-seed spreads are descriptive. Two seeds per list "
            "are enough for a small variance comparison but provide an imprecise estimate "
            "of training variance. Bootstrap CIs are over test cases, not training seeds."
        ),
    }
    write_json(run_dir / "overall_summary.json", overall)
    manifest["completed_utc"] = utc_now()
    manifest["status"] = "complete"
    write_json(run_dir / "experiment_manifest.json", manifest)
    print(f"\nAll {total_runs} list-by-seed runs are complete.", flush=True)
    print(f"List-level summary: {run_dir / 'summary.csv'}", flush=True)
    print(f"Per-seed summary: {run_dir / 'summary_by_seed.csv'}", flush=True)
    print(f"Figure: {run_dir / 'figures' / 'dice_by_selection_set.png'}", flush=True)
    print(f"Seed comparison: {run_dir / 'figures' / 'dice_by_selection_and_seed.png'}", flush=True)
    print(f"Paired comparisons: {run_dir / 'paired_vs_reference.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Check the per-arm status.json files before resuming.", file=sys.stderr)
        raise SystemExit(130)
