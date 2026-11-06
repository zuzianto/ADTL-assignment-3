#!/usr/bin/env python3
"""Generate five Promise12 SUGFW+ selection lists and compare them with Jaccard.

The script runs the released UC-SAM uncertainty and selection code without
editing the upstream checkout. Each run gets its own image-path workspace so
its uncertainty maps cannot overwrite another run's maps. SAM image features
are shared across runs because they do not depend on the augmentation seed.

Run this on Linux with a working CUDA GPU, the SUGFW+ dependencies, Promise12
preprocessed slices, and the SAM ViT-B checkpoint. See the selection_stability
README for an example command and the distinction between an imported original
list and a reconstructed reference list.
"""

from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")


DEFAULT_VARIANTS = (
    ("rerun_1", 1233, 1001),
    ("rerun_2", 1234, 1002),
    ("rerun_3", 1235, 1003),
    ("rerun_4", 1236, 1004),
)


@dataclass(frozen=True)
class RunSpec:
    label: str
    augmentation_seed: int | None
    kmeans_seed: int
    provenance: str


@dataclass
class Pool:
    frame: Any
    image_root: Path
    image_by_relative_path: dict[str, str]
    mask_by_relative_path: dict[str, str]
    relative_by_image_path: dict[str, str]
    basename_to_relative_paths: dict[str, list[str]]


def parse_variant(value: str) -> tuple[str, int, int]:
    """Parse LABEL:AUGMENTATION_SEED:KMEANS_SEED."""
    fields = value.split(":")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            "Use LABEL:AUGMENTATION_SEED:KMEANS_SEED, e.g. rerun_1:1233:1001"
        )
    label = fields[0].strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise argparse.ArgumentTypeError(
            "Labels may contain only letters, numbers, underscores, and hyphens"
        )
    try:
        augmentation_seed, kmeans_seed = int(fields[1]), int(fields[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Seeds must be integers") from exc
    return label, augmentation_seed, kmeans_seed


def resolve_from_csv(value: Any, path_base: Path) -> Path:
    path = Path(str(value).strip()).expanduser()
    if not path.is_absolute():
        path = path_base / path
    return path.resolve(strict=False)


def derive_mask_path(image_path: Path) -> str:
    """Mirror the upstream image -> label path convention."""
    text = str(image_path)
    marker = f"{os.sep}image{os.sep}"
    if marker in text:
        return text.replace(marker, f"{os.sep}label{os.sep}", 1)
    return text.replace("image", "label")


def load_pool(
    csv_path: Path,
    image_root: Path,
    path_base: Path,
    expected_pool_size: int,
) -> Pool:
    import pandas as pd

    if not csv_path.is_file():
        raise FileNotFoundError(f"Training split CSV not found: {csv_path}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_root}")

    frame = pd.read_csv(csv_path)
    if "image_pth" not in frame.columns:
        raise ValueError(f"{csv_path} must have an 'image_pth' column")
    if frame.empty:
        raise ValueError(f"{csv_path} contains no images")
    if expected_pool_size and len(frame) != expected_pool_size:
        raise ValueError(
            f"The split contains {len(frame)} rows; expected {expected_pool_size}. "
            "Pass --expected-pool-size 0 to disable this check if intentional."
        )

    image_root = image_root.resolve(strict=True)
    image_by_relative_path: dict[str, str] = {}
    mask_by_relative_path: dict[str, str] = {}
    relative_by_image_path: dict[str, str] = {}
    basename_to_relative_paths: dict[str, list[str]] = {}

    canonical_images: list[str] = []
    canonical_masks: list[str] = []
    for row_number, row in frame.iterrows():
        raw_image = row.get("image_pth")
        if not isinstance(raw_image, str) or not raw_image.strip():
            raise ValueError(f"Empty image_pth at CSV row {row_number + 2}")
        image_path = resolve_from_csv(raw_image, path_base)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image from CSV row {row_number + 2} does not exist: {image_path}"
            )
        try:
            relative = image_path.relative_to(image_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Image is outside --image-root: {image_path}\n"
                f"Image root: {image_root}"
            ) from exc
        if relative in image_by_relative_path:
            raise ValueError(f"Duplicate image in the pool CSV: {relative}")
        if image_path.suffix != ".pkl":
            raise ValueError(
                "The released SUGFW+ selector expects preprocessed .pkl slices; "
                f"found {image_path.name}"
            )

        raw_mask = row.get("mask_pth") if "mask_pth" in frame.columns else None
        if isinstance(raw_mask, str) and raw_mask.strip():
            mask_path = resolve_from_csv(raw_mask, path_base)
            canonical_mask = str(mask_path)
        else:
            canonical_mask = derive_mask_path(image_path)

        image_text = str(image_path)
        image_by_relative_path[relative] = image_text
        mask_by_relative_path[relative] = canonical_mask
        relative_by_image_path[image_text] = relative
        basename_to_relative_paths.setdefault(image_path.name, []).append(relative)
        canonical_images.append(image_text)
        canonical_masks.append(canonical_mask)

    frame = frame.copy()
    frame["image_pth"] = canonical_images
    frame["mask_pth"] = canonical_masks
    return Pool(
        frame=frame,
        image_root=image_root,
        image_by_relative_path=image_by_relative_path,
        mask_by_relative_path=mask_by_relative_path,
        relative_by_image_path=relative_by_image_path,
        basename_to_relative_paths=basename_to_relative_paths,
    )


def match_reference_image(value: Any, pool: Pool) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("The reference list contains an empty image_pth")
    normalized = value.strip().replace("\\", "/")

    try:
        candidate = Path(value.strip()).expanduser().resolve(strict=False)
        relative = candidate.relative_to(pool.image_root).as_posix()
        if relative in pool.image_by_relative_path:
            return relative
    except (ValueError, OSError):
        pass

    marker = "/image/"
    if marker in normalized:
        relative = normalized.rsplit(marker, 1)[1]
        if relative in pool.image_by_relative_path:
            return relative

    basename = PurePosixPath(normalized).name
    matches = pool.basename_to_relative_paths.get(basename, [])
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Cannot uniquely map reference slice '{value}' by filename. "
            "Use paths that retain the relative path below the image directory."
        )
    raise ValueError(
        f"Reference slice '{value}' is not present in the current training pool"
    )


def load_reference_list(
    csv_path: Path, pool: Pool, expected_budget: int
) -> Any:
    import pandas as pd

    if not csv_path.is_file():
        raise FileNotFoundError(f"Reference list CSV not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    if "image_pth" not in frame.columns:
        raise ValueError(f"{csv_path} must have an 'image_pth' column")
    relative_paths = [match_reference_image(value, pool) for value in frame["image_pth"]]
    if len(relative_paths) != expected_budget:
        raise ValueError(
            f"Reference list has {len(relative_paths)} rows; expected {expected_budget}"
        )
    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("Reference list contains duplicate slices")

    frame = frame.copy()
    frame["image_pth"] = [pool.image_by_relative_path[key] for key in relative_paths]
    frame["mask_pth"] = [pool.mask_by_relative_path[key] for key in relative_paths]
    return frame


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def stage_run_inputs(
    pool: Pool,
    run_root: Path,
    shared_feature_cache: Path | None,
) -> tuple[Path, Path]:
    """Create isolated image-path aliases so upstream caches are run-specific."""
    if run_root.exists():
        raise FileExistsError(
            f"Run workspace already exists: {run_root}. Choose a new --output-dir "
            "to avoid accidentally reusing partial uncertainty maps."
        )
    if "image" in str(run_root.resolve()).lower():
        raise ValueError(
            "The output path contains the substring 'image'. The upstream code "
            "replaces that text in cache paths; choose an output path without it."
        )
    if "splits" in str(run_root.resolve()).lower():
        raise ValueError(
            "The output path contains the substring 'splits'. The upstream code "
            "replaces that text in cache paths; choose an output path without it."
        )

    staged_image_root = run_root / "image"
    staged_split_dir = run_root / "splits"
    staged_image_root.mkdir(parents=True)
    staged_split_dir.mkdir(parents=True)

    staged_images: list[str] = []
    for relative, source_text in pool.image_by_relative_path.items():
        source = Path(source_text)
        destination = staged_image_root / Path(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source)
        staged_images.append(str(destination))

    staged_frame = pool.frame.copy()
    staged_frame["image_pth"] = staged_images
    staged_frame["mask_pth"] = [
        str((run_root / "label" / Path(relative)))
        for relative in pool.image_by_relative_path
    ]
    staged_csv = staged_split_dir / "train.csv"
    staged_frame.to_csv(staged_csv, index=False)

    if shared_feature_cache is not None:
        feature_link = run_root / "feature"
        feature_link.symlink_to(shared_feature_cache, target_is_directory=True)

    return staged_csv, staged_image_root


def run_sugfw(
    *,
    run: RunSpec,
    pool: Pool,
    run_root: Path,
    staged_csv: Path,
    staged_image_root: Path,
    checkpoint: Path,
    model_type: str,
    augmentation_count: int,
    soft_scale: int,
    select_ratio: float,
    expected_budget: int,
    upstream_uc_sam: Path,
    shared_feature_cache: Path | None,
) -> tuple[Any, Path]:
    import pandas as pd
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The released uncertainty script calls .cuda(); "
            "run this script on a Linux GPU VM."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")

    sys.path.insert(0, str(upstream_uc_sam))
    try:
        import get_uncertainty as uncertainty_module
        import select_samples as selection_module
    except Exception:
        if str(upstream_uc_sam) in sys.path:
            sys.path.remove(str(upstream_uc_sam))
        raise

    if run.augmentation_seed is not None:
        seed_everything(run.augmentation_seed, torch)
    uncertainty_args = SimpleNamespace(
        model_type=model_type,
        checkpoint=str(checkpoint),
        csv_path=str(staged_csv),
        k_times_aug=augmentation_count,
    )
    print(
        f"[{run.label}] Generating uncertainty maps: "
        f"augmentation_seed={run.augmentation_seed}"
        + (" (unseeded upstream default)" if run.augmentation_seed is None else "")
    )
    uncertainty_module.get_uncertainty(args=uncertainty_args)

    original_kmeans = selection_module.KMeans

    def seeded_kmeans(*args: Any, **kwargs: Any) -> Any:
        kwargs["random_state"] = run.kmeans_seed
        return original_kmeans(*args, **kwargs)

    selection_module.KMeans = seeded_kmeans
    try:
        selector = selection_module.Sample_Selector(
            str(staged_csv), select_ratio=select_ratio
        )
        if selector.select_num != expected_budget:
            raise ValueError(
                f"Upstream selector computes a budget of {selector.select_num}, "
                f"not the expected {expected_budget}"
            )
        print(
            f"[{run.label}] Selecting {selector.select_num} slices: "
            f"KMeans random_state={run.kmeans_seed}"
        )
        selector.sugfw_select(soft_scale=soft_scale, un=augmentation_count)
    finally:
        selection_module.KMeans = original_kmeans

    staged_output = staged_csv.parent / (
        f"my_select_{expected_budget}_scale{soft_scale}_un{augmentation_count}.csv"
    )
    if not staged_output.is_file():
        raise FileNotFoundError(
            f"Upstream selector did not create the expected list: {staged_output}"
        )

    selected = pd.read_csv(staged_output)
    if "image_pth" not in selected.columns:
        raise ValueError(f"Generated list has no image_pth column: {staged_output}")
    canonical_images: list[str] = []
    canonical_masks: list[str] = []
    for value in selected["image_pth"]:
        staged_path = Path(str(value))
        try:
            relative = staged_path.relative_to(staged_image_root).as_posix()
        except ValueError as exc:
            raise ValueError(
                f"Generated path is outside the staged image root: {value}"
            ) from exc
        if relative not in pool.image_by_relative_path:
            raise ValueError(f"Generated slice is not in the source pool: {relative}")
        canonical_images.append(pool.image_by_relative_path[relative])
        canonical_masks.append(pool.mask_by_relative_path[relative])

    if len(canonical_images) != expected_budget:
        raise ValueError(
            f"Generated {len(canonical_images)} rows; expected {expected_budget}"
        )
    if len(set(canonical_images)) != expected_budget:
        raise ValueError("Generated SUGFW+ list contains duplicate slices")

    selected["image_pth"] = canonical_images
    selected["mask_pth"] = canonical_masks
    if shared_feature_cache is None:
        shared_feature_cache = run_root / "feature"
    gc.collect()
    torch.cuda.empty_cache()
    return selected, shared_feature_cache


def environment_details(torch: Any) -> dict[str, Any]:
    import numpy as np
    import pandas as pd
    import sklearn

    return {
        "python": str(sys.version),
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "numpy": str(np.__version__),
        "pandas": str(pd.__version__),
        "scikit_learn": str(sklearn.__version__),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_comparisons(
    *,
    output_dir: Path,
    labels: list[str],
    list_frames: dict[str, Any],
    pool: Pool,
) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    sets: dict[str, set[str]] = {}
    for label in labels:
        frame = list_frames[label]
        keys = []
        for image_path in frame["image_pth"]:
            canonical = str(Path(str(image_path)).resolve(strict=False))
            relative = pool.relative_by_image_path.get(canonical)
            if relative is None:
                raise ValueError(f"Cannot map output slice to the source pool: {image_path}")
            keys.append(relative)
        if len(keys) != len(set(keys)):
            raise ValueError(f"List {label} contains duplicate slices")
        sets[label] = set(keys)

    matrix = pd.DataFrame(index=labels, columns=labels, dtype=float)
    pairwise_rows: list[dict[str, Any]] = []
    for label in labels:
        matrix.loc[label, label] = 1.0
    for left, right in itertools.combinations(labels, 2):
        intersection = sets[left] & sets[right]
        union = sets[left] | sets[right]
        score = len(intersection) / len(union) if union else 1.0
        matrix.loc[left, right] = score
        matrix.loc[right, left] = score
        pairwise_rows.append(
            {
                "list_a": left,
                "list_b": right,
                "shared_slices": len(intersection),
                "fraction_of_budget_shared": len(intersection) / len(sets[left]),
                "jaccard": score,
            }
        )

    matrix_path = output_dir / "jaccard_matrix.csv"
    pairwise_path = output_dir / "pairwise_jaccard.csv"
    summary_path = output_dir / "list_summary.csv"
    matrix.to_csv(matrix_path, index_label="list")
    pd.DataFrame(pairwise_rows).to_csv(pairwise_path, index=False)

    all_slices = set().union(*sets.values())
    summary_rows = []
    for label in labels:
        other_union = set().union(*(sets[other] for other in labels if other != label))
        summary_rows.append(
            {
                "list": label,
                "n_unique_slices": len(sets[label]),
                "n_slices_unique_to_this_list": len(sets[label] - other_union),
                "n_distinct_slices_across_all_lists": len(all_slices),
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    long_rows: list[dict[str, Any]] = []
    for label in labels:
        for row in list_frames[label].to_dict(orient="records"):
            long_rows.append({"list": label, **row})
    pd.DataFrame(long_rows).to_csv(output_dir / "selected_slices_long.csv", index=False)

    fig_width = max(7.0, 1.2 * len(labels) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.78), constrained_layout=True)
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels=labels)
    ax.set_title("SUGFW+ selection overlap (Jaccard similarity)")
    for row in range(len(labels)):
        for column in range(len(labels)):
            score = values[row, column]
            ax.text(
                column,
                row,
                f"{score:.2f}",
                ha="center",
                va="center",
                color="white" if score < 0.55 else "black",
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Jaccard similarity")
    plot_path = output_dir / "jaccard_heatmap.png"
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    return {
        "jaccard_matrix": str(matrix_path),
        "pairwise_jaccard": str(pairwise_path),
        "list_summary": str(summary_path),
        "selected_slices_long": str(output_dir / "selected_slices_long.csv"),
        "jaccard_heatmap": str(plot_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-uc-sam", required=True, type=Path)
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--image-root", required=True, type=Path)
    parser.add_argument(
        "--path-base",
        type=Path,
        help=(
            "Base directory for relative paths stored in train.csv. Defaults to "
            "--upstream-uc-sam, matching the released Promise12 CSV paths."
        ),
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--original-list",
        type=Path,
        help=(
            "Optional previously saved original SUGFW+ selection CSV. If omitted, "
            "the script creates a reference run using the upstream's unseeded augmentation "
            "behavior by default, and labels it as reconstructed."
        ),
    )
    parser.add_argument("--model-type", default="vit_b")
    parser.add_argument("--select-ratio", type=float, default=0.03)
    parser.add_argument("--expected-pool-size", type=int, default=960)
    parser.add_argument("--expected-budget", type=int, default=28)
    parser.add_argument("--soft-scale", type=int, default=5)
    parser.add_argument("--augmentations", type=int, default=10)
    parser.add_argument(
        "--reference-augmentation-seed",
        type=int,
        default=None,
        help=(
            "Optional seed for the reference run's augmentations. Omit it to preserve "
            "the upstream's default unseeded behavior."
        ),
    )
    parser.add_argument("--reference-kmeans-seed", type=int, default=42)
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        metavar="LABEL:AUG_SEED:KMEANS_SEED",
        help="Repeat exactly four times to override the four default seed pairs.",
    )
    args = parser.parse_args()

    upstream_uc_sam = args.upstream_uc_sam.expanduser().resolve(strict=True)
    train_csv = args.train_csv.expanduser().resolve(strict=True)
    image_root = args.image_root.expanduser().resolve(strict=True)
    path_base = (
        args.path_base.expanduser().resolve(strict=True)
        if args.path_base
        else upstream_uc_sam
    )
    checkpoint = args.checkpoint.expanduser().resolve(strict=False)
    output_dir = args.output_dir.expanduser().resolve(strict=False)

    if not upstream_uc_sam.is_dir():
        raise NotADirectoryError(f"UC-SAM directory not found: {upstream_uc_sam}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint}")
    if args.expected_budget < 1 or args.select_ratio <= 0:
        raise ValueError("Expected budget and select ratio must be positive")
    if args.augmentations < 1:
        raise ValueError("--augmentations must be at least 1")
    if args.variant and len(args.variant) != 4:
        raise ValueError("Provide exactly four --variant values, or omit them all")

    variants = args.variant or list(DEFAULT_VARIANTS)
    variant_labels = [item[0] for item in variants]
    if len(set(variant_labels)) != 4:
        raise ValueError("The four variant labels must be unique")
    if "reference" in variant_labels:
        raise ValueError("Variant labels cannot use the reserved label 'reference'")

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory "
            "so existing lists and caches are not overwritten."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if "image" in str((output_dir / "cache").resolve()).lower():
        raise ValueError(
            "The output directory path contains the substring 'image'. The upstream "
            "code uses string replacement on cache paths; choose another path."
        )
    if "splits" in str((output_dir / "cache").resolve()).lower():
        raise ValueError(
            "The output directory path contains the substring 'splits'. The upstream "
            "code uses string replacement on cache paths; choose another path."
        )

    pool = load_pool(train_csv, image_root, path_base, args.expected_pool_size)
    computed_budget = int(len(pool.frame) * args.select_ratio)
    if computed_budget != args.expected_budget:
        raise ValueError(
            f"With {len(pool.frame)} images and ratio {args.select_ratio}, upstream "
            f"selects {computed_budget} slices, not {args.expected_budget}. "
            "Adjust --select-ratio or --expected-budget deliberately."
        )

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The selection code uses SAM on CUDA; run this "
            "script on a Linux GPU VM after GPU quota is approved."
        )

    output_lists_dir = output_dir / "lists"
    cache_root = output_dir / "cache"
    output_lists_dir.mkdir()
    cache_root.mkdir()

    run_manifest: list[dict[str, Any]] = []
    list_frames: dict[str, Any] = {}
    labels = ["reference", *variant_labels]
    shared_feature_cache: Path | None = None

    if args.original_list:
        original_csv = args.original_list.expanduser().resolve(strict=True)
        reference = load_reference_list(original_csv, pool, args.expected_budget)
        list_frames["reference"] = reference
        reference_path = output_lists_dir / "reference.csv"
        reference.to_csv(reference_path, index=False)
        run_manifest.append(
            {
                "label": "reference",
                "provenance": "imported_previous_selection",
                "augmentation_seed": None,
                "kmeans_seed": None,
                "csv": str(reference_path),
                "source_csv": str(original_csv),
                "n_slices": len(reference),
            }
        )
    else:
        reference_run = RunSpec(
            label="reference",
            augmentation_seed=args.reference_augmentation_seed,
            kmeans_seed=args.reference_kmeans_seed,
            provenance="reconstructed_default_pipeline_not_original_author_list",
        )
        started = time.time()
        run_root = cache_root / reference_run.label
        staged_csv, staged_image_root = stage_run_inputs(pool, run_root, shared_feature_cache)
        reference, shared_feature_cache = run_sugfw(
            run=reference_run,
            pool=pool,
            run_root=run_root,
            staged_csv=staged_csv,
            staged_image_root=staged_image_root,
            checkpoint=checkpoint,
            model_type=args.model_type,
            augmentation_count=args.augmentations,
            soft_scale=args.soft_scale,
            select_ratio=args.select_ratio,
            expected_budget=args.expected_budget,
            upstream_uc_sam=upstream_uc_sam,
            shared_feature_cache=shared_feature_cache,
        )
        reference_path = output_lists_dir / "reference.csv"
        reference.to_csv(reference_path, index=False)
        list_frames["reference"] = reference
        run_manifest.append(
            {
                **asdict(reference_run),
                "csv": str(reference_path),
                "cache_dir": str(run_root),
                "elapsed_seconds": round(time.time() - started, 2),
                "n_slices": len(reference),
            }
        )

    for label, augmentation_seed, kmeans_seed in variants:
        run = RunSpec(
            label=label,
            augmentation_seed=augmentation_seed,
            kmeans_seed=kmeans_seed,
            provenance="seeded_rerun",
        )
        started = time.time()
        run_root = cache_root / run.label
        staged_csv, staged_image_root = stage_run_inputs(pool, run_root, shared_feature_cache)
        selected, shared_feature_cache = run_sugfw(
            run=run,
            pool=pool,
            run_root=run_root,
            staged_csv=staged_csv,
            staged_image_root=staged_image_root,
            checkpoint=checkpoint,
            model_type=args.model_type,
            augmentation_count=args.augmentations,
            soft_scale=args.soft_scale,
            select_ratio=args.select_ratio,
            expected_budget=args.expected_budget,
            upstream_uc_sam=upstream_uc_sam,
            shared_feature_cache=shared_feature_cache,
        )
        output_csv = output_lists_dir / f"{label}.csv"
        selected.to_csv(output_csv, index=False)
        list_frames[label] = selected
        run_manifest.append(
            {
                **asdict(run),
                "csv": str(output_csv),
                "cache_dir": str(run_root),
                "elapsed_seconds": round(time.time() - started, 2),
                "n_slices": len(selected),
            }
        )

    output_files = write_comparisons(
        output_dir=output_dir,
        labels=labels,
        list_frames=list_frames,
        pool=pool,
    )
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "SUGFW+ Promise12 selection stability pre-check",
        "source": {
            "train_csv": str(train_csv),
            "image_root": str(image_root),
            "upstream_uc_sam": str(upstream_uc_sam),
            "checkpoint": str(checkpoint),
        },
        "settings": {
            "pool_size": len(pool.frame),
            "select_ratio": args.select_ratio,
            "expected_budget": args.expected_budget,
            "soft_scale": args.soft_scale,
            "uncertainty_augmentations": args.augmentations,
            "model_type": args.model_type,
            "random_sampling": False,
            "jaccard_definition": "intersection_size / union_size",
        },
        "runs": run_manifest,
        "environment": environment_details(torch),
        "outputs": output_files,
        "fine_tuning_csv_directory": str(output_lists_dir),
        "important_note": (
            "If reference provenance is reconstructed_reference_not_original_author_list, "
            "the reference is a seeded reconstruction because no original selection CSV "
            "was supplied. Do not describe it as the authors' exact list."
        ),
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    with (output_dir / "selection_list_paths.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=["list", "csv_path", "provenance"])
        writer.writeheader()
        for run in run_manifest:
            writer.writerow(
                {
                    "list": run["label"],
                    "csv_path": run["csv"],
                    "provenance": run["provenance"],
                }
            )

    print("\nFinished generating and comparing five SUGFW+ lists.")
    print(f"Fine-tuning CSVs: {output_lists_dir}")
    print(f"Jaccard matrix:   {output_dir / 'jaccard_matrix.csv'}")
    print(f"Pairwise table:   {output_dir / 'pairwise_jaccard.csv'}")
    print(f"Heatmap:          {output_dir / 'jaccard_heatmap.png'}")
    print(f"Run manifest:     {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
