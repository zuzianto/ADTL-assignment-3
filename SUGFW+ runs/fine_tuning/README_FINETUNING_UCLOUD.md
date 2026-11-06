# SUGFW+ five-list fine-tuning on UCloud

This bundle fine-tunes the released SUGFW+ UPFT model once for each of the five
selection lists (`reference`, `rerun_1` ... `rerun_4`) produced by the earlier
selection-stability run. It reuses that run's cached uncertainty maps, the
existing Python 3.10 environment, and the SAM ViT-B checkpoint. The first
attempt showed that only the 50-case training archive had been prepared, so the
validation and test CSVs were empty. The new preparation helper downloads the
official 20-case live-challenge and 30-case test archives and rebuilds all
three official splits before training. It verifies archive checksums and the
presence of image/mask pairs; it does not install packages.

## What is compared

- Each model is trained on 28 slices (3% of the 960-slice PROMISE12 training
  pool), with the same two training seeds (`1233`, `1234`), model
  initialization, training settings, validation split, and held-out test split.
- The default is 300 epochs per run, matching the epoch count in the released
  Promise12 command. This gives 10 training runs (five lists × two seeds). On
  this single-GPU run, batch size is 16 and `n_gpu=1`; every run uses the same
  setting. This matches the released command's nominal global batch size
  (batch size 4 × 4 GPUs), while running on one GPU.
- `reference` is the reconstructed default pipeline, not a recovered copy of
  the paper's original selection CSV. The other four lists are stochastic
  reruns. The existing Jaccard figure suggests they are highly dissimilar.
- The model checkpoint is `train_best_model.pth`; that is the filename actually
  written by the upstream trainer. The published example shell script points
  inference at `mse_best_model.pth`, which the trainer in this pinned checkout
  does not write.

The crossed list-by-seed design allows a first descriptive comparison between
variation across selection lists and variation between two training seeds on
the same list. Two seeds provide only a rough estimate of training variability;
the result should be reported as exploratory rather than a precise variance
decomposition.

## Important evaluation correction

In the pinned upstream `MA-SAM-un/inference.py`, evaluation resizes a 256×256
image to 512×512 but leaves the corresponding label at its old size. That can
make the Dice calculation fail or compare misaligned arrays. The runner applies
two small, idempotent evaluation fixes: nearest-neighbor resize the label to
the evaluation size, and record HD95 as undefined (`NaN`) rather than crashing
if exactly one of prediction/ground truth is empty. Before editing, it saves the original as
`MA-SAM-un/inference.py.atdl3_backup`. No other upstream files are changed.
To restore the original later, copy that backup over `inference.py` (or use
`git checkout -- MA-SAM-un/inference.py` if the checkout has no local edits to
preserve).

The shared uncertainty prompt map used for validation and test is the mean of
all cached maps from the five runs (4,800 maps total). It is computed from
training-pool images only and held fixed across the five models. Each model's
selected training images still use the per-slice uncertainty map from their
own selection rerun. This keeps evaluation inputs identical between models
without privileging the reconstructed reference run or modifying the original
data directories.

## Upload and run in the existing UCloud job

Upload the updated `sugfw_finetuning_ucloud_bundle.zip` into the persistent
`ATDL3` folder in UCloud Files. In the terminal attached to the already
running GPU job, update the tools folder:

```bash
cd /work/ATDL3
mkdir -p /work/ATDL3/finetuning_tools
unzip -o /work/ATDL3/sugfw_finetuning_ucloud_bundle.zip -d /work/ATDL3/finetuning_tools
```

First prepare the missing evaluation data and regenerate the official splits.
This step downloads two archives (about 315 MB combined) and performs CPU
preprocessing; it does not start model training:

```bash
bash /work/ATDL3/finetuning_tools/prepare_promise12_eval_data_ucloud.sh
```

Then check the environment, GPU, non-empty CSVs, and referenced files:

```bash
bash /work/ATDL3/finetuning_tools/initialize_finetuning_ucloud.sh
```

The preflight verifies the environment, GPU, upstream files, data splits, and
checkpoint. If it reports `Preflight complete`, launch the five runs:

If the preflight reports that OpenCV cannot load `libGL.so.1`, replace the
GUI-enabled wheel with the pinned headless build (the training code uses OpenCV
image operations, not its GUI functions):

```bash
/work/ATDL3/envs/sugfw/bin/python -m pip uninstall -y opencv-python
/work/ATDL3/envs/sugfw/bin/python -m pip install --no-deps opencv-python-headless==4.7.0.72
/work/ATDL3/envs/sugfw/bin/python -c 'import cv2; print(cv2.__version__)'
bash /work/ATDL3/finetuning_tools/initialize_finetuning_ucloud.sh
```

This changes only the OpenCV wheel in the existing environment; `--no-deps`
keeps the already pinned NumPy and other packages unchanged. If a training or
inference subprocess fails, the runner now prints the end of its saved log as
well as the log path.

```bash
bash /work/ATDL3/finetuning_tools/run_five_list_finetuning_ucloud.sh
```

The script automatically finds the newest complete selection-stability run and
uses seeds `1233,1234` by default, for 10 runs in total. To use a different
pair, set `TRAIN_SEEDS` before launching, for example:

```bash
export TRAIN_SEEDS=1233,1234
bash /work/ATDL3/finetuning_tools/run_five_list_finetuning_ucloud.sh
```

To specify the run explicitly, set its path first:

```bash
export SELECTION_RUN_DIR=/work/ATDL3/results/selection_stability/run_20261103T113007Z
bash /work/ATDL3/finetuning_tools/run_five_list_finetuning_ucloud.sh
```

To watch progress from another terminal connected to the same job:

```bash
tail -f /work/ATDL3/logs/fine_tuning_run_YYYYMMDDTHHMMSSZ.log
```

Substitute the timestamp printed when the job starts. `Ctrl+C` exits `tail` only;
it does not stop the fine-tuning process. Do not launch the run script a second
time while the first process is active.

If the job is interrupted, note the printed run directory. To resume completed
arms (and continue with later ones), run again with that same directory:

```bash
export RUN_DIR=/work/ATDL3/results/selection_stability/fine_tuning/run_YYYYMMDDTHHMMSSZ
bash /work/ATDL3/finetuning_tools/run_five_list_finetuning_ucloud.sh
```

The runner reuses arms with a completion marker. It cannot resume an interrupted
training epoch; if an arm has a checkpoint but no completion marker, it stops
to avoid accidentally evaluating a partial model. Start with a fresh `RUN_DIR`
in that case.

## Outputs

Each run is saved below:

```text
/work/ATDL3/results/selection_stability/fine_tuning/run_<UTC timestamp>/
```

Main report files:

- `summary.csv`: one row per selection list, with test mean Dice, standard
-  deviation across test subjects, 95% bootstrap CI, between-seed spread, and
  HD95 summary. Dice is included on both the 0–1 scale and as a percentage.
- `summary_by_seed.csv`: one row for each of the 10 list-by-seed models.
- `per_case_metrics.csv`: per-test-volume Dice and HD95 for every list and seed.
- `paired_vs_reference.csv`: paired Dice differences against `reference`,
  averaged across the two matched training seeds, with bootstrap CIs.
- `paired_vs_reference_by_seed.csv`: the same paired comparison for each seed.
- `dice_by_selection_set.png`: patient-level Dice distributions averaged over
  the two seeds, with mean CIs.
- `dice_by_selection_and_seed.png`: mean Dice for both seeds within each list.
- `experiment_manifest.json`: paths, seeds, configuration, and provenance.
- `<list>/seed_<seed>/logs/train.log` and `test.log`: console logs for each run.

Dice is calculated per 3D test case and averaged across cases, following the
upstream inference structure. The confidence interval is a bootstrap over test
cases, not a confidence interval over training seeds. The seed-specific values
and within-list spread are also reported explicitly. HD95 follows the
upstream evaluator with spacing omitted, so its unit is voxel distance rather
than millimetres; undefined HD95 cases are counted and omitted from its mean.
The 5-list mean and spread are descriptive of these five selected sets; they
do not establish that SUGFW+ outperforms Random, because this follow-up trains
only the SUGFW+ lists. The single-GPU batch size is 16, matching the released
example's nominal global batch size of 4 per GPU across four GPUs. Hardware and
parallel execution differ, so report the actual single-GPU setup in the methods.
