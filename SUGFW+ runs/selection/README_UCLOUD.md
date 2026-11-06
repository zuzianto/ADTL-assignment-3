# Five SUGFW+ selection lists on UCloud

This package runs the selection-stability pre-check on UCloud. It stores the
Python environment, upstream code, PROMISE12 data, SAM checkpoint, caches, logs,
and results in the persistent directory selected for the UCloud job.

## 1. Upload the bundle to persistent storage

Upload `sugfw_ucloud_bundle.zip` through UCloud's Files interface into a
persistent work directory, for example a folder named `ATDL3`. Do not use a
temporary session directory. Choose a directory whose full path does not
contain `image` or `splits`, because the upstream code derives cache paths by
replacing those words.

The ZIP contains the generator and both Bash scripts. The initialization script
extracts it automatically into the selected directory.

## 2. Configure the UCloud Terminal job

Following the UCloud Terminal instructions:

- Application: **Applications → Development → Terminal**
- Job name: `SUGFW-PROMISE12-five-lists`
- Storage, Folder #1: select the `ATDL3` folder containing the ZIP; UCloud mounts it at `/work`
- Machine type: select the first MIG step, **1/7 B200** (`gpu-nvidia-b200-1-mig`), rather than a full B200; only increase to a full GPU if the job cannot access CUDA with MIG
- Time limit: allow about 12 hours if the platform permits; the earlier estimate was roughly two hours per uncertainty-map redraw, plus feature extraction and setup
- Initialization: upload `initialize_ucloud.sh`
- Batch Processing: upload `run_five_lists_ucloud.sh`

The UCloud product documentation defines one B200 MIG as 1/7 of a B200 and
accounts it as 1/7 GPU-hour per hour. This experiment uses one GPU at a time
for SAM ViT-B inference; a full B200 is unnecessary. [UCloud GPU resource
details](https://docs.cloud.sdu.dk/guide/resources-products.html)

Initialization requires a visible NVIDIA GPU. The official SUGFW+ repository
specifies Python 3.10. If the UCloud image does not provide it (for example,
if it only has Python 3.12), the script installs Miniforge under `/work/tools`
and creates the Python 3.10 environment under `/work/envs/sugfw`; both remain
in persistent storage. Do not run Batch Processing unless the initialization
log ends with `Initialization complete` and PyTorch reports
`CUDA ... available True`.

The setup installs PyTorch from the current default PyPI wheels (CUDA 13 on
the current PyTorch release), because the B200 is NVIDIA Blackwell hardware.
The current PyTorch release notes state that Blackwell users should use CUDA
13.0+ wheels and a sufficiently recent NVIDIA driver. [PyTorch 2.12 release
notes](https://pytorch.org/blog/pytorch-2-12-release-blog/). The upstream
SUGFW+ setup specifies Python 3.10. [Official SUGFW+ repository](https://github.com/HiLab-git/SUGFW-plus)

## 3. Submit and monitor

Click **Submit**. Initialization clones the upstream SUGFW+ code at the pinned
commit, downloads the PROMISE12 training archive and SAM ViT-B checkpoint,
installs the Python environment, and runs the upstream preprocessing scripts.
It creates the other two PROMISE12 split directories empty because the upstream
preprocessor expects their names, while this experiment uses only training
cases.

The script checks that the generated training CSV has exactly 960 slices. If
that check fails, do not run with the default 28-slice budget. The agreed
experiment uses 3% of 960, which is 28 slices.

Batch Processing runs the five-list generator and exits when it finishes. If
the job has a shorter maximum duration than the estimated runtime, do not start
all five runs until we adjust the generator for safe resume/checkpointing.

## 4. Output

Files remain in the persistent directory under:

```text
results/selection_stability/run_<UTC timestamp>/
```

Expected outputs include five selection CSVs, `pairwise_jaccard.csv`,
`jaccard_matrix.csv`, `jaccard_heatmap.png`, per-list summaries, and
`run_manifest.json`. Do not commit the dataset, checkpoint, caches, or results
to GitHub.

## Scientific interpretation

No original selection CSV is available in the local project or known UCloud
files. The reference run therefore uses the upstream's unseeded augmentation
behavior and KMeans `random_state=42`, and is labelled as a reconstructed
default-pipeline run. It is not the exact selection list originally produced
for the paper. Four additional lists use explicit augmentation and KMeans
seeds. This package uses the 960-slice training pool/28-slice 3% budget, not
the separate 2,595-slice pool proposal.
