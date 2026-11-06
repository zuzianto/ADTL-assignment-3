# SUGFW+ runs

This folder contains the team-maintained code used to run the PROMISE12
selection-stability and fine-tuning experiments on UCloud.

## Contents

- `selection/`: generates five SUGFW+ selection lists and compares them with
  pairwise Jaccard similarity.
- `fine_tuning/`: prepares the evaluation split and fine-tunes/evaluates each
  selection list with two training seeds.

The UCloud scripts expect persistent storage mounted at `/work`. See the
README in each subfolder for setup and run instructions.

## External requirements

The initialization script clones the upstream SUGFW+ repository and checks
out commit `9b7776f33f6d581665a2d91ca9abbe628a9d9eaa` from
https://github.com/HiLab-git/SUGFW-plus. The upstream implementation is
referenced rather than duplicated here.

PROMISE12 data, SAM checkpoints, Python environments, generated uncertainty
caches, logs, and experiment outputs remain on UCloud and are intentionally
not included in this GitHub folder.
