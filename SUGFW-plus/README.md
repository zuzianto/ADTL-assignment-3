# Random Selection Baseline

This part of the project covers the generation, training, and evaluation of the Random slice-selection baseline for comparison with SUGFW+ on PROMISE12.

## What was done

- Generated five independent Random selection lists.
- Selected 28 unique foreground slices per list from the 960-slice training pool.
- Used a fixed master selection seed of 1233.
- Trained each list using two training seeds: 1233 and 1234.
- Completed all 10 training runs.
- Evaluated the trained models on the 30-case PROMISE12 test set.
- Collected Dice and HD95 results for each run.

## Training setup

- Model: SAM ViT-B with LoRA adaptation
- LoRA rank: 32
- Epochs: 300
- Batch size: 16
- Input resolution: 512 × 512
- Optimizer: AdamW
- Learning rate: 0.0008
- One full GPU allocation

## Results

Results are reported as mean ± standard deviation across the 30 test cases for each run. The two training seeds are averaged within each selection list for the comparison between Random and SUGFW+.

## Reproducibility

Random selection lists are saved as JSON files in:

`data/Promise12/random_selections/`

Training outputs and test results are stored in:

`data/Promise12/outputs/random/`
