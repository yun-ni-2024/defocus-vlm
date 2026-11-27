# Reading Text Through Real Defocus Blur with a Vision-Language Model

CSC2529 Computational Imaging (Fall 2026) course project. Yun Ni, David Chen.

## Layout

- `src/`: all project code
- `configs/`: one config file per experiment
- `scripts/`: command-line entry points
- `slurm/`: sbatch templates for the CSLab cluster
- `runpod/`: run script for Runpod GPU pods
- `docs/`: experiment log and design notes

Data, model weights, checkpoints and intermediate runs are not in this repository; the final result of each method is in `assets/results/`. The one exception to checkpoints: a trained method keeps the adapter its final run used, with its training records, in `assets/<config name>/` (see `docs/pipeline.md`).

## Running on the CSLab cluster

All experiments run on the CSLab GPU cluster. See `docs/cluster.md` for logging in, getting the code, the folder layout and submitting jobs.

## Running on Runpod

When the cluster GPUs are busy, experiments run on rented Runpod GPU pods with the same code. See `docs/runpod.md` for the setup, syncing code over SSH and running.
