# Reading Text Through Real Defocus Blur with a Vision-Language Model

CSC2529 Computational Imaging (Fall 2026) course project. Yun Ni, David Chen.

## Layout

- `src/`: all project code
- `configs/`: one config file per experiment
- `scripts/`: command-line entry points
- `slurm/`: sbatch templates for the CSLab cluster
- `docs/`: experiment log and design notes

Data, model weights, checkpoints and results are not in this repository.

## Running on the CSLab cluster

All experiments run on the CSLab GPU cluster. See `docs/cluster.md` for logging in, getting the code, the folder layout and submitting jobs.

## Documentation

- `docs/methods.md`: the four methods compared
- `docs/dataset.md`: what is in HDC2021
- `docs/evaluation.md`: the metric and what is reported
- `docs/milestones.md`: plan and status
- `docs/findings.md`: observations kept for the report
- `docs/conventions.md`: how code, configs and runs are organised
- `docs/glossary.md`: imaging terms in plain words
- `docs/references.md`: papers and what each is used for
- `docs/cluster.md`: running on the cluster
