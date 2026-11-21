#!/bin/bash
# Run one Python script with its arguments on a GPU node, inside the project environment.
# Submit from the repository root:
#   sbatch -J NAME -o ~/csc2529/logs/%x_%j.out slurm/gpu.sbatch scripts/evaluate.py configs/baseline_a.yaml
# Options given on the command line override the defaults below, e.g. --gres=gpu:rtx_a4500:1 or -t 30.
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:rtx_4090:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 4:00:00

# environment and storage locations (docs/cluster.md, sections 3 and 4)
source /w/20261/$USER/miniforge3/bin/activate defocus-vlm
export DEFOCUS_ROOT=/w/20261/$USER/csc2529
export HF_HOME=$DEFOCUS_ROOT/models/hf_home
export TORCH_HOME=$DEFOCUS_ROOT/models/torch_home

# run from the copy of the repository the job was submitted from
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname), commit $(git rev-parse --short HEAD), $(nvidia-smi --query-gpu=name --format=csv,noheader)"
python "$@"
