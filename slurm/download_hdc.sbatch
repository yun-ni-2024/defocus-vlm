#!/bin/bash
# Download and unpack the HDC2021 data on a CPU node (no GPU is needed to wait for the network).
# Submit from the repository root, giving the output file on the command line:
#   sbatch -o ~/csc2529/logs/%x_%j.out slurm/download_hdc.sbatch [test|train|all]
#SBATCH --partition=cpunodes
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 3-00:00:00
#SBATCH -J download_hdc

# data goes under $DEFOCUS_ROOT/data/hdc2021
export DEFOCUS_ROOT="${DEFOCUS_ROOT:-/w/20261/$USER/csc2529}"

# run the download script from the copy of the repository this job was submitted from
cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname), commit $(git rev-parse --short HEAD)"
python3 scripts/download_hdc.py "${1:-all}"
