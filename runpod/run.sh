#!/bin/bash
# Run one Python script with its arguments on a Runpod pod, inside the project environment.
# The Runpod counterpart of slurm/gpu.sbatch: a pod has no queue, so this only sets up the
# environment and starts the script in the background, where it keeps running after SSH ends.
# Run from the repository root on the pod (docs/runpod.md):
#   bash runpod/run.sh NAME scripts/train_lora.py configs/train_c_real.yaml
# The output goes to $DEFOCUS_ROOT/logs/NAME_<time>.out; follow it with tail -f.
set -e

# the first argument names the run; the rest is the script and its arguments
NAME=$1
shift

# environment and storage locations, all on the network volume (docs/runpod.md)
source /workspace/venv/bin/activate
export DEFOCUS_ROOT=/workspace/csc2529
export HF_HOME=$DEFOCUS_ROOT/models/hf_home
export TORCH_HOME=$DEFOCUS_ROOT/models/torch_home

# one output file per run, like the -o of a Slurm job
mkdir -p "$DEFOCUS_ROOT/logs"
LOG="$DEFOCUS_ROOT/logs/${NAME}_$(date +%Y%m%d-%H%M%S).out"
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "no GPU")
echo "host $(hostname), commit $(git rev-parse --short HEAD), $GPU" > "$LOG"

# start in the background, detached from the terminal
nohup python "$@" >> "$LOG" 2>&1 &
echo "started $NAME (process $!), output: $LOG"
