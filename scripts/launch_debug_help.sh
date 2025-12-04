#!/bin/bash

unset ${!SLURM_*}

EXPERIMENT_NAME="debug_help_experiment"
NUM_NODES=4
TIMEOUT="08:00:00"
ENVIRONMENT="$HOME/.edf/multimodal.toml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="./.venv"
VENV_DIR=$(realpath "$VENV_DIR")

sbatch --job-name=debug_help \
       --output=reports/debug_help_%j.out \
       --error=reports/debug_help_%j.err \
       --job-name=$EXPERIMENT_NAME \
       --nodes=$NUM_NODES \
       --time=$TIMEOUT \
       --ntasks-per-node=1 \
       --cpus-per-task=280 \
       --gpus-per-node=4 \
       --partition=$PARTITION \
       --mem=380G \
       -A a127 \
       --environment=$ENVIRONMENT \
       --export=SCRIPT_DIR=$SCRIPT_DIR,VENV_DIR=$VENV_DIR \
       ${SCRIPT_DIR}/sbatch_ray_launcher.sh \
       sleep infinity