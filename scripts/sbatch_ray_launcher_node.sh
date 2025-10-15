#!/bin/bash

# Fail if any command fails
set -e

# Installing required dependencies
pip install pynvml
pip install . 
pip install third-party/verl

echo Number of nodes: $SLURM_NNODES
echo Node list: $SLURM_NODELIST
echo Number of tasks: $SLURM_NTASKS
echo CPUs per task: $SLURM_CPUS_PER_TASK
echo GPUs per node: $SLURM_GPUS_ON_NODE
echo Memory per node: $SLURM_MEM_PER_NODE
echo Working directory: $(pwd)

# Start ray HEAD node
if [[ $SLURM_NODEID -eq 0 ]]; then
    echo "Starting Ray HEAD node on $(hostname)"
    ray start \
        --head \
        --port=6379 \
        --node-ip-address=$(hostname -i) \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS_ON_NODE \
        --include-dashboard=true \
        --dashboard-host="0.0.0.0"

    # Await for the head node to initialize
    sleep 20
else
    # Sleep to ensure the head node starts before workers try to connect
    sleep 5

    # Start ray WORKER nodes
    echo "Starting Ray WORKER node on $(hostname)"
    ray start \
        --address=$(scontrol show hostnames $SLURM_NODELIST | head -n 1):6379 \
        --node-ip-address=$(hostname -i) \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS_ON_NODE
    exit 0 # Never run training script on worker nodes
fi

# Script only runs on the head node from here
ray status

echo "Running training script on $(hostname)"
$1 # Execute the training script passed as an argument
