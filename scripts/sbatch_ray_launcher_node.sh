#!/bin/bash

# Fail if any command fails
set -e

unset ROCR_VISIBLE_DEVICES # For some obscure reason this is set by CSCS environemtn
unset {HTTP,HTTPS,FTP,NO}_PROXY # Same here, having those variable set (even empty) can mess up with some tools (looking at you curl)
unset {http,https,ftp,no}_proxy # Same here, having those variable set (even empty) can mess up with some tools (looking at you curl)

echo "Activating virtual environment at $VENV_DIR"
source "$VENV_DIR/bin/activate"

# Summary
echo "Number of nodes: $SLURM_NNODES"
echo "Node list: $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Python Version: $(python --version)"
echo "Python path: $(python -c 'import sys; print(sys.executable)')"
echo "Ray path: $(which ray)"
echo "Working directory: $(pwd)"
echo "Environment variables: "

# Start ray HEAD node
if [[ $SLURM_NODEID -eq 0 ]]; then
    echo "Starting Ray HEAD node on $(hostname)"
    ray start \
        --head \
        --port=6379 \
        --log-color=false \
        --node-ip-address=$(hostname -i) \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS_ON_NODE \
        --include-dashboard=true \
        --dashboard-host="0.0.0.0" \
        --dashboard-port=8265

    # Await for the head node to initialize (await for 1 whole minues)
    sleep 20 
else
    # Sleep to ensure the head node starts before workers try to connect
    sleep 10

    # Start ray WORKER nodes
    echo "Starting Ray WORKER node on $(hostname)"
    ray start \
        --address=$(scontrol show hostnames $SLURM_NODELIST | head -n 1):6379 \
        --block \
        --log-color=false \
        --node-ip-address=$(hostname -i) \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS_ON_NODE
    exit 0 # Never run training script on worker nodes
fi

# Script only runs on the head node from here
ray status

echo "Running training script on $(hostname)"
echo "Running command: $@"
$@ # Execute the training script passed as an argument

# Finally stop the ray cluster (this should also stop all of the worker noeds)
ray stop
