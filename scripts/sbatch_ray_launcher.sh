#!/bin/bash

# Set working directory to the directory where the Slurm script is located

echo "Starting job on $(date)"

# Echo information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Allocated Nodes: $SLURM_JOB_NODELIST"
echo "Number of Nodes: $SLURM_NNODES"
echo "Number of Tasks: $SLURM_NTASKS"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "CPUs on Node: $SLURM_CPUS_ON_NODE"
echo "GPUs per Node: $SLURM_GPUS_ON_NODE"
echo "Memory per Node: $SLURM_MEM_PER_NODE"
echo "Working Directory: $(pwd)"
echo "Arguments: $@"

# Launch the Ray cluster using sbatch_ray_launcher_node.sh on multiple nodes
srun \
    --chdir=$PWD \
    "$SCRIPT_DIR/sbatch_ray_launcher_node.sh" $@

echo "Ending job on $(date)"
