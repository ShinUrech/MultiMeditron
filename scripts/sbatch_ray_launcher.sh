#!/bin/bash

# Set working directory to the directory where the Slurm script is located
echo "Starting job on $(date)"

# Unset some environment variables that might interfere with Ray
# those variables are sets for some reason on the CSCS cluster, probably due to a misconfiguration
unset ${!ROCR_*}
unset {HTTP,HTTPS,FTP}_PROXY
unset {http,https,ftp}_proxy

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
echo "Python Version: $(python --version)"
echo "Python path: $(python -c 'import sys; print(sys.executable)')"
echo "Ray path: $(which ray)"
echo "Working Directory: $(pwd)"
echo "Arguments: $@"
set -e

# Determine the stdout/stderr based on the CURRENT stdout stderr of this sbatch script
REPORT_STDOUT_FILE="${REPORT_STDOUT_FILE:-$SLURM_JOB_ID.out}"
REPORT_STDOUT_FILE="${REPORT_STDOUT_FILE%.out}-%t.log"

# Launch the Ray cluster using sbatch_ray_launcher_node.sh on multiple nodes
srun \
    --chdir=$PWD \
    --output=$REPORT_STDOUT_FILE \
    --error=$REPORT_STDOUT_FILE \
    --export=ALL \
    "$SCRIPT_DIR/sbatch_ray_launcher_node.sh" $@

echo "Ending job on $(date)"
