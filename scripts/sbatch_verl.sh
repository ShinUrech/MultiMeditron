#!/bin/bash
#SBATCH --job-name=verl-training
#SBATCH --output=logs/verl-$2-%j.out
#SBATCH --error=logs/verl-$2-%j.err
#SBATCH --cpus-per-task=200
#SBATCH --gres=gpu:4
#SBATCH --time=11:59:00
#SBATCH --partition=normal
#SBATCH --mem=380G
#SBATCH -A a127

# ATCH --ntasks=$1
# ATCH --nodes=$1
# Set working directory to the directory where the Slurm script is located
SCRIPT_DIR=$(dirname $(dirname $(realpath $0)))
cd $SCRIPT_DIR

echo "Starting job on $(date)"

# Installing required dependencies
pip install . 
pip install third-party/verl

echo Number of nodes: $SLURM_NNODES
echo Node list: $SLURM_NODELIST
echo Number of tasks: $SLURM_NTASKS
echo CPUs per task: $SLURM_CPUS_PER_TASK
echo GPUs per node: $SLURM_GPUS
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
        --num-gpus=$SLURM_GPUS \
        --include-dashboard=true \
        --dashboard-host="0.0.0.0"

    # Await for the head node to initialize
    sleep 15
else
    # Sleep to ensure the head node starts before workers try to connect
    sleep 5

    # Start ray WORKER nodes
    echo "Starting Ray WORKER node on $(hostname)"
    ray start \
        --address=$(scontrol show hostnames $SLURM_NODELIST | head -n 1):6379 \
        --node-ip-address=$(hostname -i) \
        --num-cpus=$SLURM_CPUS_PER_TASK \
        --num-gpus=$SLURM_GPUS
    exit 0 # Never run training script on worker nodes
fi

# Script only runs on the head node from here
ray status

# Run the training script
mm verl -c config/rl/grpo/train-grpo.yaml \
    trainer.nnodes=$SLURM_NNODES \
    trainer.experiment_name="$1"

echo "Ending job on $(date)"