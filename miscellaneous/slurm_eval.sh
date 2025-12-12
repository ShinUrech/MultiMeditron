#!/bin/bash
#SBATCH --chdir /users/mtang/datasets
#SBATCH --output /users/mtang/datasets/reports/benchmark_no_moove/R-%x.%j.out
#SBATCH --error /users/mtang/datasets/reports/benchmark_no_moove/R-%x.%j.err
#SBATCH --nodes 1         # number of Nodes
#SBATCH --ntasks-per-node 1     # number of MP tasks. IMPORTANT: torchrun represents just 1 Slurm task
#SBATCH --gres gpu:4        # Number of GPUs
#SBATCH --cpus-per-task 288     # number of CPUs per task.
#SBATCH --time 11:59:59       # maximum execution time (DD-HH:MM:SS)
#SBATCH --environment /users/mtang/.edf/multimodal2.toml
#SBATCH -A a127

python split_script_sglang.py --shard_id $1
