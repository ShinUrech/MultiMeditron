#!/bin/bash
#SBATCH --job-name convert-datasets
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:0
#SBATCH --cpus-per-task 64
#SBATCH --time 02:00:00
#SBATCH -A a127

export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set. Set it in your environment or ~/.bashrc"}
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:$PYTHONPATH

SCRIPT=/users/surech/meditron/MultiMeditron/scripts/convert_image_datasets.py

echo "START TIME: $(date)"
set -eo pipefail
set -x

SRUN_ARGS=" \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  "

srun $SRUN_ARGS python $SCRIPT
echo "END TIME: $(date)"
