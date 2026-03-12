#!/bin/bash
#SBATCH --job-name mm-eval-test
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 00:29:59
#SBATCH -A a127

export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set"}
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:$PYTHONPATH

CHECKPOINT=${1:?"Usage: sbatch sbatch_eval_test.sh <checkpoint_path>"}
TASKS=${2:-gmai}
LIMIT=${3:-20}

echo "START TIME: $(date)"
echo "CHECKPOINT: $CHECKPOINT"
echo "TASKS:      $TASKS"
echo "LIMIT:      $LIMIT"
set -eo pipefail
set -x

srun \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL \
  python3 -m accelerate.commands.launch \
    --num_processes 1 \
    -m lmms_eval \
    --model multimeditron \
    --model_args pretrained="$CHECKPOINT",tokenizer_type="llama" \
    --tasks $TASKS \
    --batch_size 1 \
    --limit $LIMIT \
    --output_path /users/surech/meditron/reports/lmms_eval_results

echo "END TIME: $(date)"
