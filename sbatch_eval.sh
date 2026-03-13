#!/bin/bash
#SBATCH --job-name multimeditron-eval
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 05:59:59
#SBATCH -A a127

# Usage:
#   sbatch sbatch_eval.sh <checkpoint> [tokenizer] [tasks] [limit] [num_proc]
#
# Examples:
#   Full eval:   sbatch sbatch_eval.sh /path/to/checkpoint llama gmai,slake,path_vqa
#   Quick test:  sbatch --partition=debug --time=00:29:59 sbatch_eval.sh /path/to/checkpoint llama gmai 20

CHECKPOINT=${1:?"Usage: sbatch sbatch_eval.sh <checkpoint_path> [tokenizer] [tasks] [limit] [num_proc]"}
TOKENIZER=${2:-llama}
TASKS=${3:-gmai,slake,path_vqa}
LIMIT=${4:-}
NUM_PROC=${5:-1}

export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set"}

export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:$PYTHONPATH

echo "START TIME: $(date)"
echo "CHECKPOINT: $CHECKPOINT"
echo "TOKENIZER:  $TOKENIZER"
echo "TASKS:      $TASKS"
echo "LIMIT:      ${LIMIT:-all}"
set -eo pipefail
set -x

LIMIT_ARG=""
[ -n "$LIMIT" ] && LIMIT_ARG="--limit $LIMIT"

srun \
  --cpus-per-task $SLURM_CPUS_PER_TASK \
  --jobid $SLURM_JOB_ID \
  --wait 60 \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL \
  python3 -m accelerate.commands.launch \
    --num_processes $NUM_PROC \
    -m lmms_eval \
    --model multimeditron \
    --model_args pretrained="$CHECKPOINT",tokenizer_type="$TOKENIZER",device_map="auto" \
    --tasks $TASKS \
    --batch_size 1 \
    $LIMIT_ARG \
    --output_path /users/surech/meditron/reports/lmms_eval_results

echo "END TIME: $(date)"
