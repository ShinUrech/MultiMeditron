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

export HF_HOME=/capstor/store/cscs/swissai/a127/meditron/hf_cache
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set. Set it in your environment or ~/.bashrc"}

# Load user's source tree and lmms-eval
export PYTHONPATH=/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:$PYTHONPATH

# ──────────────────────────────────────────────────
# CONFIGURE THESE before submitting:
#   CHECKPOINT  = path to the trained model checkpoint
#   TOKENIZER   = llama | apertus | qwen3
#   TASKS       = comma-separated list of benchmarks
# ──────────────────────────────────────────────────
CHECKPOINT=${1:?"Usage: sbatch sbatch_eval.sh <checkpoint_path> [tokenizer_type] [tasks] [num_proc]"}
TOKENIZER=${2:-llama}
TASKS=${3:-gmai,slake,path_vqa}
# Use 1 process + device_map=auto so the model is spread across all GPUs on a
# single process. Using >1 process with device_map=auto causes each process to
# compete for all GPUs simultaneously, leading to OOM or incorrect evaluation.
NUM_PROC=${4:-1}

echo "START TIME: $(date)"
echo "CHECKPOINT: $CHECKPOINT"
echo "TOKENIZER:  $TOKENIZER"
echo "TASKS:      $TASKS"
set -eo pipefail
set -x

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
    --batch_size 1

echo "END TIME: $(date)"
