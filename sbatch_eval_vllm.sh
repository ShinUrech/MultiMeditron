#!/bin/bash
#SBATCH --job-name multimeditron-eval-vllm
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:4
#SBATCH --cpus-per-task 288
#SBATCH --time 01:00:00
#SBATCH -A a127

# Usage:
#   sbatch sbatch_eval_vllm.sh <checkpoint> [tasks] [limit] [tp_size] [dp_size]
#
# Examples:
#   Full eval:   sbatch sbatch_eval_vllm.sh /path/to/checkpoint gmai,slake,path_vqa
#   Quick test:  sbatch --partition=debug --time=00:29:59 sbatch_eval_vllm.sh /path/to/checkpoint gmai 20
#   Multi-GPU:   sbatch sbatch_eval_vllm.sh /path/to/checkpoint gmai "" 4

MODEL_PATH=${1:?"Usage: sbatch sbatch_eval_vllm.sh <checkpoint_path> [tasks] [limit] [tp_size] [dp_size]"}
TASKS=${2:-gmai,slake,path_vqa}
LIMIT=${3:-}
export VLLM_TENSOR_PARALLEL_SIZE=${4:-4}
export VLLM_DATA_PARALLEL_SIZE=${5:-1}
export VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
export VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}

export HF_HOME=/iopsstor/scratch/cscs/surech/hf
export HF_TOKEN=${HF_TOKEN:?"HF_TOKEN is not set"}

OUTPUT_DIR="/users/surech/meditron/reports/lm_eval_results/$(basename $MODEL_PATH)"
LM_EVAL_INCLUDE_PATH="/users/surech/meditron/MultiMeditron/third-party/lmms-eval/lmms_eval/tasks"
export VLLM_PIP_DIR="/iopsstor/scratch/cscs/surech/pip-lm-eval-nodeps"

mkdir -p "$OUTPUT_DIR"

echo "START TIME: $(date)"
echo "MODEL:    $MODEL_PATH"
echo "TASKS:    $TASKS"
echo "LIMIT:    ${LIMIT:-all}"
echo "TP_SIZE:  $VLLM_TENSOR_PARALLEL_SIZE"
echo "OUTPUT:   $OUTPUT_DIR"
set -eo pipefail
set -x

NUM_GPUS="${SLURM_GPUS_ON_NODE:-4}"
TP_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-$NUM_GPUS}"
DP_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"
GPU_MEM_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"

srun \
  --nodes 1 \
  --ntasks 1 \
  --cpus-per-task "$SLURM_CPUS_PER_TASK" \
  --jobid "$SLURM_JOB_ID" \
  --wait 60 \
  --environment /users/surech/.edf/vllm.toml \
  --export=ALL \
  bash -s -- "$MODEL_PATH" "$TP_SIZE" "$DP_SIZE" "$GPU_MEM_UTIL" "$MAX_MODEL_LEN" "$TASKS" "$OUTPUT_DIR" "$LM_EVAL_INCLUDE_PATH" "$LIMIT" <<'EOS'
set -euo pipefail

MODEL_PATH="$1"
TP_SIZE="$2"
DP_SIZE="$3"
GPU_MEM_UTIL="$4"
MAX_MODEL_LEN="$5"
TASKS="$6"
OUTPUT_DIR="$7"
LM_EVAL_INCLUDE_PATH="$8"
LIMIT="$9"

mkdir -p "$VLLM_PIP_DIR"
export PYTHONPATH="/users/surech/meditron/MultiMeditron/src:/users/surech/meditron/MultiMeditron/third-party/lmms-eval:${PYTHONPATH:-}:$VLLM_PIP_DIR"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

if ! python3 -c "import accelerate, lm_eval, lmms_eval, more_itertools, sqlitedict, word2number, zstandard, pytz" >/dev/null 2>&1; then
  echo "[preflight] installing lm_eval runtime deps into $VLLM_PIP_DIR"
  rm -rf "$VLLM_PIP_DIR"
  mkdir -p "$VLLM_PIP_DIR"
  python3 -m pip install --quiet --no-deps --target "$VLLM_PIP_DIR" \
    accelerate lm-eval more-itertools sqlitedict word2number zstandard pytz
fi

python3 - <<'PY'
import importlib.util
import sys

missing = [m for m in ("torch", "vllm", "lm_eval", "lmms_eval") if importlib.util.find_spec(m) is None]
if missing:
    print(f"[preflight] missing runtime modules: {missing}", file=sys.stderr)
    sys.exit(2)

import lm_eval
import lmms_eval
print(f"[preflight] lm_eval import ok: {lm_eval.__file__}")
print(f"[preflight] lmms_eval import ok: {lmms_eval.__file__}")
PY

MODEL_ARGS="model=$MODEL_PATH,dtype=bfloat16,trust_remote_code=True,tensor_parallel_size=$TP_SIZE,data_parallel_size=$DP_SIZE,gpu_memory_utilization=$GPU_MEM_UTIL,max_model_len=$MAX_MODEL_LEN,hf_overrides={\"architectures\":[\"TransformersMultiModalForCausalLM\"]}"

LM_EVAL_CMD=(
  python3 -m lmms_eval
  --model vllm
  --model_args "$MODEL_ARGS"
  --tasks "$TASKS"
  --batch_size auto
  --verbosity DEBUG
  --log_samples
  --output_path "$OUTPUT_DIR"
  --include_path "$LM_EVAL_INCLUDE_PATH"
  --apply_chat_template
)

if [ -n "$LIMIT" ]; then
  LM_EVAL_CMD+=(--limit "$LIMIT")
fi

"${LM_EVAL_CMD[@]}"
EOS

echo "END TIME: $(date)"
