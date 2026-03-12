#!/bin/bash
#SBATCH --job-name install-lmms-eval
#SBATCH --output /users/surech/meditron/reports/R-%x.%j.out
#SBATCH --error /users/surech/meditron/reports/R-%x.%j.err
#SBATCH --partition debug
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 72
#SBATCH --time 00:30:00
#SBATCH -A a127

set -eo pipefail
set -x

srun \
  --environment /users/surech/.edf/multimeditron.toml \
  --export=ALL \
  bash -c "
    cd /users/surech/meditron/MultiMeditron/third-party/lmms-eval && \
    pip install -e . && \
    echo 'lmms-eval installed successfully' && \
    python -c 'import lmms_eval; print(\"lmms_eval version:\", lmms_eval.__version__)'
  "

echo "DONE: $(date)"
