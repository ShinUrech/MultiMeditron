#!/bin/bash

# Loop over 50 shard IDs
for i in $(seq 0 49); do
    jobname="sglang_${i}"
    sbatch --job-name=${jobname} slurm_eval.sh ${i}
done
