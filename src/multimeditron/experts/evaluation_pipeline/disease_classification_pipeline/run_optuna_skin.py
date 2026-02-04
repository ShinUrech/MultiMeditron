"""
Entry-point Script for Skin Disease CLIP Hyperparameter Optimization.

This script launches an Optuna-based hyperparameter search for CLIP-style models using
the SkinDiseaseBenchmark. It loads a YAML configuration specifying the CLIP training
setup, runs the optimization loop over multiple trials, evaluates each trial via the
skin disease classification benchmark, and saves the resulting Optuna study to disk
for later inspection and analysis.

Usage:
    python run_optuna_skin.py <config.yaml> <study_id>

The output is a serialized Optuna study containing all completed trials and their
associated hyperparameters and benchmark scores.
"""

import sys
import pickle
from skin_benchmark import SkinDiseaseBenchmark
from train_hp_opt import train

if __name__ == "__main__":
    config_path = sys.argv[1]       # your YAML for HP tuning
    script_number = sys.argv[2]     # just an id for the study pickle

    skin_bench = SkinDiseaseBenchmark(
        train_jsonl="/mloscratch/users/turan/datasets/dermnet_eval/dermnet_train.jsonl",
        test_jsonl="/mloscratch/users/turan/datasets/dermnet_eval/dermnet_val.jsonl",
        image_root="/mloscratch/users/turan/datasets/dermnet_eval",
    )

    study = train([skin_bench], config_path)

    with open(f"study_skin_{script_number}.pkl", "wb") as f:
        pickle.dump(study, f)
        print("study saved")
