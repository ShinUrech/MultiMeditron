from skin_benchmark import SkinDiseaseBenchmark

skin_bench = SkinDiseaseBenchmark(
    train_jsonl="/mloscratch/users/turan/datasets/skin_diseases_10/train_raw.jsonl",
    test_jsonl="/mloscratch/users/turan/datasets/skin_diseases_10/skin10_val_raw.jsonl",
    image_root="/mloscratch/users/turan/datasets/skin_diseases_10",
)

model_dirs = [
    "/mloscratch/users/turan/evaluation_clip/models/combined_dataset_skin_aggressive_training_config_1_lr5.418484333396616e-05_wd0.20568011432383415_nfrz2",
]

for md in model_dirs:
    print(f"\nEvaluating model: {md}")
    acc = skin_bench.evaluate(md)
    print(f"Accuracy: {acc:.4f}")
