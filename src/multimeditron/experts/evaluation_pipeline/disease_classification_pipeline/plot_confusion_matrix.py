"""
Confusion-Matrix Evaluation for Skin Disease CLIP Benchmark.

This script evaluates a trained CLIP-style vision–language model on a
multi-class skin disease classification benchmark and visualizes the
resulting confusion matrix. The evaluation is performed using the
SkinDiseaseBenchmark helper, which computes overall accuracy, per-class
accuracy, and the raw confusion matrix based on image–text matching.

The confusion matrix is row-normalized (by true class) to highlight
systematic confusions between disease categories. Long clinical labels
are optionally mapped to shorter aliases for improved readability in
figures. The resulting plot is saved as a publication-ready PNG and is
intended for diagnostic analysis and qualitative comparison between
model variants.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix 
from skin_benchmark import SkinDiseaseBenchmark

model_dir = "/mloscratch/users/turan/evaluation_clip/models/combined_dataset_skin_aggressive_training_config_1_lr5.418484333396616e-05_wd0.20568011432383415_nfrz2"

skin_bench = SkinDiseaseBenchmark(
    train_jsonl="/mloscratch/users/turan/datasets/skin_diseases_10/train_raw.jsonl",
    test_jsonl="/mloscratch/users/turan/datasets/skin_diseases_10/skin10_val_raw.jsonl",
    image_root="/mloscratch/users/turan/datasets/skin_diseases_10",
)

acc, per_class_acc, per_class_total, cm, id2label = skin_bench.evaluate_with_confusion(model_dir)

# original (long) labels
labels = [id2label[i] for i in range(len(id2label))]

# map long labels -> short labels (edit names if your strings differ)
short_label_map = {
    "Eczema": "Eczema",
    "Warts Molluscum and other Viral Infections": "Warts",
    "Melanoma": "Melanoma",
    "Atopic Dermatitis": "Dermatitis",
    "Basal Cell Carcinoma (BCC)": "BCC",
    "Melanocytic Nevi (NV)": "Nevi",
    "Benign Keratosis-like Lesions": "BKL",
    "Psoriasis pictures Lichen Planus and related diseases": "Psoriasis",
    "Seborrheic Keratoses and other Benign Tumors": "Seb. Keratoses",
    "Tinea Ringworm Candidiasis and other Fungal Infections": "Fungal",
}

short_labels = [short_label_map.get(l, l) for l in labels]

# normalize by row (true class)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(6, 6))
plt.imshow(cm_norm, interpolation="nearest", cmap="Blues")
plt.colorbar()

tick_marks = np.arange(len(short_labels))
plt.xticks(tick_marks, short_labels, rotation=45, ha="right")
plt.yticks(tick_marks, short_labels)

plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title(f"Confusion matrix (accuracy={acc:.2f})")

plt.tight_layout()
plt.savefig("confusion_matrix_10_diseases_short_labels.png", bbox_inches="tight")
print("Saved confusion matrix to confusion_matrix_10_diseases_short_labels.png")
