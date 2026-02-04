
from transformers import VisionTextDualEncoderProcessor, AutoTokenizer
from load_from_clip import load_model
from PIL import Image
from pathlib import Path
import torch
import json
import random

"""
CLIP Image–Text Alignment Evaluation (Recall@1, 4-Way Forced Choice)

This script evaluates image–text alignment for CLIP-style vision–language models
using a controlled retrieval setting suitable for medical datasets.

Evaluation Protocol
-------------------
For each query image, the model is asked to rank one ground-truth caption
against three randomly sampled negative captions drawn from the same validation
split. A prediction is counted as correct if the ground-truth caption receives
the highest cosine similarity score.

This corresponds to Recall@1 under a 4-candidate forced-choice setup
(chance level = 25%), rather than global retrieval over the full dataset.
The metric is designed to probe fine-grained cross-modal alignment while
controlling difficulty and computational cost.

Procedure
---------
1. Load a CLIP-compatible vision–language model and processor.
2. Perform text-tower and image-tower sanity probes using diverse synthetic
   inputs to detect representation collapse.
3. For each evaluation sample:
   - Use the image as the query.
   - Compare cosine similarity against one positive and three negative captions.
   - Record whether the positive caption ranks first.
4. Report average Recall@1 over all valid samples.

Notes
-----
- Similarities are computed using L2-normalized embeddings and cosine similarity.
- Negatives are randomly sampled per query (seeded for reproducibility).
- Image paths are resolved relative to the JSONL manifest location.
- This evaluation is intended for comparative analysis between models and
  fine-tuned checkpoints, not for benchmarking against large-scale retrieval
  datasets such as COCO or Flickr30k.

Typical Usage
-------------
- Compare base CLIP vs. domain-adapted (e.g., ophthalmology- or dermatology-
  fine-tuned) models.
- Sanity-check representation quality before large-scale training or deployment.
- Evaluate alignment improvements under controlled retrieval difficulty.
"""

LINE_NUMBER = 1000
EVAL_DATASETS = [
    "/mloscratch/users/turan/datasets/opthalmology_expert_datasets/eyepacs/eyepacs_val.jsonl",
]
LOG_DIR = Path("/mloscratch/users/turan/evaluation_clip/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

random.seed(14)
torch.manual_seed(14)
device = "cuda" if torch.cuda.is_available() else "cpu"

# return the similarity between an image and a text according to the given model
@torch.no_grad()
def get_similarity(text, image_path, model, processor) -> float:
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=[text], images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)

    image_embeds = outputs.image_embeds   # (1, D)
    text_embeds = outputs.text_embeds     # (1, D)

    a_norm = torch.nn.functional.normalize(image_embeds, dim=1)
    b_norm = torch.nn.functional.normalize(text_embeds, dim=1)

    similarity = torch.matmul(a_norm, b_norm.T)  # (1,1)
    return float(similarity.item())


@torch.no_grad()
def evaluate_model(model_name_or_path: str, eval_dataset: str, log_fp=None) -> float:
    # small helper to log to both stdout and file
    def log_print(msg: str):
        print(msg)
        if log_fp is not None:
            log_fp.write(msg + "\n")
            log_fp.flush()

    clip_model = load_model(model_name_or_path).to(device).eval()
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # ---- Text tower probe: test 6 diverse texts ----
    probe_texts = [
        "a cat sitting on a couch",
        "retinal fundus image with drusen",
        "an x-ray showing lung consolidation",
        "a photo of a dog running in a field",
        "a patient with diabetic retinopathy",
        "a wooden chair next to a table",
    ]

    probe_inputs = processor(
        text=probe_texts,
        images=[Image.new("RGB", (10, 10))] * 6,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        probe_out = clip_model(**probe_inputs)
        text_embs = probe_out.text_embeds  # (6, D)

    pairwise = torch.nn.functional.cosine_similarity(
        text_embs.unsqueeze(1), text_embs.unsqueeze(0), dim=-1
    )

    log_print(f"\n=== Text tower probe for {model_name_or_path} ===")
    log_print("pairwise cosine matrix:")
    for row in pairwise.cpu().numpy():
        log_print("   " + " ".join(f"{v:0.3f}" for v in row))

    unique = len(torch.unique(text_embs.cpu(), dim=0))
    log_print(f"unique text embeddings: {unique} / 6\n")

    # ---- Image tower probe: 6 simple synthetic images ----
    probe_images = []
    for i in range(6):
        img = Image.new(
            "RGB",
            (10, 10),
            color=(int((i * 40) % 256), int((i * 80) % 256), int((i * 120) % 256)),
        )
        probe_images.append(img)

    image_probe_inputs = processor(
        text=["dummy"] * 6,  # same text, different images
        images=probe_images,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        image_probe_out = clip_model(**image_probe_inputs)
        image_embs = image_probe_out.image_embeds  # (6, D)

    image_pairwise = torch.nn.functional.cosine_similarity(
        image_embs.unsqueeze(1), image_embs.unsqueeze(0), dim=-1
    )

    log_print(f"\n=== Image tower probe for {model_name_or_path} ===")
    log_print("pairwise cosine matrix:")
    for row in image_pairwise.cpu().numpy():
        log_print("   " + " ".join(f"{v:0.3f}" for v in row))

    image_unique = len(torch.unique(image_embs.cpu(), dim=0))
    log_print(f"unique image embeddings: {image_unique} / 6\n")

    # ---- load jsonl lines for this dataset ----
    with open(eval_dataset, "r", encoding="utf-8") as file:
        all_lines = file.readlines()

    if not all_lines:
        raise ValueError(f"No lines found in {eval_dataset}")

    N = min(LINE_NUMBER, len(all_lines))
    lines = all_lines[:N]
    base_dir = Path(eval_dataset).parent

    good_guess = 0
    used = 0

    for i, line in enumerate(lines):
        # sample 3 distinct negatives != i
        candidates = list(range(N))
        candidates.remove(i)
        a, b, c = random.sample(candidates, 3)

        correct_line = json.loads(line)
        a_line = json.loads(lines[a])
        b_line = json.loads(lines[b])
        c_line = json.loads(lines[c])

        texts = [
            correct_line["text"],
            a_line["text"],
            b_line["text"],
            c_line["text"],
        ]

        # ---- resolve image path ----
        rel_path = correct_line["modalities"][0]["value"]
        p = Path(rel_path)
        if not p.is_absolute():
            p = (base_dir / p).resolve()

        if not p.exists():
            # Skip this sample if the image file doesn't exist
            continue

        image_path = str(p)

        model_similarities = []
        for t in texts:
            tokens = tokenizer.encode(t, truncation=True, max_length=500)
            text_value = tokenizer.decode(tokens, skip_special_tokens=True)

            sim = get_similarity(text_value, image_path, clip_model, processor)
            model_similarities.append(sim)

        pred_idx = int(torch.tensor(model_similarities).argmax().item())
        if pred_idx == 0:
            good_guess += 1

        used += 1
        if used >= N:
            break

    if used == 0:
        raise ValueError("No valid samples used (all images missing?).")

    acc = good_guess / used
    log_print(f"Used {used} samples, accuracy = {acc:.4f}")
    return acc

def main():
    clips = [
    ("finetuned_clip_2", "/mloscratch/users/turan/training/models_opthalmology/combined_dataset_opthalmology_fine_tuning_config_2"),
    ]

    for name, model_id in clips:
        model_slug = Path(model_id).name  # last path component
        log_path = LOG_DIR / f"{model_slug}.txt"

        print(f"\n========== Logging for config: {name} ({model_id}) ==========")
        print(f"Log file: {log_path}")

        with open(log_path, "w", encoding="utf-8") as log_fp:
            for eval_dataset in EVAL_DATASETS:
                dataset_slug = Path(eval_dataset).stem
                header = f"\n\n===== Evaluating model {name} ({model_id}) on dataset {dataset_slug} ====="
                print(header)
                log_fp.write(header + "\n")
                log_fp.flush()

                acc = evaluate_model(model_id, eval_dataset, log_fp=log_fp)

                summary = f"{name} accuracy on {dataset_slug}: {acc:.4f}"
                print(summary)
                log_fp.write(summary + "\n")
                log_fp.flush()

if __name__ == "__main__":
    main()

