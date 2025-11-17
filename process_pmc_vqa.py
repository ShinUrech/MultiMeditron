from datasets import load_dataset, Image
import torch
import random
import argparse
from typing import List, Dict, Tuple
from transformers import AutoProcessor
import base64
from io import BytesIO
import os

DATA_FILES = {
    "train": "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/train.csv",
    "test":  "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/test.csv",
}

ARCHIVE_URI = "hf://datasets/RadGenome/PMC-VQA/images.zip"
MEMBER_PREFIX = "images/"

# === Utilities ===
def _normalize_choice(text: str) -> str:
    if text is None:
        return ""
    parts = str(text).strip().split(":", 1)
    cleaned = parts[-1] if len(parts) > 1 else parts[0]
    return cleaned.strip()

def extract_choices(entry: dict) -> Dict[str, str]:
    return {
        "A": _normalize_choice(entry.get("Choice A", "")),
        "B": _normalize_choice(entry.get("Choice B", "")),
        "C": _normalize_choice(entry.get("Choice C", "")),
        "D": _normalize_choice(entry.get("Choice D", "")),
    }

def correct_label_and_text(entry: dict, choices: Dict[str, str]) -> Tuple[str, str]:
    label = str(entry.get("Answer_label", "")).strip()
    text = choices.get(label) or str(entry.get("Answer", "")).strip()
    return label, text

def block_header(i: int) -> str:
    return f"*Question {i+1}*"

def format_question(q: str, choices: Dict[str, str]) -> str:
    lines = [f"*Question:* {q}", "*Options:*"]
    for k in ["A", "B", "C", "D"]:
        v = choices.get(k)
        if v:
            lines.append(f"- *{k})* {v}")
    return "\n".join(lines)

def format_rationale(label: str, rationale: str) -> str:
    return f"*Rationale ({label}):*\n{rationale.strip()}"

def format_correct(label: str, text: str) -> str:
    pretty = f"{label} — {text}" if text else label
    return f"*Correct Answer:* {pretty}"

def parse_model_output_for_rationale(generation: str) -> str:
    gen = generation.strip()
    low = gen.lower()
    if "rationale:" in low:
        start_idx = low.find("rationale:")
        body = gen[start_idx + len("rationale:"):].strip()
        fa_idx = body.lower().find("final answer:")
        if fa_idx >= 0:
            return body[:fa_idx].strip()
        return body
    return gen

def build_prompt(entry: dict) -> Tuple[str, Dict[str, str], str, str]:
    q = str(entry.get("Question", "")).strip()
    choices = extract_choices(entry)
    label, text = correct_label_and_text(entry, choices)

    opts_str = "\n".join([f"{k}. {v}" for k, v in choices.items() if v])

    prompt = (
        "Carefully inspect the provided figure and consider the multiple-choice question.\n\n"
        f"Question: {q}\n"
        f"Options:\n{opts_str}\n\n"
        f"The correct answer has already been determined to be option {label} ({text}). "
        "Do NOT change this answer and do NOT suggest a different option.\n\n"
        "Your task is to generate a rationale that explains why this answer is correct:\n"
        f"- Explicitly state in your explanation that the correct response is option {label} ({text}).\n"
        "- Infer the most plausible figure type and imaging modality when possible (e.g., X-ray, CT, MRI, FLAIR MRI, PET/CT, ultrasound, "
        "angiography, histology slide, fundus photograph, microscopy, or a biomedical chart/graph). "
        "Only specify a detailed MRI sequence such as FLAIR when this is clearly supported by the image appearance, labels, or the question/options.\n"
        "- Describe the key visual findings (location, shape, size, intensity/signal, density, color, patterns, annotations, or text labels) that support this option.\n"
        f"- Explain clearly why these findings make option {label} ({text}) the best choice.\n"
        "- Briefly explain why each of the other options is not correct or is less consistent with the visible findings or clinical context.\n"
        "- Use cautious medical language: when you mention a specific disease entity or histopathologic diagnosis, present it as likely/most consistent or suggestive, "
        "and note that definitive confirmation would require appropriate clinical correlation and, when relevant, biopsy or additional testing.\n"
        "- Keep the explanation self-contained, clinically focused, and free of meta-comments about prompts, datasets, or your own reasoning process.\n\n"
        "Write 4–7 sentences in a single coherent paragraph (no bullet points or numbered lists).\n\n"
        "Format:\n"
        "Rationale: <your paragraph explicitly stating and justifying the correct option>\n"
        f"Final Answer: {label}\n"
    )
    return prompt, choices, label, text



def pil_to_data_url(pil_img, fmt="JPEG", quality=90) -> str:
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

def maybe_downscale(pil_img, max_side: int = 1280):
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = max_side / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return pil_img.resize(new_size, Image.LANCZOS)

def gpt5_generate(pil_image, prompt_text: str, gpt5_model: str = "gpt-5.1", max_output_tokens: int = 512) -> str:
    """
    Uses OpenAI Responses API with vision input. Requires OPENAI_API_KEY and openai>=1.x.
    """
    import os
    from io import BytesIO
    import base64
    from PIL import Image

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("OpenAI package not installed. `pip install openai`") from e

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    client = OpenAI(api_key=api_key)

    def _maybe_downscale(img, max_side: int = 1280):
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS)

    pil_image = _maybe_downscale(pil_image)

    buf = BytesIO()
    pil_image.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    resp = client.responses.create(
        model=gpt5_model,
        max_output_tokens=max_output_tokens,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": data_url},
                {"type": "input_text", "text": prompt_text},
            ]
        }]
    )

    try:
        return resp.output_text.strip()
    except Exception:
        txt = ""
        if hasattr(resp, "output") and isinstance(resp.output, list) and resp.output:
            parts = resp.output[0].get("content", [])
            for p in parts:
                if p.get("type") == "output_text":
                    txt += p.get("text", "")
        return txt.strip()


# ==== Messaging builder (shared) ===
def prepare_messages_for_entry(entry: dict) -> List[dict]:
    prompt, _, _, _ = build_prompt(entry)
    image_obj_or_url = entry.get("image_url")
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": (
                            "You are a board-certified clinician and medical imaging expert. "
                            "You interpret a wide range of medical figures, including radiology (X-ray, CT, MRI, PET/CT, ultrasound, angiography), "
                            "nuclear medicine, endoscopy, histopathology, microscopy, ophthalmology (fundus and OCT), cardiology imaging, "
                            "and other biomedical figures or charts."
                    ),
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj_or_url},
                {"type": "text", "text": prompt},
            ],
        },
    ]
    return messages, prompt


# === Main ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--k", type=int, default=1)  # default: one random question
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpt5_model", type=str, default="gpt-5.1", help="OpenAI model name")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    # === Load dataset & link images from remote zip ===
    ds = load_dataset("csv", data_files=DATA_FILES)
    dset = ds[args.split]

    def to_remote_zip(example):
        example["image_url"] = f"zip://{MEMBER_PREFIX}{example['Figure_path']}::{ARCHIVE_URI}"
        return example

    dset = dset.map(to_remote_zip, desc="Linking figures in remote zip")
    dset = dset.cast_column("image_url", Image())

    # Move image_url first (nice to inspect)
    cols = dset.column_names
    if "image_url" in cols:
        cols.remove("image_url")
        dset = dset.select_columns(["image_url"] + cols)

    # === Sample k random entries (k defaults to 1 for this task) ===
    idxs = list(range(len(dset)))
    random.shuffle(idxs)
    idxs = idxs[: args.k]

    out_lines = []
    for i, idx in enumerate(idxs):
        entry = dset[int(idx)]
        q = str(entry.get("Question", "")).strip()
        print(f"Figure path: {entry['Figure_path']}")  # for debugging / traceability
        choices = extract_choices(entry)
        label, text = correct_label_and_text(entry, choices)

        # Build prompt text used for GPT-5
        _, _, _, _ = build_prompt(entry)
        messages, prompt_text = prepare_messages_for_entry(entry)

        # Get PIL image for GPT-5 (Vision requires actual pixels or http URL; zip:// won't work)
        pil_img = entry["image_url"]  # datasets.Image returns a PIL.Image here

        try:
            gpt5_raw = gpt5_generate(pil_img, prompt_text, gpt5_model=args.gpt5_model, max_output_tokens=600)
            gpt5_rationale = parse_model_output_for_rationale(gpt5_raw)
        except Exception as e:
            gpt5_rationale = f"[GPT-5 error: {e}]"

        out_lines.append(block_header(i))
        out_lines.append(format_question(q, choices))
        out_lines.append(format_rationale("GPT-5.1:", gpt5_rationale))  # GPT-5 rationale
        out_lines.append(format_correct(label, text))
        out_lines.append("---")

    final_text = "\n".join(out_lines).strip()
    print(final_text)

if __name__ == "__main__":
    main()
