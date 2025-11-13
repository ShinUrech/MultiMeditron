from datasets import load_dataset, Image
import torch
import random
import argparse
from typing import List, Dict, Tuple
from transformers import AutoProcessor, BitsAndBytesConfig
import base64
from io import BytesIO
import os

# === optional 4-bit quantization for MedGemma ===
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

DATA_FILES = {
    "train": "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/train.csv",
    "test":  "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/test.csv",
}

ARCHIVE_URI = "hf://datasets/RadGenome/PMC-VQA/images.zip"
MEMBER_PREFIX = "images/"

# =============== Utilities ===============
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

def build_prompt(entry: dict) -> Tuple[str, Dict[str,str], str, str]:
    q = str(entry.get("Question", "")).strip()
    choices = extract_choices(entry)
    label, text = correct_label_and_text(entry, choices)
    prompt = (
        "You are a Medical Expert.\n"
        "Carefully inspect the figure and answer the multiple-choice question.\n\n"
        f"Question: {q}\n"
        "Options:\n" + "\n".join([f"{k}. {v}" for k, v in choices.items() if v]) + "\n\n"
        f"Task: Explain succinctly (3–5 sentences) why the correct answer is {label} ({text}), "
        "citing visible image evidence and clinically relevant reasoning. "
        "Avoid chain-of-thought or step-by-step scratchpads; give a clear, self-contained rationale.\n"
        "Format:\nRationale: ...\nFinal Answer: <letter>"
    )
    return prompt, choices, label, text

def pil_to_data_url(pil_img, fmt="JPEG", quality=90) -> str:
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format=fmt, quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"

# =============== Model 2: MedGemma (transformers) ===============
def load_medgemma(model_name: str, use_4bit: bool):
    from transformers import AutoModelForImageTextToText
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
    if use_4bit:
        kwargs["quantization_config"] = bnb
    model = AutoModelForImageTextToText.from_pretrained(model_name, **kwargs)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def medgemma_generate(messages, model, processor, max_new_tokens=512, do_sample=False) -> str:
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device, dtype=torch.bfloat16)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
        gen_ids = out[0][input_len:]
    return processor.decode(gen_ids, skip_special_tokens=True)

def maybe_downscale(pil_img, max_side: int = 1280):
    """Downscale very large images to keep the data URL size reasonable."""
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = max_side / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return pil_img.resize(new_size, Image.LANCZOS)

# =============== Model 1: GPT-5 ===============
def gpt5_generate(pil_image, prompt_text: str, gpt5_model: str = "gpt-5", max_output_tokens: int = 512) -> str:
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

    def maybe_downscale(img, max_side: int = 1280):
        w, h = img.size
        if max(w, h) <= max_side:
            return img
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        return img.resize(new_size, Image.LANCZOS)

    pil_image = maybe_downscale(pil_image)

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



# =============== Messaging builder (shared) ===============
def prepare_messages_for_entry(entry: dict) -> List[dict]:
    prompt, _, _, _ = build_prompt(entry)
    image_obj_or_url = entry.get("image_url")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
        {"role": "user", "content": [
            {"type": "image", "image": image_obj_or_url},
            {"type": "text", "text": prompt},
        ]},
    ]
    return messages, prompt

# =============== Main ===============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_4bit", action="store_true", default=True, help="Disable 4-bit quantization for MedGemma")
    parser.add_argument("--medgemma_name", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--gpt5_model", type=str, default="gpt-5", help="OpenAI model name for Model1")
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

    # === Load Model 2 (MedGemma) ===
    use_4bit = not args.no_4bit
    medgemma_model, medgemma_proc = load_medgemma(args.medgemma_name, use_4bit)

    # === Sample k random entries ===
    idxs = list(range(len(dset)))
    random.shuffle(idxs)
    idxs = idxs[: args.k]

    out_lines = []
    for i, idx in enumerate(idxs):
        entry = dset[int(idx)]
        q = str(entry.get("Question", "")).strip()
        print(f"Figure path: {entry['Figure_path']}")  # for debugging
        choices = extract_choices(entry)
        label, text = correct_label_and_text(entry, choices)

        # Build messages & prompt (shared text prompt used for GPT-5)
        messages, prompt_text = prepare_messages_for_entry(entry)

        # Get PIL image for GPT-5 (Vision requires actual pixels or http URL; zip:// won't work)
        pil_img = entry["image_url"]  # datasets.Image returns a PIL.Image here

        # === Model 1: GPT-5 ===
        try:
            gpt5_raw = gpt5_generate(pil_img, prompt_text, gpt5_model=args.gpt5_model, max_output_tokens=600)
            gpt5_rationale = parse_model_output_for_rationale(gpt5_raw)
        except Exception as e:
            gpt5_rationale = f"[GPT-5 error: {e}]"

        # === Model 2: MedGemma-4B-IT ===
        med_raw = medgemma_generate(messages, medgemma_model, medgemma_proc, max_new_tokens=512)
        med_rationale = parse_model_output_for_rationale(med_raw)

        # === Slack-friendly block ===
        out_lines.append(block_header(i))
        out_lines.append(format_question(q, choices))
        out_lines.append(format_rationale("Model1:", gpt5_rationale)) # GPT-5
        out_lines.append(format_rationale("Model2:", med_rationale)) # MedGemma-4B-IT
        out_lines.append(format_correct(label, text))
        out_lines.append("---")  # divider

    final_text = "\n".join(out_lines).strip()
    print(final_text)

if __name__ == "__main__":
    main()
