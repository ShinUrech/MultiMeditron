#!/usr/bin/env python

from datasets import load_dataset, Image, Dataset
import argparse
import base64
from io import BytesIO
import os
import json
import time
from typing import List, Dict, Tuple
from PIL import Image as PILImage


DATA_FILES = {
    "train": "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/train.csv",
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
    return f"Question {i+1}"


def format_question(q: str, choices: Dict[str, str]) -> str:
    lines = [f"Question: {q}", "Options:"]
    for k in ["A", "B", "C", "D"]:
        v = choices.get(k)
        if v:
            lines.append(f"- {k}) {v}")
    return "\n".join(lines)


def format_rationale(label: str, rationale: str) -> str:
    return f"Rationale ({label}):\n{rationale.strip()}"


def format_correct(label: str, text: str) -> str:
    pretty = f"{label} — {text}" if text else label
    return f"Correct Answer: {pretty}"


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


def build_user_message_content(question: str, choices: Dict[str, str]) -> str:
    """
    Build the user message with the reserved special token.
    """
    lines = ["<|reserved_special_token_0|>", "\n"
        "Based on the image, answer the question:",
        "",
        question.strip(),
    ]
    for k in ["A", "B", "C", "D"]:
        v = choices.get(k)
        if v:
            lines.append(f"{k}: {v}")
    return "\n".join(lines)


def build_prompt(entry: dict) -> Tuple[str, Dict[str, str], str, str]:
    """
    Prompt used for the batch Responses API call.
    """
    q = str(entry.get("Question", "")).strip()
    choices = extract_choices(entry)
    label, text = correct_label_and_text(entry, choices)

    # User-facing question block with reserved token
    user_block = build_user_message_content(q, choices)

    prompt = (
        "Carefully inspect the provided figure and consider the multiple-choice question.\n\n"
        f"{user_block}\n\n"
        f"The correct answer has already been determined to be option {label} ({text}). "
        "Do NOT change this answer and do NOT suggest a different option.\n\n"
        "The task is to generate a rationale that explains why this answer is correct:\n"
        f"- Explicitly state in the explanation that the correct response is option {label} ({text}).\n"
        "- Infer the most plausible figure type and imaging modality when possible (e.g., X-ray, CT, MRI, FLAIR MRI, PET/CT, ultrasound, "
        "angiography, histology slide, fundus photograph, microscopy, or a biomedical chart/graph). "
        "Only specify a detailed MRI sequence such as FLAIR when this is clearly supported by the image appearance, labels, or the question/options.\n"
        "- Describe the key visual findings (location, shape, size, intensity/signal, density, color, patterns, annotations, or text labels) that support this option.\n"
        f"- Explain clearly why these findings make option {label} ({text}) the best choice.\n"
        "- Briefly explain why each of the other options is not correct or is less consistent with the visible findings or clinical context.\n"
        "- Use cautious medical language: when a specific disease entity or histopathologic diagnosis is mentioned, present it as likely/most consistent or suggestive, "
        "and note that definitive confirmation would require appropriate clinical correlation and, when relevant, biopsy or additional testing.\n"
        "- Keep the explanation self-contained, clinically focused, and free of meta-comments about prompts, datasets, or the reasoning process.\n\n"
        "Write 4–7 sentences in a single coherent paragraph (no bullet points or numbered lists).\n\n"
        "Format:\n"
        "Rationale: <a paragraph explicitly stating and justifying the correct option>\n"
        f"Final Answer: {label}\n"
    )
    return prompt, choices, label, text


def maybe_downscale(pil_img, max_side: int = 1280):
    w, h = pil_img.size
    if max(w, h) <= max_side:
        return pil_img
    scale = max_side / float(max(w, h))
    new_size = (int(w * scale), int(h * scale))
    return pil_img.resize(new_size, PILImage.LANCZOS)


def image_to_bytes_and_data_url(pil_img, fmt="JPEG", quality=90):
    pil_img = maybe_downscale(pil_img)
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format=fmt, quality=quality)
    img_bytes = buf.getvalue()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/{fmt.lower()};base64,{b64}"
    return img_bytes, data_url


def build_assistant_message_content(label: str, answer_text: str, rationale_only: str) -> str:
    rationale_only = rationale_only.strip()
    parts = []
    if rationale_only:
        parts.append(rationale_only)
    parts.append("")
    parts.append(f"Therefore, the correct answer is {label}: {answer_text}.")
    parts.append(f"**Answer**: {label}")
    return "\n".join(parts)


# ---- Optional: message builder for direct Responses API (not used in batch) ----
def prepare_messages_for_entry(entry: dict) -> List[dict]:
    prompt, _, _, _ = build_prompt(entry)
    image_obj_or_url = entry.get("image")
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


# === Batch helpers ===
def extract_output_text_from_response_obj(resp_obj: dict) -> str:
    """
    Extract raw text output from a Responses API response object.
    """
    if not resp_obj:
        return ""
    txt = resp_obj.get("output_text")
    if isinstance(txt, str) and txt.strip():
        return txt.strip()

    output = resp_obj.get("output")
    if isinstance(output, list) and output:
        msg = output[0]
        content = msg.get("content", [])
        pieces = []
        for c in content:
            if c.get("type") == "output_text":
                pieces.append(c.get("text", ""))
        if pieces:
            return "".join(pieces).strip()
    return str(resp_obj).strip()


def create_batch_requests_file(
    dataset,
    split: str,
    jsonl_path: str,
    gpt5_model: str,
    max_output_tokens: int = 2048,
):
    """
    Write one JSONL batch request file for the train split.
    """
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(dataset):
            prompt, _, _, _ = build_prompt(entry)
            pil_img = entry["image"]
            _, data_url = image_to_bytes_and_data_url(pil_img)

            body = {
                "model": gpt5_model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_image", "image_url": data_url},
                            {"type": "input_text", "text": prompt},
                        ],
                    }
                ],
                "max_output_tokens": max_output_tokens,
            }

            obj = {
                "custom_id": f"{split}-{idx}",
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
            f.write(json.dumps(obj) + "\n")


def submit_and_wait_for_batch(
    jsonl_path: str,
    endpoint: str = "/v1/responses",
    poll_interval: int = 3600,
):
    """
    Submit a batch job and block until it finishes.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    client = OpenAI(api_key=api_key)

    print(f"Uploading batch requests file: {jsonl_path}")
    batch_input_file = client.files.create(
        file=open(jsonl_path, "rb"), purpose="batch"
    )
    print(f"Created input file: {batch_input_file.id}")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=endpoint,
        completion_window="24h",
    )
    print(f"Created batch: {batch.id}, initial status: {batch.status}")

    meta_path = os.path.join(os.path.dirname(jsonl_path), "batch_meta.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump({"batch_id": batch.id, "input_file_id": batch_input_file.id}, mf)

    terminal_states = {"completed", "failed", "expired", "cancelled"}
    while batch.status not in terminal_states:
        print(f"[Batch {batch.id}] Status: {batch.status} ... sleeping {poll_interval}s")
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)

    print(f"[Batch {batch.id}] Final status: {batch.status}")
    if batch.status != "completed":
        raise RuntimeError(
            f"Batch {batch.id} did not complete successfully (status={batch.status})"
        )

    return batch


def download_batch_output(batch, output_jsonl_path: str):
    """
    Download the batch output file to disk.
    """
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    client = OpenAI(api_key=api_key)

    os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)

    if not getattr(batch, "output_file_id", None):
        raise RuntimeError(f"Batch {batch.id} has no output_file_id")

    print(f"Downloading batch output file {batch.output_file_id} -> {output_jsonl_path}")
    response_stream = client.files.content(batch.output_file_id)
    with open(output_jsonl_path, "wb") as f:
        f.write(response_stream.read())


def load_rationales_from_batch_output(output_jsonl_path: str) -> Dict[str, str]:
    """
    Parse the batch output JSONL and return {custom_id: raw_generation_text}.
    """
    mapping: Dict[str, str] = {}
    with open(output_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id")
            resp_obj = obj.get("response")
            err = obj.get("error")
            if resp_obj:
                text = extract_output_text_from_response_obj(resp_obj)
            else:
                text = f"[Batch error: {err}]"
            if cid is not None:
                mapping[cid] = text
    return mapping


def load_pmc_train_split(num_proc: int = 8):
    """
    Load the PMC-VQA train split, attach remote zip image URLs,
    and cast 'image' column to datasets.Image.
    """
    ds = load_dataset("csv", data_files=DATA_FILES)
    dataset = ds["train"]

    def to_remote_zip(example):
        rel = example["Figure_path"]
        zip_url = f"zip://{MEMBER_PREFIX}{rel}::{ARCHIVE_URI}"
        example["raw_image_url"] = zip_url
        example["image"] = zip_url
        return example

    dataset = dataset.map(
        to_remote_zip,
        desc="Linking figures in remote zip (train)",
        num_proc=num_proc,
    )
    dataset = dataset.cast_column("image", Image())

    cols = dataset.column_names
    for c in ["image", "raw_image_url"]:
        if c in cols:
            cols.remove(c)
    dataset = dataset.select_columns(["image", "raw_image_url"] + cols)
    return dataset


def build_pmc_multimodal_dataset(
    dataset,
    rats: Dict[str, str],
    split: str = "train",
) -> Dataset:
    """
    Build the final Dataset with columns ['modalities', 'conversations'].
    """
    def gen():
        for idx, entry in enumerate(dataset):
            custom_id = f"{split}-{idx}"
            raw_generation = rats.get(custom_id, "")
            rationale_only = parse_model_output_for_rationale(raw_generation)

            q = str(entry.get("Question", "")).strip()
            choices = extract_choices(entry)
            label, answer_text = correct_label_and_text(entry, choices)

            user_content = build_user_message_content(q, choices)
            assistant_content = build_assistant_message_content(
                label, answer_text, rationale_only
            )

            pil_img = entry["image"]
            img_bytes, _ = image_to_bytes_and_data_url(pil_img)
            img_path = entry["raw_image_url"]

            modalities = [
                [
                    {
                        "type": "image",
                        "value": {
                            "bytes": img_bytes,
                            "path": img_path,
                        },
                    }
                ]
            ]
            conversations = [
                {"content": user_content, "role": "user"},
                {"content": assistant_content, "role": "assistant"},
            ]

            yield {
                "modalities": modalities,
                "conversations": conversations,
            }

    pmc = Dataset.from_generator(gen)
    return pmc


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess PMC-VQA train split into Multimediset format using the OpenAI Batch API."
    )
    parser.add_argument(
        "--gpt5_model", type=str, default="gpt-5.1", help="OpenAI model name"
    )
    parser.add_argument("--max_output_tokens", type=int, default=1024)
    parser.add_argument(
        "--work_dir",
        type=str,
        default="/capstor/store/cscs/swissai/a127/homes/theoschiff/data/pmc_vqa",
        help="Directory for batch JSONL and outputs",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/capstor/store/cscs/swissai/a127/meditron/multimediset/arrow/PMC_VQA_FULL",
    )
    parser.add_argument(
        "--poll_interval",
        type=int,
        default=3600,
        help="Seconds between batch status polls (default: 3600 = 1h)",
    )
    parser.add_argument(
        "--reuse_outputs",
        action="store_true",
        help="If set, reuse existing batch results JSONL if present.",
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=8,
        help="Number of processes for Dataset map().",
    )

    args = parser.parse_args()

    split = "train"
    os.makedirs(args.work_dir, exist_ok=True)

    print("Loading PMC-VQA train split...")
    dataset = load_pmc_train_split(num_proc=args.num_proc)

    requests_jsonl = os.path.join(args.work_dir, "requests_train.jsonl")
    results_jsonl = os.path.join(args.work_dir, "results_train.jsonl")

    if args.reuse_outputs and os.path.exists(results_jsonl):
        print(f"Reusing existing batch output for train: {results_jsonl}")
    else:
        print("Building batch requests file for train...")
        create_batch_requests_file(
            dataset, split, requests_jsonl, args.gpt5_model, args.max_output_tokens
        )

        print("Submitting batch...")
        batch = submit_and_wait_for_batch(
            requests_jsonl,
            endpoint="/v1/responses",
            poll_interval=args.poll_interval,
        )
        print("Batch completed. Downloading outputs...")
        download_batch_output(batch, results_jsonl)

    print("Parsing batch outputs for train...")
    rats = load_rationales_from_batch_output(results_jsonl)
    print(f"Collected rationales for train: {len(rats)} examples")

    print("Building final PMC_VQA_FULL dataset with modalities and conversations...")
    pmc = build_pmc_multimodal_dataset(dataset, rats, split=split)
    print(pmc)

    print(f"Saving dataset to {args.out_dir}")
    pmc.save_to_disk(args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
