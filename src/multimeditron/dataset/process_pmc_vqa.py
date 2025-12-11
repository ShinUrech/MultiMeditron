import argparse
import base64
import os
import json
import time
import re
from datasets import load_dataset, Image, Dataset
from io import BytesIO
from openai import OpenAI, InternalServerError, APIConnectionError, RateLimitError
from PIL import Image as PILImage
from tqdm import tqdm
from typing import List, Dict, Tuple

DATA_FILES = {
    "train": "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/train.csv",
}

ARCHIVE_URI = "hf://datasets/RadGenome/PMC-VQA/images.zip"
MEMBER_PREFIX = "images/"

# local image directory
BASE_IMAGE_DIR = "/capstor/store/cscs/swissai/a127/homes/theoschiff/data/images"


# === Utilities ===
def _normalize_choice(text: str) -> str:
    if text is None:
        return ""
    parts = str(text).strip().split(":", 1)
    cleaned = parts[-1] if len(parts) > 1 else parts[0]
    return cleaned.strip()


def extract_choices(entry: dict) -> Dict[str, str]:
    """
    Extract and normalize the multiple-choice options from the dataset entry.
    Returns a dict mapping "A", "B", "C", "D" to the
    corresponding choice text.

    ### Args
    * entry: A dictionary representing a dataset entry.
    ### Returns
    * A dictionary mapping choice labels ("A", "B", "C", "D") to their text.
    """

    return {
        "A": _normalize_choice(entry.get("Choice A", "")),
        "B": _normalize_choice(entry.get("Choice B", "")),
        "C": _normalize_choice(entry.get("Choice C", "")),
        "D": _normalize_choice(entry.get("Choice D", "")),
    }


def correct_label_and_text(entry: dict, choices: Dict[str, str]) -> Tuple[str, str]:
    """
    Get the correct answer label and corresponding text from the dataset entry.
    ### Args
    * entry: A dictionary representing a dataset entry.
    * choices: A dictionary mapping choice labels to their text.

    ### Returns
    * A tuple containing the correct answer label and corresponding text.
    """
    label = str(entry.get("Answer_label", "")).strip()
    text = choices.get(label) or str(entry.get("Answer", "")).strip()
    return label, text


def block_header(i: int) -> str:
    """
    Generate a header string for question i (0-based).
    ### Args
    * i: The index of the question (0-based).
    ### Returns
    * A formatted header string.
    """
    return f"Question {i+1}"


def format_question(q: str, choices: Dict[str, str]) -> str:
    """
    Format the question and choices into a string.
    ### Args
    * q: The question text.
    * choices: A dictionary mapping choice labels to their text.
    ### Returns
    * A formatted string containing the question and its choices.
    """
    lines = [f"Question: {q}", "Options:"]
    for k in ["A", "B", "C", "D"]:
        v = choices.get(k)
        if v:
            lines.append(f"- {k}) {v}")
    return "\n".join(lines)


def format_rationale(label: str, rationale: str) -> str:
    """
    Format the rationale string with the correct label.
    ### Args
    * label: The correct answer label.
    * rationale: The rationale text.
    ### Returns
    * A formatted string containing the rationale.
    """
    return f"Rationale ({label}):\n{rationale.strip()}"


def format_correct(label: str, text: str) -> str:
    """
    Format the correct answer string.
    ### Args
    * label: The correct answer label.
    * text: The correct answer text.
    ### Returns
    * A formatted string containing the correct answer.
    """
    pretty = f"{label} — {text}" if text else label
    return f"Correct Answer: {pretty}"


def parse_model_output_for_rationale(generation: str) -> str:
    """
    Parse the model's generation to extract the rationale text.
    ### Args
    * generation: The raw generation text from the model.
    ### Returns
    * The extracted rationale text.
    """
    gen = generation.strip()
    low = gen.lower()
    if "rationale:" in low:
        start_idx = low.find("rationale:")
        body = gen[start_idx + len("rationale:"):].strip()
        fa_idx = body.lower().find("answer:")
        if fa_idx >= 0:
            return body[:fa_idx].strip()
        return body
    return gen


def build_user_message_content(question: str, choices: Dict[str, str]) -> str:
    """
    Build the user message with the reserved special token.
    ### Args
    * question: The question text.
    * choices: A dictionary mapping choice labels to their text.
    ### Returns
    * A formatted string for the user message content.
    """
    lines = ["Based on the image, answer the question:", question.strip()]
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

    # User-facing question block
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
        f"Answer: {label}\n"
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
    """
    Convert a PIL.Image to bytes and a data URL.
    Downscale the image if its largest side exceeds max_side.
    """
    pil_img = maybe_downscale(pil_img)
    buf = BytesIO()
    pil_img.convert("RGB").save(buf, format=fmt, quality=quality)
    img_bytes = buf.getvalue()
    b64 = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/{fmt.lower()};base64,{b64}"
    return img_bytes, data_url


def build_assistant_message_content(label: str, answer_text: str, rationale_only: str) -> str:
    """
    Build the assistant message content with rationale and final answer.
    """
    rationale_only = rationale_only.strip()
    parts = []
    if rationale_only:
        parts.append(rationale_only)
    parts.append("")
    parts.append(f"Therefore, the correct answer is {label}: {answer_text}.")
    parts.append(f"**Answer**: {label}")
    return "\n".join(parts)


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


def _entry_to_pil(entry_image):
    """
    Convert the image field from the dataset entry to a PIL.Image.
    Handles dict from Image(decode=False) and direct paths.
    """
    if isinstance(entry_image, dict):
        path = entry_image.get("path")
        return PILImage.open(path)
    if isinstance(entry_image, str):
        return PILImage.open(entry_image)
    return entry_image


def create_batch_requests_file(
    dataset,
    split: str,
    jsonl_path: str,
    gpt5_model: str,
    max_output_tokens: int = 1024,
):
    """
    Write one JSONL batch request file for the train split.
    ### Args
    * dataset: The Dataset object for the split.
    * split: The dataset split name (e.g., "train").
    * jsonl_path: Path to save the JSONL requests file.
    * gpt5_model: The OpenAI GPT-5 model name to use.
    * max_output_tokens: Maximum output tokens for the model.
    """
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, entry in tqdm(enumerate(dataset), total=len(dataset), desc="Building batch requests"):
            prompt, _, _, _ = build_prompt(entry)
            pil_img = _entry_to_pil(entry["image"])
            _, data_url = image_to_bytes_and_data_url(pil_img)

            body = {
                "model": gpt5_model,
                "input": [
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
                            },
                        ],
                    },
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
    api_key: str = os.getenv("OPENAI_API_KEY"),
    endpoint: str = "/v1/responses",
):
    """
    Submit a batch job (with retries on transient errors) and return the batch object.
    ### Args
    * jsonl_path: Path to the JSONL requests file.
    * api_key: OpenAI API key.
    * endpoint: The API endpoint for the batch (default: "/v1/responses").
    ### Returns
    * The created batch object.
    """
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    client = OpenAI(api_key=api_key)

    print(jsonl_path)

    max_retries = 5
    base_delay = 5  # seconds

    # === robust files.create with exponential backoff ===
    batch_input_file = None
    for attempt in range(max_retries):
        try:
            with open(jsonl_path, "rb") as request_file:
                batch_input_file = client.files.create(
                    file=request_file,
                    purpose="batch",
                )
            # success -> break out of retry loop
            break
        except (InternalServerError, APIConnectionError, RateLimitError) as e:
            wait = base_delay * (2 ** attempt)
            print(
                f"[WARN] files.create failed for {jsonl_path} on attempt "
                f"{attempt + 1}/{max_retries}: {e}"
            )
            if attempt == max_retries - 1:
                print("[ERROR] Giving up on files.create after max retries.")
                raise
            print(f"Retrying in {wait} seconds...")
            time.sleep(wait)

    print(f"Created input file: {batch_input_file.id}")

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint=endpoint,
        completion_window="24h",
        metadata={
            "description": f"PMC-VQA batch for {os.path.basename(jsonl_path)}"
        },
    )
    print(f"Created batch: {batch.id}, initial status: {batch.status}")

    meta_path = os.path.join(os.path.dirname(jsonl_path), "batch_meta.jsonl")

    # save batch ID and input file ID
    with open(meta_path, "a", encoding="utf-8") as mf:
        mf.write(
            json.dumps(
                {"batch_id": batch.id, "input_file_id": batch_input_file.id}
            )
            + "\n"
        )

    return batch


def download_batch_outputs(log_path: str, work_dir: str, combined_output_jsonl_path: str):
    """
    Download output files for all batch IDs found in the log, then
    concatenate them into a single JSONL at combined_output_jsonl_path.
    ### Args
    * log_path: Path to the log/out file containing batch IDs.
    * work_dir: Directory to save per-batch output files.
    * combined_output_jsonl_path: Path to save the combined JSONL output.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")

    if not os.path.exists(log_path):
        print(f"\nLog file {log_path} not found; skipping batch download from log.")
        return

    print(f"\nParsing batch IDs from log: {log_path}")
    batch_ids = parse_batch_ids_from_log(log_path)
    if not batch_ids:
        print("No batch IDs found in log; nothing to download.")
        return

    print("Found batch IDs:")
    for b in batch_ids:
        print("  ", b)

    client = OpenAI(api_key=api_key)

    per_batch_files = []

    for b in tqdm(batch_ids):
        try:
            print(f"\nRetrieving batch {b}...")
            batch = client.batches.retrieve(b)
            print(f"Batch {b} status: {batch.status}")
            print(batch.metadata)

            if getattr(batch, "output_file_id", None):
                out_path = os.path.join(work_dir, f"{b}_results.jsonl")
                print(f"Downloading batch output file {batch.output_file_id} -> {out_path}")

                response_stream = client.files.content(batch.output_file_id)
                with open(out_path, "wb") as f:
                    f.write(response_stream.read())

                print(f"Downloaded batch output to {out_path}")
                per_batch_files.append(out_path)
            else:
                print(f"Batch {b} has no output_file_id yet; skipping download.")
        except Exception as e:
            print(f"[ERROR] Failed to download results for {b}: {e}")

    # Concatenate all per-batch JSONLs into a single combined file
    if per_batch_files:
        os.makedirs(os.path.dirname(combined_output_jsonl_path), exist_ok=True)
        print(f"\nConcatenating {len(per_batch_files)} batch result files into {combined_output_jsonl_path}")
        with open(combined_output_jsonl_path, "w", encoding="utf-8") as out_f:
            for fp in per_batch_files:
                with open(fp, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
        print("Concatenation complete.")
    else:
        print("No batch output files were downloaded; combined file not created.")


def load_rationales_from_batch_output(output_jsonl_path: str) -> Dict[str, str]:
    """
    Parse the batch output JSONL and return {custom_id: raw_generation_text}.
    ### Args
    * output_jsonl_path: Path to the batch output JSONL file.
    ### Returns
    * A dictionary mapping custom IDs to their corresponding raw generation text.
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
    Load the PMC-VQA train split and map Figure_path to local image files.
    """
    from datasets import load_dataset  # local import to avoid unused warning if not used

    ds = load_dataset("csv", data_files=DATA_FILES)
    dataset = ds["train"]

    def to_local_path(example):
        filename = os.path.basename(str(example["Figure_path"]))
        local_path = os.path.join(BASE_IMAGE_DIR, filename)
        example["raw_image_url"] = local_path
        example["image"] = local_path
        return example

    dataset = dataset.map(
        to_local_path,
        desc="Linking figures to local images (train)",
        num_proc=num_proc,
    )
    dataset = dataset.cast_column("image", Image(decode=False))

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
    ### Args
    * dataset: The Dataset object for the split.
    * rats: A dictionary mapping custom IDs to rationale texts.
    * split: The dataset split name (e.g., "train").
    ### Returns
    * A Dataset object with the multimodal data.
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

            pil_img = _entry_to_pil(entry["image"])
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


def build_and_save_arrow_dataset_from_results(
    results_jsonl_path: str,
    out_dir: str,
    num_proc: int = 8,
    split: str = "train",
):
    """
    Load combined batch results, build the multimodal dataset, and save it as Arrow to out_dir.
    """
    if not os.path.exists(results_jsonl_path):
        print(f"Results JSONL {results_jsonl_path} not found; skipping Arrow build.")
        return

    print("\n=== Building Arrow dataset from batch outputs ===")
    rats = load_rationales_from_batch_output(results_jsonl_path)
    print(f"Loaded {len(rats)} rationales from {results_jsonl_path}")

    print("Loading PMC-VQA train split...")
    dataset = load_pmc_train_split(num_proc=num_proc)
    print(f"Loaded PMC-VQA train split with {len(dataset)} examples")

    pmc_dataset = build_pmc_multimodal_dataset(dataset, rats, split=split)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving Arrow dataset to {out_dir} ...")
    pmc_dataset.save_to_disk(out_dir)
    print("Done saving Arrow dataset.")


# === parse batch IDs from your log file ===
def parse_batch_ids_from_log(log_path: str) -> List[str]:
    """
    Parse batch IDs (batch_[0-9a-f]+) from a given log file.
    """
    with open(log_path, "r") as f:
        text = f.read()
    batch_ids = sorted(set(re.findall(r"batch_[0-9a-f]+", text)))
    return batch_ids


def main():
    parser = argparse.ArgumentParser(
        description="Submit PMC-VQA OpenAI Batch jobs and download outputs."
    )
    parser.add_argument(
        "--gpt5_model", type=str, default="gpt-5.1", help="OpenAI model name"
    )
    parser.add_argument("--max_output_tokens", type=int, default=2048)
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
        "--log_path",
        type=str,
        default="/users/theoschiff/meditron/reports/multimeditron/preprocess/R-preprocess-pmc.1189244.out",
        help="Log file containing batch IDs (batch_...).",
    )
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--reuse_outputs", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)

    print("Loading PMC-VQA train split...")

    split = "train"
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
        print("Done building batch requests")

    # collect all request shards
    all_files = [
        f for f in os.listdir(args.work_dir)
        if f.startswith("requests_train") and f.endswith(".jsonl")
    ]
    if not all_files:
        raise RuntimeError(f"No requests_train*.jsonl files found in {args.work_dir}")

    all_files = sorted(all_files)
    print("Found request shards:")
    for f in all_files:
        print("  ", f)

    for fname in all_files:
        req_path = os.path.join(args.work_dir, fname)

        # suffix: "", "_00", "_01", ..., based on filename
        # "requests_train_00.jsonl" -> "_00"
        suffix = fname[len("requests_train"):-len(".jsonl")]
        results_fname = f"results_train{suffix}.jsonl"
        results_path = os.path.join(args.work_dir, results_fname)

        start_time = time.time()
        print(f"\n=== Submitting batch for {fname} ===")

        batch = submit_and_wait_for_batch(
            req_path,
            endpoint="/v1/responses",
        )

        elapsed = time.time() - start_time
        print(
            f"Batch {batch.id} sent with status: {batch.status} "
            f"(submission time: {elapsed:.2f} seconds)"
        )
        # If want to download immediately (when done), you should call:
        # download_batch_output(batch, results_path)

    print("\nAll request shards processed.")

    """TO RUN AFTER ALL BATCHES ARE SUBMITTED AND COMPLETED:"""
    # # === use log file to find batch IDs and download their output ===
    # combined_results_path = os.path.join(args.work_dir, "combined_results_train.jsonl")
    # print("\n=== Downloading batch outputs ===")
    # download_batch_outputs(args.log_path, args.work_dir, combined_results_path)

    # # === build Arrow dataset from combined results ===
    # print("\n=== Building Arrow dataset and saving to disk ===")
    # build_and_save_arrow_dataset_from_results(
    #     combined_results_path,
    #     args.out_dir,
    #     num_proc=8,   
    #     split="train",
    # )


if __name__ == "__main__":
    main()
