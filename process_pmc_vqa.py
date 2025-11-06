# process_pmc_vqa.py
from datasets import load_dataset, concatenate_datasets, Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import fsspec

# 1) Load your CSVs (v1 meta files)
data_files = {
    "train": "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/train.csv",
    "test":  "https://huggingface.co/datasets/RadGenome/PMC-VQA/resolve/main/test.csv",
}


def _normalize_choice(text: str) -> str:
    """
    Strip labels like 'A:' and extra whitespace from choice strings.
    Examples:
        ' A:Maxilla ' -> 'Maxilla'
        ' B:Mandible ' -> 'Mandible'
    """
    if text is None:
        return ""
    # keep only part after the first colon if present, then strip
    parts = str(text).strip().split(":", 1)
    cleaned = parts[-1] if len(parts) > 1 else parts[0]
    return cleaned.strip()

def prepare_messages_from_entry(entry: dict) -> list:
    """
    Convert one dataset row into Qwen3-VL chat messages:
    [
      {
        "role": "user",
        "content": [
          {"type": "image", "image": <PIL image or URL>},
          {"type": "text", "text": "<descriptive prompt>"}
        ],
      }
    ]
    The prompt asks for a concise, evidence-based rationale and the final answer letter.
    """
    q = str(entry.get("Question", "")).strip()

    # Normalize choices (remove "A:", "B:" prefixes and whitespace)
    choices = {
        "A": _normalize_choice(entry.get("Choice A", "")),
        "B": _normalize_choice(entry.get("Choice B", "")),
        "C": _normalize_choice(entry.get("Choice C", "")),
        "D": _normalize_choice(entry.get("Choice D", "")),
    }

    correct_label = str(entry.get("Answer_label", "")).strip()
    # Prefer normalized choice text; fall back to raw 'Answer'
    correct_text = choices.get(correct_label) or str(entry.get("Answer", "")).strip()

    # Descriptive, efficient prompt for Qwen3-VL (rationale without verbose scratchpad)
    user_prompt = (
        "You are a Medical Expert.\n"
        "Carefully inspect the provided figure and answer the multiple-choice question.\n\n"
        f"Question: {q}\n"
        "Options:\n"
        + "\n".join([f"{k}. {v}" for k, v in choices.items() if v])
        + "\n\n"
        f"Task: Explain succinctly (3–5 sentences) why the correct answer is {correct_label} ({correct_text}), "
        "citing visible image evidence and clinically relevant reasoning. "
        "Avoid step-by-step scratchpads; give a clear, self-contained rationale. "
        "Format example:\n"
        "Rationale: ...\n"
        "Final Answer: B"
    )

    # Note: entry['image_url'] is a datasets.Image() field -> returns a PIL.Image when indexed
    # Qwen processors accept PIL.Image or an image URL.
    image_obj_or_url = entry.get("image_url")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_obj_or_url},
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    return messages


def generate_answer(messages, model, processor):
    inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    

def load_qwen_vlm(model_name: str = "Qwen/Qwen3-VL-8B-Instruct", dtype="bfloat16"):
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, dtype=dtype, use_safetensors=True)
    model.to("cuda")
    return model, processor


if __name__ == "__main__":
    ds = load_dataset("csv", data_files=data_files)
    concatenated = concatenate_datasets([ds["train"], ds["test"]]).shuffle(seed=42)
    split = concatenated.train_test_split(test_size=0.10, seed=42)
    train_dataset = split["train"]

    print("Train dataset size:", len(train_dataset))
    print("Columns:", train_dataset.column_names)
    print("Raw Figure_path example:", train_dataset[0]["Figure_path"])  # safe BEFORE any cast

    archive_uri = "hf://datasets/RadGenome/PMC-VQA/images.zip"
    member_prefix = "images/"

    print("Using archive:", archive_uri)
    print("Using internal folder prefix:", repr(member_prefix))

    def to_remote_zip(example):
        example["image_url"] = f"zip://{member_prefix}{example['Figure_path']}::{archive_uri}"
        return example

    train_dataset = train_dataset.map(to_remote_zip, desc="Linking figures in remote zip")
    train_dataset = train_dataset.cast_column("image_url", Image())

    # img = train_dataset[0]["image_uri"]  
    # img.show() 

    train_dataset = train_dataset.remove_columns(["Figure_path"])
    print("Final columns:", train_dataset.column_names)
    print("Example entry:", train_dataset[0])

    # put img_url first - knitpicking 
    cols = train_dataset.column_names
    cols.remove("image_url")
    cols = ["image_url"] + cols
    train_dataset = train_dataset.select_columns(cols)


    # Load model and processor
    model, processor = load_qwen_vlm()

    example_entry = train_dataset[0]
    print("Example entry:", example_entry)

    # prepare messages in the expected format
    test_messages = prepare_messages_from_entry(example_entry)
    print("Prepared messages schema:", test_messages)

    generate_answer(test_messages, model, processor)

