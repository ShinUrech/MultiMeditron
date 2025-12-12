# pip install "sglang>=0.3.0" zmq uvloop fastapi openai partial_json_parser sentencepiece sgl_kernel compressed_tensors msgspec nest_asyncio torchao xgrammar pyyaml json-repair tqdm
import json
import os
from tqdm import tqdm
from json_repair import repair_json
import yaml
import nest_asyncio
nest_asyncio.apply()

import sglang as sgl
from sglang.utils import stream_and_merge, async_stream_and_merge


MODEL_NAME = "Qwen/Qwen3-Next-80B-A3B-Instruct"
INPUT_FILE = "cases.jsonl"
#OUTPUT_FILE = "extracted_cases_cleaned_2.jsonl"
#N_TEST_CASES = 2  # change or comment out for full run
CONFIG_YAML = "config-sglang.yaml"


# Pass 1: extraction
PROMPT_TEMPLATE = """
You are a medical NLP assistant.

Extract information ONLY from the provided case text — do not infer or assume anything not explicitly stated.

Output **only valid JSON**, with no explanations, no notes, and no extra text.

Definitions:
- **Context**: factual details only (demographics, symptoms, history, physical findings, labs, imaging, objective results, medications).  
  - Do NOT include any diagnoses, reasoning, or treatments.  
  - Use only explicit statements of fact from the text.
- **Reasoning Trace**: explicit reasoning or diagnostic logic mentioned in the text (e.g., “X was ruled out,” “Y suspected because...”).
- **Diagnosis**: diagnoses explicitly named in the text.
- **Treatment**: treatments, interventions, procedures, or medications explicitly mentioned.

If something is not present, return an empty string (for text fields) or empty list (for reasoning_trace).

Return JSON only, in this format:
{{
  "context": "...",
  "reasoning_trace": ["...", "..."],
  "diagnosis": "...",
  "treatment": "..."
}}

Case text:
{case_text}
"""

# Pass 2: full cleaner — fixes context and reassigns misplaced content
CONTEXT_CLEAN_PROMPT = """
You are a medical text corrector preparing data to train a clinical reasoning model.

You are given:
1. A full clinical case text.
2. A previous extraction with the fields: context, reasoning_trace, diagnosis, and treatment.

Your task:
- Clean and correct the extraction so each field contains only its proper type of content.
- Move any misplaced information into the correct field.
- Keep all information explicitly stated in the case text.
- Output exactly ONE valid JSON object — not a list, not multiple objects, no explanations.

==============================
DETAILED FIELD DEFINITIONS
==============================

CONTEXT:
- Keep all **objective and factual information** that a clinician would know before making a diagnostic conclusion.
- Include demographics, medical history, symptoms, physical findings, vital signs, **laboratory data**, **imaging results**, **pathology**, **genetic/molecular findings**, and **diagnostic procedures** (e.g., biopsy, CT, MRI, FISH analysis, angiography).
- Include results of diagnostic tests (positive or negative). The model must have all the evidence needed to reason to a diagnosis.
- Include chronic or background medications, and past treatments done well before this episode (e.g., “had cataract surgery three years ago”).
- Exclude interpretive statements, reasoning, and diagnostic conclusions (e.g., “consistent with,” “was diagnosed as,” “suggesting,” “we suspected…”).
- Exclude therapeutic treatments or management decisions (e.g., “was treated with…,” “underwent surgery…”).

REASONING_TRACE:
- Include explicit reasoning, interpretation, or diagnostic logic stated by the authors.
- Examples: “X was ruled out,” “Y suspected because…,” “Findings were consistent with…,” “We concluded that…”
- If you remove reasoning or interpretation from context, move it here.

DIAGNOSIS:
- Include only explicit diagnostic statements or phrases (e.g., “was diagnosed with…,” “consistent with…”).
- Do not include the test evidence — keep those in context.
- If a diagnosis statement was removed from context, move it here.

TREATMENT:
- Include only therapeutic treatments, interventions, or medications administered during this episode.
- Include surgeries, laser procedures, drug regimens, or therapy decisions.
- Do not include diagnostic procedures.
- If a treatment statement was removed from context, move it here.

==============================
VERB AND TIMING GUIDANCE
==============================
- If something happened recently or during this clinical episode (e.g., “was performed,” “underwent,” “was started on”), it usually belongs in **treatment** (if therapeutic) or **context** (if diagnostic evidence).
- If it refers to remote or chronic history (e.g., “had undergone surgery years earlier,” “was on long-term aspirin”), keep it in **context**.
- If the sentence mixes objective evidence with reasoning or treatment, keep the factual part in **context** and move the rest appropriately.

==============================
OUTPUT REQUIREMENTS
==============================
Do not summarize or invent new information.
Do not output arrays of objects or wrap in brackets.

Return ONLY ONE valid JSON object with this structure:
{{
  "context": "cleaned factual background and diagnostic evidence",
  "reasoning_trace": ["each reasoning statement"],
  "diagnosis": "explicit diagnosis",
  "treatment": "explicit treatments or procedures"
}}

Case text:
{case_text}

Original extraction:
{extracted_json}
"""


def load_engine_from_yaml(config_path: str) -> tuple[sgl.Engine, dict]:
    """Load SGLang Engine and sampling params from the YAML file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    engine_args = cfg.get("engine", {})
    sampling_params = cfg.get("sampling_params", {})
    llm = sgl.Engine(**engine_args)
    return llm, sampling_params

def run_model(llm: sgl.Engine, sampling_params: dict, prompt: str, *, max_new_tokens: int | None = None) -> dict:
    """Generate output via SGLang and parse JSON safely."""
    sp = dict(sampling_params)
    if max_new_tokens is not None:
        sp["max_new_tokens"] = max_new_tokens

    full_text = stream_and_merge(llm, prompt, sp)

    try:
        start = full_text.find("{")
        end = full_text.rfind("}") + 1
        json_str = repair_json(full_text[start:end])
        data = json.loads(json_str)

        if isinstance(data, list):
            if len(data) > 0 and isinstance(data[0], dict):
                data = data[0]
            else:
                data = {}
        elif not isinstance(data, dict):
            data = {}

        for key in ["context", "reasoning_trace", "diagnosis", "treatment"]:
            if key not in data:
                data[key] = "" if key != "reasoning_trace" else []
    except Exception:
        data = {"context": "", "reasoning_trace": [], "diagnosis": "", "treatment": ""}

    return data

def extract_with_model(llm: sgl.Engine, sampling_params: dict, case_text: str) -> dict:
    return run_model(
        llm, sampling_params,
        PROMPT_TEMPLATE.format(case_text=case_text),
        max_new_tokens=4096
    )

def clean_and_reassign_with_model(llm: sgl.Engine, sampling_params: dict, case_text: str, extracted_json: dict) -> dict:
    prompt = CONTEXT_CLEAN_PROMPT.format(
        case_text=case_text,
        extracted_json=json.dumps(extracted_json, ensure_ascii=False, indent=2),
    )
    return run_model(llm, sampling_params, prompt, max_new_tokens=2048)



import argparse
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, required=True, help="Shard ID (0–49)")
    parser.add_argument("--num_shards", type=int, default=50, help="Total number of shards (default: 50)")
    args = parser.parse_args()

    shard_id = args.shard_id
    num_shards = args.num_shards

    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"shard_id must be between 0 and {num_shards - 1}")

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file {INPUT_FILE} not found")

    llm, sampling_params = load_engine_from_yaml(CONFIG_YAML)

    results = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue

            # process only lines assigned to this shard
            if i % num_shards != shard_id:
                continue

            case = json.loads(line)
            text = case.get("case_text", "")
            cid = case.get("case_id", "")

            extracted = extract_with_model(llm, sampling_params, text)
            corrected = clean_and_reassign_with_model(llm, sampling_params, text, extracted)

            corrected["case_id"] = cid
            results.append(corrected)

    shard_output_file = f"clean_{shard_id}.jsonl"
    with open(shard_output_file, "w", encoding="utf-8") as out:
        for r in results:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    llm.shutdown()
    print(f"\nDone")


if __name__ == "__main__":
    main()
