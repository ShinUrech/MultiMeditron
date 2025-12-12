import json
import random
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

INPUT_PATH = Path("/users/mtang/datasets/merged_clean.jsonl")
OUTPUT_PATH = Path("testtttt/medical-cases.parquet")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

records = []
with INPUT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

print(f"Loaded {len(records)} records")


processed = []
for data in tqdm(records, desc="Processing"):

    user_prompt = data.get("context", "")

    system_prompt = (
        "You are a medical assistant. Given the medical case provided by the user, "
        "write a detailed reasoning process leading to a diagnosis and treatment plan. "
        "Your final answer must be in the following format:\n"
        "Answer:\n"
        " - Diagnosis: <your diagnosis here>\n"
        " - Treatment: <your treatment plan here>."
    )

    # Skip if missing ground-truth labels
    if not data.get("diagnosis") or not data.get("treatment"):
        continue

    # new_record = {
    #     "prompt": [
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": user_prompt},
    #     ],
    #     "reward_model": {
    #         "ground_truth": {
    #             "reasoning": " ".join(data.get("reasoning_trace") or []),
    #             "diagnosis": " ".join(data.get("diagnosis") or []),
    #             "treatment": " ".join(data.get("treatment") or []),
    #         }
    #     },
    #     "data_source": "medical-cases",
    #     "extra_info": {
    #         "tools_kwargs": {"python_exec": {"dummy": "dummy"}},
    #     },
    # }


    new_record = {
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "reward_model": {
            "style": "function",
            "ground_truth": {
                "reasoning": " ".join(data.get("reasoning_trace") or []),
                "diagnosis": " ".join(data.get("diagnosis") or []),
                "treatment": " ".join(data.get("treatment") or []),
            },
        },
        "data_source": "medical-cases",
        "extra_info": {
            "tools_kwargs": {"python_exec": {"dummy": "dummy"}},
        },
    }
    processed.append(new_record)


df = pd.DataFrame(processed)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

train_df.to_parquet("testtttt/medical-cases-train.parquet", index=False)
val_df.to_parquet("testtttt/medical-cases-val.parquet", index=False)
test_df.to_parquet("testtttt/medical-cases-test.parquet", index=False)

print("Saved train/val/test Parquet files.")

df = pd.read_parquet("testtttt/medical-cases-train.parquet")
print(df.head(2))
