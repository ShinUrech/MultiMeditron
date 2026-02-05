from openai import OpenAI
from pathlib import Path
import json, time, re
from typing import List
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)
OUT_DIR = Path(config.OUTPUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def part_number_from_filename(p: Path) -> int:
    m = re.search(r"batch_id_part_(\d+)\.jsonl\.txt$", p.name)
    return int(m.group(1)) if m else 0

def wait_for_output(batch_id: str, max_wait_s: int = 86400, poll_s: int = 30):
    waited = 0
    while True:
        b = client.batches.retrieve(batch_id)
        print(f"[{batch_id}] status={b.status} out={b.output_file_id} err={b.error_file_id}")
        if b.output_file_id:
            return b
        if b.status in ("failed", "cancelled", "expired"):
            raise SystemExit(f"Batch ended with status: {b.status}")
        time.sleep(poll_s)
        waited += poll_s
        if waited >= max_wait_s:
            raise SystemExit("Timed out waiting for output_file_id")

def save_part_output(b, part_num: int) -> List[dict]:
    text = client.files.content(b.output_file_id).text
    part_path = OUT_DIR / f"api_response_part_{part_num}.jsonl"
    part_path.write_text(text)
    print(f"[saved] {part_path}")
    return [json.loads(line) for line in text.splitlines() if line.strip()]

def extract_messages(api_responses: List[dict]) -> List[str]:
    return [
        rec["response"]["body"]["choices"][0]["message"]["content"].strip()
        for rec in api_responses
    ]

def main():
    id_files = sorted(OUT_DIR.glob("batch_id_part_*.jsonl.txt"), key=part_number_from_filename)
    if not id_files:
        raise SystemExit(f"No batch id files found in {OUT_DIR}")

    all_records = []
    total_prompt = total_completion = 0

    for id_file in id_files:
        part_num = part_number_from_filename(id_file)
        batch_id = id_file.read_text().strip()

        b = wait_for_output(batch_id)
        records = save_part_output(b, part_num)
        all_records.extend(records)

        # accumulate actual usage
        for rec in records:
            usage = rec.get("response", {}).get("body", {}).get("usage", {})
            total_prompt += int(usage.get("prompt_tokens", 0))
            total_completion += int(usage.get("completion_tokens", 0))

    # merged outputs
    all_path = OUT_DIR / "api_response_all.jsonl"
    with all_path.open("w", encoding="utf-8") as fout:
        for rec in all_records:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[merged] {all_path} ({len(all_records)} responses)")

    # optional: also save the plain texts
    texts_path = OUT_DIR / "messages_all.txt"
    with texts_path.open("w", encoding="utf-8") as ftxt:
        for msg in extract_messages(all_records):
            ftxt.write(msg + "\n\n")
    print(f"[texts]  {texts_path}")

    # print ACTUAL cost
    price_in = config.PRICE_PER_MILLION_INPUT
    price_out = config.PRICE_PER_MILLION_OUTPUT
    cost_in = (total_prompt / 1e6) * price_in
    cost_out = (total_completion / 1e6) * price_out
    total_cost = cost_in + cost_out
    print(f"Currency: {config.PRICING_CURRENCY}")
    print(f"Actual input tokens: {total_prompt}")
    print(f"Actual output tokens: {total_completion}")
    print(f"Actual total cost: {total_cost:.4f} ({cost_in:.4f} in + {cost_out:.4f} out)")

if __name__ == "__main__":
    main()