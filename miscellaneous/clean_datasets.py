import json

output_file = "merged_clean.jsonl"

with open(output_file, "w", encoding="utf-8") as outfile:
    for i in range(50):
        input_file = f"clean_{i}.jsonl"
        with open(input_file, "r", encoding="utf-8") as infile:
            for line in infile:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                diagnosis = str(data.get("diagnosis", "")).strip()
                treatment = str(data.get("treatment", "")).strip()
                case_id = str(data.get("case_id", "")).strip()

                if case_id and not (diagnosis == "" or treatment == ""):
                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write("\n")
