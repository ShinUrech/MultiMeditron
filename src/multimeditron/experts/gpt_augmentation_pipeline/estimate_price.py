from pathlib import Path
import json
import config

# Expected output of an estimate-only batch run
API_RESPONSE_PATH = Path(config.OUTPUT_DIR) / "api_response_estimate.jsonl"


def calculate_token_totals(api_response_records):
    input_tokens = 0
    output_tokens = 0
    for rec in api_response_records:
        usage = rec["response"]["body"].get("usage", {})
        input_tokens += usage.get("prompt_tokens", 0)
        output_tokens += usage.get("completion_tokens", 0)
    return input_tokens, output_tokens


def main():
    if not API_RESPONSE_PATH.exists():
        raise FileNotFoundError(
            f"Estimate response file not found: {API_RESPONSE_PATH}\n"
            "Run an estimate-only batch first."
        )

    # Load estimate responses (typically a small subset, e.g. N_ITER_ESTIMATE)
    records = [
        json.loads(l)
        for l in API_RESPONSE_PATH.read_text().splitlines()
        if l.strip()
    ]

    input_tokens, output_tokens = calculate_token_totals(records)
    n_msgs = len(records)

    price_in = config.PRICE_PER_MILLION_INPUT
    price_out = config.PRICE_PER_MILLION_OUTPUT

    cost_in = (input_tokens / 1e6) * price_in
    cost_out = (output_tokens / 1e6) * price_out
    total_cost = cost_in + cost_out

    per_case_cost = total_cost / max(1, n_msgs)

    # Project to full dataset size
    from utils import load_data_raw
    total_cases = len(load_data_raw())
    projected_total = per_case_cost * total_cases

    print(f"Currency: {config.PRICING_CURRENCY}")
    print(f"Samples analysed: {n_msgs}")
    print(f"Input tokens: {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(
        f"Cost for samples: {total_cost:.4f} "
        f"({cost_in:.4f} in + {cost_out:.4f} out)"
    )
    print(f"Cost per case: {per_case_cost:.6f}")
    print(
        f"Projected total for {total_cases} cases: {projected_total:.2f}"
    )


if __name__ == "__main__":
    main()
