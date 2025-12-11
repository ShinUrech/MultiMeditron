from typing import Dict, TYPE_CHECKING
from multimeditron.verl.score_utils import *
from multimeditron.verl.infer import create_async_client
import re
import json
from rapidfuzz import fuzz


SCORE_REGISTRY = {}

def register_score(name: str):
    def decorator(func):
        SCORE_REGISTRY[name] = func
        return func
    return decorator

def get_score_by_name(name: str):
    if name not in SCORE_REGISTRY:
        raise ValueError(f"Score function '{name}' is not registered.")
    return SCORE_REGISTRY[name]

def compute_score_router(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
):
    pass

def extract_sections(text: str):
    pattern_diag = r"-\s*Diagnosis:\s*(.*?)(?=\n\s*-\s*Treatment\s*:|\Z)"
    pattern_treat = r"-\s*Treatment:\s*(.*?)(?=\Z)"
    diag = re.search(pattern_diag, text, re.I | re.S)
    treat = re.search(pattern_treat, text, re.I | re.S)
    return (diag.group(1).strip() if diag else "",
            treat.group(1).strip() if treat else "")


def score_format(solution_str: str):
    lines = [ln.strip() for ln in solution_str.splitlines() if ln.strip()]
    
    has_diag = any(re.match(r"-\s*diagnosis:\s+.+", ln, re.I) for ln in lines)
    has_treat = any(re.match(r"-\s*treatment:\s+.+", ln, re.I) for ln in lines)

    if not has_diag or not has_treat:
        return 0.0

    return 1.0


@register_score("medical_llm_judge_score")
async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: dict,
):
    print("============ Debug ============")
    print("Solution:", solution_str)
    print("Ground truth:", ground_truth)
    print("Extra:", extra_info)

    format_score = score_format(solution_str)

    pred_diag, pred_treat = extract_sections(solution_str)
    true_diag = ground_truth.get("diagnosis", "")
    true_treat = ground_truth.get("treatment", "")

    client = await create_async_client(required=False)
    if client is None:
        print("No SGLang server. Score = 0.0")
        return 0.0

    prompt = f"""
    You are an expert medical evaluator.

    Score the model output from 0.0 to 1.0 for:
    - Diagnosis correctness
    - Treatment correctness
    - Reasoning quality (logical and coherent)

    Ground truth: Diagnosis: {true_diag} | Treatment: {true_treat}
    Prediction: Diagnosis: {pred_diag} | Treatment: {pred_treat}

    Output your explanation for the score and what the model did well/badly and final score in the format below:

    Explanation: <your explanation here>
    Answer: <float between 0.0 and 1.0>

    Do not include any other text outside the specified format.
    """

    response = await client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
        },
        max_tokens=256,
    )


    print("============ LLM Judge Response ============")
    print(response)

    try:
        llm_text = response.choices[0].message.content

        print("============ Extracted Text ============")
        print(llm_text)

        if "Answer:" not in llm_text:
            raise ValueError("Answer missing")

        after = llm_text.split("Answer:")[-1].strip()
        score_str = after.split()[0].strip()

        final_score = float(score_str)
        final_score = max(0.0, min(1.0, final_score))

    except Exception as e:
        print("Judge answer parsing error", e)
        print("Fallback score = 0.25")
        return 0.25

    final_reward = 0.10 * format_score + 0.90 * final_score

    print("============ Final Reward Breakdown ============")
    print("Format:", format_score)
    print("Judge score:", final_score)
    print("Final score:", final_reward)

    return final_reward
