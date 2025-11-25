from typing import Dict
from multimeditron.verl.score_utils import *
from multimeditron.verl.infer import create_async_client
import json


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

@register_score("debug_score")
async def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
):
    # Retrieve the response and check if it is the correct format (use a regex)
    print(data_source)
    print(solution_str)
    print(ground_truth)
    print(extra_info)

    client = create_async_client(required=False)
    if client is None:
        print("No SGLang server configured, skipping score computation.")
    else:
        print("SGLang server client created.")

    # 1. Formating score
    score += 0.2 * markdown_simple_reward(solution_str)
    score += 0.2 * markdown_check_references(solution_str, lambda _: False)

    # 2. Content score
    assert client is not None
    prompt = f"""
    You are a helpful assistant that evaluates the correctness of answers to questions.

    Given the question, the correct answer, and the provided answer, please grade the following
    - capacity to speak fluently and coherently (possible values: Poor, Fair, Good, Excellent)
    - correctness of the answer (possible values: Incorrect, Partially Correct, Mostly Correct, Correct)
    - correctness of the reasoning steps (possible values: Incorrect, Partially Correct, Mostly Correct, Correct)
    Provide your answer in the following JSON format (prefixed by "ANSWER:"):

    ANSWER: {{
        "fluency": "<value>",
        "answer_correctness": "<value>",
        "reasoning_correctness": "<value>"
    }}

    Question: {extra_info.get("question", "N/A")}
    Correct Answer: {ground_truth}
    Provided Answer: {solution_str}
    """
    response = await client.completions.create(prompt)
    print("=== Prompt sent to SGLang server for scoring ===")
    print("Prompt:", prompt)
    print("SGLang response:", response)

    # Attempt to parse the response
    score = 0.0
    try:
        response_json_str = response.split("ANSWER:")[-1].strip()

        # Try to find the first and last curly braces to extract JSON
        first_brace = response_json_str.find("{")
        last_brace = response_json_str.rfind("}")
        if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
            raise ValueError("Could not find valid JSON in the response.")
        else:
            response_json_str = response_json_str[first_brace:last_brace+1]
            response_json = json.loads(response_json_str)
            fluency = response_json.get("fluency", "Poor")
            answer_correctness = response_json.get("answer_correctness", "Incorrect")
            reasoning_correctness = response_json.get("reasoning_correctness", "Incorrect")

            fluency_scores = {
                "Poor": 0.0,
                "Fair": 0.33,
                "Good": 0.66,
                "Excellent": 1.0,
            }
            correctness_scores = {
                "Incorrect": 0.0,
                "Partially Correct": 0.33,
                "Mostly Correct": 0.66,
                "Correct": 1.0,
            }

            score += 0.3 * fluency_scores.get(fluency, 0.0)
            score += 0.35 * correctness_scores.get(answer_correctness, 0.0)
            score += 0.15 * correctness_scores.get(reasoning_correctness, 0.0)
    except Exception as e:
        print(f"Error parsing response JSON: {e}")
        score += 0.5

    # response = solution_str
    # response_lower = response.lower()
    # score = response_lower.count("a") / len(response_lower) if len(response_lower) > 0 else 0
    # print(f"Score: {score}")

    return {
        "score": score,
        "acc": 0.0,
        "pred": "Maybe",
    }