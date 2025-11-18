from typing import Dict
from multimeditron.verl.score_utils import *
from multimeditron.verl.infer import create_async_client
from transformers import PreTrainedTokenizer


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

    client = create_async_client(required=True)

    response = solution_str
    response_lower = response.lower()
    score = response_lower.count("a") / len(response_lower) if len(response_lower) > 0 else 0
    print(f"Score: {score}")

    return {
        "score": score,
        "acc": 0.0,
        "pred": "Maybe",
    }