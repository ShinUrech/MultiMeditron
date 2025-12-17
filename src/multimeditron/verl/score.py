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
    print("Model's Solution:", solution_str)
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

    Score the model output:
    - A if it is fully correct
    - B if it is mostly correct with minor issues
    - C if it has some correct parts but also significant errors
    - D if it is mostly incorrect
    - F if it is completely wrong or irrelevant

    Consider the following aspects:
    - Diagnosis correctness
    - Treatment correctness

    Give a score for diagnosis and treatment seperately. 

    Ground truth: Diagnosis: {true_diag} | Treatment: {true_treat}
    Prediction: Diagnosis: {pred_diag} | Treatment: {pred_treat}

    Output your explanation for the score and what the model did well/badly and final score in the format below:

    Explanation: <your explanation here>
    Answer: Treatment score (A-F), Diagnosis score (A-F)

    Do not include any other text outside the specified format.
    """

    prompt2= f"""
        You are an expert medical response evaluator. Your task is to provide a single, holistic quality assessment of a predicted response against the authoritative ground truth.

        Criteria for Evaluation:
        1.  **Completeness:** Does the predicted response address both the Diagnosis and Treatment sections as required by the ground truth?
        2.  **Consistency:** Are the diagnosis and treatment logically consistent with each other and the underlying condition?
        3.  **Clarity and Flow:** Is the response well-structured, easy to understand, and professionally presented?
        4.  **Overall Accuracy:** Considering all elements, how closely does the predicted response match the intent and factual correctness of the ground truth?

        Score the model output:
            - A if it is fully correct
            - B if it is mostly correct with minor issues
            - C if it has some correct parts but also significant errors
            - D if it is mostly incorrect
            - F if it is completely wrong or irrelevant
        ---
        Ground Truth Diagnosis: {true_diag}
        Ground Truth Treatment: {true_treat}
        Ground Truth Reasoning: {ground_truth.get("reasoning", "")}

        Predicted Response (Full Text):
        {solution_str}
        ---

        Output your evaluation in the exact format below, including a detailed justification for the chosen grade.

        Explanation: <Your detailed reasoning, addressing all four criteria>
        Final Grade: <A, B, C, D, or F>

        Do not output anything after the final grade is given. 
    """

    response = await client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
        },
        max_tokens=512
    )

    response2 = await client.chat.completions.create(
        model="Qwen/Qwen3-30B-A3B-Instruct-2507",
        messages=[
            {"role": "user", "content": prompt2},
        ],
        temperature=0.0,
        extra_body={
        "chat_template_kwargs": {"enable_thinking": False}
        },
        max_tokens=5123
    )


    print("============ LLM Judge Response ============")
    print("Response about treatment and diagnosis:\n")
    print(response)
    print("Response about holistic evaluation:\n")
    print(response2)



    GRADE_TO_SCORE = {
    "A": 1.0,
    "B": 0.75,
    "C": 0.5,
    "D": 0.25,
    "F": 0.0
    }

    try:
        llm_text = response.choices[0].message.content

        print("============ Diagnosis/Treatment Judge Text ============")
        print(llm_text)

        if "Answer:" not in llm_text:
            treatment_grade = "D"
            diagnosis_grade = "D"
        else:
            after = llm_text.split("Answer:")[-1].strip()
            grades = re.findall(r"\b[A-F]\b", after)

            # if len(grades) != 2:
            #     treatment_grade = "C"
            #     diagnosis_grade = "C

            try:
                treatment_grade, diagnosis_grade = grades
            except:
                treatment_grade = "D"
                diagnosis_grade = "D"

            valid_grades = {"A", "B", "C", "D", "F"}
            if treatment_grade not in valid_grades or diagnosis_grade not in valid_grades:
                treatment_grade = "D"
                diagnosis_grade = "D"

    except Exception as e:
        treatment_grade = "D"
        diagnosis_grade = "D"

    try:
        llm_text2 = response2.choices[0].message.content
        print("============ Holistic Judge Text ============")
        print(llm_text2)
        match = re.search(r"Final Grade:\s*([A-F])", llm_text2)
        if match:
            holistic_grade = match.group(1)
        else:
            holistic_grade = "D"
    except Exception as e2:
        holistic_grade = "D"


    numeric_holistic = GRADE_TO_SCORE[holistic_grade]
    numeric_treatment = GRADE_TO_SCORE[treatment_grade]
    numeric_diagnosis = GRADE_TO_SCORE[diagnosis_grade]


    final_reward = 0.10 * format_score + (0.45 * numeric_treatment + 0.45 * numeric_diagnosis + 0.9 * numeric_holistic)/2

    print("============ Final Reward Breakdown ============")
    print("Format:", format_score)
    print(f"Judge answer score: Treatment {treatment_grade} ({numeric_treatment}), Diagnosis {diagnosis_grade} ({numeric_diagnosis})")
    print(f"Judge holistic score: {holistic_grade} ({numeric_holistic})")
    print("Final score:", final_reward)

    return final_reward
