"""
This script performs a diagnostic analysis of lexical overlap between ground-truth captions and randomly sampled negative 
captions in a medical image–text validation dataset. Using token-level Jaccard similarity, it quantifies how similar 
positive captions are to negatives drawn for forced-choice retrieval evaluation (4-way Recall@1). 

The analysis helps assess whether negative samples are trivially easy or excessively hard due to near-duplicate wording. 
Reported statistics (mean, median, upper quantiles, maximum) characterize the difficulty of the negative pool. 
This metric is purely lexical, not semantic, and is intended for evaluation sanity checks rather than as a standalone benchmark.

"""

import json, random, re
from pathlib import Path
import numpy as np

random.seed(14)

def norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def jaccard(a: str, b: str) -> float:
    ta = set(norm(a).split())
    tb = set(norm(b).split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(1, len(ta | tb))

eval_dataset = ""
LINE_NUMBER = 1000

with open(eval_dataset, "r", encoding="utf-8") as f:
    lines = f.readlines()

N = min(LINE_NUMBER, len(lines))
lines = lines[:N]

sims = []
for i in range(N):
    pos = json.loads(lines[i])["text"]
    candidates = list(range(N))
    candidates.remove(i)
    a, b, c = random.sample(candidates, 3)
    for j in [a, b, c]:
        neg = json.loads(lines[j])["text"]
        sims.append(jaccard(pos, neg))

arr = np.array(sims)
print("pairs:", len(arr))
print("mean:", arr.mean())
print("median:", np.median(arr))
print("p95:", np.quantile(arr, 0.95))
print("max:", arr.max())
