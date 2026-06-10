"""Custom RAG evaluation metrics: precision, recall, MRR, MAP, NDCG."""

import math
from typing import List, Set


def _to_set(items: List[str]) -> Set[str]:
    return set(items)


def precision_at_k(predicted: List[str], relevant: Set[str], k: int) -> float:
    if k <= 0 or not predicted:
        return 0.0
    top_k = predicted[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(top_k)


def recall_at_k(predicted: List[str], relevant: Set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = predicted[:k]
    hits = sum(1 for doc in top_k if doc in relevant)
    return hits / len(relevant)


def f1_at_k(predicted: List[str], relevant: Set[str], k: int) -> float:
    p = precision_at_k(predicted, relevant, k)
    r = recall_at_k(predicted, relevant, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def mrr(ranked: List[str], relevant: Set[str]) -> float:
    for i, doc in enumerate(ranked, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def map_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    score = 0.0
    hits = 0
    for i, doc in enumerate(ranked[:k], start=1):
        if doc in relevant:
            hits += 1
            score += hits / i
    if hits == 0:
        return 0.0
    return score / hits


def ndcg_at_k(ranked: List[str], relevant: Set[str], k: int) -> float:
    def dcg(items: List[str]) -> float:
        score = 0.0
        for i, doc in enumerate(items, start=1):
            rel = 1.0 if doc in relevant else 0.0
            score += (2 ** rel - 1) / math.log2(i + 1)
        return score

    top_k = ranked[:k]
    ideal = sorted(top_k, key=lambda d: 1.0 if d in relevant else 0.0, reverse=True)

    actual_dcg = dcg(top_k)
    ideal_dcg = dcg(ideal)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def extract_doc_ids(docs: List[dict], key: str = "text") -> List[str]:
    return [d.get(key, "") for d in docs]
