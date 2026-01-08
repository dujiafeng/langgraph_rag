from typing import List, Dict, Tuple

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _encoder
    if _encoder is None:
        _encoder = CrossEncoder(_MODEL_NAME)
    return _encoder


def rerank(query: str, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
    if not candidates:
        return []
    encoder = get_cross_encoder()
    pairs = [(query, doc["text"]) for doc in candidates]
    scores = encoder.predict(pairs)
    combined: List[Tuple[Dict, float]] = list(zip(candidates, scores))
    combined.sort(key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in combined[:top_k]]
