import numpy as np

from src.state import RAGState
from src.utils.embeddings import embed_text
from src.utils.similarity import cosine_similarity
from src.utils.logger import get_logger

logger = get_logger(__name__)


def mmr_selection(state: RAGState) -> RAGState:
    lambda_param = 0.5
    top_k = state.config["top_k_mmr"]
    query = state.rewritten_question or state.original_question
    query_emb = embed_text(query)

    candidates = state.reranked_results[:20]
    if not candidates:
        logger.warning("No candidates for MMR selection")
        state.final_docs = []
        return state

    doc_embs = [embed_text(doc["text"]) for doc in candidates]
    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            relevance = cosine_similarity(query_emb, doc_embs[idx])
            if selected_indices:
                max_sim_to_selected = max(
                    cosine_similarity(doc_embs[idx], doc_embs[j])
                    for j in selected_indices
                )
            else:
                max_sim_to_selected = 0
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    state.final_docs = [candidates[i] for i in selected_indices]
    logger.info(f"MMR selected {len(state.final_docs)} docs")
    return state
