import numpy as np

from src.state import RAGState
from src.utils.embeddings import embed_text
from src.utils.similarity import cosine_similarity
from src.utils.logger import get_logger

logger = get_logger(__name__)


def mmr_selection(state: RAGState) -> RAGState:
    lambda_param = 0.5  # 相关性 vs 多样性的平衡系数
    top_k = state.config["top_k_mmr"]
    query = state.rewritten_question or state.original_question
    query_emb = embed_text(query)  # 对查询做 embedding

    candidates = state.reranked_results[:20]  # 取重排后前 20 条作为候选池
    if not candidates:
        logger.warning("No candidates for MMR selection")
        state.final_docs = []
        return state

    doc_embs = [embed_text(doc["text"]) for doc in candidates]  # 预计算候选文档 embedding
    selected_indices = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):  # 逐轮挑选对 query 相关且与已选文不重复的文档
        best_idx = -1
        best_score = -np.inf
        for idx in remaining:
            relevance = cosine_similarity(query_emb, doc_embs[idx])  # 与查询的相似度
            if selected_indices:
                max_sim_to_selected = max(  # 与已选文档的最大相似度（惩罚冗余）
                    cosine_similarity(doc_embs[idx], doc_embs[j])
                    for j in selected_indices
                )
            else:
                max_sim_to_selected = 0
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected  # MMR 公式
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    state.final_docs = [candidates[i] for i in selected_indices]  # 按 MMR 排序取最终文档
    logger.info(f"MMR selected {len(state.final_docs)} docs")
    return state
