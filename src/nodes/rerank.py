from src.state import RAGState
from src.rerankers.cross_encoder import rerank
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cross_encoder_rerank(state: RAGState) -> RAGState:
    query = state.rewritten_question or state.original_question  # 取改写后的查询
    candidates = state.fused_results[:50]  # 取 RRF 融合后前 50 条候选
    if not candidates:
        logger.warning("No candidates to rerank")
        state.reranked_results = []
        return state

    logger.info(f"Reranking {len(candidates)} candidates...")
    ranked = rerank(  # 用 cross-encoder 对查询-文档对评分重排
        query=query,
        candidates=candidates,
        top_k=state.config["top_k_rerank"],
    )
    state.reranked_results = ranked
    logger.info(f"After rerank: {len(ranked)} docs")
    return state
