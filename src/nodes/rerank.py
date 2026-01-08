from src.state import RAGState
from src.rerankers.cross_encoder import rerank
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cross_encoder_rerank(state: RAGState) -> RAGState:
    query = state.rewritten_question or state.original_question
    candidates = state.fused_results[:50]
    if not candidates:
        logger.warning("No candidates to rerank")
        state.reranked_results = []
        return state

    logger.info(f"Reranking {len(candidates)} candidates...")
    ranked = rerank(
        query=query,
        candidates=candidates,
        top_k=state.config["top_k_rerank"],
    )
    state.reranked_results = ranked
    logger.info(f"After rerank: {len(ranked)} docs")
    return state
