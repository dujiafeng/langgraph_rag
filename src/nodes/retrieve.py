from src.state import RAGState
from src.retrieval.hybrid_retriever import hybrid_search
from src.utils.deduplicate import deduplicate_by_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


def hybrid_retrieve_node(state: RAGState) -> RAGState:
    queries = state.sub_questions if state.sub_questions else [state.rewritten_question]
    all_dense = []
    all_sparse = []

    for q in queries:
        logger.info(f"Hybrid search for: {q}")
        dense_hits, sparse_hits = hybrid_search(
            query=q,
            top_k=state.config["top_k_hybrid"],
        )
        all_dense.extend(dense_hits)
        all_sparse.extend(sparse_hits)

    state.dense_results = deduplicate_by_text(all_dense)
    state.sparse_results = deduplicate_by_text(all_sparse)
    logger.info(f"Dense results: {len(state.dense_results)}, Sparse results: {len(state.sparse_results)}")
    return state
