from typing import Tuple, List, Dict

from src.retrieval.dense_retriever import dense_retriever
from src.retrieval.sparse_retriever import sparse_retriever
from src.utils.deduplicate import deduplicate_by_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


def hybrid_search(query: str, top_k: int = 20) -> Tuple[List[Dict], List[Dict]]:
    dense_hits = dense_retriever.similarity_search(query, top_k)
    logger.info(f"Dense retriever returned {len(dense_hits)} hits")

    sparse_hits = sparse_retriever.bm25_search(query, top_k)
    logger.info(f"Sparse retriever returned {len(sparse_hits)} hits")

    dense_hits = deduplicate_by_text(dense_hits)
    sparse_hits = deduplicate_by_text(sparse_hits)

    return dense_hits, sparse_hits
