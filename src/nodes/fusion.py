from src.state import RAGState
from src.utils.logger import get_logger

logger = get_logger(__name__)


def rrf_fusion(state: RAGState) -> RAGState:
    k = 60
    doc_map = {}

    for rank, doc in enumerate(state.dense_results, start=1):
        doc_id = doc["text"]
        if doc_id not in doc_map:
            doc_map[doc_id] = {"doc": doc, "rrf": 0}
        doc_map[doc_id]["rrf"] += 1 / (k + rank)

    for rank, doc in enumerate(state.sparse_results, start=1):
        doc_id = doc["text"]
        if doc_id not in doc_map:
            doc_map[doc_id] = {"doc": doc, "rrf": 0}
        doc_map[doc_id]["rrf"] += 1 / (k + rank)

    sorted_docs = sorted(doc_map.values(), key=lambda x: x["rrf"], reverse=True)
    state.fused_results = [item["doc"] for item in sorted_docs]
    logger.info(f"RRF fused results: {len(state.fused_results)} docs")
    return state
