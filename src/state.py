from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class RAGState(BaseModel):
    original_question: str
    query_type: str = "rag"  # "rag" - 需检索知识库, "chat" - 闲聊直接回答

    rewritten_question: Optional[str] = None

    sub_questions: Optional[List[str]] = None
    current_sub_idx: int = 0
    sub_results: List[List[Dict]] = []

    dense_results: List[Dict] = []
    sparse_results: List[Dict] = []

    fused_results: List[Dict] = []

    reranked_results: List[Dict] = []

    final_docs: List[Dict] = []

    final_answer: Optional[str] = None

    config: Dict[str, Any] = {
        "chunk_size": 512,
        "top_k_hybrid": 20,
        "top_k_rerank": 10,
        "top_k_mmr": 5,
        "use_multi_hop": False,
    }
