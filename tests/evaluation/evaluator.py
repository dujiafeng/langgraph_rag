"""RAGEvaluator: run RAG pipeline on sample questions and compute metrics."""

import json
import os
from typing import Dict, List, Set, Tuple

from src.state import RAGState
from src.graph.builder import build_rag_graph
from tests.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
    ndcg_at_k,
    extract_doc_ids,
)


def _kw_match(text: str, keywords: List[str]) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def _find_matching_doc_ids(docs: List[dict], keywords: List[str]) -> Set[str]:
    return {d.get("text", "") for d in docs if _kw_match(d.get("text", ""), keywords)}


class RAGEvaluator:
    def __init__(self, qa_file: str | None = None):
        self.graph = build_rag_graph()
        if qa_file is None:
            qa_file = os.path.join(os.path.dirname(__file__), "sample_qa.json")
        with open(qa_file, "r", encoding="utf-8") as f:
            self.qa_pairs: List[Dict] = json.load(f)
        self.results: List[Dict] = []

    def run_single(self, qa: Dict) -> Dict:
        initial = RAGState(
            original_question=qa["question"],
            config={
                "chunk_size": 512,
                "top_k_hybrid": 20,
                "top_k_rerank": 10,
                "top_k_mmr": 5,
                "use_multi_hop": False,
            },
        )
        result = self.graph.invoke(initial)

        # Get evaluated docs for metrics
        retrieval_docs = extract_doc_ids(result.get("fused_results", [])[:20])
        final_docs = extract_doc_ids(result.get("final_docs", []))

        # Determine relevant docs by keyword matching
        relevant_keywords = qa.get("expected_doc_keywords", [])
        relevant_set = set(relevant_keywords)

        # Compute retrieval metrics
        report = {
            "id": qa["id"],
            "question": qa["question"],
            "answer": result.get("final_answer", ""),
            "retrieval_doc_count": len(retrieval_docs),
            "final_doc_count": len(final_docs),
            "precision_at_5": precision_at_k(retrieval_docs, relevant_set, 5),
            "precision_at_10": precision_at_k(retrieval_docs, relevant_set, 10),
            "recall_at_5": recall_at_k(retrieval_docs, relevant_set, 5),
            "recall_at_10": recall_at_k(retrieval_docs, relevant_set, 10),
            "mrr": mrr(retrieval_docs, relevant_set),
            "ndcg_at_10": ndcg_at_k(retrieval_docs, relevant_set, 10),
        }
        return report

    def run_all(self) -> List[Dict]:
        self.results = [self.run_single(qa) for qa in self.qa_pairs]
        return self.results

    def aggregate_report(self) -> Dict[str, float]:
        if not self.results:
            return {}
        keys = [
            "precision_at_5", "precision_at_10",
            "recall_at_5", "recall_at_10",
            "mrr", "ndcg_at_10",
        ]
        agg = {}
        for key in keys:
            values = [r[key] for r in self.results]
            agg[key] = sum(values) / len(values) if values else 0.0
        return agg

    def print_report(self):
        agg = self.aggregate_report()
        print("=" * 60)
        print("       RAG Evaluation Report")
        print("=" * 60)
        print(f"{'Metric':<20} {'Score':<10}")
        print("-" * 60)
        for key, val in sorted(agg.items()):
            print(f"{key:<20} {val:<10.4f}")
        print("-" * 60)
        print(f"Questions: {len(self.results)}")
        print(f"Avg retrieval docs: {sum(r['retrieval_doc_count'] for r in self.results) / len(self.results):.1f}")
        print("=" * 60)
