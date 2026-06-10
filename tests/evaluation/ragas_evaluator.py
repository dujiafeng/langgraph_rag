"""Ragas-based RAG evaluation wrapper.

Usage:
    from tests.evaluation.ragas_evaluator import RagasEvaluator
    evaluator = RagasEvaluator()
    df = evaluator.evaluate(samples)
    print(df)
"""

import os
from typing import Dict, List, Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    answer_correctness,
)

from src.utils.embeddings import get_embeddings
from src.nodes.rewrite import get_llm


class RagasEvaluator:
    """Compute ragas metrics on a set of (question, answer, contexts, ground_truth) samples.

    Metrics:
        - faithfulness: Is the answer faithful to the retrieved context?
        - answer_relevancy: How relevant is the answer to the question?
        - context_recall: Does the retrieved context cover the ground truth?
        - answer_correctness: How factually correct is the answer?
    """

    def __init__(self, llm=None, embeddings=None):
        self.llm = llm or get_llm()
        self.embeddings = embeddings or get_embeddings()

    def evaluate(self, samples: List[Dict]) -> "pandas.DataFrame":
        dataset = Dataset.from_dict({
            "question": [s["question"] for s in samples],
            "answer": [s.get("answer", "") for s in samples],
            "contexts": [s.get("contexts", []) for s in samples],
            "ground_truth": [s.get("ground_truth", "") for s in samples],
        })
        result = evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_recall,
                answer_correctness,
            ],
            llm=self.llm,
            embeddings=self.embeddings,
        )
        return result.to_pandas()

    @staticmethod
    def print_report(df: "pandas.DataFrame"):
        print("=" * 60)
        print("       Ragas Evaluation Report")
        print("=" * 60)
        metrics_cols = [c for c in df.columns if c != "question"]
        for col in metrics_cols:
            avg = df[col].mean()
            print(f"{col:<25} {avg:.4f}")
        print("=" * 60)
