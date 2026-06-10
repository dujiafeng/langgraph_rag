#!/usr/bin/env python
"""CLI entry point for RAG evaluation.

Usage:
    uv run python tests/evaluation/run_evaluation.py
    uv run python tests/evaluation/run_evaluation.py --qa-file custom.json
    uv run python tests/evaluation/run_evaluation.py --num-questions 5
    uv run python tests/evaluation/run_evaluation.py --use-multi-hop
    uv run python tests/evaluation/run_evaluation.py --output results.json
    uv run python tests/evaluation/run_evaluation.py --ragas-only
    uv run python tests/evaluation/run_evaluation.py --custom-only
"""

import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation")
    parser.add_argument("--qa-file", type=str, default=None,
                        help="Path to QA JSON file (default: sample_qa.json)")
    parser.add_argument("--num-questions", type=int, default=0,
                        help="Number of questions to evaluate (0 = all)")
    parser.add_argument("--use-multi-hop", action="store_true",
                        help="Enable multi-hop question splitting")
    parser.add_argument("--output", type=str, default="",
                        help="Save results to JSON file")
    parser.add_argument("--ragas-only", action="store_true",
                        help="Only run ragas evaluation")
    parser.add_argument("--custom-only", action="store_true",
                        help="Only run custom evaluation")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    # Load QA data
    qa_file = args.qa_file or os.path.join(os.path.dirname(__file__), "sample_qa.json")
    if not os.path.exists(qa_file):
        print(f"Error: QA file not found: {qa_file}")
        sys.exit(1)
    with open(qa_file, "r", encoding="utf-8") as f:
        all_qa = json.load(f)
    if args.num_questions > 0:
        all_qa = all_qa[:args.num_questions]
    print(f"Loaded {len(all_qa)} QA pairs from {qa_file}")

    run_ragas = not args.custom_only
    run_custom = not args.ragas_only

    if run_custom:
        print("\n--- Running custom evaluation ---")
        from tests.evaluation.evaluator import RAGEvaluator
        evaluator = RAGEvaluator(qa_file=qa_file)
        results = evaluator.run_all()
        evaluator.print_report()

        if args.output:
            output_path = args.output
            if not output_path.endswith(".json"):
                output_path += "_custom.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump({"results": results, "aggregate": evaluator.aggregate_report()}, f, ensure_ascii=False, indent=2)
            print(f"Custom results saved to {output_path}")

    if run_ragas:
        print("\n--- Running ragas evaluation ---")
        from tests.evaluation.ragas_evaluator import RagasEvaluator
        from src.graph.builder import build_rag_graph
        from src.state import RAGState

        graph = build_rag_graph()
        samples = []
        for qa in all_qa:
            print(f"  Running pipeline: {qa['id']} - {qa['question'][:40]}...")
            initial = RAGState(
                original_question=qa["question"],
                config={
                    "use_multi_hop": args.use_multi_hop,
                    "top_k_hybrid": 20,
                    "top_k_rerank": 10,
                    "top_k_mmr": 5,
                },
            )
            result = graph.invoke(initial)
            samples.append({
                "question": qa["question"],
                "answer": result.get("final_answer", ""),
                "contexts": [d.get("text", "") for d in result.get("final_docs", [])],
                "ground_truth": qa.get("ground_truth", ""),
            })

        ragas_eval = RagasEvaluator()
        df = ragas_eval.evaluate(samples)
        ragas_eval.print_report(df)

        if args.output:
            output_path = args.output
            if not output_path.endswith(".json"):
                output_path += "_ragas.json"
            else:
                output_path = output_path.replace(".json", "_ragas.json")
            df.to_json(output_path, orient="records", force_ascii=False)
            print(f"Ragas results saved to {output_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
