import argparse
from dotenv import load_dotenv

load_dotenv()

from config.settings import settings
from graph.builder import build_rag_graph
from state import RAGState


def main():
    parser = argparse.ArgumentParser(description="LangGraph RAG Pipeline")
    parser.add_argument("--question", type=str, required=True, help="用户问题")
    parser.add_argument(
        "--multi-hop",
        action="store_true",
        default=settings.use_multi_hop,
        help="启用问题拆分 (Multi-Hop)",
    )
    args = parser.parse_args()

    initial_state = RAGState(
        original_question=args.question,
        config={
            "chunk_size": settings.chunk_size,
            "top_k_hybrid": settings.top_k_hybrid,
            "top_k_rerank": settings.top_k_rerank,
            "top_k_mmr": settings.top_k_mmr,
            "use_multi_hop": args.multi_hop,
        },
    )

    graph = build_rag_graph()
    final_state = graph.invoke(initial_state)

    print("\n" + "=" * 60)
    print("最终答案：")
    print(final_state)


if __name__ == "__main__":
    main()
