from langgraph.graph import StateGraph, END

from src.state import RAGState
from src.nodes.classifier import classify_query, should_use_rag
from src.nodes.rewrite import rewrite_question
from src.nodes.split import split_question
from src.nodes.retrieve import hybrid_retrieve_node
from src.nodes.fusion import rrf_fusion
from src.nodes.rerank import cross_encoder_rerank
from src.nodes.mmr import mmr_selection
from src.nodes.generate import generate_answer


def should_split(state: RAGState) -> bool:
    return state.config.get("use_multi_hop", False)


def build_rag_graph() -> StateGraph:
    builder = StateGraph(RAGState)

    builder.add_node("classify", classify_query)
    builder.add_node("rewrite", rewrite_question)
    builder.add_node("split", split_question)
    builder.add_node("retrieve", hybrid_retrieve_node)
    builder.add_node("rrf", rrf_fusion)
    builder.add_node("rerank", cross_encoder_rerank)
    builder.add_node("mmr", mmr_selection)
    builder.add_node("generate", generate_answer)

    builder.set_entry_point("classify")
    builder.add_conditional_edges(
        "classify",
        should_use_rag,
        {"rag": "rewrite", "chat": "generate"},
    )
    builder.add_conditional_edges(
        "rewrite",
        should_split,
        {True: "split", False: "retrieve"},
    )
    builder.add_edge("split", "retrieve")
    builder.add_edge("retrieve", "rrf")
    builder.add_edge("rrf", "rerank")
    builder.add_edge("rerank", "mmr")
    builder.add_edge("mmr", "generate")
    builder.add_edge("generate", END)

    return builder.compile()
