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

    builder.add_node("classify", classify_query)  # 分类：判断问题属于 RAG 还是闲聊
    builder.add_node("rewrite", rewrite_question)  # 改写：优化查询语句适配检索
    builder.add_node("split", split_question)      # 拆分：将复杂问题分解为子问题
    builder.add_node("retrieve", hybrid_retrieve_node)  # 检索：混合检索（稠密+稀疏）
    builder.add_node("rrf", rrf_fusion)            # 融合：RRF 算法融合多路结果
    builder.add_node("rerank", cross_encoder_rerank)  # 重排：交叉编码器精排
    builder.add_node("mmr", mmr_selection)         # MMR：最大边际相关性去重
    builder.add_node("generate", generate_answer)  # 生成：基于上下文生成最终答案

    builder.set_entry_point("classify")            # 入口：从分类节点开始
    builder.add_conditional_edges(
        "classify",
        should_use_rag,
        {"rag": "rewrite", "chat": "generate"},  # RAG 走检索分支，闲聊直接回答
    )
    builder.add_conditional_edges(
        "rewrite",
        should_split,
        {True: "split", False: "retrieve"},      # 多跳开启则拆分，否则直接检索
    )
    builder.add_edge("split", "retrieve")          # 各子问题依次执行检索
    builder.add_edge("retrieve", "rrf")            # 检索结果送入 RRF 融合
    builder.add_edge("rrf", "rerank")              # 融合后重排
    builder.add_edge("rerank", "mmr")              # 重排后做 MMR 去重
    builder.add_edge("mmr", "generate")            # 最终文档送入生成
    builder.add_edge("generate", END)              # 生成结束

    return builder.compile()
