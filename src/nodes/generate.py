from src.state import RAGState
from src.nodes.rewrite import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_answer(state: RAGState) -> RAGState:
    if not state.final_docs:
        prompt = f"你是一个高考志愿助手。请友好地回答用户的问题。\n\n用户：{state.original_question}\n\n回答："
    else:
        context = "\n\n".join([doc["text"] for doc in state.final_docs])
        prompt = (
            f"你是一个高考志愿助手。请基于以下参考资料，为考生和家长提供准确、有用的高考志愿填报建议。"
            f"如果资料不足以回答，请明确告知。回答问题时要结合分数、位次、学校特色、专业前景等。\n\n"
            f"参考资料：\n{context}\n\n"
            f"用户问题：{state.original_question}\n\n回答："
        )

    logger.info("Generating final answer...")
    response = get_llm().invoke(prompt)
    state.final_answer = response.content.strip()
    logger.info(f"Answer generated ({len(state.final_answer)} chars)")
    return state
