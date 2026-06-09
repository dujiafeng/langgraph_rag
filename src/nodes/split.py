from src.state import RAGState
from src.nodes.rewrite import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_question(state: RAGState) -> RAGState:
    prompt = (
        f"你是一个高考志愿助手。将以下关于高考志愿的问题分解为 2~3 个相互独立的子问题，"
        f"每个子问题应能单独从高考数据库中检索回答（如分别查学校、查专业、查分数线）。\n"
        f"请每行输出一个子问题，不要添加额外说明。\n"
        f"问题：{state.rewritten_question}"
    )
    logger.info("Splitting question into sub-questions...")
    response = get_llm().invoke(prompt)  # 用 LLM 拆解复杂问题
    sub_qs = [q.strip() for q in response.content.split("\n") if q.strip()]  # 按行解析子问题
    state.sub_questions = sub_qs
    state.sub_results = [[] for _ in sub_qs]  # 为每个子问题预留结果列表
    logger.info(f"Sub-questions: {sub_qs}")
    return state
