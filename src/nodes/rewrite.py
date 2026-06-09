from langchain_openai import ChatOpenAI

from src.state import RAGState
from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_llm = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:  # 惰性初始化，避免 import 时因缺 API key 报错
        _llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    return _llm


def rewrite_question(state: RAGState) -> RAGState:
    prompt = (
        f"你是一个高考志愿助手。请将以下用户问题改写成清晰、适合高考数据库检索的正式查询，"
        f"保留学校名、专业名、年份等关键信息，只输出改写后的句子：\n{state.original_question}"
    )
    logger.info("Rewriting question...")
    response = get_llm().invoke(prompt)  # 用 DeepSeek 改写查询
    state.rewritten_question = response.content.strip()
    logger.info(f"Rewritten: {state.rewritten_question}")
    return state
