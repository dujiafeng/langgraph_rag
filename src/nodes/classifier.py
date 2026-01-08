from typing import Literal

from src.state import RAGState
from src.nodes.rewrite import get_llm
from src.utils.logger import get_logger

logger = get_logger(__name__)

_CLASSIFY_PROMPT = (
    "你是一个高考志愿助手的问题分类器。判断用户的问题是以下哪种类型：\n\n"
    "1. rag - 需要查询高考数据库才能回答的问题。包括：学校介绍、专业介绍、"
    "录取分数线、位次查询、招生计划、就业数据、志愿填报建议、选科要求、"
    "学费信息、学校排名等。\n"
    "2. chat - 日常闲聊、问候、情感交流、无需查询数据库即可回答的通用对话。"
    "包括：打招呼、感谢、开玩笑、系统功能咨询等。\n\n"
    "只输出 rag 或 chat，不要输出其他内容。\n\n"
    "用户问题：{question}"
)


def should_use_rag(state: RAGState) -> Literal["rag", "chat"]:
    return state.query_type


def classify_query(state: RAGState) -> RAGState:
    logger.info(f"Classifying query: {state.original_question}")
    response = get_llm().invoke(_CLASSIFY_PROMPT.format(question=state.original_question))
    result = response.content.strip().lower()
    state.query_type = "rag" if "rag" in result else "chat"
    logger.info(f"Query type: {state.query_type}")
    return state
