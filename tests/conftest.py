from typing import Dict, List
from unittest.mock import MagicMock

import pytest

from src.state import RAGState


@pytest.fixture
def base_state():
    return RAGState(original_question="测试问题")


@pytest.fixture
def state_with_docs():
    return RAGState(
        original_question="测试问题",
        rewritten_question="改写后测试问题",
        dense_results=[
            {"text": f"稠密文档{i}", "dense_score": 1.0 - i * 0.1}
            for i in range(5)
        ],
        sparse_results=[
            {"text": f"稀疏文档{i}", "sparse_score": 10.0 - i}
            for i in range(5)
        ],
        fused_results=[
            {"text": f"融合文档{i}"} for i in range(10)
        ],
        reranked_results=[
            {"text": f"重排文档{i}"} for i in range(10)
        ],
        final_docs=[
            {"text": f"最终文档{i}"} for i in range(3)
        ],
        final_answer="测试答案",
    )


@pytest.fixture
def mock_llm(mocker):
    mock = MagicMock()
    mock.content = ""
    mocked_invoke = mocker.patch("src.nodes.rewrite.get_llm")
    mocked_invoke.return_value.invoke.return_value = mock
    return mock


def make_mock_llm_response(mocker, text: str):
    mock = MagicMock()
    mock.content = text
    mocked_invoke = mocker.patch("src.nodes.rewrite.get_llm")
    mocked_invoke.return_value.invoke.return_value = mock
    return mock


def make_doc(text: str, **kwargs) -> Dict:
    return {"text": text, "metadata": {}, **kwargs}
