import numpy as np
import pytest

from src.state import RAGState
from src.nodes.mmr import mmr_selection
from tests.conftest import make_doc


def mock_embed_text(mocker, vectors: dict[str, list[float]]):
    def _embed(text: str) -> list[float]:
        for key, vec in vectors.items():
            if key in text:
                return vec
        return [0.0] * 10
    mocker.patch("src.nodes.mmr.embed_text", side_effect=_embed)


DIM = 4


def _mock_embed(mocker, vectors: dict[str, list[float]]):
    def _embed(text: str) -> list[float]:
        for key, vec in vectors.items():
            if key in text:
                return vec
        return [0.0] * DIM
    mocker.patch("src.nodes.mmr.embed_text", side_effect=_embed)


class TestMMRSelection:
    def test_empty_candidates_returns_empty(self, mocker):
        mocker.patch("src.nodes.mmr.embed_text", return_value=[0.0] * DIM)
        state = RAGState(
            original_question="test",
            reranked_results=[],
        )
        result = mmr_selection(state)
        assert result.final_docs == []

    def test_selects_top_k(self, mocker):
        vec = [1.0, 0.0, 0.0, 0.0]
        d = {f"doc{i}": vec for i in range(10)}
        d["test"] = vec
        _mock_embed(mocker, d)
        state = RAGState(
            original_question="test",
            rewritten_question="test",
            reranked_results=[make_doc(f"doc{i}") for i in range(10)],
            config={"top_k_mmr": 3},
        )
        result = mmr_selection(state)
        assert len(result.final_docs) == 3

    def test_lambda_one_pure_relevance(self, mocker):
        query_vec = [1.0, 0.0, 0.0, 0.0]
        _mock_embed(mocker, {
            "test": query_vec,
            "doc_a": [0.9, 0.1, 0.0, 0.0],
            "doc_b": [0.1, 0.9, 0.0, 0.0],
        })
        state = RAGState(
            original_question="test",
            rewritten_question="test",
            reranked_results=[make_doc("doc_a"), make_doc("doc_b")],
            config={"top_k_mmr": 2},
        )
        result = mmr_selection(state)
        assert len(result.final_docs) == 2

    def test_not_more_than_candidates(self, mocker):
        _mock_embed(mocker, {"test": [1.0, 0.0, 0.0, 0.0], "doc": [0.0, 1.0, 0.0, 0.0]})
        state = RAGState(
            original_question="test",
            rewritten_question="test",
            reranked_results=[make_doc("doc1"), make_doc("doc2")],
            config={"top_k_mmr": 10},
        )
        result = mmr_selection(state)
        assert len(result.final_docs) == 2

    def test_no_rewritten_uses_original(self, mocker):
        mocker.patch("src.nodes.mmr.embed_text", return_value=[0.0] * DIM)
        state = RAGState(
            original_question="原始问题",
            reranked_results=[make_doc("doc")],
            config={"top_k_mmr": 1},
        )
        result = mmr_selection(state)
        assert len(result.final_docs) == 1

    def test_preserves_doc_text(self, mocker):
        vec = [1.0, 0.0, 0.0, 0.0]
        _mock_embed(mocker, {"测试": vec, "有用文档": vec})
        state = RAGState(
            original_question="测试",
            rewritten_question="测试",
            reranked_results=[make_doc("有用文档")],
            config={"top_k_mmr": 1},
        )
        result = mmr_selection(state)
        assert result.final_docs[0]["text"] == "有用文档"
