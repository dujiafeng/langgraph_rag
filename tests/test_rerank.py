from src.state import RAGState
from src.nodes.rerank import cross_encoder_rerank
from tests.conftest import make_doc


class TestCrossEncoderRerank:
    def test_empty_candidates_returns_empty(self, mocker):
        mocker.patch("src.nodes.rerank.rerank", return_value=[])
        state = RAGState(
            original_question="test",
            fused_results=[],
        )
        result = cross_encoder_rerank(state)
        assert result.reranked_results == []

    def test_reranks_candidates(self, mocker):
        mock_rerank = mocker.patch("src.nodes.rerank.rerank")
        ranked = [make_doc("doc1"), make_doc("doc2")]
        mock_rerank.return_value = ranked

        state = RAGState(
            original_question="test",
            rewritten_question="test query",
            fused_results=[make_doc("doc1"), make_doc("doc2")],
            config={"top_k_rerank": 10},
        )
        result = cross_encoder_rerank(state)
        assert result.reranked_results == ranked

    def test_rerank_limited_to_50_candidates(self, mocker):
        mock_rerank = mocker.patch("src.nodes.rerank.rerank")

        docs = [make_doc(f"doc{i}") for i in range(100)]
        state = RAGState(
            original_question="test",
            rewritten_question="test",
            fused_results=docs,
            config={"top_k_rerank": 10},
        )
        cross_encoder_rerank(state)
        call_args = mock_rerank.call_args[1]
        assert len(call_args["candidates"]) == 50

    def test_uses_rewritten_or_original_query(self, mocker):
        mock_rerank = mocker.patch("src.nodes.rerank.rerank")

        state = RAGState(
            original_question="原始问题",
            rewritten_question=None,
            fused_results=[make_doc("doc")],
            config={"top_k_rerank": 1},
        )
        cross_encoder_rerank(state)
        call_args = mock_rerank.call_args[1]
        assert call_args["query"] == "原始问题"

    def test_top_k_from_config(self, mocker):
        mock_rerank = mocker.patch("src.nodes.rerank.rerank")

        state = RAGState(
            original_question="test",
            rewritten_question="test",
            fused_results=[make_doc(f"doc{i}") for i in range(10)],
            config={"top_k_rerank": 3},
        )
        cross_encoder_rerank(state)
        call_args = mock_rerank.call_args[1]
        assert call_args["top_k"] == 3
