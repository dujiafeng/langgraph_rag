from src.state import RAGState
from src.nodes.fusion import rrf_fusion
from tests.conftest import make_doc


class TestRRFFusion:
    def test_empty_results_returns_empty(self):
        state = RAGState(original_question="test")
        result = rrf_fusion(state)
        assert result.fused_results == []

    def test_only_dense_results_ordered(self):
        state = RAGState(
            original_question="test",
            dense_results=[
                make_doc("doc_a", dense_score=0.9),
                make_doc("doc_b", dense_score=0.8),
            ],
        )
        result = rrf_fusion(state)
        assert len(result.fused_results) == 2
        assert result.fused_results[0]["text"] == "doc_a"
        assert result.fused_results[1]["text"] == "doc_b"

    def test_only_sparse_results_ordered(self):
        state = RAGState(
            original_question="test",
            sparse_results=[
                make_doc("doc_x", sparse_score=10.0),
                make_doc("doc_y", sparse_score=5.0),
            ],
        )
        result = rrf_fusion(state)
        assert len(result.fused_results) == 2

    def test_overlap_docs_merged(self):
        state = RAGState(
            original_question="test",
            dense_results=[make_doc("common")],
            sparse_results=[make_doc("common")],
        )
        result = rrf_fusion(state)
        assert len(result.fused_results) == 1

    def test_overlap_gets_higher_score(self):
        state = RAGState(
            original_question="test",
            dense_results=[make_doc("only_dense"), make_doc("both")],
            sparse_results=[make_doc("both")],
        )
        result = rrf_fusion(state)
        texts = [d["text"] for d in result.fused_results]
        assert texts.index("both") < texts.index("only_dense")

    def test_rrf_scores_correct(self):
        state = RAGState(
            original_question="test",
            dense_results=[make_doc("doc_a"), make_doc("doc_b")],
            sparse_results=[make_doc("doc_b")],
        )
        result = rrf_fusion(state)
        fused = {d["text"]: d for d in result.fused_results}
        assert "doc_a" in fused
        assert "doc_b" in fused

    def test_preserves_doc_metadata(self):
        doc = {"text": "doc", "metadata": {"source": "dense"}, "dense_score": 0.9}
        state = RAGState(
            original_question="test",
            dense_results=[doc],
        )
        result = rrf_fusion(state)
        assert result.fused_results[0]["metadata"]["source"] == "dense"

    def test_all_unique_docs_present(self):
        state = RAGState(
            original_question="test",
            dense_results=[make_doc("a"), make_doc("b")],
            sparse_results=[make_doc("c")],
        )
        result = rrf_fusion(state)
        texts = [d["text"] for d in result.fused_results]
        assert set(texts) == {"a", "b", "c"}
        assert len(texts) == 3
