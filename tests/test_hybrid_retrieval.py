from src.retrieval.hybrid_retriever import hybrid_search
from tests.conftest import make_doc


class TestHybridSearch:
    def test_calls_both_retrievers(self, mocker):
        mocker.patch("src.retrieval.hybrid_retriever.dense_retriever.similarity_search",
                     return_value=[make_doc("dense_doc", dense_score=0.9)])
        mocker.patch("src.retrieval.hybrid_retriever.sparse_retriever.bm25_search",
                     return_value=[make_doc("sparse_doc", sparse_score=5.0)])

        dense, sparse = hybrid_search("test query", top_k=10)
        assert len(dense) == 1
        assert dense[0]["text"] == "dense_doc"
        assert len(sparse) == 1
        assert sparse[0]["text"] == "sparse_doc"

    def test_deduplicates_dense_results(self, mocker):
        mocker.patch("src.retrieval.hybrid_retriever.dense_retriever.similarity_search",
                     return_value=[
                         make_doc("doc_a", dense_score=0.9),
                         make_doc("doc_b", dense_score=0.8),
                         make_doc("doc_a", dense_score=0.7),
                     ])
        mocker.patch("src.retrieval.hybrid_retriever.sparse_retriever.bm25_search",
                     return_value=[])

        dense, _ = hybrid_search("test", top_k=10)
        assert len(dense) == 2

    def test_deduplicates_sparse_results(self, mocker):
        mocker.patch("src.retrieval.hybrid_retriever.dense_retriever.similarity_search",
                     return_value=[])
        mocker.patch("src.retrieval.hybrid_retriever.sparse_retriever.bm25_search",
                     return_value=[
                         make_doc("doc_x", sparse_score=10.0),
                         make_doc("doc_x", sparse_score=5.0),
                         make_doc("doc_y", sparse_score=3.0),
                     ])

        _, sparse = hybrid_search("test", top_k=10)
        assert len(sparse) == 2

    def test_empty_results(self, mocker):
        mocker.patch("src.retrieval.hybrid_retriever.dense_retriever.similarity_search",
                     return_value=[])
        mocker.patch("src.retrieval.hybrid_retriever.sparse_retriever.bm25_search",
                     return_value=[])

        dense, sparse = hybrid_search("nonexistent", top_k=10)
        assert dense == []
        assert sparse == []

    def test_passes_top_k_to_both(self, mocker):
        mock_dense = mocker.patch(
            "src.retrieval.hybrid_retriever.dense_retriever.similarity_search",
            return_value=[])
        mock_sparse = mocker.patch(
            "src.retrieval.hybrid_retriever.sparse_retriever.bm25_search",
            return_value=[])

        hybrid_search("test", top_k=5)
        assert mock_dense.call_args[0][1] == 5
        assert mock_sparse.call_args[0][1] == 5
