from src.graph.builder import build_rag_graph, should_split
from src.state import RAGState


class TestGraphCompilation:
    def test_builds_without_error(self):
        graph = build_rag_graph()
        assert graph is not None

    def test_has_all_nodes(self):
        graph = build_rag_graph()
        nodes = list(graph.nodes.keys())
        assert "classify" in nodes
        assert "rewrite" in nodes
        assert "split" in nodes
        assert "retrieve" in nodes
        assert "rrf" in nodes
        assert "rerank" in nodes
        assert "mmr" in nodes
        assert "generate" in nodes

    def test_entry_point_is_classify(self):
        graph = build_rag_graph()
        nodes = list(graph.nodes.keys())
        assert "classify" in nodes
        assert nodes.index("classify") == 1


class TestShouldSplit:
    def test_returns_true_when_multi_hop_enabled(self):
        state = RAGState(original_question="test", config={"use_multi_hop": True})
        assert should_split(state) is True

    def test_returns_false_when_multi_hop_disabled(self):
        state = RAGState(original_question="test", config={"use_multi_hop": False})
        assert should_split(state) is False

    def test_defaults_to_false(self):
        state = RAGState(original_question="test")
        assert should_split(state) is False


class TestGraphRouting:
    def test_rag_route_goes_to_rewrite(self, mocker):
        mocker.patch("src.nodes.rewrite.get_llm")
        mocker.patch("src.nodes.classifier.get_llm")
        mocker.patch("src.nodes.split.get_llm")
        mocker.patch("src.nodes.mmr.embed_text", return_value=[0.0] * 10)
        mocker.patch("src.nodes.rerank.rerank", return_value=[])
        mocker.patch("src.retrieval.hybrid_retriever.hybrid_search",
                     return_value=([], []))
        mocker.patch("src.nodes.generate.get_llm")

        graph = build_rag_graph()
        state = RAGState(original_question="华南理工计算机分数线")
        result = graph.invoke(state)
        assert "final_answer" in result

    def test_graph_runs_with_all_nodes_mocked(self, mocker):
        mocker.patch("src.nodes.rewrite.get_llm").return_value.invoke.return_value.content = "test"
        mocker.patch("src.nodes.classifier.get_llm").return_value.invoke.return_value.content = "rag"
        mocker.patch("src.nodes.split.get_llm").return_value.invoke.return_value.content = "子问题1"
        mocker.patch("src.nodes.generate.get_llm").return_value.invoke.return_value.content = "最终答案"
        mocker.patch("src.nodes.mmr.embed_text", return_value=[0.0] * 4)
        mocker.patch("src.nodes.rerank.rerank", return_value=[])
        mocker.patch("src.nodes.retrieve.hybrid_search",
                     return_value=([], []))

        graph = build_rag_graph()
        state = RAGState(original_question="计算机专业排名")
        result = graph.invoke(state)
        assert result.get("final_answer") == "最终答案"
