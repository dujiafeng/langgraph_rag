from src.state import RAGState
from src.nodes.classifier import classify_query, should_use_rag


class TestShouldUseRag:
    def test_returns_query_type(self, base_state):
        base_state.query_type = "rag"
        assert should_use_rag(base_state) == "rag"

        base_state.query_type = "chat"
        assert should_use_rag(base_state) == "chat"


class TestClassifyQuery:
    def test_classifies_as_rag(self, mocker):
        mock = mocker.patch("src.nodes.classifier.get_llm")
        mock.return_value.invoke.return_value.content = "rag"

        state = classify_query(RAGState(original_question="北大录取分数线"))
        assert state.query_type == "rag"


    def test_classifies_as_chat(self, mocker):
        mock = mocker.patch("src.nodes.classifier.get_llm")
        mock.return_value.invoke.return_value.content = "chat"

        state = classify_query(RAGState(original_question="你好"))
        assert state.query_type == "chat"

    def test_case_insensitive(self, mocker):
        mock = mocker.patch("src.nodes.classifier.get_llm")
        mock.return_value.invoke.return_value.content = "RAG"

        state = classify_query(RAGState(original_question="分数线"))
        assert state.query_type == "rag"

    def test_unknown_falls_to_chat(self, mocker):
        mock = mocker.patch("src.nodes.classifier.get_llm")
        mock.return_value.invoke.return_value.content = "unknown"

        state = classify_query(RAGState(original_question="???"))
        assert state.query_type == "chat"

    def test_prompt_includes_question(self, mocker):
        mock = mocker.patch("src.nodes.classifier.get_llm")
        mock.return_value.invoke.return_value.content = "rag"

        classify_query(RAGState(original_question="华南理工计算机分数线"))
        call_args = mock.return_value.invoke.call_args[0][0]
        assert "华南理工计算机分数线" in call_args
