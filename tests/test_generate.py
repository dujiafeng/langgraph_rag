from src.state import RAGState
from src.nodes.generate import generate_answer
from tests.conftest import make_doc


class TestGenerateAnswer:
    def test_generates_without_docs_chat_path(self, mocker):
        mock = mocker.patch("src.nodes.generate.get_llm")
        mock.return_value.invoke.return_value.content = "你好！有什么可以帮助你的吗？"

        state = RAGState(original_question="你好")
        result = generate_answer(state)
        assert "你好" in result.final_answer

    def test_generates_with_docs_rag_path(self, mocker):
        mock = mocker.patch("src.nodes.generate.get_llm")
        mock.return_value.invoke.return_value.content = (
            "华东理工大学位于上海，是211高校，计算机专业录取分数约560分。"
        )

        state = RAGState(
            original_question="华东理工大学计算机专业分数线",
            final_docs=[
                make_doc("华东理工大学是211高校，位于上海。"),
                make_doc("计算机专业录取分数线约560分。"),
            ],
        )
        result = generate_answer(state)
        assert result.final_answer is not None

    def test_context_included_in_prompt(self, mocker):
        mock = mocker.patch("src.nodes.generate.get_llm")
        mock.return_value.invoke.return_value.content = "答案"

        state = RAGState(
            original_question="大连理工怎么样",
            final_docs=[
                make_doc("大连理工大学是985高校。"),
            ],
        )
        generate_answer(state)
        call_args = mock.return_value.invoke.call_args[0][0]
        assert "大连理工大学是985高校" in call_args
        assert "大连理工怎么样" in call_args

    def test_chat_path_no_context_in_prompt(self, mocker):
        mock = mocker.patch("src.nodes.generate.get_llm")
        mock.return_value.invoke.return_value.content = "今天天气不错！"

        state = RAGState(original_question="今天天气怎么样")
        generate_answer(state)
        call_args = mock.return_value.invoke.call_args[0][0]
        assert "高考志愿助手" in call_args
        assert "今天天气怎么样" in call_args

    def test_empty_final_docs_is_chat_path(self, mocker):
        mock = mocker.patch("src.nodes.generate.get_llm")
        mock.return_value.invoke.return_value.content = "好的"

        state = RAGState(
            original_question="谢谢",
            final_docs=[],
        )
        result = generate_answer(state)
        assert result.final_answer == "好的"
