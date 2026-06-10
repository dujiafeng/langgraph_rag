from src.state import RAGState
from src.nodes.split import split_question


class TestSplitQuestion:
    def test_splits_into_sub_questions(self, mocker):
        mock = mocker.patch("src.nodes.split.get_llm")
        mock.return_value.invoke.return_value.content = (
            "计算机专业的就业前景如何？\n"
            "计算机专业的录取分数线是多少？"
        )

        state = RAGState(
            original_question="学计算机怎么样",
            rewritten_question="计算机专业的前景和分数线",
        )
        result = split_question(state)
        assert result.sub_questions == ["计算机专业的就业前景如何？", "计算机专业的录取分数线是多少？"]

    def test_sub_results_initialized(self, mocker):
        mock = mocker.patch("src.nodes.split.get_llm")
        mock.return_value.invoke.return_value.content = "Q1\nQ2\nQ3"

        state = RAGState(
            original_question="test",
            rewritten_question="test rewrite",
        )
        result = split_question(state)
        assert len(result.sub_results) == 3
        assert all(r == [] for r in result.sub_results)

    def test_single_line_result(self, mocker):
        mock = mocker.patch("src.nodes.split.get_llm")
        mock.return_value.invoke.return_value.content = "单一子问题"

        state = RAGState(
            original_question="test",
            rewritten_question="test",
        )
        result = split_question(state)
        assert result.sub_questions == ["单一子问题"]
        assert len(result.sub_results) == 1

    def test_empty_lines_skipped(self, mocker):
        mock = mocker.patch("src.nodes.split.get_llm")
        mock.return_value.invoke.return_value.content = "\nQ1\n\nQ2\n"

        state = RAGState(
            original_question="test",
            rewritten_question="test",
        )
        result = split_question(state)
        assert result.sub_questions == ["Q1", "Q2"]

    def test_strips_whitespace(self, mocker):
        mock = mocker.patch("src.nodes.split.get_llm")
        mock.return_value.invoke.return_value.content = "  子问题A  \n  子问题B  "

        state = RAGState(
            original_question="test",
            rewritten_question="test",
        )
        result = split_question(state)
        assert result.sub_questions == ["子问题A", "子问题B"]
