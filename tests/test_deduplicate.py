from src.utils.deduplicate import deduplicate_by_text
from tests.conftest import make_doc


class TestDeduplicateByText:
    def test_empty_list_returns_empty(self):
        assert deduplicate_by_text([]) == []

    def test_all_unique_keeps_order(self):
        docs = [make_doc("a"), make_doc("b"), make_doc("c")]
        result = deduplicate_by_text(docs)
        assert len(result) == 3
        assert [d["text"] for d in result] == ["a", "b", "c"]

    def test_duplicates_removed_keep_first(self):
        docs = [make_doc("a"), make_doc("b"), make_doc("a"), make_doc("c"), make_doc("b")]
        result = deduplicate_by_text(docs)
        assert len(result) == 3
        assert [d["text"] for d in result] == ["a", "b", "c"]

    def test_all_duplicates_returns_single(self):
        docs = [make_doc("x"), make_doc("x"), make_doc("x")]
        result = deduplicate_by_text(docs)
        assert len(result) == 1
        assert result[0]["text"] == "x"

    def test_metadata_preserved_on_first_occurrence(self):
        docs = [
            {"text": "a", "metadata": {"source": "first"}},
            {"text": "a", "metadata": {"source": "second"}},
        ]
        result = deduplicate_by_text(docs)
        assert len(result) == 1
        assert result[0]["metadata"]["source"] == "first"

    def test_case_sensitive(self):
        docs = [make_doc("A"), make_doc("a")]
        result = deduplicate_by_text(docs)
        assert len(result) == 2

    def test_empty_string_handling(self):
        docs = [make_doc(""), make_doc("hello"), make_doc("")]
        result = deduplicate_by_text(docs)
        assert len(result) == 2
