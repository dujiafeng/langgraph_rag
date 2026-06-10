from src.state import RAGState


class TestRAGStateDefaults:
    def test_default_query_type_is_rag(self):
        state = RAGState(original_question="test")
        assert state.query_type == "rag"

    def test_original_question_is_required(self):
        state = RAGState(original_question="北大分数线")
        assert state.original_question == "北大分数线"

    def test_optional_fields_default_to_none(self):
        state = RAGState(original_question="test")
        assert state.rewritten_question is None
        assert state.sub_questions is None
        assert state.final_answer is None

    def test_list_fields_default_to_empty(self):
        state = RAGState(original_question="test")
        assert state.dense_results == []
        assert state.sparse_results == []
        assert state.fused_results == []
        assert state.reranked_results == []
        assert state.final_docs == []
        assert state.sub_results == []


class TestRAGStateConfig:
    def test_default_config_values(self):
        state = RAGState(original_question="test")
        assert state.config["chunk_size"] == 512
        assert state.config["top_k_hybrid"] == 20
        assert state.config["top_k_rerank"] == 10
        assert state.config["top_k_mmr"] == 5
        assert state.config["use_multi_hop"] is False

    def test_custom_config_overrides_defaults(self):
        state = RAGState(
            original_question="test",
            config={"top_k_mmr": 8, "use_multi_hop": True},
        )
        assert state.config["top_k_mmr"] == 8
        assert state.config["use_multi_hop"] is True

    def test_full_state_creation(self, state_with_docs):
        s = state_with_docs
        assert s.original_question == "测试问题"
        assert s.rewritten_question == "改写后测试问题"
        assert len(s.dense_results) == 5
        assert len(s.sparse_results) == 5
        assert len(s.fused_results) == 10
        assert len(s.reranked_results) == 10
        assert len(s.final_docs) == 3
        assert s.final_answer == "测试答案"

    def test_state_dict_roundtrip(self):
        state = RAGState(original_question="test", query_type="chat", final_answer="你好")
        d = state.model_dump()
        restored = RAGState(**d)
        assert restored.original_question == "test"
        assert restored.query_type == "chat"
        assert restored.final_answer == "你好"
