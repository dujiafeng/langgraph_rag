import pytest

from tests.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    f1_at_k,
    mrr,
    map_at_k,
    ndcg_at_k,
)


class TestPrecisionAtK:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "b"], {"a"}, 2) == 0.5

    def test_none_relevant(self):
        assert precision_at_k(["a", "b"], {"c"}, 2) == 0.0

    def test_k_larger_than_list(self):
        assert precision_at_k(["a"], {"a"}, 5) == 1.0

    def test_empty_list(self):
        assert precision_at_k([], {"a"}, 3) == 0.0

    def test_zero_k(self):
        assert precision_at_k(["a", "b"], {"a"}, 0) == 0.0


class TestRecallAtK:
    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "b"], {"a", "b", "c"}, 2) == pytest.approx(2 / 3)

    def test_no_relevant(self):
        assert recall_at_k(["a"], {"b"}, 2) == 0.0

    def test_empty_relevant(self):
        assert recall_at_k(["a", "b"], set(), 2) == 0.0


class TestF1AtK:
    def test_perfect(self):
        assert f1_at_k(["a", "b"], {"a", "b"}, 2) == 1.0

    def test_tradeoff(self):
        f1 = f1_at_k(["a", "b"], {"a", "c"}, 2)
        expected_p = 0.5
        expected_r = 0.5
        expected_f1 = 2 * expected_p * expected_r / (expected_p + expected_r)
        assert f1 == pytest.approx(expected_f1)

    def test_no_match(self):
        assert f1_at_k(["a"], {"b"}, 1) == 0.0


class TestMRR:
    def test_first_is_relevant(self):
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_is_relevant(self):
        assert mrr(["a", "b", "c"], {"b"}) == 0.5

    def test_multiple_relevant_first_wins(self):
        assert mrr(["a", "b"], {"a", "b"}) == 1.0

    def test_no_relevant(self):
        assert mrr(["a", "b"], {"c"}) == 0.0

    def test_empty_ranked(self):
        assert mrr([], {"a"}) == 0.0


class TestMAPAtK:
    def test_all_relevant(self):
        assert map_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0

    def test_single_relevant(self):
        assert map_at_k(["a", "b", "c"], {"a"}, 3) == 1.0

    def test_late_relevant(self):
        score = map_at_k(["x", "y", "a"], {"a"}, 3)
        assert score == pytest.approx(1 / 3)

    def test_no_relevant(self):
        assert map_at_k(["a", "b"], {"c"}, 2) == 0.0


class TestNDCGAtK:
    def test_perfect(self):
        assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)

    def test_no_relevant(self):
        assert ndcg_at_k(["a", "b"], {"c"}, 2) == 0.0

    def test_partial(self):
        score = ndcg_at_k(["a", "b", "c"], {"a"}, 3)
        assert score > 0
        assert score <= 1.0

    def test_all_relevant_at_k(self):
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == pytest.approx(1.0)
