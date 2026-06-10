import pytest
from src.utils.similarity import cosine_similarity


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_same_direction_different_magnitude(self):
        assert cosine_similarity([1, 2], [2, 4]) == pytest.approx(1.0)

    def test_partial_similarity(self):
        sim = cosine_similarity([1, 0], [0.707, 0.707])
        assert sim == pytest.approx(0.707, abs=0.01)

    def test_zero_vector_does_not_crash(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == pytest.approx(0.0, abs=1e-6)
        assert cosine_similarity([1, 2], [0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_both_zero_vectors(self):
        assert cosine_similarity([0, 0], [0, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_large_vectors(self):
        import numpy as np
        a = [1.0] * 1000
        b = [2.0] * 1000
        assert cosine_similarity(a, b) == pytest.approx(1.0)

    def test_negative_values(self):
        sim = cosine_similarity([-1, 0], [1, 0])
        assert sim == pytest.approx(-1.0)
