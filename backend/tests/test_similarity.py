import pytest

from src.shared.embeddings import cosine_similarity


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
    ],
)
def test_cosine_similarity(left, right, expected):
    assert cosine_similarity(left, right) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("left", "right", "message"),
    [
        ([[1.0, 0.0]], [1.0, 0.0], "one-dimensional"),
        ([1.0, 0.0], [1.0], "equal dimensions"),
        ([0.0, 0.0], [1.0, 0.0], "zero vector"),
        ([float("nan"), 0.0], [1.0, 0.0], "finite"),
    ],
)
def test_cosine_similarity_rejects_invalid_vectors(left, right, message):
    with pytest.raises(ValueError, match=message):
        cosine_similarity(left, right)
