import numpy as np
import pytest
from datetime import datetime, timezone

from src.shared.types import RawNews
from src.news_pipeline.dedup import _intra_batch_dedup


def _make_news(title: str = "Test News", desc: str = "Description") -> RawNews:
    return RawNews(
        title=title,
        description=desc,
        full_text="Full text content",
        source="test-source",
        published_at=datetime.now(timezone.utc),
        url="https://example.com",
    )


class TestIntraBatchDedup:
    def test_no_duplicates_all_kept(self, sample_embeddings):
        """All unique items survive dedup."""
        news = [_make_news(f"News {i}") for i in range(5)]
        result = _intra_batch_dedup(news, sample_embeddings, threshold=0.90)
        assert len(result) == 5

    def test_near_duplicates_removed(self, duplicate_embeddings):
        """Near-duplicate (cosine ≥ 0.90) is removed, keeping the first."""
        news = [_make_news(f"News {i}") for i in range(3)]
        result = _intra_batch_dedup(news, duplicate_embeddings, threshold=0.90)

        # base and near_dup are similar, so only 2 should survive (first + different)
        assert len(result) == 2
        assert result[0][0].title == "News 0"
        assert result[1][0].title == "News 2"

    def test_empty_input(self):
        """Empty input returns empty."""
        result = _intra_batch_dedup([], [], threshold=0.90)
        assert result == []

    def test_single_item(self, sample_embeddings):
        """Single item always survives."""
        news = [_make_news("Only")]
        result = _intra_batch_dedup(news, [sample_embeddings[0]], threshold=0.90)
        assert len(result) == 1

    def test_threshold_boundary(self):
        """Items exactly at threshold should be deduped."""
        # Create two vectors with known cosine similarity
        v1 = np.zeros(1536)
        v1[0] = 1.0
        v2 = np.zeros(1536)
        v2[0] = 0.9
        v2[1] = 0.1
        v2 = v2 / np.linalg.norm(v2)

        cosine = float(np.dot(v1, v2))
        news = [_make_news("A"), _make_news("B")]

        # Use a threshold below the actual cosine → should dedup
        result = _intra_batch_dedup(news, [v1.tolist(), v2.tolist()], threshold=cosine - 0.01)
        assert len(result) == 1

    def test_high_threshold_keeps_all(self, duplicate_embeddings):
        """Very high threshold (0.99) keeps near-duplicates."""
        news = [_make_news(f"News {i}") for i in range(3)]
        result = _intra_batch_dedup(news, duplicate_embeddings, threshold=0.99)
        assert len(result) >= 2
