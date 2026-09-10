from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
from datetime import datetime, timezone

from src.shared.types import RawNews
from src.news_pipeline.dedup import _intra_batch_dedup, _top1_intra_batch, deduplicate


def _make_news(title: str = "Test News", desc: str = "Description") -> RawNews:
    return RawNews(
        title=title,
        description=desc,
        full_text="Full text content",
        source="test-source",
        published_at=datetime.now(timezone.utc),
        url=f"https://example.com/{title.replace(' ', '-')}",
    )


class TestIntraBatchDedup:
    def test_top1_match_excludes_self(self):
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ]

        result = _top1_intra_batch(embeddings)

        assert result[0][0:2] == (0, 1)
        assert result[1][0:2] == (1, 0)
        assert all(left_index != right_index for left_index, right_index, _ in result)

    def test_top1_match_empty_and_singleton(self):
        assert _top1_intra_batch([]) == []
        assert _top1_intra_batch([[1.0, 0.0]]) == []

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


class _Observation:
    def __init__(self, owner, kwargs):
        self.owner = owner
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, **kwargs):
        self.owner.updates.append((self.kwargs, kwargs))


class _LangfuseStub:
    def __init__(self):
        self.observations = []
        self.updates = []

    def start_as_current_observation(self, **kwargs):
        self.observations.append(kwargs)
        return _Observation(self, kwargs)


@pytest.mark.asyncio
async def test_deduplicate_audits_all_items_and_preserves_filtered_path(monkeypatch):
    news = [_make_news("A"), _make_news("B"), _make_news("C")]
    embeddings = [
        [1.0, 0.0],
        [0.95, 0.05],
        [0.0, 1.0],
    ]
    search_top1 = AsyncMock(
        return_value=[SimpleNamespace(points=[]), SimpleNamespace(points=[]), SimpleNamespace(points=[])]
    )
    search_similar = AsyncMock(return_value=[])
    langfuse = _LangfuseStub()

    monkeypatch.setattr("src.news_pipeline.dedup.embed_texts", AsyncMock(return_value=embeddings))
    monkeypatch.setattr("src.news_pipeline.dedup.search_top1_batch", search_top1)
    monkeypatch.setattr("src.news_pipeline.dedup.search_similar", search_similar)
    monkeypatch.setattr("src.news_pipeline.dedup.langfuse_client", lambda: langfuse)

    result = await deduplicate(news)

    assert [item[0].title for item in result] == ["A", "C"]
    search_top1.assert_awaited_once_with(embeddings)
    assert search_similar.await_count == 4
    assert [call.kwargs["score_threshold"] for call in search_similar.await_args_list] == [
        0.9,
        0.5,
        0.9,
        0.5,
    ]
    assert all(call.kwargs["filter_conditions"]["published_at"]["range"]["gte"] for call in search_similar.await_args_list)
    assert len(langfuse.observations) == 3
    assert all(observation["name"] == "news-similarity-pair" for observation in langfuse.observations)
    assert all(set(observation["input"]) == {"left", "right"} for observation in langfuse.observations)
    assert all(
        set(kwargs["output"]) == {"similar", "cosine_similarity", "threshold_used"}
        for _, kwargs in langfuse.updates
    )
    assert all(kwargs["output"]["threshold_used"] == 0.9 for _, kwargs in langfuse.updates)
    assert all(
        "similarity_score" not in observation["metadata"]
        and "similarity_threshold" not in observation["metadata"]
        for observation in langfuse.observations
    )
