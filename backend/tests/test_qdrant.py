from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.shared.qdrant import search_top1_batch


class _Observation:
    def __init__(self, owner):
        self.owner = owner

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def update(self, **kwargs):
        self.owner.updated = kwargs


class _LangfuseStub:
    def __init__(self):
        self.inputs = []
        self.updated = None

    def start_as_current_observation(self, **kwargs):
        self.inputs.append(kwargs)
        return _Observation(self)


@pytest.mark.asyncio
async def test_search_top1_batch_uses_unfiltered_queries(monkeypatch):
    qdrant_client = SimpleNamespace(
        query_batch_points=AsyncMock(
            return_value=[
                SimpleNamespace(points=[]),
                SimpleNamespace(points=[]),
            ]
        )
    )
    langfuse = _LangfuseStub()

    async def fake_client():
        return qdrant_client

    monkeypatch.setattr("src.shared.qdrant.client", fake_client)
    monkeypatch.setattr("src.shared.qdrant.langfuse_client", lambda: langfuse)

    vectors = [[1.0, 0.0], [0.0, 1.0]]
    responses = await search_top1_batch(vectors)

    assert len(responses) == len(vectors)
    qdrant_client.query_batch_points.assert_awaited_once()
    requests = qdrant_client.query_batch_points.await_args.kwargs["requests"]
    assert len(requests) == 2
    assert all(request.limit == 1 for request in requests)
    assert all(request.score_threshold is None for request in requests)
    assert all(request.filter is None for request in requests)
    assert all(request.with_payload is True for request in requests)
    assert langfuse.inputs[0]["input"]["score_threshold"] is None
    assert langfuse.inputs[0]["input"]["filter_conditions"] is None
