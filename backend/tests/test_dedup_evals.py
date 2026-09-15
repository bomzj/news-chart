from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from evals.run_dedup_evals import _parser, detect_duplicate, run_dedup_evals


@pytest.mark.asyncio
async def test_detect_duplicate_uses_embedding_dimension_override(monkeypatch):
    embed_texts = AsyncMock(
        return_value=[
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )
    monkeypatch.setattr("evals.run_dedup_evals.embed_texts", embed_texts)

    item = SimpleNamespace(
        input={"news_1": "First article", "news_2": "Second article"}
    )
    result = await detect_duplicate(item=item, threshold=0.9, dimensions=512)

    assert result == {"duplicate": True}
    embed_texts.assert_awaited_once_with(
        ["First article", "Second article"],
        observe=True,
        dimensions=512,
    )


def test_parser_accepts_embedding_dimensions():
    args = _parser().parse_args(["--threshold", "0.95", "--dimensions", "1024"])

    assert args.threshold == 0.95
    assert args.dimensions == 1024


def test_run_dedup_evals_records_embedding_dimensions(monkeypatch):
    dataset = Mock(items=[object()])
    dataset.run_experiment.return_value = object()
    client = Mock()
    client.get_dataset.return_value = dataset
    config = SimpleNamespace(
        dedup=SimpleNamespace(cosine_threshold=0.9),
        embeddings=SimpleNamespace(
            deployment="text-embedding-3-large",
            dimensions=256,
        ),
    )
    monkeypatch.setattr("evals.run_dedup_evals.app_config", lambda: config)
    monkeypatch.setattr("evals.run_dedup_evals.langfuse_client", lambda: client)

    result = run_dedup_evals(
        dataset="test-dataset",
        threshold=0.95,
        dimensions=1024,
    )

    assert result is dataset.run_experiment.return_value
    experiment = dataset.run_experiment.call_args.kwargs
    assert experiment["name"] == "dedup-threshold-0.95-dimensions-1024"
    assert experiment["metadata"]["embedding_dimensions"] == 1024
    assert experiment["task"].keywords == {"threshold": 0.95, "dimensions": 1024}
