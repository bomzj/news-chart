from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

import src.shared.azure_ai as azure_ai
from src.shared.embeddings import embed_texts


def test_azure_ai_client_defaults_to_standard_and_opts_into_langfuse(monkeypatch):
    azure_ai._clients.clear()
    standard_client = object()
    observed_client = object()
    standard_factory = Mock(return_value=standard_client)
    observed_factory = Mock(return_value=observed_client)
    langfuse_factory = Mock()

    monkeypatch.setattr(azure_ai, "AsyncAzureOpenAI", standard_factory)
    monkeypatch.setattr(azure_ai, "LangfuseAsyncAzureOpenAI", observed_factory)
    monkeypatch.setattr(azure_ai, "langfuse_client", langfuse_factory)
    monkeypatch.setattr(
        azure_ai,
        "secrets",
        lambda: SimpleNamespace(
            azure_ai_endpoint="https://example.test",
            azure_ai_api_key="test-key",
        ),
    )

    try:
        standard = azure_ai.azure_ai_client("2025-04-01-preview")
        observed = azure_ai.azure_ai_client("2025-04-01-preview", observe=True)

        assert standard is standard_client
        assert observed is observed_client
        assert azure_ai.azure_ai_client("2025-04-01-preview") is standard_client
        assert azure_ai.azure_ai_client("2025-04-01-preview", observe=True) is observed_client
        standard_factory.assert_called_once()
        observed_factory.assert_called_once()
        langfuse_factory.assert_called_once_with()
    finally:
        azure_ai._clients.clear()


@pytest.mark.asyncio
async def test_embed_texts_only_adds_langfuse_options_when_observing(monkeypatch):
    cfg = SimpleNamespace(
        embeddings=SimpleNamespace(
            deployment="text-embedding-3-large",
            dimensions=256,
            api_version="2023-05-15",
        )
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            data=[
                SimpleNamespace(index=0, embedding=[1.0, 0.0]),
                SimpleNamespace(index=1, embedding=[0.0, 1.0]),
            ]
        )
    )
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    client_factory = Mock(return_value=client)

    monkeypatch.setattr("src.shared.embeddings.app_config", lambda: cfg)
    monkeypatch.setattr("src.shared.embeddings.azure_ai_client", client_factory)

    await embed_texts(["first", "second"])
    plain_kwargs = create.await_args.kwargs
    assert "name" not in plain_kwargs
    assert "metadata" not in plain_kwargs
    assert client_factory.call_args.kwargs == {"observe": False}

    create.reset_mock()
    await embed_texts(["first", "second"], observe=True)
    observed_kwargs = create.await_args.kwargs
    assert observed_kwargs["name"] == "embed-news"
    assert observed_kwargs["metadata"] == {"input_count": 2}
    assert client_factory.call_args.kwargs == {"observe": True}
