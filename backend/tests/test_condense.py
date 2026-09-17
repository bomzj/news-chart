from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.config import app_config

import pytest

from src.news_collector.condense import _condense_single, condense_texts


@pytest.mark.asyncio
async def test_condense_uses_condense_reasoning_effort():
    create = AsyncMock(return_value=SimpleNamespace(output_text="short"))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    with patch(
        "src.news_collector.condense.azure_ai_client",
        return_value=client,
    ) as client_factory:
        result = await _condense_single("long article text", 5)

    assert result == "short"
    client_factory.assert_called_once_with()
    assert create.await_args.kwargs["model"] == app_config().llm.name
    assert create.await_args.kwargs["reasoning"] == {
        "effort": app_config().llm.reasoning_effort.condense
    }


@pytest.mark.asyncio
async def test_condense_texts_leaves_text_at_limit_unchanged():
    text = "x" * 5000
    config = SimpleNamespace(
        collector=SimpleNamespace(max_full_text_chars=5000),
    )

    with (
        patch("src.news_collector.condense.app_config", return_value=config),
        patch(
            "src.news_collector.condense._condense_single",
            new_callable=AsyncMock,
        ) as condense,
    ):
        result = await condense_texts([text])

    assert result == [text]
    condense.assert_not_awaited()


@pytest.mark.asyncio
async def test_condense_texts_condenses_text_over_limit():
    text = "x" * 5001
    config = SimpleNamespace(
        collector=SimpleNamespace(max_full_text_chars=5000),
    )

    with (
        patch("src.news_collector.condense.app_config", return_value=config),
        patch(
            "src.news_collector.condense._condense_single",
            new_callable=AsyncMock,
            return_value="short",
        ) as condense,
    ):
        result = await condense_texts([text])

    assert result == ["short"]
    condense.assert_awaited_once_with(text, 5000)


@pytest.mark.asyncio
async def test_condense_caps_llm_output_to_limit():
    create = AsyncMock(return_value=SimpleNamespace(output_text="x" * 10))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    with patch(
        "src.news_collector.condense.azure_ai_client",
        return_value=client,
    ):
        result = await _condense_single("long article text", 5)

    assert result == "xxxxx"


@pytest.mark.asyncio
async def test_condense_truncates_when_llm_returns_no_output(caplog):
    create = AsyncMock(return_value=SimpleNamespace(output_text=""))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    with patch(
        "src.news_collector.condense.azure_ai_client",
        return_value=client,
    ):
        with caplog.at_level("WARNING"):
            result = await _condense_single("long article text", 5)

    assert result == "long "
    assert "No text output from condense LLM" in caplog.text


@pytest.mark.asyncio
async def test_condense_truncates_when_llm_fails(caplog):
    create = AsyncMock(side_effect=RuntimeError("service unavailable"))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    with patch(
        "src.news_collector.condense.azure_ai_client",
        return_value=client,
    ):
        with caplog.at_level("WARNING"):
            result = await _condense_single("long article text", 5)

    assert result == "long "
    assert "Condense LLM failed" in caplog.text
