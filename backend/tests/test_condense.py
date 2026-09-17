from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.news_collector.condense import _condense_single


@pytest.mark.asyncio
async def test_condense_uses_lite_reasoning_effort():
    create = AsyncMock(return_value=SimpleNamespace(output_text="short"))
    client = SimpleNamespace(responses=SimpleNamespace(create=create))

    with patch(
        "src.news_collector.condense.azure_ai_client",
        return_value=client,
    ) as client_factory:
        result = await _condense_single("long article text", 5)

    assert result == "short"
    client_factory.assert_called_once_with("2025-04-01-preview")
    assert create.await_args.kwargs["model"] == "gpt-5.4-nano"
    assert create.await_args.kwargs["reasoning"] == {"effort": "high"}
