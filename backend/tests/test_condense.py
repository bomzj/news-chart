from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.config import app_config

import pytest

from src.news_collector.condense import _condense_single


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
