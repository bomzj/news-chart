import httpx
import pytest

import src.shared.http as http


@pytest.mark.asyncio
async def test_close_http_client_closes_shared_client(monkeypatch):
    client = httpx.AsyncClient()
    monkeypatch.setattr(http, "_client", client)

    await http.close_http_client()

    assert client.is_closed
    assert http._client is None
