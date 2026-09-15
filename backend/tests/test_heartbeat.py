import httpx
import pytest

from src.main import app


@pytest.mark.asyncio
async def test_heartbeat_returns_ok_without_external_dependencies():
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        response = await client.get("/api/heartbeat")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
