from langfuse.openai import AsyncAzureOpenAI

from src.config import secrets
from src.shared.observability import langfuse_client

_clients: dict[str, AsyncAzureOpenAI] = {}


def azure_ai_client(api_version: str) -> AsyncAzureOpenAI:
    langfuse_client()

    if api_version not in _clients:
        cfg = secrets()
        _clients[api_version] = AsyncAzureOpenAI(
            azure_endpoint=cfg.azure_ai_endpoint,
            api_key=cfg.azure_ai_api_key,
            api_version=api_version,
            max_retries=0,
        )

    return _clients[api_version]


async def close_azure_ai_clients() -> None:
    clients = list(_clients.values())
    _clients.clear()

    for client in clients:
        await client.close()
