from typing import TypeAlias

from langfuse.openai import AsyncAzureOpenAI as LangfuseAsyncAzureOpenAI
from openai import AsyncAzureOpenAI

from src.config import secrets
from src.shared.observability import langfuse_client, langfuse_tracing_enabled

AzureAIClient: TypeAlias = AsyncAzureOpenAI | LangfuseAsyncAzureOpenAI
_clients: dict[tuple[str, bool], AzureAIClient] = {}


def azure_ai_client(api_version: str, *, observe: bool = False) -> AzureAIClient:
    """Return an Azure client, opting into Langfuse only for retained observations."""
    tracing = observe and langfuse_tracing_enabled()
    cache_key = (api_version, tracing)
    if cache_key in _clients:
        return _clients[cache_key]

    cfg = secrets()
    if tracing:
        langfuse_client()
        _clients[cache_key] = LangfuseAsyncAzureOpenAI(
            azure_endpoint=cfg.azure_ai_endpoint,
            api_key=cfg.azure_ai_api_key,
            api_version=api_version,
            max_retries=0,
        )
    else:
        _clients[cache_key] = AsyncAzureOpenAI(
            azure_endpoint=cfg.azure_ai_endpoint,
            api_key=cfg.azure_ai_api_key,
            api_version=api_version,
            max_retries=0,
        )

    return _clients[cache_key]


async def close_azure_ai_clients() -> None:
    clients = list(_clients.values())
    _clients.clear()

    for client in clients:
        await client.close()
