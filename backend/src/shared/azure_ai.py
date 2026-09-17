from typing import TypeAlias

from langfuse.openai import AsyncOpenAI as LangfuseAsyncOpenAI
from openai import AsyncOpenAI

from src.config import secrets
from src.shared.observability import langfuse_client, langfuse_tracing_enabled

AzureAIClient: TypeAlias = AsyncOpenAI | LangfuseAsyncOpenAI
_clients: dict[bool, AzureAIClient] = {}


def azure_ai_base_url(endpoint: str) -> str:
    base_url = endpoint.rstrip("/")
    if not base_url.endswith("/openai/v1"):
        raise ValueError("AZURE_AI_ENDPOINT must include the /openai/v1/ path")
    return f"{base_url}/"


def azure_ai_client(*, observe: bool = False) -> AzureAIClient:
    """Return an Azure v1 client, optionally wrapped for Langfuse observations."""
    tracing = observe and langfuse_tracing_enabled()
    cache_key = tracing
    if cache_key in _clients:
        return _clients[cache_key]

    cfg = secrets()
    base_url = azure_ai_base_url(cfg.azure_ai_endpoint)
    if tracing:
        langfuse_client()
        _clients[cache_key] = LangfuseAsyncOpenAI(
            api_key=cfg.azure_ai_api_key,
            base_url=base_url,
            max_retries=0,
        )
    else:
        _clients[cache_key] = AsyncOpenAI(
            api_key=cfg.azure_ai_api_key,
            base_url=base_url,
            max_retries=0,
        )

    return _clients[cache_key]


async def close_azure_ai_clients() -> None:
    clients = list(_clients.values())
    _clients.clear()

    for client in clients:
        await client.close()
