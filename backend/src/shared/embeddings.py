from src.config import app_config
from src.shared.azure_ai import azure_ai_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts via Azure AI (OpenAI-compatible endpoint)."""
    cfg = app_config().embeddings
    client = azure_ai_client(cfg.api_version)
    response = await client.embeddings.create(
        model=cfg.deployment,
        input=texts,
        dimensions=cfg.dimensions,
        name="embed-news",
        metadata={"input_count": len(texts)},
        timeout=60.0,
    )

    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
