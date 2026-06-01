from src.config import app_config, secrets
from src.shared.http import http_client


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed texts via Azure AI (OpenAI-compatible endpoint)."""
    cfg = app_config().embeddings
    sec = secrets()

    url = (
        f"{sec.azure_ai_endpoint}/openai/deployments/{cfg.deployment}"
        f"/embeddings?api-version={cfg.api_version}"
    )

    client = await http_client()
    response = await client.post(
        url,
        headers={"api-key": sec.azure_ai_api_key},
        json={"input": texts, "dimensions": cfg.dimensions},
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
