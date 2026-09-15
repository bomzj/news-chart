from collections.abc import Sequence

import numpy as np

from src.config import app_config
from src.shared.azure_ai import azure_ai_client


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two non-zero vectors."""
    left_vector = np.asarray(left, dtype=float)
    right_vector = np.asarray(right, dtype=float)

    if left_vector.ndim != 1 or right_vector.ndim != 1:
        raise ValueError("Cosine similarity requires one-dimensional vectors")
    if left_vector.shape != right_vector.shape:
        raise ValueError("Cosine similarity requires vectors with equal dimensions")
    if not np.all(np.isfinite(left_vector)) or not np.all(np.isfinite(right_vector)):
        raise ValueError("Cosine similarity requires finite vector values")

    left_norm = float(np.linalg.norm(left_vector))
    right_norm = float(np.linalg.norm(right_vector))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cannot calculate cosine similarity for a zero vector")

    return float(np.dot(left_vector, right_vector) / (left_norm * right_norm))


async def embed_texts(
    texts: list[str],
    *,
    observe: bool = False,
    dimensions: int | None = None,
) -> list[list[float]]:
    """Batch embed texts, optionally retaining a Langfuse embedding observation."""
    cfg = app_config().embeddings
    dimensions_used = cfg.dimensions if dimensions is None else dimensions
    if dimensions_used <= 0:
        raise ValueError("Embedding dimensions must be positive")

    client = azure_ai_client(cfg.api_version, observe=observe)
    if observe:
        response = await client.embeddings.create(
            model=cfg.deployment,
            input=texts,
            dimensions=dimensions_used,
            name="embed-news",
            metadata={"input_count": len(texts)},
            timeout=60.0,
        )
    else:
        response = await client.embeddings.create(
            model=cfg.deployment,
            input=texts,
            dimensions=dimensions_used,
            timeout=60.0,
        )

    return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]
