from datetime import datetime, timezone, timedelta

import numpy as np

from src.config import app_config
from src.shared.embeddings import embed_texts
from src.shared.qdrant import search_similar
from src.shared.types import RawNews


async def deduplicate(news_items: list[RawNews]) -> list[tuple[RawNews, list[dict]]]:
    """
    Two-stage dedup:
    1. Intra-batch: pairwise cosine within current batch, cluster duplicates (keep first)
    2. RAG dedup: check each survivor against Qdrant (last 24h), discard if ≥ threshold

    Returns non-duplicate news paired with similar past context.
    """
    if not news_items:
        return []

    cfg = app_config().dedup
    threshold = cfg.cosine_threshold

    texts = [f"{n.title} {n.description}" for n in news_items]
    embeddings = await embed_texts(texts)

    # Stage 1: intra-batch dedup
    survivors = _intra_batch_dedup(news_items, embeddings, threshold)

    # Stage 2: RAG dedup against vector DB
    results: list[tuple[RawNews, list[dict]]] = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=cfg.lookback_hours)

    for news, embedding in survivors:
        filter_conditions = {
            "published_at": {"range": {"gte": cutoff.isoformat()}}
        }

        similar = await search_similar(
            vector=embedding,
            limit=5,
            score_threshold=threshold,
            filter_conditions=filter_conditions,
        )

        if similar and similar[0].score >= threshold:
            continue  # duplicate found in DB, discard

        # Collect similar context (lower threshold hits) for agent enrichment
        context_hits = await search_similar(
            vector=embedding,
            limit=3,
            score_threshold=0.5,
            filter_conditions=filter_conditions,
        )
        context = [hit.payload for hit in context_hits if hit.payload]
        results.append((news, context))

    return results


def _intra_batch_dedup(
    news_items: list[RawNews],
    embeddings: list[list[float]],
    threshold: float,
) -> list[tuple[RawNews, list[float]]]:
    """Remove duplicates within a batch via pairwise cosine similarity."""
    if len(news_items) <= 1:
        return [(news_items[0], embeddings[0])] if news_items else []

    vectors = np.array(embeddings)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / norms

    keep = set(range(len(news_items)))

    for i in range(len(news_items)):
        if i not in keep:
            continue
        for j in range(i + 1, len(news_items)):
            if j not in keep:
                continue
            similarity = float(np.dot(normalized[i], normalized[j]))
            if similarity >= threshold:
                keep.discard(j)

    return [(news_items[i], embeddings[i]) for i in sorted(keep)]
