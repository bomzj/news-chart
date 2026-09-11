import logging
from datetime import datetime, timezone, timedelta
from typing import Literal

from src.config import app_config
from src.shared.embeddings import cosine_similarity, embed_texts
from src.shared.observability import langfuse_client
from src.shared.qdrant import search_similar, search_top1_batch
from src.shared.types import RawNews

logger = logging.getLogger(__name__)

_EMBEDDING_INPUT_VERSION = "title_description_v1"
DedupStage = Literal["qdrant", "intra_batch"]


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
    if len(embeddings) != len(news_items):
        raise ValueError("Embedding count does not match fetched news count")

    embedding_cfg = app_config().embeddings

    # Audit every fetched article independently of production dedup filters.
    qdrant_matches = await search_top1_batch(embeddings)
    for news, response in zip(news_items, qdrant_matches):
        if not response.points:
            continue

        point = response.points[0]
        pair_fields = _qdrant_pair_fields(point)
        if pair_fields is None:
            continue

        right_published_at, right_description = pair_fields
        _observe_duplicate_check(
            left=news,
            right_published_at=right_published_at,
            right_description=right_description,
            right_key=f"qdrant:{point.id}",
            score=float(point.score),
            threshold=threshold,
            stage="qdrant",
            embedding_model=embedding_cfg.deployment,
            embedding_dimensions=embedding_cfg.dimensions,
        )

    # Capture the best intra-batch partner before the production survivor filter.
    for left_index, right_index, score in _top1_intra_batch(embeddings):
        _observe_duplicate_check(
            left=news_items[left_index],
            right_published_at=news_items[right_index].published_at.isoformat(),
            right_description=news_items[right_index].description,
            right_key=f"intra_batch:{_article_key(news_items[right_index])}",
            score=score,
            threshold=threshold,
            stage="intra_batch",
            embedding_model=embedding_cfg.deployment,
            embedding_dimensions=embedding_cfg.dimensions,
        )

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


def _top1_intra_batch(
    embeddings: list[list[float]],
) -> list[tuple[int, int, float]]:
    """Return each item's best non-self intra-batch match and its cosine score."""
    if len(embeddings) <= 1:
        return []

    matches: list[tuple[int, int, float]] = []
    for left_index, left_embedding in enumerate(embeddings):
        best_match = max(
            (
                (right_index, cosine_similarity(left_embedding, right_embedding))
                for right_index, right_embedding in enumerate(embeddings)
                if right_index != left_index
            ),
            key=lambda match: match[1],
        )
        matches.append((left_index, best_match[0], best_match[1]))
    return matches


def _intra_batch_dedup(
    news_items: list[RawNews],
    embeddings: list[list[float]],
    threshold: float,
) -> list[tuple[RawNews, list[float]]]:
    """Remove duplicates within a batch via pairwise cosine similarity."""
    if len(news_items) <= 1:
        return [(news_items[0], embeddings[0])] if news_items else []

    keep = set(range(len(news_items)))

    for i in range(len(news_items)):
        if i not in keep:
            continue
        for j in range(i + 1, len(news_items)):
            if j not in keep:
                continue
            similarity = cosine_similarity(embeddings[i], embeddings[j])
            if similarity >= threshold:
                keep.discard(j)

    return [(news_items[i], embeddings[i]) for i in sorted(keep)]


def _observe_duplicate_check(
    *,
    left: RawNews,
    right_published_at: str,
    right_description: str,
    right_key: str,
    score: float,
    threshold: float,
    stage: DedupStage,
    embedding_model: str,
    embedding_dimensions: int,
) -> None:
    left_key = _article_key(left)
    metadata = {
        "source": "production",
        "stage": stage,
        "embedding_model": embedding_model,
        "embedding_dimensions": embedding_dimensions,
        "embedding_input_version": _EMBEDDING_INPUT_VERSION,
        "left_article_key": left_key,
        "right_article_key": right_key,
        "pair_key": _pair_key(left_key, right_key),
    }

    with langfuse_client().start_as_current_observation(
        as_type="span",
        name="duplicate-check",
        input={
            "left": {
                "published_at": left.published_at.isoformat(),
                "description": left.description,
            },
            "right": {
                "published_at": right_published_at,
                "description": right_description,
            },
        },
        metadata=metadata,
    ) as observation:
        observation.update(
            output={
                "duplicate": score >= threshold,
                "cosine_similarity": score,
                "threshold": threshold,
            }
        )


def _qdrant_pair_fields(point) -> tuple[str, str] | None:
    payload = point.payload or {}
    published_at = payload.get("published_at")
    description = payload.get("news_full_text")
    if isinstance(published_at, str) and isinstance(description, str):
        return published_at, description

    logger.warning(
        "Skipping similarity audit for Qdrant point %s: missing published_at or news_full_text",
        point.id,
    )
    return None


def _article_key(news: RawNews) -> str:
    return news.url or f"{news.source}|{news.published_at.isoformat()}|{news.title}"


def _pair_key(left_key: str, right_key: str) -> str:
    return "::".join(sorted((left_key, right_key)))
