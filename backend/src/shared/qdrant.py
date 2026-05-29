from datetime import datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Range,
    DatetimeRange,
    PayloadSchemaType,
)

from src.config import app_config, secrets


def _build_range(params: dict) -> Range | DatetimeRange:
    """Route to DatetimeRange when values are ISO-8601 strings, else numeric Range."""
    sample = next((v for v in params.values() if v is not None), None)
    if isinstance(sample, str):
        return DatetimeRange(**{k: datetime.fromisoformat(v) for k, v in params.items()})
    return Range(**params)


_client: AsyncQdrantClient | None = None

COLLECTION_NAME = "news_vectors"

_REQUIRED_INDEXES: dict[str, PayloadSchemaType] = {
    "published_at": PayloadSchemaType.DATETIME,
    "ticker": PayloadSchemaType.KEYWORD,
    "impact": PayloadSchemaType.INTEGER,
    "realized_price_delta_pct_1h": PayloadSchemaType.FLOAT,
    "realized_price_delta_pct_24h": PayloadSchemaType.FLOAT,
    "realized_price_delta_pct_7d": PayloadSchemaType.FLOAT,
    "realized_price_delta_pct_30d": PayloadSchemaType.FLOAT,
}


async def _ensure_payload_indexes(qc: AsyncQdrantClient, info) -> None:
    """Create any missing payload indexes on an existing collection."""
    existing_indexes = info.payload_schema or {}
    for field, schema_type in _REQUIRED_INDEXES.items():
        if field not in existing_indexes:
            await qc.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name=field,
                field_schema=schema_type,
            )


async def client() -> AsyncQdrantClient:
    global _client
    if _client is None:
        sec = secrets()
        _client = AsyncQdrantClient(url=sec.qdrant_url, api_key=sec.qdrant_api_key)
    return _client


async def ensure_collection() -> None:
    """Create collection if it doesn't exist, or recreate if dimensions changed."""
    qc = await client()
    cfg = app_config().embeddings
    collections = await qc.get_collections()
    existing = [c.name for c in collections.collections]

    if COLLECTION_NAME in existing:
        info = await qc.get_collection(COLLECTION_NAME)
        current_size = info.config.params.vectors.size
        if current_size != cfg.dimensions:
            await qc.delete_collection(COLLECTION_NAME)
        else:
            await _ensure_payload_indexes(qc, info)
            return

    await qc.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=cfg.dimensions,
            distance=Distance.COSINE,
        ),
    )
    for field, schema_type in _REQUIRED_INDEXES.items():
        await qc.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=schema_type,
        )


async def upsert_news(points: list[PointStruct]) -> None:
    """Batch upsert news vectors with payloads."""
    qc = await client()
    await qc.upsert(collection_name=COLLECTION_NAME, points=points)


async def search_similar(
    vector: list[float],
    *,
    limit: int = 5,
    score_threshold: float | None = None,
    filter_conditions: dict | None = None,
) -> list:
    """Search for similar vectors with optional filtering."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    qc = await client()

    search_filter = None
    if filter_conditions:
        conditions = []
        for key, value in filter_conditions.items():
            if isinstance(value, dict) and "range" in value:
                conditions.append(FieldCondition(key=key, range=_build_range(value["range"])))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        search_filter = Filter(must=conditions)

    response = await qc.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
        score_threshold=score_threshold,
        query_filter=search_filter,
    )
    return response.points


async def batch_update_payload(point_ids: list[str], payloads: list[dict]) -> None:
    """Batch update payloads for existing points."""
    qc = await client()
    for point_id, payload in zip(point_ids, payloads):
        await qc.set_payload(
            collection_name=COLLECTION_NAME,
            payload=payload,
            points=[point_id],
        )


async def scroll_with_filter(filter_conditions: dict, limit: int = 100) -> list:
    """Scroll through points matching filter conditions."""
    from qdrant_client.models import Filter, FieldCondition, MatchValue, IsNullCondition, PayloadField

    qc = await client()

    conditions = []
    for key, value in filter_conditions.items():
        if isinstance(value, dict):
            if "range" in value:
                conditions.append(FieldCondition(key=key, range=_build_range(value["range"])))
            elif "is_null" in value:
                conditions.append(IsNullCondition(is_null=PayloadField(key=key)))
        else:
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))

    search_filter = Filter(must=conditions)

    results, _ = await qc.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=search_filter,
        limit=limit,
        with_payload=True,
    )
    return results
