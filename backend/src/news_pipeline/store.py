import uuid

from qdrant_client.models import PointStruct

from src.shared.qdrant import upsert_news
from src.shared.types import NewsRecord


async def store_news_batch(records: list[NewsRecord], embeddings: list[list[float]]) -> None:
    """Batch upsert news records into Qdrant with their embeddings."""
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload=record.model_dump(mode="json"),
        )
        for record, embedding in zip(records, embeddings)
    ]
    await upsert_news(points)
