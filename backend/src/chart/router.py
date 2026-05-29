import logging
from datetime import datetime

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    DatetimeRange,
    Range,
)

from src.shared.qdrant import client, ensure_collection, COLLECTION_NAME

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/news")
async def news(
    ticker: str = Query(default="BTC"),
    from_ts: str = Query(default=None, description="ISO 8601 start timestamp"),
    to_ts: str = Query(default=None, description="ISO 8601 end timestamp"),
    min_impact: int = Query(default=1, ge=1, le=3),
):
    """
    Fetch analyzed news for the chart frontend.
    Returns news items with sentiment, impact, summary, and timing.
    """
    qc = await client()

    conditions = [
        FieldCondition(key="ticker", match=MatchValue(value=ticker)),
    ]

    if min_impact > 1:
        conditions.append(
            FieldCondition(key="impact", range=Range(gte=min_impact))
        )

    if from_ts or to_ts:
        dt_range_params = {}
        if from_ts:
            dt_range_params["gte"] = datetime.fromisoformat(from_ts)
        if to_ts:
            dt_range_params["lte"] = datetime.fromisoformat(to_ts)
        conditions.append(
            FieldCondition(key="published_at", range=DatetimeRange(**dt_range_params))
        )

    try:
        results, _ = await qc.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=conditions),
            limit=500,
            with_payload=True,
        )
    except Exception as exc:
        logger.error("Qdrant scroll failed: %s", exc)
        # Attempt to fix missing indexes and retry once
        try:
            await ensure_collection()
            results, _ = await qc.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(must=conditions),
                limit=500,
                with_payload=True,
            )
        except Exception as retry_exc:
            logger.error("Qdrant retry failed: %s", retry_exc)
            return JSONResponse(
                status_code=502,
                content={"error": "News storage unavailable", "items": [], "count": 0},
            )

    items = [
        {
            "published_at": point.payload.get("published_at"),
            "sentiment": point.payload.get("sentiment"),
            "impact": point.payload.get("impact"),
            "news_summary": point.payload.get("news_summary"),
            "confidence": point.payload.get("confidence"),
            "predicted_by_model": point.payload.get("predicted_by_model"),
            "price_at_ingestion": point.payload.get("price_at_ingestion"),
        }
        for point in results
    ]

    return {"items": items, "count": len(items)}
