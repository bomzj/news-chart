import asyncio
import logging

from fastapi import APIRouter
from starlette.responses import JSONResponse

from src.config import app_config
from src.shared.binance import mark_price
from src.shared.embeddings import embed_texts
from src.shared.qdrant import ensure_collection
from src.shared.types import NewsRecord
from src.news_pipeline.fetch_news import fetch_news
from src.news_pipeline.dedup import deduplicate
from src.news_pipeline.agents import analyze_batch
from src.news_pipeline.models import AnalysisInput
from src.news_pipeline.store import store_news_batch

router = APIRouter()
logger = logging.getLogger(__name__)


async def _run_pipeline():
    """Execute the full news pipeline in the background."""
    cfg = app_config()
    await ensure_collection()

    total_processed = 0
    discarded_count = 0

    for ticker in cfg.tickers:
        raw_news = await fetch_news(ticker)
        if not raw_news:
            continue

        unique_news = await deduplicate(raw_news)
        if not unique_news:
            continue

        analysis_inputs = [
            AnalysisInput(raw_news=news, similar_context=context)
            for news, context in unique_news
        ]

        analysis_results = await analyze_batch(analysis_inputs)

        kept = [
            (news_ctx, result)
            for news_ctx, result in zip(unique_news, analysis_results)
            if result is not None
        ]
        discarded_count += len(unique_news) - len(kept)

        if not kept:
            continue

        symbol = f"{ticker}USDT"
        current_price = await mark_price(symbol)

        texts = [f"{n.title} {n.description}" for (n, _), _ in kept]
        embeddings = await embed_texts(texts)

        records = [
            NewsRecord(
                ticker=ticker,
                source=news.source,
                published_at=news.published_at,
                news_summary=result.news_summary,
                news_full_text=news.full_text,
                sentiment=result.sentiment,
                impact=result.impact,
                confidence=result.confidence,
                predicted_by_model=result.predicted_by_model,
                price_at_ingestion=current_price,
            )
            for ((news, _), result), embedding in zip(kept, embeddings)
        ]

        await store_news_batch(records, embeddings)
        total_processed += len(records)

    logger.info(
        "Pipeline finished: processed=%d discarded=%d tickers=%s",
        total_processed,
        discarded_count,
        cfg.tickers,
    )


@router.post("/api/read-news")
async def read_news():
    """
    Trigger news pipeline in background, return 202 immediately.
    Keeps cron services happy (tiny response, no timeout).
    """
    asyncio.create_task(_run_pipeline())
    return JSONResponse(status_code=202, content={"status": "accepted"})
