import asyncio
import logging

import httpx
from fastapi import APIRouter
from starlette.responses import JSONResponse

from src.config import app_config
from src.shared.binance import mark_price
from src.shared.embeddings import embed_texts
from src.shared.observability import langfuse_client
from src.shared.qdrant import ensure_collection
from src.shared.types import NewsRecord
from src.news_pipeline.fetch_news import fetch_news
from src.news_pipeline.dedup import deduplicate
from src.news_pipeline.agents import analyze_batch
from src.news_pipeline.models import AnalysisInput
from src.news_pipeline.store import store_news_batch

router = APIRouter()
logger = logging.getLogger(__name__)
_background_tasks: set[asyncio.Task[None]] = set()


async def _process_ticker(ticker: str) -> tuple[int, int]:
    raw_news = await fetch_news(ticker)
    if not raw_news:
        return 0, 0

    with langfuse_client().start_as_current_observation(
        as_type="chain",
        name="deduplicate-news",
        input={"article_count": len(raw_news)},
    ) as dedup_observation:
        unique_news = await deduplicate(raw_news)
        dedup_observation.update(
            output={
                "unique_count": len(unique_news),
                "duplicate_count": len(raw_news) - len(unique_news),
            }
        )

    if not unique_news:
        return 0, 0

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
    discarded_count = len(unique_news) - len(kept)

    if not kept:
        return 0, discarded_count

    symbol = f"{ticker}USDT"
    try:
        current_price = await mark_price(symbol)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(
            "Skipping %d analyzed news items for %s because price lookup failed: %s",
            len(kept),
            ticker,
            exc,
        )
        return 0, discarded_count

    texts = [f"{n.title} {n.description}" for (n, _), _ in kept]
    embeddings = await embed_texts(texts)
    if len(embeddings) != len(kept):
        raise ValueError("Embedding count does not match analyzed news count")

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
        for (news, _), result in kept
    ]

    await store_news_batch(records, embeddings)
    return len(records), discarded_count


async def _run_pipeline():
    """Execute the full news pipeline in the background."""
    cfg = app_config()
    await ensure_collection()

    results = [await _process_ticker(ticker) for ticker in cfg.tickers]
    total_processed = sum(processed for processed, _ in results)
    discarded_count = sum(discarded for _, discarded in results)

    logger.info(
        "Pipeline finished: processed=%d discarded=%d tickers=%s",
        total_processed,
        discarded_count,
        cfg.tickers,
    )


def _report_pipeline_task(task: asyncio.Task[None]) -> None:
    try:
        task.result()
    except asyncio.CancelledError:
        logger.warning("News pipeline task was cancelled")
    except Exception:
        logger.exception("News pipeline failed before completion")


@router.post("/api/read-news")
async def read_news():
    """
    Trigger news pipeline in background, return 202 immediately.
    Keeps cron services happy (tiny response, no timeout).
    """
    task = asyncio.create_task(_run_pipeline())
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    task.add_done_callback(_report_pipeline_task)
    return JSONResponse(status_code=202, content={"status": "accepted"})
