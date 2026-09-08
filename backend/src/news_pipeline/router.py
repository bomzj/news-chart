import asyncio
import logging

from fastapi import APIRouter
from langfuse import propagate_attributes
from starlette.responses import JSONResponse

from src.config import app_config
from src.shared.binance import mark_price
from src.shared.embeddings import embed_texts
from src.shared.observability import langfuse_client
from src.shared.qdrant import COLLECTION_NAME, ensure_collection
from src.shared.types import NewsRecord
from src.news_pipeline.fetch_news import fetch_news
from src.news_pipeline.dedup import deduplicate
from src.news_pipeline.agents import analyze_batch
from src.news_pipeline.models import AnalysisInput
from src.news_pipeline.store import store_news_batch

router = APIRouter()
logger = logging.getLogger(__name__)


async def _process_ticker(ticker: str) -> tuple[int, int]:
    langfuse = langfuse_client()
    stats = {
        "fetched": 0,
        "unique": 0,
        "kept": 0,
        "discarded": 0,
        "stored": 0,
    }

    with langfuse.start_as_current_observation(
        as_type="chain",
        name="process-ticker",
        input={"ticker": ticker},
    ) as ticker_observation:
        with langfuse.start_as_current_observation(
            as_type="chain",
            name="collect-news",
            input={"ticker": ticker},
        ) as fetch_observation:
            raw_news = await fetch_news(ticker)
            stats["fetched"] = len(raw_news)
            fetch_observation.update(output={"article_count": len(raw_news)})

        if not raw_news:
            ticker_observation.update(output=stats)
            return 0, 0

        with langfuse.start_as_current_observation(
            as_type="chain",
            name="deduplicate-news",
            input={"article_count": len(raw_news)},
        ) as dedup_observation:
            unique_news = await deduplicate(raw_news)
            stats["unique"] = len(unique_news)
            dedup_observation.update(
                output={
                    "unique_count": len(unique_news),
                    "duplicate_count": len(raw_news) - len(unique_news),
                }
            )

        if not unique_news:
            ticker_observation.update(output=stats)
            return 0, 0

        analysis_inputs = [
            AnalysisInput(raw_news=news, similar_context=context)
            for news, context in unique_news
        ]

        with langfuse.start_as_current_observation(
            as_type="chain",
            name="analyze-news-batch",
            input={"article_count": len(analysis_inputs)},
        ) as analysis_observation:
            analysis_results = await analyze_batch(analysis_inputs)
            kept = [
                (news_ctx, result)
                for news_ctx, result in zip(unique_news, analysis_results)
                if result is not None
            ]
            stats["kept"] = len(kept)
            stats["discarded"] = len(unique_news) - len(kept)
            analysis_observation.update(
                output={
                    "kept_count": stats["kept"],
                    "discarded_count": stats["discarded"],
                }
            )

        if not kept:
            ticker_observation.update(output=stats)
            return 0, stats["discarded"]

        symbol = f"{ticker}USDT"
        with langfuse.start_as_current_observation(
            as_type="tool",
            name="fetch-mark-price",
            input={"symbol": symbol},
        ) as price_observation:
            current_price = await mark_price(symbol)
            price_observation.update(output={"price": current_price})

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

        with langfuse.start_as_current_observation(
            as_type="tool",
            name="store-news",
            input={"record_count": len(records)},
        ) as store_observation:
            await store_news_batch(records, embeddings)
            store_observation.update(output={"stored_count": len(records)})

        stats["stored"] = len(records)
        ticker_observation.update(output=stats)
        return stats["stored"], stats["discarded"]


async def _run_pipeline():
    """Execute the full news pipeline in the background."""
    cfg = app_config()
    langfuse = langfuse_client()

    with propagate_attributes(
        trace_name="process-news-pipeline",
        tags=["news-pipeline", "langgraph"],
        metadata={"trigger": "cron", "framework": "langgraph"},
    ):
        with langfuse.start_as_current_observation(
            as_type="chain",
            name="process-news-pipeline",
            input={"tickers": cfg.tickers},
        ) as pipeline_observation:
            with langfuse.start_as_current_observation(
                as_type="tool",
                name="prepare-news-store",
                input={"collection": COLLECTION_NAME},
            ) as store_observation:
                await ensure_collection()
                store_observation.update(output={"ready": True})

            results = [await _process_ticker(ticker) for ticker in cfg.tickers]
            total_processed = sum(processed for processed, _ in results)
            discarded_count = sum(discarded for _, discarded in results)

            pipeline_observation.update(
                output={
                    "processed_count": total_processed,
                    "discarded_count": discarded_count,
                }
            )

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
