from fastapi import APIRouter

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


@router.post("/api/read-news")
async def read_news():
    """
    Full news pipeline: fetch → dedup → analyze → filter noise → snapshot price → store.
    Triggered by cron every 10 minutes.
    """
    cfg = app_config()
    await ensure_collection()

    all_records: list[NewsRecord] = []
    all_embeddings: list[list[float]] = []
    discarded_count = 0

    for ticker in cfg.tickers:
        raw_news = await fetch_news(ticker)
        if not raw_news:
            continue

        # Deduplicate (intra-batch + RAG)
        unique_news = await deduplicate(raw_news)
        if not unique_news:
            continue

        # Prepare analysis inputs with context
        analysis_inputs = [
            AnalysisInput(raw_news=news, similar_context=context)
            for news, context in unique_news
        ]

        # Run junior/senior agents in parallel
        analysis_results = await analyze_batch(analysis_inputs)

        # Filter out noise (discarded by LLM)
        kept = [
            (news_ctx, result)
            for news_ctx, result in zip(unique_news, analysis_results)
            if result is not None
        ]
        discarded_count += len(unique_news) - len(kept)

        if not kept:
            continue

        # Snapshot current price
        symbol = f"{ticker}USDT"
        current_price = await mark_price(symbol)

        # Build records and get embeddings for storage
        texts = [f"{n.title} {n.description}" for (n, _), _ in kept]
        embeddings = await embed_texts(texts)

        for ((news, _), result), embedding in zip(kept, embeddings):
            record = NewsRecord(
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
            all_records.append(record)
            all_embeddings.append(embedding)

    # Batch store all
    if all_records:
        await store_news_batch(all_records, all_embeddings)

    return {
        "status": "ok",
        "processed": len(all_records),
        "discarded": discarded_count,
        "tickers": cfg.tickers,
    }
