import asyncio
import gc
import logging

import trafilatura

from src.shared.http import http_client

logger = logging.getLogger(__name__)

_MAX_CONCURRENCY = 5
_FETCH_TIMEOUT = 15.0


async def extract_full_texts(articles: list[dict]) -> list[str | None]:
    """
    Fetch and extract full article text from URLs in parallel.
    Returns None for articles whose URL is unavailable (4xx, 5xx, timeout).
    Runs gc.collect() after all extractions to free lxml DOM trees retained by trafilatura.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENCY)

    async def _extract_one(article: dict) -> str | None:
        url = article.get("url", "")

        if not url:
            logger.warning("Article '%s' has no URL, skipping", article.get("title", "unknown"))
            return None

        async with semaphore:
            return await _fetch_and_extract(url)

    results = await asyncio.gather(*[_extract_one(a) for a in articles])
    gc.collect()
    return results


async def _fetch_and_extract(url: str) -> str | None:
    """Fetch HTML page and extract main text content via trafilatura.
    Returns None when the URL is unreachable or content extraction fails."""
    try:
        client = await http_client()
        response = await client.get(
            url,
            timeout=_FETCH_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsPipeline/1.0)"},
        )
        response.raise_for_status()
        html = response.text

        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        del html

        if text and len(text) > 50:
            return text

        logger.warning("Extraction too short for %s, skipping", url)
        return None

    except Exception as exc:
        logger.warning("Failed to extract %s: %s, skipping", url, exc)
        return None
