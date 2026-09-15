import asyncio
import logging

import httpx

from src.config import secrets
from src.shared.http import http_client
from src.news_pipeline.extract import extract_full_texts
from src.news_pipeline.condense import condense_texts
from src.shared.types import RawNews

logger = logging.getLogger(__name__)


# Tickers that require Marketaux crypto prefix "CC:"
CRYPTO_TICKERS: set[str] = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "LINK", "MATIC"}
_MARKETAUX_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_MARKETAUX_MAX_ATTEMPTS = 2
_MARKETAUX_RETRY_DELAY_SECONDS = 1.0


def _marketaux_symbol(ticker: str) -> str:
    """Map system ticker to Marketaux symbol format. Crypto gets CC: prefix."""
    if ticker.upper() in CRYPTO_TICKERS:
        return f"CC:{ticker.upper()}"
    return ticker


async def fetch_news(ticker: str) -> list[RawNews]:
    """Fetch recent news from MarketAux API for a given ticker."""
    sec = secrets()
    symbol = _marketaux_symbol(ticker)

    data = await _fetch_marketaux_data(symbol, sec.marketaux_api_key)
    if data is None:
        return []

    articles = data.get("data", [])
    full_texts = await extract_full_texts(articles)

    # Skip articles whose URL was unavailable (403, 404, timeout, etc.)
    available = [(a, t) for a, t in zip(articles, full_texts) if t is not None]
    skipped = len(articles) - len(available)
    if skipped:
        logger.info("Skipped %d/%d articles for %s (URL unavailable)", skipped, len(articles), ticker)
    if not available:
        return []

    articles_ok, texts_ok = zip(*available)
    condensed = await condense_texts(list(texts_ok))

    return [_parse_article(article, full_text) for article, full_text in zip(articles_ok, condensed)]


async def _fetch_marketaux_data(symbol: str, api_token: str) -> dict | None:
    """Fetch MarketAux data, retrying transient request failures once."""
    client = await http_client()
    params = {
        "symbols": symbol,
        "filter_entities": "true",
        "language": "en",
        "api_token": api_token,
    }

    for attempt in range(1, _MARKETAUX_MAX_ATTEMPTS + 1):
        try:
            response = await client.get(
                "https://api.marketaux.com/v1/news/all",
                params=params,
                timeout=_MARKETAUX_TIMEOUT,
            )
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.NetworkError) as exc:
            if attempt == _MARKETAUX_MAX_ATTEMPTS:
                logger.warning(
                    "MarketAux request failed for %s after %d attempts; skipping ticker: %s",
                    symbol,
                    attempt,
                    exc,
                )
                return None

            logger.warning(
                "MarketAux request failed for %s; retrying: %s",
                symbol,
                exc,
            )
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code not in {408, 425, 429} and status_code < 500:
                raise

            if attempt == _MARKETAUX_MAX_ATTEMPTS:
                logger.warning(
                    "MarketAux returned HTTP %d for %s after %d attempts; skipping ticker",
                    status_code,
                    symbol,
                    attempt,
                )
                return None

            logger.warning(
                "MarketAux returned HTTP %d for %s; retrying",
                status_code,
                symbol,
            )

        await asyncio.sleep(_MARKETAUX_RETRY_DELAY_SECONDS * attempt)

    return None


def _parse_article(article: dict, full_text: str) -> RawNews:
    return RawNews(
        title=article.get("title", ""),
        description=article.get("description", ""),
        full_text=full_text,
        source=article.get("source", "unknown"),
        published_at=article["published_at"],
        url=article.get("url", ""),
    )
