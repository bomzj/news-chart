import logging

from src.config import secrets
from src.shared.http import http_client
from src.news_pipeline.extract import extract_full_texts
from src.news_pipeline.condense import condense_texts
from src.shared.types import RawNews

logger = logging.getLogger(__name__)


# Tickers that require Marketaux crypto prefix "CC:"
CRYPTO_TICKERS: set[str] = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "DOT", "AVAX", "LINK", "MATIC"}


def _marketaux_symbol(ticker: str) -> str:
    """Map system ticker to Marketaux symbol format. Crypto gets CC: prefix."""
    if ticker.upper() in CRYPTO_TICKERS:
        return f"CC:{ticker.upper()}"
    return ticker


async def fetch_news(ticker: str) -> list[RawNews]:
    """Fetch recent news from MarketAux API for a given ticker."""
    sec = secrets()
    symbol = _marketaux_symbol(ticker)

    client = await http_client()
    response = await client.get(
        "https://api.marketaux.com/v1/news/all",
        params={
            "symbols": symbol,
            "filter_entities": "true",
            "language": "en",
            "api_token": sec.marketaux_api_key,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    data = response.json()

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


def _parse_article(article: dict, full_text: str) -> RawNews:
    return RawNews(
        title=article.get("title", ""),
        description=article.get("description", ""),
        full_text=full_text,
        source=article.get("source", "unknown"),
        published_at=article["published_at"],
        url=article.get("url", ""),
    )
