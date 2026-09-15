import logging
import math
from typing import Any

import httpx

from src.shared.http import http_client

logger = logging.getLogger(__name__)

FUTURES_API_URL = "https://fapi.binance.com"
SPOT_API_URL = "https://api.binance.com"
REQUEST_TIMEOUT = 10.0


def fallback_allowed(status_code: int) -> bool:
    """Return whether a Futures API response can use the Spot API fallback."""
    return status_code in {403, 418, 429, 451} or status_code >= 500


async def request_json(url: str, params: dict[str, str | int]) -> Any:
    """Request and decode a Binance JSON response."""
    client = await http_client()
    response = await client.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


async def request_with_spot_fallback(
    futures_path: str,
    spot_path: str,
    params: dict[str, str | int],
) -> Any:
    """Request Futures market data, falling back to the equivalent Spot route."""
    try:
        return await request_json(f"{FUTURES_API_URL}{futures_path}", params)
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        if not fallback_allowed(status_code):
            raise
        logger.warning(
            "Binance Futures returned HTTP %d for %s; using Spot fallback",
            status_code,
            futures_path,
        )
    except httpx.RequestError as exc:
        logger.warning(
            "Binance Futures request failed for %s (%s); using Spot fallback",
            futures_path,
            exc,
        )

    return await request_json(f"{SPOT_API_URL}{spot_path}", params)


def price_value(value: Any, source: str) -> float:
    """Validate and convert a Binance price value."""
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid price returned by Binance {source}") from exc

    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"Invalid price returned by Binance {source}")
    return price


async def mark_price(symbol: str = "BTCUSDT") -> float:
    """Fetch current mark price from Binance USDⓈ-M perpetual futures."""
    data = await request_with_spot_fallback(
        futures_path="/fapi/v1/premiumIndex",
        spot_path="/api/v3/ticker/price",
        params={"symbol": symbol.upper()},
    )
    if not isinstance(data, dict):
        raise ValueError("Invalid mark price response from Binance")
    return price_value(data.get("markPrice", data.get("price")), "price endpoint")


async def historical_price(symbol: str, timestamp_ms: int) -> float:
    """Fetch close price at a specific timestamp from Binance klines."""
    klines = await request_with_spot_fallback(
        futures_path="/fapi/v1/klines",
        spot_path="/api/v3/klines",
        params={
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": timestamp_ms,
            "limit": 1,
        },
    )

    if (
        not isinstance(klines, list)
        or not klines
        or not isinstance(klines[0], list)
        or len(klines[0]) < 5
    ):
        raise ValueError(f"No kline data for {symbol} at {timestamp_ms}")
    # kline[4] = close price
    return price_value(klines[0][4], "kline endpoint")
