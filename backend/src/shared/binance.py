import logging
import math
from bisect import bisect_left
from typing import Any

import httpx

from src.shared.http import http_client

logger = logging.getLogger(__name__)

FUTURES_API_URL = "https://fapi.binance.com"
SPOT_API_URL = "https://api.binance.com"
REQUEST_TIMEOUT = 10.0
KLINE_INTERVAL_MS = 60_000
MAX_KLINES_PER_REQUEST = 1000


def fallback_allowed(status_code: int) -> bool:
    """Return whether a Futures API response can use the Spot API fallback."""
    return status_code in {403, 418, 429, 451} or status_code >= 500


async def request_json(url: str, params: dict[str, str | int]) -> Any:
    """Request and decode a Binance JSON response."""
    client = await http_client()
    response = await client.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


async def request_with_source_fallback(
    futures_path: str,
    spot_path: str,
    params: dict[str, str | int],
) -> tuple[Any, bool]:
    """Request market data and return the response with whether Futures was used."""
    try:
        return await request_json(f"{FUTURES_API_URL}{futures_path}", params), True
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

    return await request_json(f"{SPOT_API_URL}{spot_path}", params), False


async def request_with_spot_fallback(
    futures_path: str,
    spot_path: str,
    params: dict[str, str | int],
) -> Any:
    """Request Futures market data, falling back to the equivalent Spot route."""
    data, _ = await request_with_source_fallback(futures_path, spot_path, params)
    return data


def price_value(value: Any, source: str) -> float:
    """Validate and convert a Binance price value."""
    try:
        price = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid price returned by Binance {source}") from exc

    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"Invalid price returned by Binance {source}")
    return price


def kline_open_time(timestamp_ms: int) -> int:
    """Return the first one-minute kline boundary at or after a timestamp."""
    return ((timestamp_ms + KLINE_INTERVAL_MS - 1) // KLINE_INTERVAL_MS) * KLINE_INTERVAL_MS


def kline_batches(timestamps_ms: list[int]) -> list[list[int]]:
    """Group timestamps into requests containing at most 1000 one-minute klines."""
    ordered_timestamps = sorted(set(timestamps_ms))
    batches: list[list[int]] = []
    current_batch: list[int] = []
    first_open_time = 0

    for timestamp_ms in ordered_timestamps:
        open_time = kline_open_time(timestamp_ms)
        if (
            current_batch
            and open_time - first_open_time >= KLINE_INTERVAL_MS * MAX_KLINES_PER_REQUEST
        ):
            batches.append(current_batch)
            current_batch = []

        if not current_batch:
            first_open_time = open_time
        current_batch.append(timestamp_ms)

    if current_batch:
        batches.append(current_batch)

    return batches


def parse_kline_prices(
    klines: Any,
    timestamps_ms: list[int],
    symbol: str,
) -> dict[int, float]:
    """Map each requested timestamp to the first returned kline at or after it."""
    if not isinstance(klines, list):
        raise ValueError(f"Invalid kline response for {symbol}")

    open_times: list[int] = []
    close_prices: list[float] = []
    for kline in klines:
        if not isinstance(kline, list) or len(kline) < 5:
            raise ValueError(f"Invalid kline response for {symbol}")
        try:
            open_time = int(kline[0])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid kline response for {symbol}") from exc
        open_times.append(open_time)
        close_prices.append(price_value(kline[4], "kline endpoint"))

    prices: dict[int, float] = {}
    for timestamp_ms in timestamps_ms:
        index = bisect_left(open_times, timestamp_ms)
        if index < len(close_prices):
            prices[timestamp_ms] = close_prices[index]
    return prices


async def historical_prices(symbol: str, timestamps_ms: list[int]) -> dict[int, float]:
    """Fetch close prices for many timestamps using batched one-minute kline ranges."""
    if not timestamps_ms:
        return {}

    prices: dict[int, float] = {}
    use_futures = True
    for timestamp_batch in kline_batches(timestamps_ms):
        first_open_time = kline_open_time(timestamp_batch[0])
        last_open_time = kline_open_time(timestamp_batch[-1])
        limit = (last_open_time - first_open_time) // KLINE_INTERVAL_MS + 1
        params = {
            "symbol": symbol.upper(),
            "interval": "1m",
            "startTime": timestamp_batch[0],
            "limit": limit,
        }

        if use_futures:
            klines, use_futures = await request_with_source_fallback(
                futures_path="/fapi/v1/klines",
                spot_path="/api/v3/klines",
                params=params,
            )
        else:
            klines = await request_json(f"{SPOT_API_URL}/api/v3/klines", params)
        prices.update(parse_kline_prices(klines, timestamp_batch, symbol))

    return prices


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
    prices = await historical_prices(symbol, [timestamp_ms])
    if timestamp_ms not in prices:
        raise ValueError(f"No kline data for {symbol} at {timestamp_ms}")
    return prices[timestamp_ms]
