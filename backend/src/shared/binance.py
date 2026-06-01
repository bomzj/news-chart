from src.shared.http import http_client


async def mark_price(symbol: str = "BTCUSDT") -> float:
    """Fetch current mark price from Binance USDⓈ-M perpetual futures."""
    client = await http_client()
    response = await client.get(
        "https://fapi.binance.com/fapi/v1/premiumIndex",
        params={"symbol": symbol},
        timeout=10.0,
    )
    response.raise_for_status()
    return float(response.json()["markPrice"])


async def historical_price(symbol: str, timestamp_ms: int) -> float:
    """Fetch close price at a specific timestamp from Binance klines."""
    client = await http_client()
    response = await client.get(
        "https://fapi.binance.com/fapi/v1/klines",
        params={
            "symbol": symbol,
            "interval": "1m",
            "startTime": timestamp_ms,
            "limit": 1,
        },
        timeout=10.0,
    )
    response.raise_for_status()
    klines = response.json()
    if not klines:
        raise ValueError(f"No kline data for {symbol} at {timestamp_ms}")
    # kline[4] = close price
    return float(klines[0][4])
