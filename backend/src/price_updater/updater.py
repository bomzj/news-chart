from datetime import datetime, timezone, timedelta

from src.config import app_config, DeltaWindow
from src.shared.binance import historical_price
from src.shared.qdrant import scroll_with_filter, batch_update_payload


# Maps delta window config strings to timedelta and payload field name
DELTA_MAP: dict[DeltaWindow, tuple[timedelta, str]] = {
    "1h": (timedelta(hours=1), "realized_price_delta_pct_1h"),
    "24h": (timedelta(hours=24), "realized_price_delta_pct_24h"),
    "7d": (timedelta(days=7), "realized_price_delta_pct_7d"),
    "30d": (timedelta(days=30), "realized_price_delta_pct_30d"),
}


def price_delta_pct(price_at_ingestion: float, price_after: float) -> float:
    """Calculate price change as a fraction (e.g., 1% = 0.01, -5% = -0.05)."""
    return (price_after - price_at_ingestion) / price_at_ingestion


async def update_deltas_for_window(ticker: str, window: DeltaWindow) -> int:
    """
    Find news eligible for a specific delta window update, fetch historical price,
    calculate delta, and batch-update Qdrant payloads.
    Returns count of updated records.
    """
    delta_duration, field_name = DELTA_MAP[window]
    now = datetime.now(timezone.utc)

    # News must be old enough for this window AND not yet have this delta filled
    cutoff = now - delta_duration
    filter_conditions = {
        "ticker": ticker,
        field_name: {"is_null": True},
        "published_at": {"range": {"lte": cutoff.isoformat()}},
    }

    points = await scroll_with_filter(filter_conditions, limit=100)
    if not points:
        return 0

    symbol = f"{ticker}USDT"
    point_ids: list[str] = []
    payloads: list[dict] = []

    for point in points:
        payload = point.payload
        published_at = datetime.fromisoformat(payload["published_at"])
        target_time = published_at + delta_duration
        target_ts_ms = int(target_time.timestamp() * 1000)

        try:
            price_after = await historical_price(symbol, target_ts_ms)
        except (ValueError, Exception):
            continue

        delta = price_delta_pct(payload["price_at_ingestion"], price_after)
        point_ids.append(str(point.id))
        payloads.append({field_name: round(delta, 6)})

    if point_ids:
        await batch_update_payload(point_ids, payloads)

    return len(point_ids)


async def update_all_prices() -> dict[str, int]:
    """Run price delta updates for all tickers and all configured delta windows."""
    cfg = app_config()
    results: dict[str, int] = {}

    for ticker in cfg.tickers:
        for window in cfg.price_updater.deltas:
            key = f"{ticker}_{window}"
            count = await update_deltas_for_window(ticker, window)
            results[key] = count

    return results
