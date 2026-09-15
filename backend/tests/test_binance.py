from src.shared.binance import historical_price, mark_price


async def test_mark_price_uses_futures_mark_price(httpx_mock):
    httpx_mock.add_response(
        url="https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
        json={"markPrice": "68250.123"},
    )

    assert await mark_price("btcusdt") == 68250.123


async def test_mark_price_falls_back_to_spot_after_rate_limit(httpx_mock):
    httpx_mock.add_response(
        url="https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
        status_code=418,
    )
    httpx_mock.add_response(
        url="https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
        json={"symbol": "BTCUSDT", "price": "68250.123"},
    )

    assert await mark_price() == 68250.123


async def test_historical_price_falls_back_to_spot_after_rate_limit(httpx_mock):
    params = "symbol=BTCUSDT&interval=1m&startTime=1710000000000&limit=1"
    httpx_mock.add_response(
        url=f"https://fapi.binance.com/fapi/v1/klines?{params}",
        status_code=418,
    )
    httpx_mock.add_response(
        url=f"https://api.binance.com/api/v3/klines?{params}",
        json=[[1710000000000, "68000", "68300", "67900", "68250.123", "1"]],
    )

    assert await historical_price("btcusdt", 1710000000000) == 68250.123
