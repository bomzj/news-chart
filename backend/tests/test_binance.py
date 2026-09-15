from src.shared.binance import historical_price, historical_prices, mark_price


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


async def test_historical_prices_batches_timestamps(httpx_mock):
    start_time = 1710000000000
    params = "symbol=BTCUSDT&interval=1m&startTime=1710000000000&limit=3"
    httpx_mock.add_response(
        url=f"https://fapi.binance.com/fapi/v1/klines?{params}",
        json=[
            [start_time, "68000", "68300", "67900", "68250.123", "1"],
            [start_time + 60000, "68250", "68400", "68100", "68300.000", "1"],
            [start_time + 120000, "68300", "68500", "68200", "68450.000", "1"],
        ],
    )

    prices = await historical_prices(
        "btcusdt",
        [start_time, start_time + 60000, start_time + 120000],
    )

    assert prices == {
        start_time: 68250.123,
        start_time + 60000: 68300.0,
        start_time + 120000: 68450.0,
    }
    assert len(httpx_mock.get_requests()) == 1


async def test_historical_prices_stays_on_spot_after_futures_rate_limit(httpx_mock):
    start_time = 1710000000000
    second_start_time = start_time + 1000 * 60000
    first_params = "symbol=BTCUSDT&interval=1m&startTime=1710000000000&limit=1"
    second_params = (
        "symbol=BTCUSDT&interval=1m&startTime=1710060000000&limit=1"
    )
    httpx_mock.add_response(
        url=f"https://fapi.binance.com/fapi/v1/klines?{first_params}",
        status_code=418,
    )
    httpx_mock.add_response(
        url=f"https://api.binance.com/api/v3/klines?{first_params}",
        json=[[start_time, "68000", "68300", "67900", "68250.123", "1"]],
    )
    httpx_mock.add_response(
        url=f"https://api.binance.com/api/v3/klines?{second_params}",
        json=[[second_start_time, "68300", "68500", "68200", "68450.000", "1"]],
    )

    prices = await historical_prices("btcusdt", [start_time, second_start_time])

    assert prices[start_time] == 68250.123
    assert prices[second_start_time] == 68450.0
    assert [
        request.url.host
        for request in httpx_mock.get_requests()
    ] == ["fapi.binance.com", "api.binance.com", "api.binance.com"]
