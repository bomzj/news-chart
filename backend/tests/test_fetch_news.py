import asyncio
from types import SimpleNamespace

import pytest
import httpx

from src.news_collector.fetch_news import CRYPTO_TICKERS, _marketaux_symbol, fetch_news
from src.news_collector.extract import extract_full_texts, _fetch_and_extract


class TestMarketauxSymbol:
    def test_crypto_ticker_gets_prefix(self):
        assert _marketaux_symbol("BTC") == "CC:BTC"
        assert _marketaux_symbol("ETH") == "CC:ETH"
        assert _marketaux_symbol("SOL") == "CC:SOL"

    def test_crypto_ticker_case_insensitive(self):
        assert _marketaux_symbol("btc") == "CC:BTC"
        assert _marketaux_symbol("Eth") == "CC:ETH"

    def test_non_crypto_ticker_unchanged(self):
        assert _marketaux_symbol("AAPL") == "AAPL"
        assert _marketaux_symbol("TSLA") == "TSLA"

    def test_all_known_crypto_tickers_mapped(self):
        for ticker in CRYPTO_TICKERS:
            assert _marketaux_symbol(ticker) == f"CC:{ticker}"


class TestFetchNews:
    async def test_retries_marketaux_timeout(self, monkeypatch):
        class FlakyClient:
            calls = 0

            async def get(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise httpx.ReadTimeout("timed out")
                return httpx.Response(
                    200,
                    json={"data": []},
                    request=httpx.Request("GET", "https://api.marketaux.com/v1/news/all"),
                )

        client = FlakyClient()

        async def fake_http_client():
            return client

        sleep = asyncio.sleep
        monkeypatch.setattr("src.news_collector.fetch_news.secrets", lambda: SimpleNamespace(marketaux_api_key="test"))
        monkeypatch.setattr("src.news_collector.fetch_news.http_client", fake_http_client)
        monkeypatch.setattr("src.news_collector.fetch_news.asyncio.sleep", lambda _: sleep(0))

        assert await fetch_news("BTC") == []
        assert client.calls == 2

    async def test_skips_ticker_after_repeated_marketaux_timeout(self, monkeypatch, caplog):
        class TimeoutClient:
            calls = 0

            async def get(self, *args, **kwargs):
                self.calls += 1
                raise httpx.ReadTimeout("timed out")

        client = TimeoutClient()

        async def fake_http_client():
            return client

        sleep = asyncio.sleep
        monkeypatch.setattr("src.news_collector.fetch_news.secrets", lambda: SimpleNamespace(marketaux_api_key="test"))
        monkeypatch.setattr("src.news_collector.fetch_news.http_client", fake_http_client)
        monkeypatch.setattr("src.news_collector.fetch_news.asyncio.sleep", lambda _: sleep(0))

        with caplog.at_level("WARNING"):
            assert await fetch_news("BTC") == []

        assert client.calls == 2
        assert "skipping ticker" in caplog.text


class TestExtractFullTexts:
    async def test_unexpected_article_failure_does_not_abort_batch(self, monkeypatch):
        async def fail_for_one(url):
            if url == "bad":
                raise RuntimeError("unexpected extraction failure")
            return "extracted article text"

        monkeypatch.setattr("src.news_collector.extract._fetch_and_extract", fail_for_one)

        results = await extract_full_texts([{"url": "bad"}, {"url": "good"}])

        assert results == [None, "extracted article text"]

    async def test_skip_on_empty_url(self):
        articles = [{"url": "", "description": "fallback text", "snippet": "snip"}]
        results = await extract_full_texts(articles)
        assert results == [None]

    async def test_skip_on_missing_url(self):
        articles = [{"description": "desc only"}]
        results = await extract_full_texts(articles)
        assert results == [None]

    async def test_skip_on_http_error(self, httpx_mock):
        httpx_mock.add_response(url="https://example.com/article", status_code=500)
        articles = [{"url": "https://example.com/article", "description": "fallback"}]
        results = await extract_full_texts(articles)
        assert results == [None]

    async def test_skip_on_timeout(self, httpx_mock):
        httpx_mock.add_exception(
            httpx.ReadTimeout("timed out"),
            url="https://example.com/slow",
        )
        articles = [{"url": "https://example.com/slow", "description": "timeout fallback"}]
        results = await extract_full_texts(articles)
        assert results == [None]

    async def test_successful_extraction(self, httpx_mock):
        html = """
        <html><body>
        <article>
            <p>This is the full article text that should be extracted by trafilatura.
            It needs to be long enough to pass the 50-character minimum threshold check.</p>
        </article>
        </body></html>
        """
        httpx_mock.add_response(url="https://example.com/good", text=html)
        articles = [{"url": "https://example.com/good", "description": "short desc"}]
        results = await extract_full_texts(articles)
        # Should extract something longer than fallback
        assert len(results[0]) > 50

    async def test_skip_on_short_extraction(self, httpx_mock):
        html = "<html><body><p>Hi</p></body></html>"
        httpx_mock.add_response(url="https://example.com/tiny", text=html)
        articles = [{"url": "https://example.com/tiny", "description": "better fallback text"}]
        results = await extract_full_texts(articles)
        assert results == [None]
