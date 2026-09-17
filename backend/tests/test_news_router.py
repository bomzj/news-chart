import asyncio
import json

import pytest
from fastapi.routing import APIRoute

import src.news_collector.router as news_router


class TestReportCollectorTask:
    async def test_logs_background_collector_failure(self, caplog):
        async def fail():
            raise RuntimeError("collector failure")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)

        with caplog.at_level("ERROR"):
            news_router._report_collector_task(task)

        assert "News collector failed before completion" in caplog.text

    async def test_logs_background_collector_cancellation(self, caplog):
        async def wait_forever():
            await asyncio.Event().wait()

        task = asyncio.create_task(wait_forever())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with caplog.at_level("WARNING"):
            news_router._report_collector_task(task)

        assert "News collector task was cancelled" in caplog.text


class TestCollectNews:
    def test_exposes_canonical_and_legacy_routes(self):
        routes = {
            route.path: route
            for route in news_router.router.routes
            if isinstance(route, APIRoute)
        }

        assert routes["/api/collect-news"].endpoint is news_router.collect_news
        assert routes["/api/read-news"].endpoint is news_router.collect_news
        assert routes["/api/collect-news"].include_in_schema is True
        assert routes["/api/read-news"].include_in_schema is False

    async def test_skips_overlapping_collector(self, monkeypatch):
        started = asyncio.Event()
        release = asyncio.Event()

        async def run_collector():
            started.set()
            await release.wait()

        monkeypatch.setattr(news_router, "_run_collector", run_collector)

        first_response = await news_router.collect_news()
        first_task = next(iter(news_router._background_tasks))

        try:
            second_response = await news_router.collect_news()

            assert first_response.status_code == 202
            assert second_response.status_code == 202
            assert json.loads(second_response.body) == {"status": "already_running"}
            assert len(news_router._background_tasks) == 1

            await asyncio.sleep(0)
            assert started.is_set()
        finally:
            release.set()
            await first_task
            await asyncio.sleep(0)
