import asyncio
import json

import pytest

import src.news_pipeline.router as news_router


class TestReportPipelineTask:
    async def test_logs_background_pipeline_failure(self, caplog):
        async def fail():
            raise RuntimeError("pipeline failure")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)

        with caplog.at_level("ERROR"):
            news_router._report_pipeline_task(task)

        assert "News pipeline failed before completion" in caplog.text

    async def test_logs_background_pipeline_cancellation(self, caplog):
        async def wait_forever():
            await asyncio.Event().wait()

        task = asyncio.create_task(wait_forever())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with caplog.at_level("WARNING"):
            news_router._report_pipeline_task(task)

        assert "News pipeline task was cancelled" in caplog.text


class TestReadNews:
    async def test_skips_overlapping_pipeline(self, monkeypatch):
        started = asyncio.Event()
        release = asyncio.Event()

        async def run_pipeline():
            started.set()
            await release.wait()

        monkeypatch.setattr(news_router, "_run_pipeline", run_pipeline)

        first_response = await news_router.read_news()
        first_task = next(iter(news_router._background_tasks))

        try:
            second_response = await news_router.read_news()

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
