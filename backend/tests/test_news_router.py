import asyncio

import pytest

from src.news_pipeline.router import _report_pipeline_task


class TestReportPipelineTask:
    async def test_logs_background_pipeline_failure(self, caplog):
        async def fail():
            raise RuntimeError("pipeline failure")

        task = asyncio.create_task(fail())
        await asyncio.sleep(0)

        with caplog.at_level("ERROR"):
            _report_pipeline_task(task)

        assert "News pipeline failed before completion" in caplog.text

    async def test_logs_background_pipeline_cancellation(self, caplog):
        async def wait_forever():
            await asyncio.Event().wait()

        task = asyncio.create_task(wait_forever())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        with caplog.at_level("WARNING"):
            _report_pipeline_task(task)

        assert "News pipeline task was cancelled" in caplog.text
