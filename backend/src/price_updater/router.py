import asyncio
import logging

from fastapi import APIRouter
from starlette.responses import JSONResponse

from src.price_updater.updater import update_all_prices

router = APIRouter()
logger = logging.getLogger(__name__)
price_update_tasks: set[asyncio.Task[None]] = set()
price_update_lock = asyncio.Lock()


async def _run_price_update():
    """Execute price backfill in the background."""
    if price_update_lock.locked():
        logger.warning("Skipping price update because another run is still in progress")
        return

    async with price_update_lock:
        results = await update_all_prices()
        total = sum(results.values())
        logger.info("Price update finished: updated=%d details=%s", total, results)


def report_price_update_task(task: asyncio.Task[None]) -> None:
    if task.cancelled():
        logger.warning("Price update task was cancelled")
        return

    error = task.exception()
    if error is not None:
        logger.error(
            "Price update failed before completion",
            exc_info=(type(error), error, error.__traceback__),
        )


@router.post("/api/update-prices")
async def update_prices():
    """
    Trigger price backfill in background, return 202 immediately.
    Keeps cron services happy (tiny response, no timeout).
    """
    task = asyncio.create_task(_run_price_update())
    price_update_tasks.add(task)
    task.add_done_callback(price_update_tasks.discard)
    task.add_done_callback(report_price_update_task)
    return JSONResponse(status_code=202, content={"status": "accepted"})
