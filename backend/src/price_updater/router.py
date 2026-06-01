import asyncio
import logging

from fastapi import APIRouter
from starlette.responses import JSONResponse

from src.price_updater.updater import update_all_prices

router = APIRouter()
logger = logging.getLogger(__name__)


async def _run_price_update():
    """Execute price backfill in the background."""
    results = await update_all_prices()
    total = sum(results.values())
    logger.info("Price update finished: updated=%d details=%s", total, results)


@router.post("/api/update-prices")
async def update_prices():
    """
    Trigger price backfill in background, return 202 immediately.
    Keeps cron services happy (tiny response, no timeout).
    """
    asyncio.create_task(_run_price_update())
    return JSONResponse(status_code=202, content={"status": "accepted"})
