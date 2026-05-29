from fastapi import APIRouter

from src.price_updater.updater import update_all_prices

router = APIRouter()


@router.post("/api/update-prices")
async def update_prices():
    """
    Backfill realized price deltas for news that have been stored long enough.
    Triggered by cron every 15 minutes.
    """
    results = await update_all_prices()
    total = sum(results.values())
    return {"status": "ok", "updated": total, "details": results}
