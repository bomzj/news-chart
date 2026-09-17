import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.news_collector.router import router as news_router
from src.price_updater.router import router as price_router
from src.chart.router import router as chart_router
from src.shared.azure_ai import close_azure_ai_clients
from src.shared.http import close_http_client
from src.shared.observability import shutdown_langfuse
from src.shared.qdrant import ensure_collection

logger = logging.getLogger(__name__)

app = FastAPI(
    title="News Chart API",
    description="Crypto news collector with AI sentiment scoring and price tracking",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(news_router)
app.include_router(price_router)
app.include_router(chart_router)


@app.on_event("startup")
async def startup():
    try:
        await ensure_collection()
    except Exception as exc:
        logger.warning("Failed to ensure Qdrant indexes on startup: %s", exc)


@app.on_event("shutdown")
async def shutdown():
    await close_azure_ai_clients()
    await close_http_client()
    shutdown_langfuse()


@app.get("/health")
@app.get("/api/heartbeat")
async def heartbeat():
    """Return a dependency-free liveness response for uptime monitors."""
    return {"status": "ok"}
