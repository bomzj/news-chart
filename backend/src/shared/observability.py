import logging

from langfuse import Langfuse

from src.config import secrets

logger = logging.getLogger(__name__)

_client: Langfuse | None = None


def langfuse_tracing_enabled() -> bool:
    cfg = secrets()
    return bool(
        cfg.langfuse_tracing_enabled
        and cfg.langfuse_public_key
        and cfg.langfuse_secret_key
    )


def langfuse_client() -> Langfuse:
    global _client
    if _client is None:
        cfg = secrets()
        enabled = langfuse_tracing_enabled()

        if cfg.langfuse_tracing_enabled and not enabled:
            logger.info("Langfuse tracing disabled: API keys are not configured")

        _client = Langfuse(
            public_key=cfg.langfuse_public_key or "disabled",
            secret_key=cfg.langfuse_secret_key or "disabled",
            base_url=cfg.langfuse_base_url,
            environment=cfg.langfuse_tracing_environment,
            tracing_enabled=enabled,
        )

    return _client


def shutdown_langfuse() -> None:
    global _client
    if _client is not None:
        _client.shutdown()
        _client = None
