import asyncio
import logging

from src.config import app_config
from src.shared.azure_ai import azure_ai_client

logger = logging.getLogger(__name__)

CONDENSE_SYSTEM_PROMPT = """You are a news condensation assistant. Summarize the following news article text while preserving all key facts, entities, numbers, and quotes. Output ONLY the condensed text — no preamble, no labels."""


async def condense_texts(texts: list[str]) -> list[str]:
    """
    Condense texts exceeding max_full_text_chars via the Lite model.
    Short texts pass through unchanged.
    """
    limit = app_config().collector.max_full_text_chars

    async def _maybe_condense(text: str) -> str:
        if len(text) <= limit:
            return text
        return await _condense_single(text, limit)

    return await asyncio.gather(*[_maybe_condense(t) for t in texts])


async def _condense_single(text: str, limit: int) -> str:
    """Call the Lite model to summarize a single oversized article."""
    cfg = app_config().agents

    user_prompt = (
        f"Condense this article to under {limit} characters while keeping all important facts:\n\n{text}"
    )

    try:
        client = azure_ai_client(cfg.api_version)
        response = await client.responses.create(
            model=cfg.lite_model,
            input=[
                {"role": "system", "content": CONDENSE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            name="condense-article",
            metadata={"max_output_chars": limit},
            timeout=60.0,
        )

        if response.output_text:
            return response.output_text[:limit]

        logger.warning("No text output from condense LLM, truncating instead")
        return text[:limit]

    except Exception as exc:
        logger.warning("Condense LLM failed: %s, truncating instead", exc)
        return text[:limit]
